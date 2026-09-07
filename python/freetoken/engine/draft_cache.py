"""A fixed-address KV cache for the DFlash2 sliding-window draft.

The draft forward is captured into a CUDA graph (``engine/draft_graph.py``), and a graph bakes
the address of every tensor it touches. transformers' ``DynamicCache`` grows by ``torch.cat``
and shrinks by slicing, so its K/V live at a new address after every block: a replay would
read whatever the capture-time allocation now holds. This cache allocates once and never
again; the draft's five layers write into it and read from it at the same addresses forever.

It is a ring over absolute positions: position ``p`` lives in slot ``p % ring``. The ring is
just wide enough to hold the window plus one block in flight, so the newest positions can
overwrite only positions the window no longer shows (the proof is at ``_ring_size``). Which
slots are live is tracked by ``slot_pos``, the position each slot holds, and the attention
mask is derived from it by the same position rule the eager model derives from index
distances (``dflash/model.py`` ``_attention_mask``): key visible iff ``|q_pos - k_pos| <
window``, plus ``k_pos <= q_pos`` on a causal draft. A retired or never-written slot holds
``NEG``, a position so far in the past that the rule masks it.

Every method is a handful of device kernels with no host synchronisation, which is what lets
the eager path and the captured graph run the identical sequence ``stage -> update x layers
-> retire`` on the same object. The eager path must use this cache too: a request whose first
block was drafted into a DynamicCache would replay its later blocks against an empty ring.

Duck-typed for the model: it implements only ``update(k, v, layer_idx, cache_kwargs)``, the
one method the DFlash layer calls (``dflash/model.py:396``).
"""

from __future__ import annotations

import torch

# What an unwritten or retired slot holds. Far enough below any real position that the
# distance rule masks it (|q - NEG| >= window for every q >= 0), and far enough from the
# int64 limits that q - NEG never overflows.
NEG = -(1 << 40)

DEFAULT_BLOCK = 8
_RING_ALIGN = 16


def _static_cache_supported(config) -> bool:
    """Whether every draft layer is a sliding-window layer over one integer window.

    The mask rule and the ring width assume one window shared by all layers. A full-attention
    layer (DFlash1) has no window: its keys never expire, and no ring can hold them.
    """
    window = getattr(config, "sliding_window", None)
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        return False
    layer_types = getattr(config, "layer_types", None)
    if not layer_types:
        # The model treats a missing layer_types as full_attention on every layer
        # (dflash/model.py:363-364).
        return False
    return all(layer_type == "sliding_attention" for layer_type in layer_types)


def _is_causal(config) -> bool:
    """The layer's causality rule of dflash/model.py:365-366, for a sliding-attention layer."""
    is_causal = getattr(config, "is_causal", None)
    return True if is_causal is None else bool(is_causal)


def _ring_size(window: int, block: int) -> int:
    """Slots needed so that staging never overwrites a key the window still shows.

    After staging a block at ``seq_len`` the live positions span ``[seq_len - window,
    seq_len + block)``: the staged rows reach down to ``seq_len - c`` with ``c <= window``,
    and row ``i`` can see down to ``seq_len + i - window + 1``. That is ``window + block``
    consecutive positions, which need ``window + block`` distinct slots. Rounded up to a
    multiple of 16 because torch's SDPA copies a boolean mask whose last dim is not
    16-aligned into a padded buffer on every call.
    """
    needed = window + 2 * block - 1
    return -(-needed // _RING_ALIGN) * _RING_ALIGN


class StaticDraftCache:
    """Fixed-address ring KV cache for a DFlash draft whose layers all share one window.

    Per layer: ``keys[l]`` and ``values[l]`` are ``[1, kv_heads, ring, head_dim]`` in the
    model's dtype, zero-filled at construction (a NaN in a masked slot would still poison
    ``P @ V``: 0 * NaN is NaN). ``slot_pos`` is ``int64 [ring]``, the position each slot
    holds or ``NEG``.

    A block is one ``stage(row_pos)``, then ``mask(row_pos[c:])``, then one ``update`` per
    layer from inside the model forward, then ``retire()``. ``reset()`` starts a new request.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        window: int,
        block: int = DEFAULT_BLOCK,
        causal: bool = False,
        device,
        dtype,
    ):
        if isinstance(window, bool) or int(window) <= 0 or int(block) <= 0:
            raise ValueError(f"window and block must be positive, got {window!r} and {block!r}")
        self.window: int = int(window)
        self.block: int = int(block)
        self.causal: bool = bool(causal)
        self.layers: int = int(num_layers)
        self.kv_heads: int = int(num_kv_heads)
        self.head_dim: int = int(head_dim)
        self.ring: int = _ring_size(self.window, self.block)
        self.device = torch.device(device)
        self.dtype = dtype

        shape = (1, self.kv_heads, self.ring, self.head_dim)
        self.keys: list[torch.Tensor] = [
            torch.zeros(shape, dtype=dtype, device=self.device) for _ in range(self.layers)
        ]
        self.values: list[torch.Tensor] = [
            torch.zeros(shape, dtype=dtype, device=self.device) for _ in range(self.layers)
        ]
        self.slot_pos: torch.Tensor = torch.full(
            (self.ring,), NEG, dtype=torch.int64, device=self.device
        )
        # Slots of the rows in flight, set by stage() and consumed by update() and retire().
        self._rows: torch.Tensor | None = None

    @classmethod
    def from_config(cls, config, *, device, dtype, block: int = DEFAULT_BLOCK) -> StaticDraftCache:
        """Build from the draft config, deriving the geometry the way the model does.

        The runner may instead pass the geometry explicitly, reading ``causal`` off the
        instantiated layers; both derive from the same config fields (dflash/model.py:363-367).
        """
        if not _static_cache_supported(config):
            raise ValueError(
                "StaticDraftCache needs an integer sliding_window and sliding_attention on "
                "every layer; DFlash1-style full-attention drafts keep the DynamicCache"
            )
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads
        return cls(
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            window=config.sliding_window,
            block=block,
            causal=_is_causal(config),
            device=device,
            dtype=dtype,
        )

    def stage(self, row_pos: torch.Tensor) -> None:
        """Claim the slots of this block's rows: ``c`` context rows then ``block`` noise rows.

        ``row_pos`` is ``int64 [c + block]`` of consecutive positions ``[seq_len - c, seq_len
        + block)``. Only the length is inspected on the host; the caller (the runner)
        truncates a first block longer than the window, which is what keeps the slots unique.
        """
        rows = row_pos.shape[0]
        if row_pos.dim() != 1 or rows < self.block or rows > self.window + self.block:
            raise ValueError(
                f"stage() expects int64 [c + {self.block}] with 0 <= c <= {self.window}, "
                f"got shape {tuple(row_pos.shape)}"
            )
        slots = row_pos % self.ring
        self.slot_pos.index_copy_(0, slots, row_pos)
        self._rows = slots

    def mask(self, q_pos: torch.Tensor) -> torch.Tensor:
        """The bool attention mask ``[1, 1, len(q_pos), ring]`` of the staged ring.

        ``q_pos`` is ``int64 [block]`` = ``seq_len + arange(block)``. Same visible set as the
        index rule the model would derive from a contiguous DynamicCache (P2 of the plan):
        with contiguous positions, index distance equals position distance.
        """
        d = q_pos[:, None] - self.slot_pos[None, :]
        visible = (d < self.window) & (d > -self.window)
        if self.causal:
            visible &= d >= 0
        return visible[None, None]

    def update(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int, cache_kwargs=None):
        """Write this block's rows of layer ``layer_idx`` and return the full static K/V.

        ``k`` and ``v`` are ``[1, kv_heads, c + block, head_dim]`` in the ring's dtype, rows
        in the order of the staged positions. The model reads the returned views through
        the mask; their address never changes, which is what the graph relies on.
        """
        if self._rows is None:
            raise RuntimeError("update() before stage(): no rows are in flight")
        keys, values = self.keys[layer_idx], self.values[layer_idx]
        keys.index_copy_(2, self._rows, k)
        values.index_copy_(2, self._rows, v)
        return keys, values

    def retire(self) -> None:
        """Forget the ``block`` noise rows of the staged block: today's ``crop(-block)``.

        Only ``slot_pos`` changes; the stale K/V stay in the ring, finite and masked, until
        a later position lands on the slot.
        """
        if self._rows is None:
            raise RuntimeError("retire() before stage(): no rows are in flight")
        self.slot_pos.index_fill_(0, self._rows[-self.block :], NEG)
        self._rows = None

    def reset(self) -> None:
        """Empty the ring for a new request without touching K/V or reallocating.

        Marking every slot NEG is enough: the mask hides the old K/V, and they are finite,
        so nothing they hold can leak into the attention output.
        """
        self.slot_pos.fill_(NEG)
        self._rows = None
