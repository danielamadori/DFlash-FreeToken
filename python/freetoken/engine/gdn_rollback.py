"""Rewinding the GDN recurrent state to the accepted prefix of a draft block.

A speculative block drafts K tokens and the target accepts k <= K of them. Full attention
recovers by freeing the KV pages of the rejected positions, but a gated-delta-net layer has
advanced a recurrent state over the whole verification window, and that state carries no
per-position structure to free: afterwards it reflects every drafted position, while the
sequence continues from the accepted prefix. Restoring the pre-block snapshot is not the answer
either, since that is the state before any of it.

The window is K + 1 rows, not K: it opens with the token sampled the step before, whose KV this
forward computes (see engine/speculative.py). Committing k candidates therefore advances the
sequence by k + 1 rows, and a rewind that walks forward only k leaves the recurrent state one
token behind the KV cache on every block -- which reads as a sequence re-emitting what it has
already said, not as an error.

llama.cpp solves this by keeping one state snapshot per token and rolling back by index
(``llama-memory-recurrent.cpp``, ``n_rs_seq``). We cannot: the vendored fla chunk kernel emits
one state per chunk and ``CHUNK_SIZE`` is 64, so a draft block of 8 produces a single state.

So we re-run the recurrence over the accepted prefix instead -- but only the recurrence. The
projections, which is where a 27B model spends its time reading weights, are not repeated: the
verification forward already computed q/k/v/g/beta for every drafted position, and this module
keeps those rows so the rewind is a scan over k tokens with no weight traffic at all.

The cost of a rewind is one chunk-kernel launch per linear layer over k tokens. Blocks that are
fully accepted cost nothing: the live state is already correct and no rewind is issued.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, NamedTuple

import torch

if TYPE_CHECKING:
    from freetoken.kvcache.linear_state_pool import LinearStatePool


class LayerStash(NamedTuple):
    """One linear layer's inputs to the recurrence, for the positions of one draft block.

    These are the kernel's inputs after the convolution and the in-kernel l2norm, i.e. exactly
    what a rescan needs; ``conv_in`` is kept raw because the convolution state is a window over
    raw inputs, not over ``mixed``.

    ``W`` is the verification window -- the pending token plus K candidates, so K + 1 rows --
    not the number of candidates.
    """

    rescan: Callable[..., None]
    local_index: int       # this layer's row in the state pool
    head_k_dim: int
    q: torch.Tensor        # [1, W, num_k_heads, head_k_dim]
    k: torch.Tensor        # [1, W, num_k_heads, head_k_dim]
    v: torch.Tensor        # [1, W, num_v_heads, head_v_dim]
    g: torch.Tensor        # [1, W, num_v_heads]
    beta: torch.Tensor     # [1, W, num_v_heads]
    conv_in: torch.Tensor  # [W, conv_dim] raw convolution input


class GDNRollback:
    """Per-block recorder and rewinder for the linear-attention state of one request.

    Lifecycle, driven by the scheduler around a speculative block:
    ``open()`` before the verification forward, the layers ``stash()`` into it during that
    forward, then exactly one of ``rewind(k)`` or ``close()``.

    Restricted to a single request per block, which is what the speculative scheduler drives
    today; ``open()`` refuses anything else rather than silently rewinding one sequence's state
    with another's rows.
    """

    def __init__(self, pool: LinearStatePool) -> None:
        self._pool = pool
        self._stash: dict[int, LayerStash] = {}
        self._live_slot: int | None = None
        self._scratch_slot: int | None = None
        self._open = False
        self._fused: Callable[..., None] | None = None

    @property
    def recording(self) -> bool:
        """Whether layers should stash on this forward."""
        return self._open

    def open(self, live_slot: int) -> None:
        """Snapshot the live state of ``live_slot`` into the scratch slot and start recording.

        The scratch slot is taken once and held for the scheduler's lifetime, not borrowed per
        block. Borrowing put it in competition with the donated-snapshot cache, which fills as
        requests complete: the pool then ran out mid-run even after being sized one slot larger,
        because that one slot was exactly what the cache had grown into.
        """
        if self._open:
            raise RuntimeError("GDNRollback.open() called twice without a rewind or close")
        if self._scratch_slot is None:
            (self._scratch_slot,) = self._pool.alloc(1)
        self._pool.copy_from(live_slot, self._scratch_slot)
        self._live_slot = live_slot
        self._open = True
        self._stash.clear()

    def stash(
        self,
        layer_id: int,
        rescan: Callable[..., None],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        conv_in: torch.Tensor,
        local_index: int,
        head_k_dim: int,
        fused: Callable[..., None] | None = None,
    ) -> None:
        """Record one layer's recurrence inputs for this block. No-op unless recording."""
        if not self._open:
            return
        self._fused = fused
        self._stash[layer_id] = LayerStash(
            rescan, local_index, head_k_dim, q, k, v, g, beta, conv_in
        )

    def rewind(self, accepted: int) -> None:
        """Rewind every recorded layer's state to the tokens this block actually committed.

        A block of K candidates is verified over a window of K + 1 positions: the token sampled
        the step before, whose KV this forward computes, followed by the K candidates
        (see engine/speculative.py). Committing ``accepted`` of them advances the sequence by
        ``accepted + 1`` -- that pending token plus the accepted candidates -- so the state has
        to walk forward over ``accepted + 1`` rows of the window, not ``accepted``.

        Getting that wrong leaves the recurrent state one token behind the KV cache on every
        single block, and the sequence re-emits what it has already said.
        """
        if not self._open:
            raise RuntimeError("GDNRollback.rewind() without an open block")
        if accepted < 0:
            raise ValueError(f"accepted must not be negative, got {accepted}")
        live, scratch = self._live_slot, self._scratch_slot
        assert live is not None
        try:
            window = self._window_len()
            committed = accepted + 1
            if window is None or committed >= window:
                # The whole window was committed: the live state already reflects it exactly.
                # Rewinding would be a no-op scan, but it would still cost a kernel launch per
                # layer on the common good case.
                return
            # Back to the pre-block state, then forward again over the prefix that survived.
            self._pool.copy_from(scratch, live)
            if self._fused is not None:
                # One launch for every layer: they are independent sequences over the same
                # kernel, and per-layer calls were nearly all launch overhead.
                self._fused(
                    self._stash.values(),
                    live_slot=live,
                    scratch_slot=scratch,
                    committed=committed,
                )
            else:
                for entry in self._stash.values():
                    entry.rescan(
                        live_slot=live,
                        scratch_slot=scratch,
                        committed=committed,
                        stash=entry,
                    )
        finally:
            self.close()

    def close(self) -> None:
        """End the block and stop recording. Idempotent; the scratch slot is kept."""
        self._open = False
        self._live_slot = None
        self._stash.clear()

    def release(self) -> None:
        """Give the scratch slot back. For teardown, or a pool rebuild that reclaims slots."""
        self.close()
        if self._scratch_slot is not None:
            self._pool.free([self._scratch_slot])
            self._scratch_slot = None

    def _window_len(self) -> int | None:
        """Rows in the verification window: the pending token plus the drafted candidates."""
        for entry in self._stash.values():
            return int(entry.q.shape[1])
        return None


__all__ = ["GDNRollback", "LayerStash"]
