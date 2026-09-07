"""Rewinding the GDN recurrent state to the accepted prefix of a draft block.

A speculative block drafts N tokens and the target accepts k <= N of them. Full attention
recovers by freeing the KV pages of the rejected positions, but a gated-delta-net layer has
advanced a recurrent state N times, and that state carries no per-position structure to free:
after the verification forward it reflects all N tokens, while the sequence continues from k.
Restoring the pre-block snapshot is not the answer either, since that is the state after 0.

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
    """

    rescan: Callable[..., None]
    q: torch.Tensor        # [1, N, num_k_heads, head_k_dim]
    k: torch.Tensor        # [1, N, num_k_heads, head_k_dim]
    v: torch.Tensor        # [1, N, num_v_heads, head_v_dim]
    g: torch.Tensor        # [1, N, num_v_heads]
    beta: torch.Tensor     # [1, N, num_v_heads]
    conv_in: torch.Tensor  # [N, conv_dim] raw convolution input


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

    @property
    def recording(self) -> bool:
        """Whether layers should stash on this forward."""
        return self._scratch_slot is not None

    def open(self, live_slot: int) -> None:
        """Snapshot the live state of ``live_slot`` into a scratch slot and start recording."""
        if self._scratch_slot is not None:
            raise RuntimeError("GDNRollback.open() called twice without a rewind or close")
        (scratch,) = self._pool.alloc(1)
        self._pool.copy_from(live_slot, scratch)
        self._live_slot = live_slot
        self._scratch_slot = scratch
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
    ) -> None:
        """Record one layer's recurrence inputs for this block. No-op unless recording."""
        if self._scratch_slot is None:
            return
        self._stash[layer_id] = LayerStash(rescan, q, k, v, g, beta, conv_in)

    def rewind(self, accepted: int) -> None:
        """Rewind every recorded layer's state to just after ``accepted`` drafted tokens.

        ``accepted == N`` needs no work, but is still accepted here so the caller does not have
        to special-case it and risk leaking the scratch slot on the fully-accepted path.
        """
        if self._scratch_slot is None:
            raise RuntimeError("GDNRollback.rewind() without an open block")
        if accepted < 0:
            raise ValueError(f"accepted must not be negative, got {accepted}")
        live, scratch = self._live_slot, self._scratch_slot
        assert live is not None
        try:
            drafted = self._drafted_len()
            if drafted is None or accepted >= drafted:
                # Every drafted token was accepted: the live state already reflects exactly the
                # committed sequence. Rewinding here would be a no-op scan over all N rows, but
                # doing it anyway would burn a kernel launch per layer on the common good case.
                return
            # Back to the pre-block state, then forward again over the prefix that survived.
            self._pool.copy_from(scratch, live)
            for entry in self._stash.values():
                entry.rescan(
                    live_slot=live,
                    scratch_slot=scratch,
                    accepted=accepted,
                    stash=entry,
                )
        finally:
            self.close()

    def close(self) -> None:
        """Release the scratch slot and stop recording. Idempotent."""
        if self._scratch_slot is not None:
            self._pool.free([self._scratch_slot])
            self._scratch_slot = None
        self._live_slot = None
        self._stash.clear()

    def _drafted_len(self) -> int | None:
        for entry in self._stash.values():
            return int(entry.q.shape[1])
        return None


__all__ = ["GDNRollback", "LayerStash"]
