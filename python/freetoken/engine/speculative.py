"""Bookkeeping for one speculative decode step.

Kept apart from the engine and the scheduler on purpose: the arithmetic that decides
which positions survive a draft block, and which KV pages go back to the allocator, is
the part that corrupts a sequence when it is off by one, and it is also the only part
that can be tested without a GPU.

A decode step arrives with the request holding one uncommitted token::

    positions   0 .. cached_len-1   KV already in cache
    positions   cached_len          the token sampled last step, KV computed this forward
    device_len  = cached_len + 1

Drafting K candidates extends that window to ``cached_len + 1 + K``, so the verification
forward scores K + 1 positions: one per candidate plus the bonus that follows the last
accepted token. Accepting ``a`` of them commits ``a + 1`` tokens (the accepted drafts and
the target's own next token) and rejects the rest, whose pages must be freed.

The candidates live in the scheduler's token pool, which is what the forward reads; the
request's host ids are appended only for tokens that survive verification, through the same
per-token path a plain decode step uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from freetoken.core import Req


@dataclass(frozen=True)
class DraftBlock:
    """A drafted block that has been written into the request's host token buffer."""

    size: int
    first_position: int

    @property
    def positions(self) -> range:
        return range(self.first_position, self.first_position + self.size)


def open_draft_block(req: Req, drafted: int) -> DraftBlock:
    """Extend the request's verification window over `drafted` candidate positions.

    Only the device window moves. The candidates themselves live in the scheduler's token
    pool until they are accepted: the host ids stay the authoritative record of what the
    request has actually generated, so a rejected candidate never has to be un-said.

    The block is truncated to the remaining output budget -- and to leave room for the bonus
    token the target contributes -- so speculation never runs past max_device_len.
    """
    budget = req.max_device_len - req.device_len
    size = max(min(drafted, budget - 1), 0)
    if size == 0:
        return DraftBlock(size=0, first_position=req.device_len)

    first_position = req.device_len
    req.device_len += size
    return DraftBlock(size=size, first_position=first_position)


def commit_verified(req: Req, block: DraftBlock, accepted: int) -> range:
    """Keep the accepted prefix of the block; return the positions whose KV must be freed.

    Leaves the request exactly where a plain decode step would leave it after emitting
    ``accepted + 1`` tokens: everything up to the last accepted candidate is cached, and one
    position stays pending for the bonus token the caller appends through the normal
    token-commit path.

    The returned range is the positions computed for rejected candidates. Their pages are
    still in the request's page-table row and must be handed back by the caller, which owns
    the pools.
    """
    if not 0 <= accepted <= block.size:
        raise ValueError(f"accepted={accepted} outside the drafted block of {block.size}")

    new_cached_len = block.first_position + accepted
    rejected = range(new_cached_len, block.first_position + block.size)
    req.cached_len = new_cached_len
    req.device_len = new_cached_len + 1
    return rejected


def unsupported_reason(
    *,
    page_size: int,
    is_swa: bool,
    is_hybrid: bool,
    tp_size: int,
    overlap_scheduling: bool,
    hybrid_rollback: bool = False,
) -> str | None:
    """Why this configuration cannot run speculative decoding yet, or None if it can.

    Every one of these would need its own rollback path -- a page holding several tokens,
    the sliding-window and GDN state pools, a sharded vocabulary, and a scheduler that runs
    a step ahead of its own results. Refusing here keeps a half-supported configuration from
    looking like a working one with a poor acceptance rate.
    """
    if page_size != 1:
        return f"page_size={page_size}: rolling back a rejected candidate would free a page still holding accepted tokens"
    if is_swa:
        return "sliding-window KV: the swa pool slots of a rejected candidate need their own rollback"
    if is_hybrid and not hybrid_rollback:
        return "hybrid GDN state: the recurrent state advanced by a rejected candidate cannot be rewound"
    if tp_size != 1:
        return f"tensor parallel size {tp_size}: the draft runs against a sharded vocabulary"
    if overlap_scheduling:
        return "overlap scheduling: the draft needs the previous step's sampled token, which is still in flight"
    return None


__all__ = ["DraftBlock", "open_draft_block", "commit_verified", "unsupported_reason"]
