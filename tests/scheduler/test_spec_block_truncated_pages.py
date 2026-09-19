"""A speculative block that ends early must not leave KV pages stranded.

``commit_verified`` advances ``cached_len`` over the WHOLE accepted block, assuming every
committed token gets appended. When EOS (or a stop string) lands among the accepted
candidates the commit loop breaks, ``input_ids`` stops short, and the request reaches
``cache_req`` with ``cached_len > len(input_ids)``. ``HybridRadixCache.insert`` then aligned
key AND pages down to the shorter key: the surplus pages went neither into the tree nor back
to the allocator, and the returned match length did not name them, so the caller could not
free them either. One page per occurrence, until ``check_integrity`` stopped the engine.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Req, SamplingParams
from freetoken.engine.speculative import commit_verified, open_draft_block
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.scheduler import Scheduler

NUM_PAGES = 64


def _pool(num_slots=16):
    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    return LinearStatePool(group=g, num_slots=num_slots, dtype=torch.bfloat16,
                           device=torch.device("cpu"), tp_size=1)


def _stage_pages(cm, req, upto):
    """Hand the request real pages for [cached_len, upto), with allocate_paged's accounting.

    Not allocate_paged itself: that writes the page table through a pinned host buffer
    (`pin = torch.cuda.is_available()`), so on a box with a GPU this CPU-only test would
    depend on a live CUDA context -- and one poisoned by an earlier GPU test in the same
    process would fail it for a reason that has nothing to do with page accounting.
    """
    n = upto - req.cached_len
    if n > 0:
        cm.page_table[req.table_idx, req.cached_len : upto] = cm._allocate(n)


def _pend(ids):
    t = torch.tensor(ids, dtype=torch.int32)
    return SimpleNamespace(input_ids=t, input_len=len(ids), mm_items=None,
                           cache_ns=None, cache_public_len=0)


def _truncated_req(cm, pool, prompt=(1, 2, 3, 4, 5, 6), drafted=4, accepted=3, eos_at=1):
    """A request left exactly where an EOS inside a speculative block leaves it.

    ``accepted`` candidates were committed, but the commit loop broke at index ``eos_at``,
    so only ``eos_at + 1`` tokens were ever appended to input_ids.
    """
    n = len(prompt)
    mr = cm.match_req(_pend(list(prompt)))
    req = Req(input_ids=torch.tensor(list(prompt), dtype=torch.int32), table_idx=0,
              cached_len=0, output_len=16, uid=0, sampling_params=SamplingParams(),
              cache_handle=mr.cuda_handle)
    cm.lock(mr.cuda_handle)
    req.linear_slot_idx = pool.alloc(1)[0]
    req.mamba_ping_pong = tuple(pool.alloc(2))

    # Mid-decode, as every other path leaves a request: pages for [0, n-1), the token at
    # n-1 pending (its KV is what the next forward computes).
    _stage_pages(cm, req, n - 1)
    req.cached_len, req.device_len = n - 1, n

    # the verify forward: pages for the pending token plus every candidate
    block = open_draft_block(req, drafted)
    _stage_pages(cm, req, req.device_len)
    rejected = commit_verified(req, block, accepted=accepted)
    cm.free_rejected_positions(req, rejected)

    # the commit loop appends one token per committed candidate and breaks on the EOS
    for _ in range(eos_at + 1):
        req.append_host(torch.tensor([99], dtype=torch.int32))
    return req


def _fresh():
    pool = _pool()
    page_table = torch.zeros(4, NUM_PAGES, dtype=torch.int32)
    cm = CacheManager(NUM_PAGES, 1, page_table, "hybrid_radix", linear_state_pool=pool)
    return cm, pool


def test_the_bad_state_is_refused_instead_of_silently_truncated():
    """Without the scheduler's rewind, cache_req must fail loudly, not leak."""
    cm, pool = _fresh()
    req = _truncated_req(cm, pool)
    assert req.cached_len > req.input_ids.numel()   # the state the defect produced
    with pytest.raises(AssertionError, match="pages -- the caller is naming pages"):
        cm.cache_req(req, finished=True)


def test_rewind_restores_the_plain_decode_invariant():
    cm, pool = _fresh()
    req = _truncated_req(cm, pool, accepted=3, eos_at=1)
    before = len(cm.free_slots)

    Scheduler._rewind_truncated_block(SimpleNamespace(cache_manager=cm), req)

    emitted = req.input_ids.numel()
    assert req.device_len == emitted            # what every other path assumes
    assert req.cached_len == emitted - 1        # the last token stays pending
    assert req.spec_block_truncated is True
    assert len(cm.free_slots) > before          # the stranded positions came back


def test_finishing_a_truncated_block_loses_no_page():
    cm, pool = _fresh()
    req = _truncated_req(cm, pool, accepted=3, eos_at=1)

    Scheduler._rewind_truncated_block(SimpleNamespace(cache_manager=cm), req)
    cm.cache_req(req, finished=True)

    cm.check_integrity()                        # the assertion that stopped the engine
    tree_pages = cm.prefix_cache.full_evictable + cm.prefix_cache.full_protected
    assert len(cm.free_slots) + tree_pages == NUM_PAGES


@pytest.mark.parametrize("accepted,eos_at", [(4, 0), (4, 1), (4, 2), (3, 0), (1, 0), (4, 4)])
def test_no_page_is_lost_wherever_the_eos_lands(accepted, eos_at):
    cm, pool = _fresh()
    req = _truncated_req(cm, pool, drafted=4, accepted=accepted, eos_at=eos_at)
    if req.cached_len > req.input_ids.numel():
        Scheduler._rewind_truncated_block(SimpleNamespace(cache_manager=cm), req)
    cm.cache_req(req, finished=True)

    cm.check_integrity()
    tree_pages = cm.prefix_cache.full_evictable + cm.prefix_cache.full_protected
    assert len(cm.free_slots) + tree_pages == NUM_PAGES


def test_a_truncated_block_does_not_donate_its_over_advanced_gdn_state():
    """rewind cannot rewind the GDN live state with cached_len, so it must not be donated."""
    cm, pool = _fresh()
    req = _truncated_req(cm, pool, accepted=3, eos_at=1)
    Scheduler._rewind_truncated_block(SimpleNamespace(cache_manager=cm), req)

    free_before = pool.num_free_slots
    cm.cache_req(req, finished=True)

    # all three slots (live + both ping-pong) came back; none went into the tree
    assert pool.num_free_slots == free_before + 3
    assert cm.prefix_cache.mamba_evictable_size + cm.prefix_cache.mamba_protected == 0
