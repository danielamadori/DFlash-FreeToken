"""Who may read a cached prefix back, and who may only learn that they cannot.

The KV of a prefix does not depend on who sent it -- the same tokens give the same tensors.
What depends on the sender is what a cache HIT means. Without an owner on the node, a session
can send a guessed prefix and read off the prefill length whether somebody else had already
sent it: an oracle over other people's prompts, and scripts/probe_cache_isolation.py measured
it at eight hits out of eight.

The other half is that isolation must not cost the shared prefix. The system section of the
prompt is byte-identical for every session, so sharing it leaks nothing, and re-prefilling it
per session would not fit anyway: the KV cache holds 92,693 tokens against a system prompt
with 81 MCP tools that runs to tens of thousands. Hence two tiers rather than two trees.
"""
from __future__ import annotations

import torch

from freetoken.kvcache.hybrid_radix_cache import HybridRadixCache

PAGE = 4


def _tutti(c: HybridRadixCache) -> list:
    """Every node of the tree, for assertions about ownership rather than about reuse."""
    out, stack = [], [c.root]
    while stack:
        n = stack.pop()
        out.append(n)
        stack.extend(n.children.values())
    return out


def _cache() -> HybridRadixCache:
    return HybridRadixCache(torch.device("cpu"), page_size=PAGE)


def ids(*pagine: int) -> torch.Tensor:
    """One distinct token repeated per page, which is how the page key is formed."""
    return torch.tensor([p for p in pagine for _ in range(PAGE)], dtype=torch.int64)


def slots(n: int, base: int = 0) -> torch.Tensor:
    return torch.arange(base, base + n * PAGE, dtype=torch.int32)


def test_without_a_namespace_nothing_changes():
    """The default has to be the old behaviour exactly, or every deployment that does not ask
    for isolation pays for it."""
    c = _cache()
    c.insert(ids(1, 2), slots(2), mamba_value=7)
    assert c.match_prefix(ids(1, 2)).cached_len == 2 * PAGE


def test_a_session_cannot_read_another_session_prefix_back():
    """The oracle, closed. B sends exactly what A sent and must still prefill it itself."""
    c = _cache()
    c.insert(ids(1, 2), slots(2), mamba_value=7, ns="alice")
    assert c.match_prefix(ids(1, 2), ns="alice").cached_len == 2 * PAGE, "her own, reusable"
    assert c.match_prefix(ids(1, 2), ns="bob").cached_len == 0, "his guess tells him nothing"


def test_the_public_prefix_stays_shared_across_sessions():
    """One page of system prompt, then private content. Bob reuses what Alice paid for and
    nothing more -- the whole point of two tiers instead of two trees.

    The public span is inserted on its own, with its own snapshot, because that is what the
    engine does: a prefill commits a snapshot every CHUNK_SIZE tokens
    (scheduler/cache.py, "the x64 prefill snapshots remain as reuse points"). A hybrid match
    can only resume from such a boundary, so a public node without one would be correct and
    useless.
    """
    c = _cache()
    c.insert(ids(1), slots(1), mamba_value=6)                       # the shared system section
    c.insert(ids(1, 2), slots(2), mamba_value=7, ns="alice", public_len=PAGE)

    assert c.match_prefix(ids(1, 3), ns="bob").cached_len == PAGE, "the shared page, no more"
    assert c.match_prefix(ids(1, 2), ns="bob").cached_len == PAGE, "not her second page"
    assert c.match_prefix(ids(1, 2), ns="alice").cached_len == 2 * PAGE, "hers in full"


def test_a_span_crossing_the_boundary_is_cut_not_swallowed():
    """A single insert covering system + private must not make the system half private too:
    the first session to arrive would otherwise take the shared prefix with it.

    Asserted on the tree rather than on a match, because the public half of a fresh insert
    carries no snapshot and a hybrid match cannot resume there yet. Ownership is the property
    under test; reusability arrives with the next x64 commit.
    """
    c = _cache()
    c.insert(ids(1, 2, 3), slots(3), mamba_value=7, ns="alice", public_len=2 * PAGE)
    pubblici = [n for n in _tutti(c) if n.ns is None and n is not c.root]
    privati = [n for n in _tutti(c) if n.ns == "alice"]
    assert sum(n.length for n in pubblici) == 2 * PAGE, "the system section stayed public"
    assert sum(n.length for n in privati) == PAGE, "and only her own page became hers"


def test_two_sessions_sending_the_same_tokens_do_not_clobber_each_other():
    """The case that matters, and the one a page-keyed children dict gets wrong.

    Bob sends exactly what Alice sent. His node has the same page key as hers, so filed under
    that key alone it would take her slot in the parent and orphan her subtree: her KV leaks
    and she loses a cache she is still entitled to. Both must survive, each reaching only its
    own.
    """
    c = _cache()
    c.insert(ids(1, 2), slots(2), mamba_value=7, ns="alice")
    assert c.match_prefix(ids(1, 2), ns="bob").cached_len == 0
    c.insert(ids(1, 2), slots(2, base=100), mamba_value=8, ns="bob")

    assert c.match_prefix(ids(1, 2), ns="alice").cached_len == 2 * PAGE, "hers survived"
    assert c.match_prefix(ids(1, 2), ns="bob").cached_len == 2 * PAGE, "and his exists"
    alice = c.match_prefix(ids(1, 2), ns="alice")
    bob = c.match_prefix(ids(1, 2), ns="bob")
    assert alice.mamba_value != bob.mamba_value, "two trees, two snapshots"
    assert not torch.equal(alice.kv_indices, bob.kv_indices), "and two sets of pages"
