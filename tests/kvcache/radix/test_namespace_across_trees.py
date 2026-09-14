"""Il namespace si comporta uguale nei tre alberi, o dice chiaramente dove non arriva.

Il campo ``ns`` sta sul nodo, che e' condiviso da tutti e tre. Un albero che lo portasse e lo
ignorasse nel cammino sarebbe la peggiore delle tre situazioni: sembrerebbe isolare, e non lo
farebbe. Questo modulo tiene ferma la scelta fatta per ciascuno --

    HybridRadixCache   isola, e ha il doppio strato (e' quello servito qui)
    RadixPrefixCache   isola, e ha il doppio strato
    SWARadixCache      isola, e RIFIUTA il doppio strato invece di fingerlo: un taglio a
                       public_len dovrebbe dividere anche il tombstone e la corsa di finestra
                       del nodo, e nessun modello servito qui esercita quella finestra.
"""
from __future__ import annotations

import pytest
import torch

from freetoken.kvcache.hybrid_radix_cache import HybridRadixCache
from freetoken.kvcache.radix_cache import RadixPrefixCache
from freetoken.kvcache.swa_radix_cache import SWARadixCache

PAGE = 4


def ids(*pagine: int) -> torch.Tensor:
    return torch.tensor([p for p in pagine for _ in range(PAGE)], dtype=torch.int64)


def slots(n: int, base: int = 0) -> torch.Tensor:
    return torch.arange(base, base + n * PAGE, dtype=torch.int32)


def test_the_plain_tree_isolates_like_the_hybrid_one():
    c = RadixPrefixCache(torch.device("cpu"), page_size=PAGE)
    c.insert_prefix(ids(1, 2), slots(2), ns="alice")
    assert c.match_prefix(ids(1, 2), ns="alice").cuda_handle.cached_len == 2 * PAGE
    assert c.match_prefix(ids(1, 2), ns="bob").cuda_handle.cached_len == 0
    assert c.match_prefix(ids(1, 2)).cuda_handle.cached_len == 0, "nor does a caller with no name of its own"


def test_the_plain_tree_shares_the_public_prefix():
    c = RadixPrefixCache(torch.device("cpu"), page_size=PAGE)
    c.insert_prefix(ids(1, 2), slots(2), ns="alice", public_len=PAGE)
    assert c.match_prefix(ids(1, 3), ns="bob").cuda_handle.cached_len == PAGE, "the shared page"
    assert c.match_prefix(ids(1, 2), ns="bob").cuda_handle.cached_len == PAGE, "and not her second"


def test_the_plain_tree_reports_what_it_already_held():
    """The same distinction the hybrid tree lost twelve pages to."""
    c = RadixPrefixCache(torch.device("cpu"), page_size=PAGE)
    r = c.insert_prefix(ids(1, 2, 3), slots(3), ns="alice", public_len=PAGE)
    assert r.cached_len == 0, "the tree held nothing; nothing of this request was redundant"
    r = c.insert_prefix(ids(1, 9), slots(2, base=50), ns="bob", public_len=PAGE)
    assert r.cached_len == PAGE, "one page was already there"


def test_two_namespaces_with_the_same_tokens_do_not_clobber_each_other():
    """The bug a page-only child key hides: the second node takes the first one's slot."""
    for c in (RadixPrefixCache(torch.device("cpu"), page_size=PAGE),):
        c.insert_prefix(ids(1, 2), slots(2), ns="alice")
        c.insert_prefix(ids(1, 2), slots(2, base=100), ns="bob")
        assert c.match_prefix(ids(1, 2), ns="alice").cuda_handle.cached_len == 2 * PAGE
        assert c.match_prefix(ids(1, 2), ns="bob").cuda_handle.cached_len == 2 * PAGE


def test_the_swa_tree_isolates_too():
    c = SWARadixCache(torch.device("cpu"), page_size=PAGE, sliding_window_size=4 * PAGE)
    c.insert(ids(1, 2), slots(2), ns="alice")
    assert c.match_prefix(ids(1, 2), ns="alice").cached_len > 0
    assert c.match_prefix(ids(1, 2), ns="bob").cached_len == 0


@pytest.mark.parametrize("chiamata", ("match", "insert"))
def test_the_swa_tree_refuses_the_public_tier_rather_than_ignoring_it(chiamata):
    """Silently accepting it would leave a caller believing the system section was shared."""
    c = SWARadixCache(torch.device("cpu"), page_size=PAGE, sliding_window_size=4 * PAGE)
    with pytest.raises(NotImplementedError, match="no public tier"):
        if chiamata == "match":
            c.match_prefix(ids(1, 2), ns="alice", public_len=PAGE)
        else:
            c.insert(ids(1, 2), slots(2), ns="alice", public_len=PAGE)


def test_all_three_default_to_the_shared_tree():
    """Every deployment that does not partition must see exactly the tree it saw before."""
    h = HybridRadixCache(torch.device("cpu"), page_size=PAGE)
    h.insert(ids(1, 2), slots(2), mamba_value=7)
    assert h.match_prefix(ids(1, 2)).cached_len == 2 * PAGE

    r = RadixPrefixCache(torch.device("cpu"), page_size=PAGE)
    r.insert_prefix(ids(1, 2), slots(2))
    assert r.match_prefix(ids(1, 2)).cuda_handle.cached_len == 2 * PAGE

    s = SWARadixCache(torch.device("cpu"), page_size=PAGE, sliding_window_size=4 * PAGE)
    s.insert(ids(1, 2), slots(2))
    assert s.match_prefix(ids(1, 2)).cached_len > 0
