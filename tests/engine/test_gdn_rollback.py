"""The bookkeeping of rewinding a linear-attention state to a draft block's accepted prefix.

These cover the parts that do not need a GPU: that the pre-block snapshot is taken and given
back, that a fully accepted block does no work, and that a partly rejected one restores the
snapshot and rescans exactly the accepted rows. The numerical half -- that the rescanned state
equals the state of a non-speculative run -- cannot be asserted here and is checked by
comparing tokenised output against the baseline on a GPU.
"""

from __future__ import annotations

import torch

from freetoken.engine.gdn_rollback import GDNRollback


class FakePool:
    """Enough LinearStatePool to exercise the slot lifecycle: a free list and slot copies."""

    def __init__(self, num_slots: int = 8) -> None:
        self._free = list(range(1, num_slots))
        self.copies: list[tuple[int, int]] = []
        self.freed: list[int] = []

    def alloc(self, n: int = 1) -> list[int]:
        if n > len(self._free):
            raise RuntimeError("exhausted")
        return [self._free.pop() for _ in range(n)]

    def free(self, slots) -> None:
        for s in slots:
            self.freed.append(s)
            self._free.append(s)

    def copy_from(self, src: int, dst: int) -> None:
        self.copies.append((src, dst))

    @property
    def num_free(self) -> int:
        return len(self._free)


def _stash_into(rb: GDNRollback, layer_id: int, n: int, calls: list) -> None:
    """Record a layer whose verification window is ``n`` rows (pending token + candidates)."""

    def rescan(*, live_slot, scratch_slot, committed, stash):
        calls.append((layer_id, live_slot, scratch_slot, committed, int(stash.q.shape[1])))

    rb.stash(
        layer_id,
        rescan,
        q=torch.zeros(1, n, 2, 4),
        k=torch.zeros(1, n, 2, 4),
        v=torch.zeros(1, n, 2, 4),
        g=torch.zeros(1, n, 2),
        beta=torch.zeros(1, n, 2),
        conv_in=torch.zeros(n, 6),
        local_index=layer_id,
        head_k_dim=4,
    )


def test_open_snapshots_live_into_a_scratch_slot():
    pool = FakePool()
    rb = GDNRollback(pool)
    assert not rb.recording
    rb.open(live_slot=3)
    assert rb.recording
    assert len(pool.copies) == 1
    src, dst = pool.copies[0]
    assert src == 3 and dst != 3
    rb.close()


def test_the_scratch_slot_is_held_across_blocks_not_borrowed():
    """It competed with the donated-snapshot cache when borrowed, and the pool ran dry.

    The cache grows as requests complete, so a slot handed back between blocks is a slot the
    cache can take. Holding one for the scheduler's lifetime is what keeps a rewind possible on
    the hundredth block as much as the first.
    """
    pool = FakePool()
    before = pool.num_free
    rb = GDNRollback(pool)
    rb.open(live_slot=3)
    assert pool.num_free == before - 1
    rb.close()
    assert pool.num_free == before - 1, "close() must keep the slot"
    assert not rb.recording
    rb.open(live_slot=3)
    assert pool.num_free == before - 1, "and reopening must not take a second one"
    rb.release()
    assert pool.num_free == before


def test_close_is_idempotent_and_release_frees_once():
    pool = FakePool()
    rb = GDNRollback(pool)
    rb.open(live_slot=1)
    rb.close()
    rb.close()
    assert pool.freed == []
    rb.release()
    rb.release()
    assert len(pool.freed) == 1


def test_stash_is_ignored_when_not_recording():
    """A layer may run outside a speculative block; stashing then must not accumulate rows."""
    pool = FakePool()
    rb = GDNRollback(pool)
    calls: list = []
    _stash_into(rb, 0, 8, calls)
    rb.open(live_slot=2)
    rb.rewind(accepted=0)
    assert calls == []


def test_full_acceptance_does_no_rescan_and_no_restore():
    """The whole point of the fast path: an accepted block leaves the live state alone.

    A window of 8 rows carries 7 candidates, so accepting all 7 commits all 8 rows.
    """
    pool = FakePool()
    rb = GDNRollback(pool)
    calls: list = []
    rb.open(live_slot=5)
    for layer in (0, 1, 2):
        _stash_into(rb, layer, 8, calls)
    pool.copies.clear()
    rb.rewind(accepted=7)
    assert calls == []
    assert pool.copies == []      # no restore
    assert pool.freed == []       # and the scratch slot stays held for the next block


def test_partial_acceptance_restores_then_rescans_every_layer():
    pool = FakePool()
    rb = GDNRollback(pool)
    calls: list = []
    rb.open(live_slot=5)
    scratch = pool.copies[0][1]
    for layer in (0, 1, 2):
        _stash_into(rb, layer, 8, calls)
    pool.copies.clear()
    rb.rewind(accepted=3)
    # Restored from the pre-block snapshot, once, before any layer walks forward again.
    assert pool.copies == [(scratch, 5)]
    assert [c[0] for c in calls] == [0, 1, 2]
    for _, live, scr, committed, window in calls:
        # 3 accepted candidates commit 4 rows: them plus the token pending from the step before.
        assert (live, scr, committed, window) == (5, scratch, 4, 8)


def test_rejecting_everything_still_commits_the_pending_token():
    """accepted == 0 does not mean the state goes back to the pre-block snapshot untouched."""
    pool = FakePool()
    rb = GDNRollback(pool)
    calls: list = []
    rb.open(live_slot=4)
    scratch = pool.copies[0][1]
    _stash_into(rb, 0, 5, calls)
    pool.copies.clear()
    rb.rewind(accepted=0)
    assert pool.copies == [(scratch, 4)]
    # Rejecting every candidate still commits one row: the token pending from the step before.
    assert calls == [(0, 4, scratch, 1, 5)]


def test_a_failed_rescan_still_ends_the_block():
    """Otherwise the next open() raises "called twice" and the request never recovers."""
    pool = FakePool()
    rb = GDNRollback(pool)

    def boom(*, live_slot, scratch_slot, committed, stash):
        raise RuntimeError("kernel failed")

    rb.open(live_slot=1)
    rb.stash(
        0, boom,
        q=torch.zeros(1, 4, 2, 4), k=torch.zeros(1, 4, 2, 4), v=torch.zeros(1, 4, 2, 4),
        g=torch.zeros(1, 4, 2), beta=torch.zeros(1, 4, 2), conv_in=torch.zeros(4, 6),
        local_index=0, head_k_dim=4,
    )
    try:
        rb.rewind(accepted=2)
    except RuntimeError:
        pass
    assert not rb.recording, "a failed rescan must still end the block"
    rb.open(live_slot=1)  # and must leave the rollback usable for the next one


def test_the_window_carries_one_row_more_than_the_candidates():
    """The off-by-one that corrupted real output: committing k candidates advances k + 1 rows.

    A block of K candidates is verified over K + 1 positions -- the token sampled the step
    before, whose KV this forward computes, then the candidates. Walking the state forward by
    the number of ACCEPTED CANDIDATES leaves it one token behind the KV cache on every block,
    and the sequence starts re-emitting text it has already produced. The first version of this
    file asserted the wrong quantity here, which is why its tests passed while the 27B produced
    "Qual e'Qual e'Qual e'Qual e'Qual e'".
    """
    pool = FakePool()
    rb = GDNRollback(pool)
    calls: list = []
    rb.open(live_slot=2)
    _stash_into(rb, 0, 5, calls)  # window of 5 rows = 4 candidates
    rb.rewind(accepted=2)
    assert len(calls) == 1
    committed = calls[0][3]
    assert committed == 3, "2 accepted candidates commit 3 rows, not 2"


def test_accepting_every_candidate_commits_the_whole_window():
    """Boundary of the fast path: accepted == window - 1 means nothing needs rewinding."""
    pool = FakePool()
    rb = GDNRollback(pool)
    calls: list = []
    rb.open(live_slot=2)
    _stash_into(rb, 0, 5, calls)  # window of 5 rows = 4 candidates
    pool.copies.clear()
    rb.rewind(accepted=4)
    assert calls == [], "the whole window is committed; there is nothing to walk back to"
    assert pool.copies == []


def test_one_fused_call_replaces_the_per_layer_ones():
    """48 per-layer launches for a three-token rescan were almost entirely launch overhead.

    The layers are independent sequences over the same kernel, so the rewind issues one call
    for all of them; the per-layer path stays for anything that does not supply a fused one.
    """
    pool = FakePool()
    rb = GDNRollback(pool)
    per_layer: list = []
    fused_calls: list = []

    def fused(entries, *, live_slot, scratch_slot, committed):
        fused_calls.append((len(list(entries)), live_slot, committed))

    rb.open(live_slot=3)
    for layer in range(4):
        def rescan(*, live_slot, scratch_slot, committed, stash):
            per_layer.append(1)

        rb.stash(
            layer, rescan,
            q=torch.zeros(1, 5, 2, 4), k=torch.zeros(1, 5, 2, 4), v=torch.zeros(1, 5, 2, 4),
            g=torch.zeros(1, 5, 2), beta=torch.zeros(1, 5, 2), conv_in=torch.zeros(5, 6),
            local_index=layer, head_k_dim=4, fused=fused,
        )
    rb.rewind(accepted=2)

    assert per_layer == [], "the per-layer path must not run when a fused one is supplied"
    assert fused_calls == [(4, 3, 3)], "one call carrying every layer, committing accepted + 1"


def test_opening_twice_is_refused():
    pool = FakePool()
    rb = GDNRollback(pool)
    rb.open(live_slot=1)
    try:
        rb.open(live_slot=2)
    except RuntimeError as e:
        assert "twice" in str(e)
    else:
        raise AssertionError("a second open must not be silently accepted")
    rb.close()


def test_rewind_without_open_is_refused():
    rb = GDNRollback(FakePool())
    try:
        rb.rewind(accepted=1)
    except RuntimeError as e:
        assert "without an open block" in str(e)
    else:
        raise AssertionError("rewinding with no block must raise")
