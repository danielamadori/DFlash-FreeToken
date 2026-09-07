"""The static ring KV cache of the DFlash2 draft sees what the eager DynamicCache saw.

The draft forward will be replayed from a CUDA graph, so its KV cache must keep one address
for the life of the runner. The ring cache replaces transformers' growing DynamicCache and
derives the attention mask from stored positions instead of from tensor shapes. What has to
hold is that, block after block -- through the ring wrap, a first block longer than the
window, a request starting mid-sequence after a prefix hit -- every query row sees exactly
the keys the eager model's index-distance rule (``dflash/model.py`` ``_attention_mask``)
would have shown it over the DynamicCache. The reference below is that eager pipeline in
miniature: a chronological list of positions cropped like ``DynamicSlidingWindowLayer``,
and the index rule copied from the model.

CPU tensors only; no weights, no model.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest
import torch

from freetoken.engine.draft_cache import NEG, StaticDraftCache, _static_cache_supported

BLOCK = 8


def _config(window=64, layers=2, kv_heads=2, head_dim=4, is_causal=False, layer_types="sliding"):
    if layer_types == "sliding":
        layer_types = ["sliding_attention"] * layers
    return SimpleNamespace(
        sliding_window=window,
        layer_types=layer_types,
        num_hidden_layers=layers,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        is_causal=is_causal,
    )


def _cache(**kw) -> StaticDraftCache:
    return StaticDraftCache.from_config(_config(**kw), device="cpu", dtype=torch.float32)


# --- the eager reference -----------------------------------------------------------------


def _index_rule(kv_len: int, q_len: int, *, is_causal: bool, window: int) -> torch.Tensor:
    """dflash/model.py ``_attention_mask`` over lengths alone: bool [q_len, kv_len]."""
    query_position = kv_len - q_len + torch.arange(q_len)[:, None]
    key_position = torch.arange(kv_len)[None, :]
    visible = torch.ones((q_len, kv_len), dtype=torch.bool)
    if is_causal:
        visible &= key_position <= query_position
    visible &= query_position - key_position < window
    if not is_causal:
        visible &= key_position - query_position < window
    return visible


class _EagerReference:
    """The DynamicCache pipeline over positions: update returns stored + new, the model masks
    by index distance, crop(-block) drops the noise rows and keeps the last window - 1 keys
    (transformers cache_utils.py DynamicSlidingWindowLayer.update/crop)."""

    def __init__(self, window: int, is_causal: bool):
        self.window = window
        self.is_causal = is_causal
        self.stored = torch.empty(0, dtype=torch.int64)

    def block(self, row_pos: torch.Tensor) -> list[set[int]]:
        kv = torch.cat([self.stored, row_pos])
        visible = _index_rule(kv.numel(), BLOCK, is_causal=self.is_causal, window=self.window)
        rows = [set(kv[visible[i]].tolist()) for i in range(BLOCK)]
        self.stored = kv[:-BLOCK][-(self.window - 1) :]
        return rows


def _static_block(cache: StaticDraftCache, row_pos: torch.Tensor) -> list[set[int]]:
    """One draft block through the static cache, returning the visible positions per row."""
    cache.stage(row_pos)
    mask = cache.mask(row_pos[-BLOCK:])
    assert mask.shape == (1, 1, BLOCK, cache.ring) and mask.dtype == torch.bool
    rows = [set(cache.slot_pos[mask[0, 0, i]].tolist()) for i in range(BLOCK)]
    for i in range(BLOCK):
        # A visible slot is a live one; the ring never shows a NEG or an overwritten slot.
        assert NEG not in rows[i]
        assert len(rows[i]) == int(mask[0, 0, i].sum())
    cache.retire()
    return rows


def _check_slot_invariant(cache: StaticDraftCache) -> None:
    """I2: a slot holds NEG or a position that maps to it."""
    live = cache.slot_pos != NEG
    slots = torch.arange(cache.ring)
    assert torch.equal(cache.slot_pos[live] % cache.ring, slots[live])


def _run_sequence(cache: StaticDraftCache, ref: _EagerReference, seq_len: int, cs: list[int]):
    """Feed blocks of ``cs`` context rows starting at ``seq_len``; the next block's context
    rows are the ones this block committed (scheduler: accepted + 1), so seq_len advances by
    the next c."""
    for n, c in enumerate(cs):
        if n > 0:
            seq_len += c
        row_pos = torch.arange(seq_len - c, seq_len + BLOCK, dtype=torch.int64)
        expected = ref.block(row_pos)
        # The runner feeds the static path at most `window` context rows (plan 2.2): the
        # rows beyond it are invisible to every query row anyway, and they would not fit
        # the ring.
        got = _static_block(cache, row_pos[-(min(c, cache.window) + BLOCK) :])
        assert got == expected, f"block {n} at seq_len {seq_len} c={c}"
        _check_slot_invariant(cache)


# --- (1) reference simulation ------------------------------------------------------------


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("is_causal", [False, True])
def test_visible_sets_match_eager_over_random_blocks(seed, is_causal):
    rng = random.Random(seed)
    window = 64
    cache = _cache(window=window, is_causal=is_causal)
    ref = _EagerReference(window, is_causal)
    first = rng.choice([1, 5, 8, 30, 63, 64, 65, 200, 300])
    start = rng.choice([first, first + 17, 1000])  # a prefix-gap start feeds only its own rows
    cs = [first] + [rng.randint(1, BLOCK) for _ in range(600)]  # positions past 6 x ring
    _run_sequence(cache, ref, start, cs)
    assert start + sum(cs[1:]) > 6 * cache.ring


def test_visible_sets_match_eager_at_production_geometry():
    """window 2048, ring 2064: a 3000-row first block (truncated to 2048 on the static side)
    and enough steady-state blocks to wrap the ring twice."""
    rng = random.Random(2064)
    cache = _cache(window=2048, layers=1, kv_heads=1, head_dim=1)
    assert cache.ring == 2064
    ref = _EagerReference(2048, False)
    cs = [3000] + [rng.randint(1, BLOCK) for _ in range(300)]
    _run_sequence(cache, ref, 3000, cs)
    assert 3000 + sum(cs[1:]) > 2 * cache.ring


def test_first_block_after_prefix_hit_sees_only_its_rows():
    """After a prefix hit the draft is fed the last chunk only; the ring holds nothing of the
    prefix and the mask must not invent it."""
    cache = _cache(window=64)
    ref = _EagerReference(64, False)
    _run_sequence(cache, ref, 1500, [5, 3, 8, 1, 1, 7])
    live = cache.slot_pos[cache.slot_pos != NEG]
    assert int(live.min()) == 1495


# --- (2) slot uniqueness -----------------------------------------------------------------


@pytest.mark.parametrize("rows", [BLOCK, BLOCK + 1, 64 + BLOCK - 1, 64 + BLOCK])
@pytest.mark.parametrize("start", [0, 7, 79, 80, 12345])
def test_staged_slots_are_unique(rows, start):
    cache = _cache(window=64)
    row_pos = torch.arange(start, start + rows, dtype=torch.int64)
    cache.stage(row_pos)
    slots = cache._rows
    assert slots.numel() == rows and torch.unique(slots).numel() == rows
    assert torch.equal(cache.slot_pos[slots], row_pos)


def test_stage_rejects_more_rows_than_the_ring_can_keep_apart():
    cache = _cache(window=64)
    with pytest.raises(ValueError):
        cache.stage(torch.arange(0, 64 + BLOCK + 1, dtype=torch.int64))
    with pytest.raises(ValueError):
        cache.stage(torch.arange(0, BLOCK - 1, dtype=torch.int64))
    with pytest.raises(ValueError):
        cache.stage(torch.arange(0, 12, dtype=torch.int64)[None])


# --- (3) retire and overwrite ------------------------------------------------------------


def test_retire_masks_exactly_the_noise_slots_and_stage_reclaims_them():
    cache = _cache(window=64)
    row_pos = torch.arange(100 - 3, 100 + BLOCK, dtype=torch.int64)
    cache.stage(row_pos)
    q_pos = row_pos[-BLOCK:]
    before = cache.mask(q_pos)[0, 0]
    assert bool(before[:, row_pos % cache.ring].all())

    cache.retire()
    after = cache.mask(q_pos)[0, 0]
    noise_slots = row_pos[-BLOCK:] % cache.ring
    ctx_slots = row_pos[:3] % cache.ring
    assert not after[:, noise_slots].any()
    assert bool(after[:, ctx_slots].all())
    assert torch.equal(cache.slot_pos[noise_slots], torch.full((BLOCK,), NEG, dtype=torch.int64))
    assert torch.equal(before & ~after, before ^ after)  # nothing else changed
    assert cache._rows is None

    # The next block (say 4 accepted: c = 5, seq_len 105) lands on the retired slots.
    next_pos = torch.arange(105 - 5, 105 + BLOCK, dtype=torch.int64)
    cache.stage(next_pos)
    assert torch.equal(cache.slot_pos[next_pos % cache.ring], next_pos)
    assert torch.equal(cache.slot_pos[ctx_slots], row_pos[:3])


def test_update_and_retire_need_a_staged_block():
    cache = _cache(window=64)
    k = torch.zeros(1, 2, BLOCK, 4)
    with pytest.raises(RuntimeError):
        cache.update(k, k, 0)
    with pytest.raises(RuntimeError):
        cache.retire()


# --- (4) reset keeps the addresses ---------------------------------------------------------


def test_reset_keeps_addresses_and_kv_contents():
    cache = _cache(window=64)
    ptrs = (
        [t.data_ptr() for t in cache.keys],
        [t.data_ptr() for t in cache.values],
        cache.slot_pos.data_ptr(),
    )
    row_pos = torch.arange(10, 10 + 2 + BLOCK, dtype=torch.int64)
    cache.stage(row_pos)
    k = torch.randn(1, 2, 2 + BLOCK, 4)
    cache.update(k, k, 1)
    cache.retire()
    cache.reset()
    assert (
        [t.data_ptr() for t in cache.keys],
        [t.data_ptr() for t in cache.values],
        cache.slot_pos.data_ptr(),
    ) == ptrs
    assert bool((cache.slot_pos == NEG).all())
    assert not cache.mask(row_pos[-BLOCK:]).any()
    # I5: K/V are neither reallocated nor zeroed; the mask alone hides them.
    assert torch.equal(cache.keys[1][:, :, row_pos % cache.ring], k)
    assert cache._rows is None


# --- (5) production shapes ---------------------------------------------------------------


def test_production_geometry_shapes():
    # The runner's explicit-geometry construction (draft_runner.py), production values.
    cache = StaticDraftCache(
        num_layers=5,
        num_kv_heads=8,
        head_dim=128,
        window=2048,
        block=8,
        causal=False,
        device="cpu",
        dtype=torch.bfloat16,
    )
    assert cache.ring == 2064 and cache.ring % 16 == 0
    assert len(cache.keys) == len(cache.values) == 5
    for keys, values in zip(cache.keys, cache.values):
        assert keys.shape == values.shape == (1, 8, 2064, 128)
        assert keys.dtype == values.dtype == torch.bfloat16
        assert not keys.any() and not values.any()
    assert cache.slot_pos.shape == (2064,) and cache.slot_pos.dtype == torch.int64
    assert bool((cache.slot_pos == NEG).all())
    q_pos = torch.arange(16, 24, dtype=torch.int64)
    cache.stage(torch.arange(15, 24, dtype=torch.int64))
    mask = cache.mask(q_pos)
    assert mask.shape == (1, 1, 8, 2064) and mask.dtype == torch.bool
    assert mask.shape[-1] % 16 == 0


def test_ring_is_the_smallest_16_aligned_width_over_window_plus_block():
    for window, expect in [(2048, 2064), (64, 80), (16, 32), (1, 16), (100, 128)]:
        cache = _cache(window=window, layers=1, kv_heads=1, head_dim=1)
        assert cache.ring == expect
        assert cache.ring >= window + BLOCK


# --- (6) update ----------------------------------------------------------------------------


def test_update_returns_the_same_views_and_writes_the_staged_rows():
    cache = _cache(window=64, layers=2, kv_heads=2, head_dim=4)
    row_pos = torch.arange(200 - 3, 200 + BLOCK, dtype=torch.int64)
    cache.stage(row_pos)
    k = torch.randn(1, 2, 3 + BLOCK, 4)
    v = torch.randn(1, 2, 3 + BLOCK, 4)
    k_out, v_out = cache.update(k, v, 1, {"sin": None, "cos": None, "cache_position": None})
    assert k_out is cache.keys[1] and v_out is cache.values[1]
    k_again, v_again = cache.update(k, v, 1)
    assert k_again is k_out and v_again is v_out

    slots = row_pos % cache.ring
    assert torch.equal(cache.keys[1][:, :, slots], k)
    assert torch.equal(cache.values[1][:, :, slots], v)
    untouched = torch.ones(cache.ring, dtype=torch.bool)
    untouched[slots] = False
    assert not cache.keys[1][:, :, untouched].any()
    assert not cache.values[1][:, :, untouched].any()
    assert not cache.keys[0].any()  # the other layer is not written


def test_block_runs_under_inference_mode():
    """draft() is decorated with inference_mode; the ring is allocated outside it."""
    cache = _cache(window=64)
    with torch.inference_mode():
        row_pos = torch.arange(5, 5 + 1 + BLOCK, dtype=torch.int64)
        cache.stage(row_pos)
        cache.mask(row_pos[-BLOCK:])
        k = torch.ones(1, 2, 1 + BLOCK, 4)
        for layer in range(2):
            cache.update(k, k, layer)
        cache.retire()
        cache.reset()
    assert not cache.keys[0].is_inference()


# --- (7) causal flag ---------------------------------------------------------------------


def test_causal_flag_follows_the_model_rule():
    # dflash/model.py:365-366: a missing is_causal makes a sliding layer causal.
    assert _cache(is_causal=None).causal is True
    assert _cache(is_causal=False).causal is False
    assert _cache(is_causal=True).causal is True


def test_causal_mask_hides_later_noise_rows():
    cache = _cache(window=64, is_causal=True)
    row_pos = torch.arange(40 - 2, 40 + BLOCK, dtype=torch.int64)
    cache.stage(row_pos)
    mask = cache.mask(row_pos[-BLOCK:])[0, 0]
    slots = row_pos % cache.ring
    for i in range(BLOCK):
        seen = mask[i, slots]
        assert bool(seen[: 2 + i + 1].all()) and not seen[2 + i + 1 :].any()
    non_causal = _cache(window=64, is_causal=False)
    non_causal.stage(row_pos)
    assert bool(non_causal.mask(row_pos[-BLOCK:])[0, 0][:, slots].all())


# --- gate ----------------------------------------------------------------------------------


def test_static_cache_supported_gate():
    assert _static_cache_supported(_config(window=2048, layers=5))
    assert not _static_cache_supported(_config(window=None))
    assert not _static_cache_supported(_config(window=True))
    assert not _static_cache_supported(_config(window=0))
    assert not _static_cache_supported(_config(layer_types=None))
    assert not _static_cache_supported(_config(layer_types=[]))
    assert not _static_cache_supported(
        _config(layer_types=["sliding_attention", "full_attention"])
    )
    assert not _static_cache_supported(SimpleNamespace())
    with pytest.raises(ValueError):
        StaticDraftCache.from_config(_config(layer_types=None), device="cpu", dtype=torch.float32)
    with pytest.raises(ValueError):
        StaticDraftCache(
            num_layers=1, num_kv_heads=1, head_dim=1, window=0, device="cpu", dtype=torch.float32
        )


def test_from_config_matches_explicit_geometry():
    config = _config(window=2048, layers=5, kv_heads=8, head_dim=128, is_causal=None)
    a = StaticDraftCache.from_config(config, device="cpu", dtype=torch.bfloat16)
    b = StaticDraftCache(
        num_layers=5, num_kv_heads=8, head_dim=128, window=2048, block=8, causal=True,
        device="cpu", dtype=torch.bfloat16,
    )
    assert (a.layers, a.kv_heads, a.head_dim, a.window, a.block, a.causal, a.ring) == (
        b.layers, b.kv_heads, b.head_dim, b.window, b.block, b.causal, b.ring
    )
    assert a.keys[0].shape == b.keys[0].shape == (1, 8, 2064, 128)
    # head_dim derived from hidden_size / num_attention_heads when the config omits it.
    bare = _config(window=64, layers=1, kv_heads=1, head_dim=None)
    bare.hidden_size, bare.num_attention_heads = 5120, 40
    assert StaticDraftCache.from_config(bare, device="cpu", dtype=torch.float32).head_dim == 128
