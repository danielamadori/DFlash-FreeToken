"""Host-built chunk bookkeeping for the fla chunk kernels, and the capture guard.

The kernels derive ``chunk_indices`` / ``chunk_offsets`` from the device ``cu_seqlens`` with a
``.tolist()`` -- a host sync per forward (two, since chunk_fwd_o tiles short inputs with its own
block size) and an invalid operation under CUDA-graph capture. ``build_fla_metadata`` now builds
them on the host and hands them down; these tests pin that the host formulas match the device
ones bit for bit, that the plumbing carries them without recomputing, and that a device-side
derivation under capture is refused instead of corrupting the graph.
"""

from __future__ import annotations

import contextlib
import random

import pytest
import torch

import freetoken.kernel.fla.chunk as fla_chunk
import freetoken.kernel.fla.utils as fla_utils
from freetoken.attention.linear import build_fla_chunk_indices, build_fla_metadata
from freetoken.core import Batch, Req
from freetoken.kernel.fla.chunk import CHUNK_SIZE
from freetoken.kernel.fla.chunk_o import chunk_fwd_o_block_size
from freetoken.kernel.fla.index import (
    chunk_indices_from_lens,
    chunk_offsets_from_lens,
    prepare_chunk_indices,
    prepare_chunk_offsets,
)

CPU = torch.device("cpu")


def _cu(lens):
    # A fresh tensor per call: the device functions cache by identity, and a cache hit would
    # compare the host formula against an earlier answer rather than the formula itself.
    return torch.tensor([0, *lens], dtype=torch.int64).cumsum_(0)


def _random_cases(seed: int = 0, n: int = 40):
    rng = random.Random(seed)
    for _ in range(n):
        yield [rng.randint(1, 200) for _ in range(rng.randint(1, 6))]


@pytest.mark.parametrize("bt", [16, 32, 64])
def test_host_indices_match_the_device_formula(bt):
    for lens in _random_cases():
        cu = _cu(lens)
        assert chunk_indices_from_lens(lens, bt) == prepare_chunk_indices(cu, bt).tolist()
        assert chunk_offsets_from_lens(lens, bt) == prepare_chunk_offsets(cu, bt).tolist()


def test_a_chunkless_sequence_takes_no_sequence_number():
    """The device formula numbers sequences with `eq(0).cumsum() - 1`, i.e. only sequences that
    own a chunk; a naive enumerate() would hand the third sequence index 2 here."""
    lens = [3, 0, 5]
    assert chunk_indices_from_lens(lens, 64) == prepare_chunk_indices(_cu(lens), 64).tolist()
    assert chunk_indices_from_lens(lens, 64) == [[0, 0], [1, 0]]
    assert chunk_offsets_from_lens(lens, 64) == [0, 1, 1, 2]


def test_build_fla_chunk_indices_matches_every_kernel_key():
    """chunk_fwd_o keys its own indices on a block size that depends on the row count."""
    for lens in _random_cases(seed=1):
        cu = _cu(lens)
        got = build_fla_chunk_indices(lens, CPU, pin_memory=False)
        bt_o = chunk_fwd_o_block_size(sum(lens), CHUNK_SIZE)
        assert torch.equal(got["chunk_indices"], prepare_chunk_indices(cu, CHUNK_SIZE))
        assert torch.equal(got["chunk_indices_o"], prepare_chunk_indices(cu, bt_o))
        assert torch.equal(got["chunk_offsets"], prepare_chunk_offsets(cu, CHUNK_SIZE))
        for t in got.values():
            assert t.dtype == torch.int64
        assert got["chunk_indices"].shape[1] == 2


def test_the_verify_block_bookkeeping():
    """One sequence of 8 rows: one chunk under both block sizes (64, and chunk_fwd_o's 16)."""
    got = build_fla_chunk_indices([8], CPU, pin_memory=False)
    assert got["chunk_indices"].tolist() == [[0, 0]]
    assert got["chunk_indices_o"].tolist() == [[0, 0]]
    assert got["chunk_offsets"].tolist() == [0, 1]
    assert chunk_fwd_o_block_size(8) == 16


def test_long_inputs_share_one_indices_tensor():
    """At >= 64 rows chunk_fwd_o uses the shared block size, so the second copy is the first."""
    got = build_fla_chunk_indices([100, 30], CPU, pin_memory=False)
    assert got["chunk_indices_o"] is got["chunk_indices"]
    assert got["chunk_indices"].tolist() == [[0, 0], [0, 1], [1, 0]]
    assert got["chunk_offsets"].tolist() == [0, 2, 3]


def _verify_batch(rows: int = 8, cached_len: int = 1, table_idx: int = 3) -> Batch:
    req = Req(
        input_ids=torch.zeros(rows + cached_len, dtype=torch.int32),
        table_idx=table_idx, cached_len=cached_len, output_len=1, uid=0,
        sampling_params=None, cache_handle=None,
    )
    assert req.extend_len == rows
    batch = Batch([req], "decode")
    batch.padded_reqs = batch.reqs
    return batch


def test_build_fla_metadata_for_a_verify_block(monkeypatch):
    """A decode-phase batch of one 8-row request is the speculative verification forward.

    Without a GPU the staging is plain host memory (pinning is gated on CUDA), and `.to(cpu)`
    is the identity, so the tensors here are the exact values the device would receive.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    fla = build_fla_metadata(_verify_batch(), CPU)
    assert fla.cu_seqlens.tolist() == [0, 8] and fla.cu_seqlens.dtype == torch.int64
    assert fla.cache_indices.tolist() == [3] and fla.cache_indices.dtype == torch.int32
    assert fla.has_initial_state.tolist() == [True]
    assert fla.fresh_state_indices is None
    assert fla.chunk_indices.tolist() == [[0, 0]]
    assert fla.chunk_indices_o.tolist() == [[0, 0]]
    assert fla.chunk_offsets.tolist() == [0, 1]
    for t in (fla.chunk_indices, fla.chunk_indices_o, fla.chunk_offsets):
        assert t.dtype == torch.int64
    assert fla.track_dst is None and fla.track_h_row is None


def test_decode_metadata_carries_no_chunk_fields(monkeypatch):
    """One token per request takes the recurrent kernel, which has no chunks."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    batch = _verify_batch(rows=1)
    batch.linear_table_idx = torch.tensor([3], dtype=torch.int32)
    fla = build_fla_metadata(batch, CPU)
    assert fla.chunk_indices is None and fla.chunk_indices_o is None
    assert fla.chunk_offsets is None


def test_chunk_gated_delta_rule_forwards_precomputed_indices_untouched(monkeypatch):
    """The public entry point hands all three tensors to the fwd and derives nothing itself."""
    monkeypatch.setattr(fla_utils, "custom_device_ctx", lambda index: contextlib.nullcontext())
    seen = {}

    def fake_fwd(**kw):
        seen.update(kw)
        return kw["g"], kw["q"], None, None, torch.zeros(1), None

    def refuse(*a, **k):
        raise AssertionError("chunk indices were derived from cu_seqlens on the device path")

    monkeypatch.setattr(fla_chunk, "chunk_gated_delta_rule_fwd", fake_fwd)
    monkeypatch.setattr(fla_chunk, "prepare_chunk_indices", refuse)

    t = 8
    q = torch.zeros(1, t, 2, 4, dtype=torch.bfloat16)
    g = torch.zeros(1, t, 2)
    ci = torch.tensor([[0, 0]])
    ci_o = torch.tensor([[0, 0]])
    co = torch.tensor([0, 1])
    fla_chunk.chunk_gated_delta_rule(
        q, q.clone(), q.clone(), g, g.clone(), scale=0.5,
        initial_state=torch.zeros(1, 2, 4, 4),
        initial_state_indices=torch.zeros(1, dtype=torch.int32),
        cu_seqlens=torch.tensor([0, t]),
        chunk_indices=ci, chunk_indices_o=ci_o, chunk_offsets=co,
    )
    assert seen["chunk_indices"] is ci
    assert seen["chunk_indices_o"] is ci_o
    assert seen["chunk_offsets"] is co


def test_chunk_gated_delta_rule_still_derives_indices_when_none_are_given(monkeypatch):
    """Today's path, untouched: callers that pass only cu_seqlens get the derived indices."""
    monkeypatch.setattr(fla_utils, "custom_device_ctx", lambda index: contextlib.nullcontext())
    seen = {}

    def fake_fwd(**kw):
        seen.update(kw)
        return kw["g"], kw["q"], None, None, torch.zeros(1), None

    monkeypatch.setattr(fla_chunk, "chunk_gated_delta_rule_fwd", fake_fwd)
    t = 8
    q = torch.zeros(1, t, 2, 4, dtype=torch.bfloat16)
    g = torch.zeros(1, t, 2)
    fla_chunk.chunk_gated_delta_rule(
        q, q.clone(), q.clone(), g, g.clone(), scale=0.5,
        initial_state=torch.zeros(1, 2, 4, 4),
        initial_state_indices=torch.zeros(1, dtype=torch.int32),
        cu_seqlens=torch.tensor([0, t]),
    )
    assert seen["chunk_indices"].tolist() == [[0, 0]]
    assert seen["chunk_indices_o"] is None and seen["chunk_offsets"] is None


def test_gdn_prefill_chunk_fla_forwards_the_chunk_kwargs(monkeypatch):
    import freetoken.kernel.fla as fla_pkg
    from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_prefill_chunk_fla

    seen = {}

    def fake(**kw):
        seen.update(kw)
        return kw["q"], None, None

    monkeypatch.setattr(fla_pkg, "chunk_gated_delta_rule", fake)
    q = torch.zeros(1, 8, 2, 4, dtype=torch.bfloat16)
    ci, co = torch.tensor([[0, 0]]), torch.tensor([0, 1])
    common = dict(
        state_source=torch.zeros(4, 2, 4, 4), indices=torch.zeros(1, dtype=torch.int32),
        cu_seqlens=torch.tensor([0, 8]), scale=0.5,
    )
    gdn_prefill_chunk_fla(q, q, q, q, q, **common)
    assert seen["chunk_indices"] is None and seen["chunk_offsets"] is None
    gdn_prefill_chunk_fla(
        q, q, q, q, q, **common, chunk_indices=ci, chunk_indices_o=ci, chunk_offsets=co
    )
    assert seen["chunk_indices"] is ci and seen["chunk_indices_o"] is ci
    assert seen["chunk_offsets"] is co


def test_deriving_indices_from_a_device_tensor_under_capture_is_refused(monkeypatch):
    """A `.tolist()` mid-capture is an invalid stream operation that surfaces as a broken graph
    later; refusing at the source names the fix. A meta tensor stands in for a device one."""
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    cu = torch.empty(2, dtype=torch.int64, device="meta")
    with pytest.raises(RuntimeError, match="capture"):
        prepare_chunk_indices(cu, 64)
    with pytest.raises(RuntimeError, match="capture"):
        prepare_chunk_offsets(cu, 64)


def test_host_tensors_are_not_guarded(monkeypatch):
    """The scheduler's track metadata derives offsets from a host cu_seqlens; that is a host
    op and must keep working whatever the stream is doing (and without asking CUDA)."""

    def boom():
        raise AssertionError("asked CUDA about capture state for a host tensor")

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", boom)
    assert prepare_chunk_offsets(_cu([70, 3]), 64).tolist() == [0, 2, 3]
    assert prepare_chunk_indices(_cu([70, 3]), 64).tolist() == [[0, 0], [0, 1], [1, 0]]


def test_an_identity_cache_hit_needs_no_readback(monkeypatch):
    """Why the guard sits after the cache: a hit returns the stored tensor with no readback,
    which is what a capture warmed with the same cu_seqlens object relies on."""
    import freetoken.kernel.fla.index as fla_index

    calls = []
    monkeypatch.setattr(fla_index, "_refuse_under_capture", lambda cu: calls.append(cu))
    cu = _cu([8])
    first = prepare_chunk_indices(cu, 64)
    assert prepare_chunk_indices(cu, 64) is first
    assert len(calls) == 1, "the second call is a cache hit and must not reach the guard"
