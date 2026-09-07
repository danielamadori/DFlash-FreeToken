"""Which decode batches the captured CUDA graphs may serve.

A speculative verify batch is a decode batch (the scheduler keeps the phase) whose single
request extends by the whole block: the pending token plus every candidate. The graphs were
captured over one row per request, so the dispatch predicate has to look past the phase and
the batch size at how many rows each request actually contributes. Host-side only: the
predicate and the padding decision never touch a device.
"""

from __future__ import annotations

import torch

from freetoken.core import Batch, Req
from freetoken.engine.graph import GraphRunner
from freetoken.engine.speculative import open_draft_block
from freetoken.env import ENV, EnvBool


def _runner(graph_bs_list: list[int]) -> GraphRunner:
    """A GraphRunner with the dispatch state only; capture needs a GPU."""
    runner = GraphRunner.__new__(GraphRunner)
    runner.graph_bs_list = sorted(graph_bs_list)
    runner.max_graph_bs = max(graph_bs_list) if graph_bs_list else 0
    runner.dummy_req = Req(
        input_ids=torch.tensor([0], dtype=torch.int32),
        table_idx=99,
        cached_len=0,
        output_len=1,
        uid=-1,
        sampling_params=None,
        cache_handle=None,
    )
    return runner


def _decoding_req(uid: int = 1, prompt_len: int = 10, output_len: int = 16) -> Req:
    """A request mid-decode: one sampled token appended, its KV not yet computed."""
    ids = torch.arange(prompt_len + 1, dtype=torch.int32)
    return Req(
        input_ids=ids,
        table_idx=uid,
        cached_len=prompt_len,
        output_len=output_len,
        uid=uid,
        sampling_params=None,
        cache_handle=None,
    )


def _decode_batch(reqs: list[Req]) -> Batch:
    batch = Batch(reqs=reqs, phase="decode")
    batch.padded_reqs = batch.reqs
    return batch


def test_plain_decode_batch_uses_the_graph():
    runner = _runner([1, 2, 4])
    batch = _decode_batch([_decoding_req()])
    assert batch.reqs[0].extend_len == 1
    assert runner.can_use_cuda_graph(batch)


def test_verify_batch_is_not_routed_into_a_one_row_graph():
    runner = _runner([1, 2, 4])
    req = _decoding_req()
    block = open_draft_block(req, 7)
    assert block.size == 7 and req.extend_len == 8
    batch = _decode_batch([req])
    assert batch.is_decode and batch.size <= runner.max_graph_bs
    assert not runner.can_use_cuda_graph(batch)


def test_prefill_and_oversized_batches_stay_eager():
    runner = _runner([1, 2])
    prefill = Batch(reqs=[_decoding_req()], phase="prefill")
    prefill.padded_reqs = prefill.reqs
    assert not runner.can_use_cuda_graph(prefill)

    too_big = _decode_batch([_decoding_req(uid=i) for i in range(3)])
    assert not runner.can_use_cuda_graph(too_big)

    disabled = _runner([])
    assert not disabled.can_use_cuda_graph(_decode_batch([_decoding_req()]))


def test_one_extended_request_makes_the_whole_batch_eager():
    runner = _runner([1, 2, 4])
    plain, extended = _decoding_req(uid=1), _decoding_req(uid=2)
    open_draft_block(extended, 3)
    assert not runner.can_use_cuda_graph(_decode_batch([plain, extended]))


def test_pad_batch_pads_only_what_the_graph_will_serve():
    runner = _runner([1, 2, 4])

    plain = _decode_batch([_decoding_req(uid=i) for i in range(3)])
    runner.pad_batch(plain)
    assert plain.padded_size == 4
    assert plain.padded_reqs[3] is runner.dummy_req

    verify_req = _decoding_req()
    open_draft_block(verify_req, 7)
    verify = _decode_batch([verify_req])
    runner.pad_batch(verify)
    assert verify.padded_reqs == verify.reqs

    # Eager batches are never padded, whatever their size.
    big = _decode_batch([_decoding_req(uid=i) for i in range(5)])
    runner.pad_batch(big)
    assert big.padded_reqs == big.reqs


def test_verify_graph_flag_defaults_off_and_parses_like_the_other_spec_flags(monkeypatch):
    assert not ENV.SPEC_VERIFY_GRAPH
    assert ENV.SPEC_VERIFY_GRAPH.value is False

    monkeypatch.setenv("FREETOKEN_SPEC_VERIFY_GRAPH", "1")
    flag = EnvBool(False)
    flag._init("FREETOKEN_SPEC_VERIFY_GRAPH")
    assert flag

    monkeypatch.setenv("FREETOKEN_SPEC_VERIFY_GRAPH", "0")
    flag = EnvBool(False)
    flag._init("FREETOKEN_SPEC_VERIFY_GRAPH")
    assert not flag
