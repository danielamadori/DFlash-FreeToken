"""Decode-graph replays must publish the hidden states the draft reads next.

A captured forward rewrites its hidden-state tensors in place on every replay, but the
Python assignment that names them (``self._captured_hidden_states = captured``) only ran at
capture time. Any eager forward in between -- a prefill, a verify -- rebinds the model's
store to a fresh list, so after the next replay the scheduler would draft from the previous
step's context. These tests drive GraphRunner with a fake model, a fake graph and CPU
buffers; the capture itself is exercised with torch.cuda's graph entry points stubbed out.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import torch

import freetoken.engine.graph as graph_mod
from freetoken.core import Batch, Context, Req
from freetoken.engine.graph import GraphCaptureBuffer, GraphRunner
from freetoken.models.blocks import BaseLLMModel, HiddenStateCapture

HIDDEN = 4
VOCAB = 8
NUM_LAYERS = 3
CAPTURE_LAYERS = (1,)


class FakeStack(HiddenStateCapture):
    """The inner ``*Model``: owns ``layers`` and the capture store, like Qwen3_5Model."""

    def __init__(self) -> None:
        self.layers = SimpleNamespace(op_list=[None] * NUM_LAYERS)


class FakeModel(BaseLLMModel):
    """Publishes a fresh store per forward, the way the real stacks do (residual + x)."""

    def __init__(self) -> None:
        self.model = FakeStack()
        self.forwards = 0
        self.stores: list[list[torch.Tensor | None]] = []
        self.rows = 1

    def forward(self) -> torch.Tensor:
        self.forwards += 1
        logits = torch.full((self.rows, VOCAB), float(self.forwards))
        if not self.model._capture_layer_ids:
            return logits  # capture off: the real stacks return before touching the store
        store = self.model._new_capture_store()
        for layer_id in self.model._capture_layer_ids:
            store[layer_id + 1] = torch.full((self.rows, HIDDEN), float(self.forwards))
        self.model._captured_hidden_states = store
        self.stores.append(store)
        return logits


class FakeGraph:
    """Replays by rewriting the captured tensors in place, running no Python of the model."""

    def __init__(self, captured: list[torch.Tensor | None]) -> None:
        self.captured = captured
        self.replays = 0

    def replay(self) -> None:
        self.replays += 1
        for tensor in self.captured:
            if tensor is not None:
                tensor.fill_(100.0 + self.replays)


class FakeAttnBackend:
    def __init__(self) -> None:
        self.prepared: list[str] = []

    def init_capture_graph(self, max_seq_len: int, bs_list: list[int]) -> None:
        self.prepared.append("init")

    def prepare_for_capture(self, batch: Batch) -> None:
        self.prepared.append("capture")

    def prepare_for_replay(self, batch: Batch) -> None:
        self.prepared.append("replay")


def _req(uid: int = 1) -> Req:
    return Req(
        input_ids=torch.tensor([0, 1], dtype=torch.int32),
        table_idx=uid,
        cached_len=1,
        output_len=4,
        uid=uid,
        sampling_params=None,
        cache_handle=None,
    )


def _decode_batch(bs: int = 1) -> Batch:
    batch = Batch(reqs=[_req(uid=i + 1) for i in range(bs)], phase="decode")
    batch.padded_reqs = batch.reqs
    batch.input_ids = torch.arange(bs, dtype=torch.int32)
    batch.positions = torch.ones(bs, dtype=torch.int32)
    batch.out_loc = torch.arange(bs, dtype=torch.int32)
    batch.linear_table_idx = None
    return batch


def _runner(model: FakeModel, graph: FakeGraph, hidden) -> GraphRunner:
    """A replay-ready runner over CPU buffers, as if bs=1 had been captured."""
    runner = GraphRunner.__new__(GraphRunner)
    runner.model = model
    runner.attn_backend = FakeAttnBackend()
    runner.graph_bs_list = [1]
    runner.max_graph_bs = 1
    runner.dummy_req = _req(uid=-1)
    runner.buffer = GraphCaptureBuffer.init(1, VOCAB, torch.device("cpu"))
    runner.graph_map = {1: graph}
    runner.hidden_map = {1: hidden}
    return runner


def _captured_model() -> tuple[FakeModel, list[torch.Tensor | None]]:
    """A model whose last forward is the captured one, so its store is the static list."""
    model = FakeModel()
    model.enable_hidden_state_capture(CAPTURE_LAYERS)
    model.forward()
    return model, model.last_hidden_states


def test_replay_rebinds_the_store_after_an_interleaved_eager_forward():
    model, static = _captured_model()
    graph = FakeGraph(static)
    runner = _runner(model, graph, static)

    model.forward()  # an eager forward (a prefill, a verify) names a fresh list
    fresh = model.last_hidden_states
    assert fresh is not static

    logits = runner.replay(_decode_batch())

    assert graph.replays == 1
    assert model.last_hidden_states is static
    assert torch.equal(static[CAPTURE_LAYERS[0] + 1], torch.full((1, HIDDEN), 101.0))
    # The eager list is untouched by the replay: reading it would have been the stale draft.
    assert torch.equal(fresh[CAPTURE_LAYERS[0] + 1], torch.full((1, HIDDEN), 2.0))
    assert logits.shape == (1, VOCAB)
    assert runner.attn_backend.prepared == ["replay"]


def test_replay_of_a_graph_captured_before_enabling_publishes_nothing():
    # What the engine did before enable_hidden_state_capture moved ahead of the capture: the
    # captured forward ran with capture off, so there is no static list to rebind to and the
    # model keeps whatever the last eager forward left. This is the hole S0 closes by
    # ordering, not by code in replay -- the None entry must stay a no-op.
    model = FakeModel()
    model.forward()
    assert model.last_hidden_states is None
    graph = FakeGraph([])
    runner = _runner(model, graph, None)

    model.enable_hidden_state_capture(CAPTURE_LAYERS)
    model.forward()
    stale = model.last_hidden_states

    runner.replay(_decode_batch())

    assert graph.replays == 1
    assert model.last_hidden_states is stale


def test_capture_keeps_the_captured_forwards_store_not_the_warmups(monkeypatch):
    model = FakeModel()
    model.enable_hidden_state_capture(CAPTURE_LAYERS)

    class StubCUDAGraph:
        def __init__(self) -> None:
            self.fake = None

        def pool(self):
            return "pool"

    @contextmanager
    def stub_graph(graph, pool=None, stream=None):
        yield
        # the forward that ran inside is the captured one: its store is what a replay
        # rewrites, so the fake graph adopts it
        graph.fake = FakeGraph(model.last_hidden_states)

    monkeypatch.setattr(torch.cuda, "CUDAGraph", StubCUDAGraph)
    monkeypatch.setattr(torch.cuda, "graph", stub_graph)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(graph_mod, "get_free_memory", lambda device: 0)
    monkeypatch.setattr(graph_mod, "get_global_ctx", lambda: Context(1))
    monkeypatch.setattr(
        graph_mod, "get_tp_info", lambda: SimpleNamespace(is_primary=lambda: True)
    )

    runner = GraphRunner.__new__(GraphRunner)
    runner.model = model
    runner.attn_backend = FakeAttnBackend()
    runner.graph_bs_list = [1, 2]
    runner.max_graph_bs = 2
    runner.dummy_req = _req(uid=-1)
    runner.moe_offload_cache = None
    runner.stream = None
    runner.device = torch.device("cpu")

    # Graphs are captured largest first, warm-up then capture: bs 2, 2, 1, 1.
    original_forward = model.forward
    rows_by_call = iter([2, 2, 1, 1])

    def forward_with_rows() -> torch.Tensor:
        model.rows = next(rows_by_call)
        return original_forward()

    model.forward = forward_with_rows

    runner._capture_graphs(max_seq_len=16, vocab_size=VOCAB, model=model)

    assert model.forwards == 4  # warm-up + capture for each of bs 2 and bs 1
    assert sorted(runner.graph_map) == [1, 2]
    # stores[1] and stores[3] are the captured forwards; stores[0] and [2] the warm-ups
    assert runner.hidden_map[2] is model.stores[1]
    assert runner.hidden_map[1] is model.stores[3]
    assert runner.hidden_map[2] is not model.stores[0]
    assert runner.hidden_map[2][CAPTURE_LAYERS[0] + 1].shape == (2, HIDDEN)
    assert runner.hidden_map[1][CAPTURE_LAYERS[0] + 1].shape == (1, HIDDEN)
    assert runner.attn_backend.prepared == ["init", "capture", "capture"]

    # and a replay through the runner rebinds to the bs-1 captured store
    for bs, graph in runner.graph_map.items():
        runner.graph_map[bs] = graph.fake
    model.forward = original_forward
    model.rows = 1
    model.forward()
    runner.replay(_decode_batch())
    assert model.last_hidden_states is runner.hidden_map[1]


def test_capture_with_hidden_state_capture_off_records_none(monkeypatch):
    # Models that do not publish (capture disabled, or no draft) must not make replay fail.
    model, static = _captured_model()
    model.model.set_capture_layer_ids(())
    model.forward()
    assert model.last_hidden_states is None
    graph = FakeGraph([])
    runner = _runner(model, graph, None)
    runner.replay(_decode_batch())
    assert model.last_hidden_states is None


def test_destroy_unbinds_a_store_that_lives_in_the_graph_pool():
    model, static = _captured_model()
    runner = _runner(model, FakeGraph(static), static)
    assert model.last_hidden_states is static

    runner.destroy_cuda_graphs()

    assert runner.hidden_map == {}
    assert runner.graph_map == {}
    assert model.last_hidden_states is None


def test_destroy_leaves_an_eager_store_alone():
    model, static = _captured_model()
    runner = _runner(model, FakeGraph(static), static)
    model.forward()
    eager = model.last_hidden_states

    runner.destroy_cuda_graphs()

    assert model.last_hidden_states is eager
