"""Engine.forward_logits routes a verify batch to the captured graph only when the runner
says the batch is the one it was captured for, and the runner is built only in the
configurations whose rollback can take its stash. Host-side only: the Engine is built without
__init__ and the runner is a fake.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import freetoken.engine.engine as engine_mod
from freetoken.attention.fi import FlashInferBackend
from freetoken.core import Batch, Context, Req
from freetoken.engine.engine import Engine
from freetoken.env import ENV


class FakeModel:
    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self.batches: list[Batch] = []
        self.logits = torch.zeros(8, 4)

    def forward(self) -> torch.Tensor:
        self.batches.append(self.ctx.batch)
        return self.logits


class FakeVerifyGraph:
    def __init__(self, ctx: Context, can: bool) -> None:
        self.ctx = ctx
        self.can = can
        self.asked: list[Batch] = []
        self.replayed: list[tuple[Batch, Batch | None]] = []
        self.logits = torch.ones(8, 4)

    def can_replay(self, batch: Batch) -> bool:
        self.asked.append(batch)
        return self.can

    def replay(self, batch: Batch) -> torch.Tensor:
        self.replayed.append((batch, self.ctx._batch))
        return self.logits


def _batch() -> Batch:
    req = Req(
        input_ids=torch.arange(11, dtype=torch.int32), table_idx=1, cached_len=10,
        output_len=16, uid=1, sampling_params=None, cache_handle=None,
    )
    batch = Batch(reqs=[req], phase="decode")
    batch.padded_reqs = batch.reqs
    return batch


def _engine(monkeypatch, verify_graph) -> tuple[Engine, FakeModel]:
    engine = Engine.__new__(Engine)  # bypass __init__/GPU
    engine.stream = object()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: engine.stream)
    engine.ctx = Context(1)
    engine.model = FakeModel(engine.ctx)
    engine.cpu_moe_executor = None
    engine.verify_graph = verify_graph
    return engine, engine.model


def test_no_verify_graph_runs_eagerly(monkeypatch):
    engine, model = _engine(monkeypatch, None)
    batch = _batch()
    assert engine.forward_logits(batch) is model.logits
    assert model.batches == [batch] and engine.ctx._batch is None


def test_batch_the_runner_refuses_runs_eagerly(monkeypatch):
    engine, model = _engine(monkeypatch, None)
    graph = FakeVerifyGraph(engine.ctx, can=False)
    engine.verify_graph = graph
    batch = _batch()
    assert engine.forward_logits(batch) is model.logits
    assert graph.asked == [batch] and graph.replayed == []
    assert model.batches == [batch]


def test_batch_the_runner_accepts_is_replayed_inside_the_forward_context(monkeypatch):
    engine, model = _engine(monkeypatch, None)
    graph = FakeVerifyGraph(engine.ctx, can=True)
    engine.verify_graph = graph
    batch = _batch()
    assert engine.forward_logits(batch) is graph.logits
    assert model.batches == []
    assert graph.replayed == [(batch, batch)] and engine.ctx._batch is None


def test_forward_logits_asserts_the_engine_stream(monkeypatch):
    engine, _ = _engine(monkeypatch, None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: object())
    with pytest.raises(AssertionError):
        engine.forward_logits(_batch())


# --------------------------------------------------------------------------- building


class RecordingRunner:
    instances: list["RecordingRunner"] = []
    fail = False

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.captured = 0
        self.destroyed = 0
        RecordingRunner.instances.append(self)

    def capture(self) -> None:
        if RecordingRunner.fail:
            raise RuntimeError("capture exploded")
        self.captured += 1

    def destroy(self) -> None:
        self.destroyed += 1


@pytest.fixture()
def buildable(monkeypatch):
    """An engine with everything _build_verify_graph reads, in a supported configuration."""
    RecordingRunner.instances = []
    RecordingRunner.fail = False
    monkeypatch.setattr(engine_mod, "VerifyGraphRunner", RecordingRunner)
    monkeypatch.setattr(ENV.SPEC_VERIFY_GRAPH, "value", True)
    monkeypatch.setattr(ENV.SPEC_GDN_ROLLBACK, "value", True)
    engine = Engine.__new__(Engine)
    engine.stream = "stream"
    engine.device = torch.device("cpu")
    engine.model = object()
    engine.attn_backend = FlashInferBackend.__new__(FlashInferBackend)
    engine.linear_state_pool = None
    engine.moe_offload_cache = None
    # --spec-block-size 8: the block's first slot is the anchor, 7 candidates, 8 verify rows.
    engine.draft_runner = SimpleNamespace(block_size=8, target_layer_ids=[1, 2])
    engine.config = SimpleNamespace(
        model_config=SimpleNamespace(vocab_size=32), page_size=1, cache_type="hybrid_radix"
    )
    engine.dummy_req = SimpleNamespace(table_idx=9)
    engine.num_pages = 64
    return engine


def test_build_captures_with_the_engine_geometry(buildable):
    engine = buildable
    engine._build_verify_graph(128)
    (runner,) = RecordingRunner.instances
    assert engine.verify_graph is runner and runner.captured == 1
    assert runner.kwargs == dict(
        stream="stream", device=torch.device("cpu"), model=engine.model,
        attn_backend=engine.attn_backend, linear_state_pool=None, moe_offload_cache=None,
        rows=8, vocab_size=32, max_seq_len=128, capture_table_idx=9, sink_loc=64,
        target_layer_ids=[1, 2],
    )


def test_build_with_a_state_pool_needs_hybrid_radix_and_the_rollback_switch(buildable, monkeypatch):
    engine = buildable
    engine.linear_state_pool = object()
    engine._build_verify_graph(128)
    (runner,) = RecordingRunner.instances
    assert engine.verify_graph is runner
    assert runner.kwargs["linear_state_pool"] is engine.linear_state_pool

    engine.config.cache_type = "radix"
    engine._build_verify_graph(128)
    assert engine.verify_graph is None and len(RecordingRunner.instances) == 1

    engine.config.cache_type = "hybrid_radix"
    monkeypatch.setattr(ENV.SPEC_GDN_ROLLBACK, "value", False)
    engine._build_verify_graph(128)
    assert engine.verify_graph is None and len(RecordingRunner.instances) == 1


@pytest.mark.parametrize(
    "unsupported",
    [
        pytest.param(lambda e, mp: mp.setattr(ENV.SPEC_VERIFY_GRAPH, "value", False), id="flag-off"),
        pytest.param(lambda e, mp: setattr(e, "draft_runner", None), id="no-draft"),
        pytest.param(lambda e, mp: setattr(e, "attn_backend", object()), id="not-fi"),
    ],
)
def test_build_stays_eager_when_unsupported(buildable, monkeypatch, unsupported):
    engine = buildable
    unsupported(engine, monkeypatch)
    engine._build_verify_graph(128)
    assert engine.verify_graph is None and RecordingRunner.instances == []


def test_a_failed_capture_falls_back_to_eager(buildable):
    engine = buildable
    RecordingRunner.fail = True
    engine._build_verify_graph(128)
    (runner,) = RecordingRunner.instances
    assert engine.verify_graph is None and runner.destroyed == 1


def test_destroy_drops_the_runner_once(buildable):
    engine = buildable
    engine._build_verify_graph(128)
    (runner,) = RecordingRunner.instances
    engine._destroy_verify_graph()
    assert engine.verify_graph is None and runner.destroyed == 1
    engine._destroy_verify_graph()
    assert runner.destroyed == 1


def test_flag_off_by_default_builds_nothing_silently(buildable, monkeypatch):
    # The default configuration must never reach the runner: the feature is a no-op when off.
    engine = buildable
    monkeypatch.setattr(ENV.SPEC_VERIFY_GRAPH, "value", False)
    engine.draft_runner = None
    engine._build_verify_graph(128)
    assert engine.verify_graph is None and RecordingRunner.instances == []
