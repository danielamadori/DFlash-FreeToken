"""Engine builds the draft graph only behind the flag, with a draft runner that owns the
static ring, wires it onto the runner, and tears it down before the verify graph on a
rebuild and at shutdown. Host-side only: the Engine is built without __init__ and the graph
runner is a fake (one test uses the real DraftGraphRunner with capture stubbed out, to check
the wiring contract both ends rely on).
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

import freetoken.engine.engine as engine_mod
from freetoken.engine.draft_graph import DraftGraphRunner
from freetoken.engine.engine import Engine
from freetoken.env import ENV


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
        # The real destroy() unwires the runner it was installed on (draft_graph.py); the
        # fake does the same so the engine's teardown can be asserted end to end.
        self.destroyed += 1
        runner = self.kwargs["runner"]
        if getattr(runner, "graph", None) is self:
            runner.graph = None


def _draft_runner(static: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        block_size=8, target_layer_ids=[1, 2], graph=None,
        _static_cache=object() if static else None,
    )


@pytest.fixture()
def buildable(monkeypatch):
    """An engine with everything _build_draft_graph reads, in a supported configuration."""
    RecordingRunner.instances = []
    RecordingRunner.fail = False
    monkeypatch.setattr(engine_mod, "DraftGraphRunner", RecordingRunner)
    monkeypatch.setattr(ENV.SPEC_DRAFT_GRAPH, "value", True)
    engine = Engine.__new__(Engine)
    engine.stream = "stream"
    engine.device = torch.device("cpu")
    engine.draft_runner = _draft_runner()
    return engine


# --------------------------------------------------------------------------- building


def test_build_captures_on_the_engine_stream_and_wires_the_runner(buildable):
    engine = buildable
    engine._build_draft_graph()
    (runner,) = RecordingRunner.instances
    assert engine.draft_graph is runner and runner.captured == 1
    assert runner.kwargs == dict(
        stream="stream", device=torch.device("cpu"), runner=engine.draft_runner
    )
    # draft() dispatches on this attribute; capture() alone does not set it.
    assert engine.draft_runner.graph is runner


@pytest.mark.parametrize(
    "unsupported, reason",
    [
        pytest.param(
            lambda e, mp: mp.setattr(ENV.SPEC_DRAFT_GRAPH, "value", False),
            None, id="flag-off",
        ),
        pytest.param(
            lambda e, mp: setattr(e, "draft_runner", None), "no draft model", id="no-draft"
        ),
        pytest.param(
            lambda e, mp: setattr(e, "draft_runner", _draft_runner(static=False)),
            "DynamicCache", id="dynamic-cache",
        ),
    ],
)
def test_build_stays_eager_when_unsupported(buildable, monkeypatch, caplog, unsupported, reason):
    engine = buildable
    unsupported(engine, monkeypatch)
    with caplog.at_level("INFO"):
        engine._build_draft_graph()
    assert engine.draft_graph is None and RecordingRunner.instances == []
    if engine.draft_runner is not None:
        assert engine.draft_runner.graph is None
    eager = [r.message for r in caplog.records if "Speculative draft stays eager" in r.message]
    if reason is None:
        # The default configuration is silent: the feature is a no-op when off.
        assert eager == []
    else:
        (line,) = eager
        assert reason in line


def test_a_failed_capture_falls_back_to_eager_without_raising(buildable, caplog):
    engine = buildable
    RecordingRunner.fail = True
    with caplog.at_level("WARNING"):
        engine._build_draft_graph()
    (runner,) = RecordingRunner.instances
    assert engine.draft_graph is None and runner.destroyed == 1
    # Never wired: a half-built graph must not be reachable from draft().
    assert engine.draft_runner.graph is None
    (line,) = [r.message for r in caplog.records if "stays eager" in r.message]
    assert "Draft graph capture failed" in line and "capture exploded" in line


def test_rebuild_after_a_failed_capture_retries(buildable):
    engine = buildable
    RecordingRunner.fail = True
    engine._build_draft_graph()
    RecordingRunner.fail = False
    engine._destroy_draft_graph()  # nothing to drop, must not raise
    engine._build_draft_graph()
    failed, retried = RecordingRunner.instances
    assert engine.draft_graph is retried and engine.draft_runner.graph is retried
    assert failed.destroyed == 1 and retried.captured == 1


# --------------------------------------------------------------------------- teardown


def test_destroy_drops_the_runner_once_and_unwires_the_draft(buildable):
    engine = buildable
    engine._build_draft_graph()
    (runner,) = RecordingRunner.instances
    engine._destroy_draft_graph()
    assert engine.draft_graph is None and runner.destroyed == 1
    assert engine.draft_runner.graph is None
    engine._destroy_draft_graph()
    assert runner.destroyed == 1


def test_rebuild_order_is_destroy_then_build(buildable):
    engine = buildable
    engine._build_draft_graph()
    (first,) = RecordingRunner.instances
    # The rebuild's sequence: free the old graph before capturing the new one.
    engine._destroy_draft_graph()
    engine._build_draft_graph()
    first_again, second = RecordingRunner.instances
    assert first_again is first and first.destroyed == 1
    assert engine.draft_graph is second and engine.draft_runner.graph is second
    assert second.captured == 1


def test_real_runner_destroy_unwires_the_draft(monkeypatch):
    """The contract between the engine and draft_graph.py: the engine wires runner.graph and
    relies on DraftGraphRunner.destroy() to unwire it."""
    monkeypatch.setattr(ENV.SPEC_DRAFT_GRAPH, "value", True)
    monkeypatch.setattr(DraftGraphRunner, "capture", lambda self: None)
    engine = Engine.__new__(Engine)
    engine.stream = "stream"
    engine.device = torch.device("cpu")
    engine.draft_runner = _draft_runner()
    engine._build_draft_graph()
    graph = engine.draft_graph
    assert isinstance(graph, DraftGraphRunner)
    assert graph.runner is engine.draft_runner and engine.draft_runner.graph is graph
    engine._destroy_draft_graph()
    assert engine.draft_graph is None and engine.draft_runner.graph is None


def test_shutdown_tears_down_the_draft_graph_before_the_verify_graph(monkeypatch):
    order: list[str] = []
    engine = Engine.__new__(Engine)
    engine.draft_graph = SimpleNamespace(destroy=lambda: order.append("draft"))
    engine.verify_graph = SimpleNamespace(destroy=lambda: order.append("verify"))
    engine.graph_runner = SimpleNamespace(destroy_cuda_graphs=lambda: order.append("decode"))
    monkeypatch.setattr(
        torch.distributed, "destroy_process_group", lambda: order.append("pg")
    )
    monkeypatch.setattr(engine_mod, "destroy_distributed", lambda: order.append("dist"))
    engine.shutdown()
    assert order == ["draft", "verify", "decode", "pg", "dist"]
    assert engine.draft_graph is None and engine.verify_graph is None


def test_lifecycle_call_sites_keep_the_free_before_alloc_order():
    """rebuild_runtime_cache frees the draft graph before the verify graph (and both before
    the caches are resized) and recaptures the verify graph before the draft graph; __init__
    captures in the same order. The sequence is the plan's free-before-alloc contract, so a
    reordering must fail here rather than only on a rebuild under memory pressure."""
    rebuild = inspect.getsource(Engine.rebuild_runtime_cache)
    assert (
        rebuild.index("self._destroy_draft_graph()")
        < rebuild.index("self._destroy_verify_graph()")
        < rebuild.index("self.graph_runner.destroy_cuda_graphs()")
        < rebuild.index("self._build_verify_graph(")
        < rebuild.index("self._build_draft_graph()")
    )
    init = inspect.getsource(Engine.__init__)
    assert init.index("self._build_verify_graph(") < init.index("self._build_draft_graph()")
