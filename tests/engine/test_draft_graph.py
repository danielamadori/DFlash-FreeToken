"""The draft graph runner: capture, the replay gate, the replay copies, teardown.

Everything runs on CPU against a fake DFlashRunner whose ``_run_block`` records what it was
handed and returns tensors of the right shape; torch.cuda's graph entry points are stubbed
the way tests/engine/test_verify_graph.py stubs them. What is asserted is the wiring the
plan's silent failures hide behind: which rows land where in the static input, which graph
runs, that the ring is reset around the capture, and that nothing replays past the gate.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import freetoken.engine.draft_graph as dg
from freetoken.engine.draft_graph import CAPTURE_SEQ_LEN, DraftCaptureBuffer, DraftGraphRunner

BLOCK = 8
HIDDEN = 4
VOCAB = 10
TARGET_LAYERS = (1, 3)
FEATURES = HIDDEN * len(TARGET_LAYERS)
MASK = 9
CPU = torch.device("cpu")


class FakeRing:
    def __init__(self, log: list) -> None:
        self.log = log
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1
        self.log.append(("reset",))


class FakeRunner:
    """The DFlashRunner fields the graph runner reads, and a recording ``_run_block``."""

    def __init__(self, log: list, *, bad_shapes: bool = False) -> None:
        self.log = log
        self.block_size = BLOCK
        self.target_layer_ids = list(TARGET_LAYERS)
        self.mask_token_id = MASK
        self.dtype = torch.float32
        self.device = CPU
        self.draft_model = SimpleNamespace(fc=SimpleNamespace(in_features=FEATURES))
        self._static_cache = FakeRing(log)
        self._cache_owner: int | None = 7
        self._seq_len_t = torch.zeros(1, dtype=torch.int64)
        self.graph = None
        self.forwards = 0
        self.bad_shapes = bad_shapes

    def reset_cache(self) -> None:
        self._static_cache.reset()

    def _run_block(self, th, ids, c, temperature, top_p, top_k):  # noqa: ANN001 - fake
        self.forwards += 1
        self.log.append(
            ("run", c, th, ids.clone(), int(self._seq_len_t), temperature, top_p, top_k,
             torch.is_inference_mode_enabled())
        )
        n = float(self.forwards)
        candidates = 1 if self.bad_shapes else BLOCK - 1
        tokens = torch.full((1, candidates), self.forwards, dtype=torch.int64)
        logits = torch.full((1, candidates, VOCAB), n)
        return tokens, logits + 0.5, logits


class FakeGraph:
    """Rewrites the static outputs in place, running no Python of the model."""

    def __init__(self, c: int, bufs: DraftCaptureBuffer, log: list) -> None:
        self.c = c
        self.bufs = bufs
        self.log = log
        self.replays = 0

    def replay(self) -> None:
        self.replays += 1
        self.log.append(("graph", self.c, self.replays))
        # What the captured body would compute: a function of the inputs it reads.
        self.bufs.tokens.fill_(100 * self.c + self.replays)
        self.bufs.probs.fill_(float(self.bufs.seq_len) + float(self.bufs.ids[0, 0]))


@pytest.fixture()
def cuda_stubs(monkeypatch):
    """torch.cuda graph entry points that record their arguments and run nothing."""
    calls: list = []

    class StubCUDAGraph:
        def pool(self):
            return ("pool", id(self))

    @contextmanager
    def stub_graph(graph, pool=None, stream=None):
        calls.append(("graph", graph, pool, stream))
        yield

    monkeypatch.setattr(torch.cuda, "CUDAGraph", StubCUDAGraph)
    monkeypatch.setattr(torch.cuda, "graph", stub_graph)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: calls.append(("sync",)))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **k: "stream")
    return calls


def _runner(log: list | None = None, **runner_kwargs) -> tuple[DraftGraphRunner, FakeRunner, list]:
    log = [] if log is None else log
    fake = FakeRunner(log, **runner_kwargs)
    return DraftGraphRunner(stream="stream", device=CPU, runner=fake), fake, log


def _captured(cuda_stubs) -> tuple[DraftGraphRunner, FakeRunner, list]:
    """A runner as capture leaves it, with fake graphs standing in for the captured ones."""
    runner, fake, log = _runner()
    runner.capture()
    fake.graph = runner
    runner.graphs = {c: FakeGraph(c, runner.bufs, log) for c in range(1, BLOCK + 1)}
    log.clear()
    return runner, fake, log


def _store(c: int, *, rows: dict[int, int] | None = None, dtype=torch.float32, batch: int = 1):
    """The target's hidden-state store: entry layer + 1 is [1, c, HIDDEN], the rest None."""
    store: list = [None] * (max(TARGET_LAYERS) + 2)
    for layer_id in TARGET_LAYERS:
        n = (rows or {}).get(layer_id, c)
        store[layer_id + 1] = (
            torch.arange(n * HIDDEN, dtype=dtype).view(1, n, HIDDEN) + 1000 * layer_id
        ).repeat(batch, 1, 1)
    return store


# ------------------------------------------------------------------------------- buffers


def test_buffer_shapes_and_capture_values():
    seq_len = torch.zeros(1, dtype=torch.int64)
    bufs = DraftCaptureBuffer.init(
        block=BLOCK, features=FEATURES, num_target_layers=len(TARGET_LAYERS),
        mask_token_id=MASK, seq_len=seq_len, device=CPU, dtype=torch.bfloat16,
    )
    assert bufs.th.shape == (1, BLOCK, FEATURES) and bufs.th.dtype == torch.bfloat16
    assert bufs.ids.shape == (1, BLOCK) and bufs.ids.dtype == torch.int64
    assert bufs.hidden == HIDDEN and bufs.seq_len is seq_len
    assert bufs.tokens is None and bufs.probs is None

    bufs.th.fill_(3.0)
    bufs.ids.fill_(4)
    bufs.set_capture_inputs()
    assert int(seq_len) == CAPTURE_SEQ_LEN and CAPTURE_SEQ_LEN >= BLOCK
    assert bool((bufs.th == 0).all())
    assert bufs.ids.tolist() == [[0] + [MASK] * (BLOCK - 1)]


def test_buffer_refuses_a_feature_width_that_is_not_layers_times_hidden():
    with pytest.raises(AssertionError, match="not a multiple"):
        DraftCaptureBuffer.init(
            block=BLOCK, features=FEATURES + 1, num_target_layers=len(TARGET_LAYERS),
            mask_token_id=MASK, seq_len=torch.zeros(1, dtype=torch.int64), device=CPU,
            dtype=torch.float32,
        )


def test_buffer_outputs_are_sized_from_the_first_forward_and_copies_refuse_broadcasts():
    bufs = DraftCaptureBuffer.init(
        block=BLOCK, features=FEATURES, num_target_layers=len(TARGET_LAYERS),
        mask_token_id=MASK, seq_len=torch.zeros(1, dtype=torch.int64), device=CPU,
        dtype=torch.float32,
    )
    tokens = torch.ones(1, BLOCK - 1, dtype=torch.int64)
    probs = torch.ones(1, BLOCK - 1, VOCAB, dtype=torch.bfloat16)
    bufs.alloc_outputs(tokens, probs)
    assert bufs.tokens.shape == tokens.shape and bufs.tokens.dtype == torch.int64
    assert bufs.probs.shape == probs.shape and bufs.probs.dtype == torch.bfloat16
    assert bufs.tokens is not tokens and bufs.probs is not probs
    with pytest.raises(AssertionError):
        bufs.copy_outputs(tokens[:, :1], probs)  # [1, 1] would broadcast into [1, 7]
    with pytest.raises(AssertionError):
        bufs.copy_outputs(tokens, probs[:, :, :1])
    bufs.copy_outputs(tokens * 5, probs * 2)
    assert bufs.tokens.tolist() == [[5] * (BLOCK - 1)]
    assert float(bufs.probs[0, 0, 0]) == 2.0


# ------------------------------------------------------------------------------- capture


def test_capture_warms_up_then_captures_each_row_count_into_one_shared_pool(cuda_stubs):
    runner, fake, log = _runner()

    runner.capture()

    # a warm-up then a capture per c, in order, over the same buffer view
    runs = [entry for entry in log if entry[0] == "run"]
    assert [entry[1] for entry in runs] == [c for c in range(1, BLOCK + 1) for _ in (0, 1)]
    bufs = runner.bufs
    for _, c, th, ids, seq_len, temperature, top_p, top_k, inference in runs:
        assert th.shape == (1, c, FEATURES) and th.data_ptr() == bufs.th.data_ptr()
        assert ids.tolist() == [[0] + [MASK] * (BLOCK - 1)]
        assert seq_len == CAPTURE_SEQ_LEN
        assert (temperature, top_p, top_k) == (0.0, 1.0, 0)
        assert inference, "the captured body must run under inference_mode like draft()"
    assert bool((bufs.th == 0).all()), "zero context: finite through the whole body"

    # one graph per c on the engine stream, the first one's pool shared by the rest
    graph_calls = [call for call in cuda_stubs if call[0] == "graph"]
    assert len(graph_calls) == BLOCK
    assert set(runner.graphs) == set(range(1, BLOCK + 1))
    assert [call[1] for call in graph_calls] == [runner.graphs[c] for c in range(1, BLOCK + 1)]
    assert graph_calls[0][2] is None
    pool = runner.graphs[1].pool()
    assert all(call[2] == pool for call in graph_calls[1:]) and runner.pool == pool
    assert all(call[3] == "stream" for call in graph_calls)
    # a synchronize between every warm-up and its capture
    syncs = [i for i, call in enumerate(cuda_stubs) if call[0] == "sync"]
    graphs = [i for i, call in enumerate(cuda_stubs) if call[0] == "graph"]
    assert len(syncs) == BLOCK and all(s < g for s, g in zip(syncs, graphs))

    # the outputs hold the last captured forward, not a warm-up
    assert fake.forwards == 2 * BLOCK
    assert bufs.tokens.tolist() == [[2 * BLOCK] * (BLOCK - 1)]
    assert float(bufs.probs[0, 0, 0]) == 2 * BLOCK + 0.5
    assert bufs.probs.shape == (1, BLOCK - 1, VOCAB)
    # the scheduler keeps the outputs past draft()'s inference_mode; an inference tensor
    # would refuse the next in-place update there
    assert not bufs.tokens.is_inference() and not bufs.probs.is_inference()

    # the ring was reset and the owner cleared before the first warm-up and after the last
    # capture, so the rows the captures staged are never a request's context
    assert fake._static_cache.resets == 2 and fake._cache_owner is None
    assert log[0] == ("reset",) and log[-1] == ("reset",)
    assert fake.graph is None, "the engine installs the runner; capture only builds it"


def test_capture_leaves_the_ring_reset_when_a_forward_returns_the_wrong_shapes(cuda_stubs):
    runner, fake, log = _runner(bad_shapes=True)
    with pytest.raises(AssertionError, match="for a block of"):
        runner.capture()
    assert fake._static_cache.resets == 2 and fake._cache_owner is None
    assert runner.graphs == {} and runner.bufs is None and runner.pool is None
    assert runner.can_replay(_store(3), BLOCK, 0.0) is None


def test_capture_refuses_a_runner_without_the_static_ring(cuda_stubs):
    runner, fake, log = _runner()
    fake._static_cache = None
    with pytest.raises(AssertionError, match="static ring"):
        runner.capture()
    assert fake.forwards == 0 and runner.graphs == {}


def test_capture_twice_is_refused(cuda_stubs):
    runner, *_ = _captured(cuda_stubs)
    with pytest.raises(AssertionError):
        runner.capture()


# ------------------------------------------------------------------------------- gate


def test_can_replay_returns_the_row_count_of_a_greedy_full_block(cuda_stubs):
    runner, fake, log = _captured(cuda_stubs)
    for c in range(1, BLOCK + 1):
        assert runner.can_replay(_store(c), BLOCK, 0.0) == c
    assert runner.can_replay(_store(3), BLOCK, -1.0) == 3


@pytest.mark.parametrize(
    "store_kwargs, k, temperature",
    [
        (dict(c=0), BLOCK, 0.0),  # no context rows
        (dict(c=BLOCK + 1), BLOCK, 0.0),  # first block after a prefill
        (dict(c=3, rows={TARGET_LAYERS[0]: 4}), BLOCK, 0.0),  # layers disagree on c
        (dict(c=3), BLOCK - 1, 0.0),  # truncated block
        (dict(c=3), BLOCK, 0.7),  # sampled: a greedy graph would hand one-hot probs
        (dict(c=3, dtype=torch.bfloat16), BLOCK, 0.0),  # another dtype than the buffer
        (dict(c=3, batch=2), BLOCK, 0.0),  # not one request
    ],
)
def test_can_replay_refuses_what_no_graph_was_captured_for(
    cuda_stubs, store_kwargs, k, temperature
):
    runner, *_ = _captured(cuda_stubs)
    assert runner.can_replay(_store(**store_kwargs), k, temperature) is None


def test_can_replay_refuses_a_two_dimensional_or_narrow_hidden_state(cuda_stubs):
    runner, *_ = _captured(cuda_stubs)
    store = _store(3)
    store[TARGET_LAYERS[1] + 1] = store[TARGET_LAYERS[1] + 1][0]
    assert runner.can_replay(store, BLOCK, 0.0) is None
    store = _store(3)
    store[TARGET_LAYERS[1] + 1] = store[TARGET_LAYERS[1] + 1][:, :, :-1]
    assert runner.can_replay(store, BLOCK, 0.0) is None
    store = _store(3)
    store[TARGET_LAYERS[1] + 1] = None
    assert runner.can_replay(store, BLOCK, 0.0) is None
    assert runner.can_replay(_store(3)[: TARGET_LAYERS[1] + 1], BLOCK, 0.0) is None


def test_can_replay_is_none_without_graphs(cuda_stubs):
    runner, *_ = _runner()
    assert runner.can_replay(_store(3), BLOCK, 0.0) is None
    captured, *_ = _captured(cuda_stubs)
    captured.destroy()
    assert captured.can_replay(_store(3), BLOCK, 0.0) is None


# ------------------------------------------------------------------------------- replay


def test_replay_copies_the_context_rows_fills_the_start_and_runs_the_matching_graph(
    cuda_stubs, caplog
):
    runner, fake, log = _captured(cuda_stubs)
    bufs = runner.bufs
    bufs.th.fill_(-1.0)
    bufs.ids.fill_(MASK)
    c, seq_len = 3, 1234
    store = _store(c)
    anchor = torch.tensor(5, dtype=torch.int32)

    with caplog.at_level("INFO"):
        tokens, probs = runner.replay(store, anchor, seq_len, c)

    # rows [:c] of every target layer land in their column block, nothing else moves
    for i, layer_id in enumerate(TARGET_LAYERS):
        cols = slice(i * HIDDEN, (i + 1) * HIDDEN)
        assert torch.equal(bufs.th[0, :c, cols], store[layer_id + 1][0])
    assert bool((bufs.th[0, c:] == -1.0).all()), "rows past c are not the graph's to read"
    assert int(bufs.seq_len) == seq_len and bufs.seq_len is fake._seq_len_t
    # the anchor id with its dtype cast, the mask tokens untouched
    assert bufs.ids.dtype == torch.int64 and bufs.ids.tolist() == [[5] + [MASK] * (BLOCK - 1)]
    # graphs[c] ran, once, and the static buffers came back
    assert log == [("graph", c, 1)]
    assert tokens is bufs.tokens and probs is bufs.probs
    assert tokens.tolist() == [[100 * c + 1] * (BLOCK - 1)]
    assert float(probs[0, 0, 0]) == seq_len + 5
    assert runner.replays == 1
    assert "Draft graph replay active" in caplog.text

    # a second block through another graph: counted, not announced again
    caplog.clear()
    with caplog.at_level("INFO"):
        runner.replay(_store(1), torch.tensor(6, dtype=torch.int32), seq_len + 3, 1)
    assert log[-1] == ("graph", 1, 1) and runner.replays == 2
    assert "replay active" not in caplog.text
    assert bufs.ids[0, 0].item() == 6 and int(bufs.seq_len) == seq_len + 3


def test_replay_copies_only_the_rows_of_the_store_it_is_handed(cuda_stubs):
    """The store changes identity between blocks; every replay reads the current one."""
    runner, fake, log = _captured(cuda_stubs)
    first, second = _store(2), _store(2)
    for layer_id in TARGET_LAYERS:
        second[layer_id + 1] += 0.5
    runner.replay(first, torch.tensor(1, dtype=torch.int32), 10, 2)
    runner.replay(second, torch.tensor(1, dtype=torch.int32), 12, 2)
    assert torch.equal(runner.bufs.th[0, :2, :HIDDEN], second[TARGET_LAYERS[0] + 1][0])


def test_replay_asserts_the_engine_stream(cuda_stubs, monkeypatch):
    runner, *_ = _captured(cuda_stubs)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **k: "other")
    with pytest.raises(AssertionError):
        runner.replay(_store(3), torch.tensor(5, dtype=torch.int32), 10, 3)


def test_replay_refuses_a_row_count_without_a_graph(cuda_stubs):
    runner, *_ = _captured(cuda_stubs)
    del runner.graphs[3]
    with pytest.raises(AssertionError, match="c=3"):
        runner.replay(_store(3), torch.tensor(5, dtype=torch.int32), 10, 3)


# ------------------------------------------------------------------------------- destroy


def test_destroy_drops_the_graphs_and_unwires_the_runner(cuda_stubs):
    runner, fake, log = _captured(cuda_stubs)
    assert fake.graph is runner
    runner.destroy()
    assert runner.graphs == {} and runner.pool is None and runner.bufs is None
    assert fake.graph is None
    assert runner.can_replay(_store(3), BLOCK, 0.0) is None


def test_destroy_leaves_another_installed_runner_alone(cuda_stubs):
    runner, fake, log = _captured(cuda_stubs)
    other = object()
    fake.graph = other
    runner.destroy()
    assert fake.graph is other


def test_free_memory_is_zero_off_the_gpu():
    assert dg._free_memory(CPU) == 0
