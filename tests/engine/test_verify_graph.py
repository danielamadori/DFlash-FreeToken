"""The captured K+1-row verify forward: its static buffers, what it replays, and what it
republishes afterwards.

A replay runs none of the Python the eager verify runs, so everything that Python did
besides launching kernels has to be redone by the runner: copy the block's inputs into the
static buffers, plan the attention wrapper outside the graph, rebind the model's hidden-state
store, and install the capture-time rollback stash into the scheduler's open block. These
tests drive the runner on CPU with a fake model, a fake graph, a fake attention backend and
the rollback's FakePool; the capture itself runs with torch.cuda's graph entry points stubbed.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import freetoken.engine.verify_graph as vg
from freetoken.attention.fi import FIMetadata
from freetoken.attention.linear import FLAMetadata
from freetoken.core import Batch, Context, Req
from freetoken.engine.gdn_rollback import GDNRollback, LayerStash
from freetoken.engine.speculative import open_draft_block
from freetoken.engine.verify_graph import VerifyCaptureBuffer, VerifyGraphRunner
from freetoken.models.blocks import BaseLLMModel, HiddenStateCapture

ROWS = 8
VOCAB = 16
HIDDEN = 4
MAX_SEQ = 32
NUM_LAYERS = 4
TARGET_LAYERS = (1, 2)
GDN_LAYERS = (0, 3)
SINK = 100
DUMMY_TABLE = 7
CPU = torch.device("cpu")


class FakePool:
    """Enough LinearStatePool for the capture: a free list, slot copies and the layer count."""

    def __init__(self, num_slots: int = 8) -> None:
        self.padding_slot = 0
        self.num_linear_layers = len(GDN_LAYERS)
        self._free = list(range(1, num_slots))
        self.copies: list[tuple[int, int]] = []
        self.freed: list[int] = []

    def alloc(self, n: int = 1) -> list[int]:
        return [self._free.pop() for _ in range(n)]

    def free(self, slots) -> None:
        for s in slots:
            self.freed.append(s)
            self._free.append(s)

    def copy_from(self, src: int, dst: int) -> None:
        self.copies.append((src, dst))


class FakeStack(HiddenStateCapture):
    def __init__(self) -> None:
        self.layers = SimpleNamespace(op_list=[None] * NUM_LAYERS)


def _fused(entries, *, live_slot, scratch_slot, committed):
    pass


class FakeModel(BaseLLMModel):
    """Stashes into the open rollback and publishes a fresh store, like the real stack."""

    def __init__(self, ctx: Context, gdn_layers=GDN_LAYERS, hidden_rows_delta: int = 0) -> None:
        self.model = FakeStack()
        self.ctx = ctx
        self.gdn_layers = gdn_layers
        self.hidden_rows_delta = hidden_rows_delta
        self.forwards = 0
        self.stashed: list[dict[int, torch.Tensor]] = []

    def forward(self) -> torch.Tensor:
        self.forwards += 1
        batch = self.ctx.batch
        rows = int(batch.input_ids.shape[0])
        rb = self.ctx.gdn_rollback
        stashed = {}
        if rb is not None and rb.recording:
            for layer_id in self.gdn_layers:
                q = torch.full((1, rows, 2, 4), float(self.forwards))
                stashed[layer_id] = q
                rb.stash(
                    layer_id, lambda **kw: None, q, torch.zeros_like(q), torch.zeros_like(q),
                    torch.zeros(1, rows, 2), torch.zeros(1, rows, 2), torch.zeros(rows, 6),
                    local_index=layer_id, head_k_dim=4, fused=_fused,
                )
        self.stashed.append(stashed)
        store = self.model._new_capture_store()
        for layer_id in self.model._capture_layer_ids:
            store[layer_id + 1] = torch.full(
                (rows + self.hidden_rows_delta, HIDDEN), float(self.forwards)
            )
        self.model._captured_hidden_states = store
        return torch.full((rows, VOCAB), float(self.forwards))


class FakeAttnBackend:
    def __init__(self, log: list | None = None) -> None:
        self.log = [] if log is None else log

    def prepare_verify_capture(self, batch, bufs) -> None:
        self.log.append(("capture", batch, bufs))
        batch.attn_metadata = _fi_metadata(initialized=True)

    def prepare_verify_replay(self, batch) -> None:
        assert not batch.attn_metadata.initialized
        batch.attn_metadata.initialized = True
        self.log.append(("replay", batch))


class FakeGraph:
    """Rewrites the static outputs in place, running no Python of the model."""

    def __init__(self, bufs: VerifyCaptureBuffer, log: list) -> None:
        self.bufs = bufs
        self.log = log
        self.replays = 0

    def replay(self) -> None:
        self.replays += 1
        self.log.append(("graph", self.replays))
        self.bufs.logits.fill_(100.0 + self.replays)
        for h in self.bufs.hidden:
            if h is not None:
                h.fill_(100.0 + self.replays)


def _fi_metadata(initialized: bool = False) -> FIMetadata:
    metadata = object.__new__(FIMetadata)
    metadata.initialized = initialized
    return metadata


def _verify_req(prompt_len: int = 10, drafted: int = ROWS - 1, output_len: int = 32) -> Req:
    req = Req(
        input_ids=torch.arange(prompt_len + 1, dtype=torch.int32),
        table_idx=1,
        cached_len=prompt_len,
        output_len=output_len,
        uid=1,
        sampling_params=None,
        cache_handle=None,
    )
    req.linear_slot_idx = 5
    open_draft_block(req, drafted)
    return req


def _verify_batch(req: Req | None = None, *, with_pool: bool = True) -> Batch:
    """The batch _prepare_batch hands forward_logits for one drafted block."""
    req = req or _verify_req()
    batch = Batch(reqs=[req], phase="decode")
    batch.padded_reqs = batch.reqs
    batch.all_logits = True
    rows = req.extend_len
    batch.input_ids = torch.arange(10, 10 + rows, dtype=torch.int32)
    batch.positions = torch.arange(req.cached_len, req.device_len, dtype=torch.int32)
    batch.out_loc = torch.arange(200, 200 + rows, dtype=torch.int32)
    batch.attn_metadata = _fi_metadata()
    if with_pool:
        batch.linear_table_idx = torch.tensor([req.linear_slot_idx], dtype=torch.int32)
        batch.fla_metadata = FLAMetadata(
            cu_seqlens=torch.tensor([0, rows]), cache_indices=batch.linear_table_idx
        )
    return batch


def _runner(model, attn, pool, **overrides) -> VerifyGraphRunner:
    kwargs = dict(
        stream="stream",
        device=CPU,
        model=model,
        attn_backend=attn,
        linear_state_pool=pool,
        moe_offload_cache=None,
        rows=ROWS,
        vocab_size=VOCAB,
        max_seq_len=MAX_SEQ,
        capture_table_idx=DUMMY_TABLE,
        sink_loc=SINK,
        target_layer_ids=TARGET_LAYERS,
    )
    kwargs.update(overrides)
    return VerifyGraphRunner(**kwargs)


def _replay_ready(pool, log: list) -> tuple[VerifyGraphRunner, FakeModel, Context]:
    """A runner as capture leaves it, over CPU buffers and a fake graph."""
    ctx = Context(1)
    model = FakeModel(ctx)
    model.enable_hidden_state_capture(TARGET_LAYERS)
    runner = _runner(model, FakeAttnBackend(log), pool)
    bufs = VerifyCaptureBuffer.init(ROWS, VOCAB, MAX_SEQ, CPU, with_state_pool=pool is not None)
    bufs.hidden = [None] * (NUM_LAYERS + 1)
    for layer_id in TARGET_LAYERS:
        bufs.hidden[layer_id + 1] = torch.zeros(ROWS, HIDDEN)
    runner.bufs = bufs
    runner.graph = FakeGraph(bufs, log)
    if pool is not None:
        runner.stash = {
            layer_id: LayerStash(
                lambda **kw: None, layer_id, 4,
                q=torch.zeros(1, ROWS, 2, 4), k=torch.zeros(1, ROWS, 2, 4),
                v=torch.zeros(1, ROWS, 2, 4), g=torch.zeros(1, ROWS, 2),
                beta=torch.zeros(1, ROWS, 2), conv_in=torch.zeros(ROWS, 6),
            )
            for layer_id in GDN_LAYERS
        }
        runner.fused = _fused
    return runner, model, ctx


# ----------------------------------------------------------------------------- buffers


def test_buffer_shapes_and_constants():
    bufs = VerifyCaptureBuffer.init(ROWS, VOCAB, MAX_SEQ, CPU, with_state_pool=True)
    for t in (bufs.input_ids, bufs.positions, bufs.out_loc):
        assert t.shape == (ROWS,) and t.dtype == torch.int32
    assert bufs.table_idx.shape == (1,) and bufs.table_idx.dtype == torch.int32
    assert bufs.logits.shape == (ROWS, VOCAB) and bufs.logits.dtype == torch.float32
    assert bufs.hidden is None

    fla = bufs.fla
    assert torch.equal(fla.cu_seqlens, torch.tensor([0, ROWS])) and fla.cu_seqlens.dtype == torch.int64
    assert fla.cache_indices is bufs.table_idx
    assert fla.has_initial_state.dtype == torch.bool and bool(fla.has_initial_state.all())
    assert fla.fresh_state_indices is None and fla.track_dst is None
    assert torch.equal(fla.chunk_indices, torch.tensor([[0, 0]]))
    assert torch.equal(fla.chunk_indices_o, torch.tensor([[0, 0]]))
    assert torch.equal(fla.chunk_offsets, torch.tensor([0, 1]))

    fi = bufs.fi
    assert torch.equal(fi.qo_indptr, torch.tensor([0, ROWS], dtype=torch.int32))
    assert fi.kv_indptr.shape == (2,) and fi.kv_indices.shape == (MAX_SEQ,)
    assert torch.equal(fi.last_page_len, torch.tensor([1], dtype=torch.int32))


def test_buffer_without_state_pool_has_no_gdn_metadata():
    bufs = VerifyCaptureBuffer.init(ROWS, VOCAB, MAX_SEQ, CPU, with_state_pool=False)
    assert bufs.fla is None
    batch = _verify_batch(with_pool=False)
    bufs.set_batch(batch, sink_loc=SINK, gdn_slot=None)
    assert batch.linear_table_idx is None and batch.fla_metadata is None


def test_set_batch_fills_the_capture_geometry():
    bufs = VerifyCaptureBuffer.init(ROWS, VOCAB, MAX_SEQ, CPU, with_state_pool=True)
    batch = _verify_batch()
    bufs.set_batch(batch, sink_loc=SINK, gdn_slot=0)
    assert batch.input_ids is bufs.input_ids and batch.positions is bufs.positions
    assert batch.out_loc is bufs.out_loc and batch.linear_table_idx is bufs.table_idx
    assert batch.fla_metadata is bufs.fla
    assert torch.equal(bufs.positions, torch.arange(1, ROWS + 1, dtype=torch.int32))
    assert bool((bufs.out_loc == SINK).all()) and int(bufs.table_idx) == 0


def test_alloc_hidden_refuses_a_store_with_the_wrong_row_count():
    bufs = VerifyCaptureBuffer.init(ROWS, VOCAB, MAX_SEQ, CPU, with_state_pool=False)
    store = [None] * (NUM_LAYERS + 1)
    store[TARGET_LAYERS[0] + 1] = torch.zeros(ROWS - 1, HIDDEN)
    store[TARGET_LAYERS[1] + 1] = torch.zeros(ROWS, HIDDEN)
    with pytest.raises(AssertionError):
        bufs.alloc_hidden(store, TARGET_LAYERS)
    with pytest.raises(AssertionError):
        bufs.alloc_hidden(None, TARGET_LAYERS)


# ---------------------------------------------------------------------------- can_replay


def test_can_replay_accepts_exactly_the_captured_shape():
    runner, _, _ = _replay_ready(FakePool(), [])
    assert runner.can_replay(_verify_batch())


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda b: setattr(b, "reqs", [_verify_req(drafted=ROWS - 2)]), id="rows-7"),
        pytest.param(lambda b: setattr(b, "reqs", b.reqs + [_verify_req()]), id="size-2"),
        pytest.param(
            lambda b: setattr(b, "padded_reqs", b.reqs + [_verify_req()]), id="padded-2"
        ),
        pytest.param(lambda b: setattr(b, "phase", "prefill"), id="prefill"),
        pytest.param(lambda b: setattr(b, "all_logits", False), id="all-logits-off"),
        pytest.param(
            lambda b: setattr(b, "attn_metadata", _fi_metadata(initialized=True)),
            id="already-planned",
        ),
        pytest.param(lambda b: setattr(b, "attn_metadata", object()), id="not-fi"),
        pytest.param(
            lambda b: setattr(b.fla_metadata, "fresh_state_indices", torch.tensor([5])),
            id="fresh-state",
        ),
        pytest.param(
            lambda b: setattr(b.fla_metadata, "track_dst", torch.tensor([2])), id="track"
        ),
        pytest.param(
            lambda b: setattr(b, "linear_table_idx", torch.tensor([5, 6], dtype=torch.int32)),
            id="two-slots",
        ),
        pytest.param(lambda b: setattr(b, "fla_metadata", None), id="no-fla"),
    ],
)
def test_can_replay_rejects_every_other_batch(mutate):
    runner, _, _ = _replay_ready(FakePool(), [])
    batch = _verify_batch()
    mutate(batch)
    assert not runner.can_replay(batch)


def test_can_replay_rejects_a_fresh_request():
    # cached_len 0 with 8 rows: the graph was captured continuing a prefix, and a fresh
    # sequence needs its state zeroed first, which the captured forward never does.
    req = Req(
        input_ids=torch.zeros(ROWS, dtype=torch.int32), table_idx=1, cached_len=0,
        output_len=8, uid=1, sampling_params=None, cache_handle=None,
    )
    req.linear_slot_idx = 5
    assert req.extend_len == ROWS
    runner, _, _ = _replay_ready(FakePool(), [])
    assert not runner.can_replay(_verify_batch(req))


def test_can_replay_without_a_state_pool_ignores_gdn_fields():
    runner, _, _ = _replay_ready(None, [])
    batch = _verify_batch(with_pool=False)
    assert runner.can_replay(batch)
    batch.fla_metadata = FLAMetadata(
        cu_seqlens=torch.tensor([0, ROWS]), cache_indices=torch.tensor([5]),
        fresh_state_indices=torch.tensor([5]),
    )
    assert runner.can_replay(batch)


def test_can_replay_is_false_before_capture():
    ctx = Context(1)
    runner = _runner(FakeModel(ctx), FakeAttnBackend(), FakePool())
    assert not runner.can_replay(_verify_batch())


# -------------------------------------------------------------------------------- replay


def test_replay_copies_inputs_plans_once_rebinds_hidden_and_adopts_the_stash(monkeypatch):
    log: list = []
    pool = FakePool()
    runner, model, ctx = _replay_ready(pool, log)
    monkeypatch.setattr(vg, "get_global_ctx", lambda: ctx)
    model.forward = lambda: None  # never called by a replay
    stale = [None] * (NUM_LAYERS + 1)
    model.model._captured_hidden_states = stale

    # The scheduler opened its own rollback on the request's live slot before the forward.
    rb = GDNRollback(pool)
    rb.open(5)
    ctx.gdn_rollback = rb
    batch = _verify_batch()

    with ctx.forward_batch(batch):
        logits = runner.replay(batch)

    bufs = runner.bufs
    assert torch.equal(bufs.input_ids, batch.input_ids)
    assert torch.equal(bufs.positions, batch.positions)
    assert torch.equal(bufs.out_loc, batch.out_loc)
    assert int(bufs.table_idx) == 5
    # plan, then replay, nothing in between (the int workspace is shared by every wrapper)
    assert log == [("replay", batch), ("graph", 1)]
    assert logits is bufs.logits and float(logits[0, 0]) == 101.0
    assert model.last_hidden_states is bufs.hidden
    assert float(bufs.hidden[TARGET_LAYERS[0] + 1][0, 0]) == 101.0
    # adopted: the same entries, in a dict of the rollback's own
    assert rb._stash == runner.stash and rb._stash is not runner.stash
    assert rb._fused is _fused
    # and a second block after the rewind still has the runner's entries to adopt
    rb.rewind(ROWS - 1)
    assert runner.stash and len(runner.stash) == len(GDN_LAYERS)
    rb.open(5)
    with ctx.forward_batch(batch):
        batch.attn_metadata = _fi_metadata()
        runner.replay(batch)
    assert rb._stash == runner.stash
    rb.close()


def test_replay_without_an_open_rollback_adopts_nothing(monkeypatch):
    log: list = []
    runner, model, ctx = _replay_ready(FakePool(), log)
    monkeypatch.setattr(vg, "get_global_ctx", lambda: ctx)
    ctx.gdn_rollback = None
    batch = _verify_batch()
    with ctx.forward_batch(batch):
        runner.replay(batch)
    assert log == [("replay", batch), ("graph", 1)]

    # a closed (not recording) rollback is left alone too
    rb = GDNRollback(FakePool())
    ctx.gdn_rollback = rb
    batch.attn_metadata = _fi_metadata()
    with ctx.forward_batch(batch):
        runner.replay(batch)
    assert rb._stash == {}


def test_replay_of_a_batch_can_replay_refuses_is_a_bug(monkeypatch):
    runner, _, ctx = _replay_ready(FakePool(), [])
    monkeypatch.setattr(vg, "get_global_ctx", lambda: ctx)
    batch = _verify_batch(_verify_req(drafted=ROWS - 2))
    with pytest.raises(AssertionError):
        runner.replay(batch)


# ------------------------------------------------------------------------------- capture


@pytest.fixture()
def cuda_stubs(monkeypatch):
    """torch.cuda graph entry points that record their arguments and run nothing."""
    calls: list = []

    class StubCUDAGraph:
        pass

    @contextmanager
    def stub_graph(graph, pool=None, stream=None):
        calls.append(("graph", graph, pool, stream))
        assert torch.cuda.is_current_stream_capturing() is False
        yield

    monkeypatch.setattr(torch.cuda, "CUDAGraph", StubCUDAGraph)
    monkeypatch.setattr(torch.cuda, "graph", stub_graph)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: calls.append(("sync",)))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return calls


def _captured(monkeypatch, cuda_stubs, *, pool, model_kwargs=None, runner_kwargs=None):
    ctx = Context(1)
    monkeypatch.setattr(vg, "get_global_ctx", lambda: ctx)
    model = FakeModel(ctx, **(model_kwargs or {}))
    model.enable_hidden_state_capture(TARGET_LAYERS)
    attn = FakeAttnBackend()
    runner = _runner(model, attn, pool, **(runner_kwargs or {}))
    return runner, model, attn, ctx


def test_capture_records_the_captured_forward_not_the_warmup(monkeypatch, cuda_stubs):
    pool = FakePool()
    runner, model, attn, ctx = _captured(monkeypatch, cuda_stubs, pool=pool)

    runner.capture()

    # warm-up then capture, both on the static batch
    assert model.forwards == 2
    (kind, batch, fi_bufs), = attn.log
    assert kind == "capture" and fi_bufs is runner.bufs.fi
    req = batch.reqs[0]
    assert batch.is_decode and batch.all_logits and batch.padded_reqs == batch.reqs
    assert req.extend_len == ROWS and req.cached_len == 1
    assert req.table_idx == DUMMY_TABLE and req.linear_slot_idx == pool.padding_slot
    bufs = runner.bufs
    assert batch.input_ids is bufs.input_ids and batch.fla_metadata is bufs.fla
    assert torch.equal(bufs.positions, torch.arange(1, ROWS + 1, dtype=torch.int32))
    assert bool((bufs.out_loc == SINK).all()) and int(bufs.table_idx) == pool.padding_slot

    # a private pool on the engine stream
    graph_calls = [c for c in cuda_stubs if c[0] == "graph"]
    assert len(graph_calls) == 1
    _, graph, mempool, stream = graph_calls[0]
    assert runner.graph is graph and mempool is None and stream == "stream"
    assert cuda_stubs.index(("sync",)) < cuda_stubs.index(graph_calls[0])

    # outputs of the captured (second) forward
    assert float(bufs.logits[0, 0]) == 2.0
    for layer_id in TARGET_LAYERS:
        assert bufs.hidden[layer_id + 1].shape == (ROWS, HIDDEN)
        assert float(bufs.hidden[layer_id + 1][0, 0]) == 2.0
    assert bufs.hidden[0] is None and bufs.hidden[TARGET_LAYERS[0]] is None

    # the stash is the captured forward's entries, snapshotted before the block closed
    assert set(runner.stash) == set(GDN_LAYERS)
    for layer_id, entry in runner.stash.items():
        assert entry.q is model.stashed[1][layer_id]
        assert entry.q.shape[1] == ROWS
    assert runner.fused is _fused

    # the capture-only rollback opened twice on the padding slot and gave its slot back
    scratch = pool.freed[0]
    assert pool.copies == [(pool.padding_slot, scratch), (pool.padding_slot, scratch)]
    assert pool.freed == [scratch] and ctx.gdn_rollback is None
    assert ctx._batch is None


def test_capture_without_a_state_pool_uses_no_rollback(monkeypatch, cuda_stubs):
    runner, model, attn, ctx = _captured(monkeypatch, cuda_stubs, pool=None)
    runner.capture()
    assert model.forwards == 2 and runner.stash == {} and runner.fused is None
    assert runner.bufs.fla is None
    batch = attn.log[0][1]
    assert batch.linear_table_idx is None and batch.fla_metadata is None
    assert batch.reqs[0].linear_slot_idx is None
    assert runner.can_replay(_verify_batch(with_pool=False))


def test_capture_refuses_hidden_states_of_the_wrong_shape(monkeypatch, cuda_stubs):
    pool = FakePool()
    runner, model, attn, ctx = _captured(
        monkeypatch, cuda_stubs, pool=pool, model_kwargs={"hidden_rows_delta": -1}
    )
    with pytest.raises(AssertionError):
        runner.capture()
    # the failed capture still gave the scratch slot back and left no rollback behind
    assert len(pool.freed) == 1 and ctx.gdn_rollback is None and ctx._batch is None
    assert runner.graph is None and not runner.can_replay(_verify_batch())


def test_capture_refuses_a_stash_missing_a_layer(monkeypatch, cuda_stubs):
    pool = FakePool()
    runner, model, attn, ctx = _captured(
        monkeypatch, cuda_stubs, pool=pool, model_kwargs={"gdn_layers": GDN_LAYERS[:1]}
    )
    with pytest.raises(AssertionError, match="stashed 1 of 2"):
        runner.capture()
    assert runner.graph is None


def test_capture_resets_the_moe_offload_cache_around_the_forwards(monkeypatch, cuda_stubs):
    resets: list[int] = []
    runner, model, attn, ctx = _captured(
        monkeypatch, cuda_stubs, pool=None,
        runner_kwargs={"moe_offload_cache": SimpleNamespace(reset=lambda: resets.append(model.forwards))},
    )
    runner.capture()
    # before the warm-up (so the capture replays cold-cache copies) and after the capture
    assert resets == [0, 2]


def test_capture_twice_is_refused(monkeypatch, cuda_stubs):
    runner, *_ = _captured(monkeypatch, cuda_stubs, pool=None)
    runner.capture()
    with pytest.raises(AssertionError):
        runner.capture()


# ------------------------------------------------------------------------------- destroy


def test_destroy_drops_the_graph_and_unbinds_the_static_store():
    runner, model, ctx = _replay_ready(FakePool(), [])
    model.model._captured_hidden_states = runner.bufs.hidden
    runner.destroy()
    assert runner.graph is None and runner.bufs is None
    assert runner.stash == {} and runner.fused is None
    assert model.last_hidden_states is None
    assert not runner.can_replay(_verify_batch())


def test_destroy_leaves_an_eager_store_alone():
    runner, model, ctx = _replay_ready(FakePool(), [])
    with ctx.forward_batch(_verify_batch()):
        model.forward()
    eager = model.last_hidden_states
    runner.destroy()
    assert model.last_hidden_states is eager
