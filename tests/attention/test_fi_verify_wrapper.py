"""FlashInferBackend.prepare_verify_capture / prepare_verify_replay: the graph-mode
prefill wrapper for the K+1-row speculative verify (plan runs outside the graph).

CPU only: ``flashinfer`` is replaced by a fake module whose wrapper classes record
constructor kwargs and plan() calls, the backend is built without ``__init__``, and
pinned host allocations are disabled (they would initialize a CUDA context).
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROWS = 8
DEVICE_LEN = 37
MAX_SEQ = 64


class FakePrefillWrapper:
    def __init__(self, float_workspace_buffer, **kwargs):
        self.float_workspace_buffer = float_workspace_buffer
        self.kwargs = kwargs
        self._int_workspace_buffer = torch.empty(4, dtype=torch.uint8)
        self.plans = []

    def plan(self, **kwargs):
        self.plans.append(kwargs)


class FakeDecodeWrapper:
    def __init__(self, *args, **kwargs):
        self.plans = []

    def plan(self, **kwargs):
        self.plans.append(kwargs)


class FakeGraphDecodeWrapper(FakeDecodeWrapper):
    pass


class FakeEvent:
    def __init__(self):
        self.synchronized = 0
        self.recorded = 0

    def synchronize(self):
        self.synchronized += 1

    def record(self):
        self.recorded += 1


def _strip_pin(fn):
    def wrapped(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return fn(*args, **kwargs)

    return wrapped


@pytest.fixture()
def fi(monkeypatch):
    fake = ModuleType("flashinfer")
    fake.BatchPrefillWithPagedKVCacheWrapper = FakePrefillWrapper
    fake.BatchDecodeWithPagedKVCacheWrapper = FakeDecodeWrapper
    fake.CUDAGraphBatchDecodeWithPagedKVCacheWrapper = FakeGraphDecodeWrapper
    monkeypatch.setitem(sys.modules, "flashinfer", fake)

    from freetoken.attention import fi as mod

    monkeypatch.setattr(mod.FIMetadata, "__post_init__", lambda self: None)
    for name in ("tensor", "arange", "ones"):
        monkeypatch.setattr(torch, name, _strip_pin(getattr(torch, name)))
    page_table = torch.arange(4 * MAX_SEQ, dtype=torch.int32).view(4, MAX_SEQ)
    monkeypatch.setattr(mod, "get_global_ctx", lambda: SimpleNamespace(page_table=page_table))
    return mod


@pytest.fixture()
def backend(fi):
    b = fi.FlashInferBackend.__new__(fi.FlashInferBackend)
    b.config = SimpleNamespace(head_dim=128, num_qo_heads=32, num_kv_heads=8)
    b.kvcache = SimpleNamespace(dtype=torch.bfloat16, device=torch.device("cpu"))
    b.device = torch.device("cpu")
    b.float_workspace_buffer = torch.empty(16, dtype=torch.uint8)
    b.prefill_wrapper = FakePrefillWrapper(b.float_workspace_buffer)
    b.decode_wrappers = FakeDecodeWrapper()
    b.int_workspace_buffer = b.prefill_wrapper._int_workspace_buffer
    b.qo_head_local, b.kv_head_local = 32, 8
    b.cached_ones_cpu = torch.tensor([], dtype=torch.int32)
    b.capture_bs, b.max_graph_bs, b.graph_wrappers, b.capture = [], 0, {}, None
    b.verify_wrapper, b.verify_bufs, b.verify_rows = None, None, 0
    b.last_event = FakeEvent()
    return b


def _batch(rows=ROWS, device_len=DEVICE_LEN, table_idx=1, is_decode=True):
    req = SimpleNamespace(
        extend_len=rows, device_len=device_len, cached_len=device_len - rows, table_idx=table_idx
    )
    return SimpleNamespace(padded_reqs=[req], is_decode=is_decode, attn_metadata=None)


def _bufs(fi):
    return fi.FIVerifyBuffers.create(MAX_SEQ, torch.device("cpu"))


def test_verify_buffers_create_shapes(fi):
    bufs = _bufs(fi)
    for t, shape in (
        (bufs.qo_indptr, (2,)),
        (bufs.kv_indptr, (2,)),
        (bufs.kv_indices, (MAX_SEQ,)),
        (bufs.last_page_len, (1,)),
    ):
        assert t.shape == shape and t.dtype == torch.int32
    assert bufs.qo_indptr.tolist() == [0, ROWS] and bufs.last_page_len.tolist() == [1]


def test_prepare_verify_capture_builds_graph_wrapper_and_plans_outside(fi, backend):
    bufs = _bufs(fi)
    batch = _batch()
    backend.prepare_verify_capture(batch, bufs)

    w = backend.verify_wrapper
    assert isinstance(w, FakePrefillWrapper) and w is not backend.prefill_wrapper
    assert w.float_workspace_buffer is backend.float_workspace_buffer
    assert w.kwargs["use_cuda_graph"] is True
    assert w.kwargs["backend"] == "fa2" and w.kwargs["kv_layout"] == "NHD"
    assert w.kwargs["qo_indptr_buf"] is bufs.qo_indptr
    assert w.kwargs["paged_kv_indptr_buf"] is bufs.kv_indptr
    assert w.kwargs["paged_kv_indices_buf"] is bufs.kv_indices
    assert w.kwargs["paged_kv_last_page_len_buf"] is bufs.last_page_len
    # every wrapper plans into the one int workspace (same hack as the decode graphs)
    assert w._int_workspace_buffer is backend.int_workspace_buffer
    assert backend.verify_bufs is bufs and backend.verify_rows == ROWS

    m = batch.attn_metadata
    assert isinstance(m, fi.FIMetadata) and m.wrapper is w and m.initialized
    assert len(w.plans) == 1 and backend.prefill_wrapper.plans == []
    plan = w.plans[0]
    assert plan["qo_indptr"].tolist() == [0, ROWS]
    assert plan["paged_kv_indptr"].tolist() == [0, DEVICE_LEN]
    assert plan["paged_kv_last_page_len"].tolist() == [1]
    assert plan["paged_kv_indices"].tolist() == list(range(MAX_SEQ, MAX_SEQ + DEVICE_LEN))
    assert plan["causal"] is True and plan["non_blocking"] is True
    assert backend.last_event.synchronized == 1 and backend.last_event.recorded == 1


def test_prepare_verify_capture_twice_raises(fi, backend):
    backend.prepare_verify_capture(_batch(), _bufs(fi))
    with pytest.raises(AssertionError):
        backend.prepare_verify_capture(_batch(), _bufs(fi))


def test_prepare_verify_capture_rejects_single_row_batch(fi, backend):
    # a 1-row capture would freeze _max_total_num_rows at 1 and refuse every verify
    with pytest.raises(AssertionError):
        backend.prepare_verify_capture(_batch(rows=1), _bufs(fi))
    # nothing was built: the engine can retry after logging the failed capture
    assert backend.verify_wrapper is None and backend.verify_rows == 0
    backend.prepare_verify_capture(_batch(), _bufs(fi))
    assert backend.verify_rows == ROWS


def test_prepare_verify_replay_swaps_wrapper_and_plans_once(fi, backend):
    backend.prepare_verify_capture(_batch(), _bufs(fi))
    w = backend.verify_wrapper

    batch = _batch(device_len=DEVICE_LEN + 5, table_idx=2)
    backend.prepare_metadata(batch)
    m = batch.attn_metadata
    assert m.wrapper is backend.prefill_wrapper and not m.initialized

    backend.prepare_verify_replay(batch)
    assert m.wrapper is w and m.initialized
    assert len(w.plans) == 2 and backend.prefill_wrapper.plans == []
    plan = w.plans[-1]
    assert plan["qo_indptr"].tolist() == [0, ROWS]
    assert plan["paged_kv_indptr"].tolist() == [0, DEVICE_LEN + 5]
    assert plan["paged_kv_indices"].tolist() == list(range(2 * MAX_SEQ, 2 * MAX_SEQ + DEVICE_LEN + 5))
    assert backend.last_event.synchronized == 2 and backend.last_event.recorded == 2

    # the same metadata is planned once: a second call must not re-plan silently
    with pytest.raises(AssertionError):
        backend.prepare_verify_replay(batch)
    assert len(w.plans) == 2


def test_prepare_verify_replay_rejects_initialized_metadata(fi, backend):
    backend.prepare_verify_capture(_batch(), _bufs(fi))
    batch = _batch()
    backend.prepare_metadata(batch)
    backend._initialize_metadata_once(batch.attn_metadata)  # eager path already planned
    assert len(backend.prefill_wrapper.plans) == 1
    with pytest.raises(AssertionError):
        backend.prepare_verify_replay(batch)
    assert batch.attn_metadata.wrapper is backend.prefill_wrapper


def test_prepare_verify_replay_rejects_decode_wrapper_metadata(fi, backend):
    backend.prepare_verify_capture(_batch(), _bufs(fi))
    batch = _batch(rows=1)  # plain decode: prepare_metadata selects the decode wrapper
    backend.prepare_metadata(batch)
    assert batch.attn_metadata.wrapper is backend.decode_wrappers
    with pytest.raises(AssertionError):
        backend.prepare_verify_replay(batch)
    assert not batch.attn_metadata.initialized and len(backend.verify_wrapper.plans) == 1


def test_prepare_verify_replay_rejects_row_count_mismatch(fi, backend):
    # flashinfer only refuses MORE rows than the first plan; 7 rows would replay a
    # graph captured for 8
    backend.prepare_verify_capture(_batch(), _bufs(fi))
    batch = _batch(rows=ROWS - 1)
    backend.prepare_metadata(batch)
    assert batch.attn_metadata.wrapper is backend.prefill_wrapper
    with pytest.raises(AssertionError):
        backend.prepare_verify_replay(batch)
    assert len(backend.verify_wrapper.plans) == 1


def test_prepare_verify_replay_rejects_kv_beyond_static_buffer(fi, backend):
    backend.prepare_verify_capture(_batch(), _bufs(fi))
    page_table = torch.arange(4 * 2 * MAX_SEQ, dtype=torch.int32).view(4, 2 * MAX_SEQ)
    fi.get_global_ctx = lambda: SimpleNamespace(page_table=page_table)
    batch = _batch(device_len=MAX_SEQ + 1)
    backend.prepare_metadata(batch)
    with pytest.raises(AssertionError):
        backend.prepare_verify_replay(batch)


def test_prepare_verify_replay_before_capture_raises(fi, backend):
    batch = _batch()
    backend.prepare_metadata(batch)
    with pytest.raises(AssertionError):
        backend.prepare_verify_replay(batch)
    assert batch.attn_metadata.wrapper is backend.prefill_wrapper and not batch.attn_metadata.initialized


def test_reset_capture_drops_verify_wrapper(fi, backend):
    backend.prepare_verify_capture(_batch(), _bufs(fi))
    backend.graph_wrappers = {1: FakeGraphDecodeWrapper()}
    backend.capture, backend.capture_bs, backend.max_graph_bs = object(), [1], 1
    backend.reset_capture()
    assert backend.verify_wrapper is None and backend.verify_bufs is None
    assert backend.verify_rows == 0
    assert backend.graph_wrappers == {} and backend.capture is None
    assert backend.capture_bs == [] and backend.max_graph_bs == 0
    # re-armable after a rebuild, like init_capture_graph
    backend.prepare_verify_capture(_batch(), _bufs(fi))
    assert backend.verify_wrapper is not None


def test_decode_replay_path_ignores_verify_wrapper(fi, backend):
    backend.prepare_verify_capture(_batch(), _bufs(fi))
    graph_w = FakeGraphDecodeWrapper()
    backend.graph_wrappers, backend.capture_bs, backend.capture = {1: graph_w}, [1], object()
    batch = _batch(rows=1)
    batch.padded_size = 1
    backend.prepare_metadata(batch)
    backend.prepare_for_replay(batch)
    assert batch.attn_metadata.wrapper is graph_w and len(graph_w.plans) == 1
    assert len(backend.verify_wrapper.plans) == 1
