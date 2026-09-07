"""Replaying the K+1-row speculative verify forward as one CUDA graph.

A draft block is verified by a single forward over the pending token plus the K candidates:
one request, K + 1 rows, in the decode phase. Eagerly that forward is a few thousand kernel
launches, and on a small block the launch gaps are a quarter of its span. The decode graphs in
``engine/graph.py`` cannot serve it: they were captured over one row per request.

This runner captures that forward once, over static buffers, and replays it. What makes it
different from the decode graphs is what the eager verify does in Python besides launching
kernels, none of which a replay repeats:

- the attention wrapper plans inside the forward; here it plans outside, into the static index
  buffers of a graph-mode wrapper, right before every replay (``FlashInferBackend``);
- the linear layers stash their recurrence inputs into the open ``GDNRollback`` so a partly
  rejected block can be rewound; a replay stashes nothing, so the entries recorded at capture
  time -- views into the graph's private pool, rewritten by every replay -- are adopted into
  the scheduler's open block after each replay;
- the model publishes its hidden states by assigning a fresh list; the captured forward copies
  them into static buffers and the model's store is rebound to those after every replay.

The graph owns a private memory pool: the stash entries are views into allocations made
during capture, and a later capture sharing the pool would hand those addresses to something
else.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence

import torch
from freetoken.attention.fi import FIMetadata, FIVerifyBuffers
from freetoken.attention.linear import FLAMetadata, build_fla_chunk_indices
from freetoken.core import Batch, Req, get_global_ctx
from freetoken.engine.gdn_rollback import GDNRollback, LayerStash
from freetoken.utils import init_logger, mem_GB

if TYPE_CHECKING:
    from freetoken.attention.fi import FlashInferBackend
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.models import BaseLLMModel
    from freetoken.moe.offload_cache import OffloadMoeCache

logger = init_logger(__name__)


@dataclass
class VerifyCaptureBuffer:
    """Static inputs and outputs of the captured verify forward, one request of ``rows``."""

    rows: int
    input_ids: torch.Tensor  # int32 [rows]
    positions: torch.Tensor  # int32 [rows]
    out_loc: torch.Tensor  # int32 [rows]
    table_idx: torch.Tensor  # int32 [1]: the request's GDN state slot
    logits: torch.Tensor  # float32 [rows, vocab]
    fi: FIVerifyBuffers
    # Static GDN metadata over the buffers above: everything but the slot is a constant of
    # the shape (one sequence of ``rows`` continuing a cached prefix). None without a pool.
    fla: FLAMetadata | None
    # The model's hidden-state store layout (entry i + 1 = layer i), with a static tensor at
    # every layer the draft reads and None elsewhere. Allocated from the warm-up forward,
    # which is the first time the shapes are known.
    hidden: list[torch.Tensor | None] | None = None

    @classmethod
    def init(
        cls,
        rows: int,
        vocab_size: int,
        max_seq_len: int,
        device: torch.device,
        *,
        with_state_pool: bool,
    ) -> VerifyCaptureBuffer:
        i32 = {"dtype": torch.int32, "device": device}
        table_idx = torch.zeros(1, **i32)
        fla = None
        if with_state_pool:
            # The chunk bookkeeping is what the fla kernels would otherwise derive from
            # cu_seqlens with a host readback, which the capture guard refuses.
            chunks = build_fla_chunk_indices([rows], device, pin_memory=False)
            fla = FLAMetadata(
                cu_seqlens=torch.tensor([0, rows], dtype=torch.int64, device=device),
                cache_indices=table_idx,
                has_initial_state=torch.ones(1, dtype=torch.bool, device=device),
                **chunks,
            )
        return cls(
            rows=rows,
            input_ids=torch.zeros(rows, **i32),
            positions=torch.zeros(rows, **i32),
            out_loc=torch.zeros(rows, **i32),
            table_idx=table_idx,
            logits=torch.empty(rows, vocab_size, dtype=torch.float32, device=device),
            fi=FIVerifyBuffers.create(max_seq_len, device, rows=rows),
            fla=fla,
        )

    def set_batch(self, batch: Batch, *, sink_loc: int, gdn_slot: int | None) -> None:
        """Point the capture batch at the buffers and fill them with the capture values."""
        batch.input_ids = self.input_ids
        batch.positions = self.positions
        batch.out_loc = self.out_loc
        # The capture request continues a one-token prefix, so its rows sit at 1..rows and
        # its KV lands on the sink page nothing reads back.
        self.positions.copy_(torch.arange(1, self.rows + 1, dtype=torch.int32))
        self.out_loc.fill_(sink_loc)
        if self.fla is not None:
            assert gdn_slot is not None
            self.table_idx.fill_(gdn_slot)
            batch.linear_table_idx = self.table_idx
            batch.fla_metadata = self.fla

    def copy_from(self, batch: Batch) -> None:
        self.input_ids.copy_(batch.input_ids)
        self.positions.copy_(batch.positions)
        self.out_loc.copy_(batch.out_loc)
        if self.fla is not None:
            # The only GDN input that changes between blocks: which slot the state lives in.
            self.table_idx.copy_(batch.linear_table_idx)

    def alloc_hidden(
        self, store: Sequence[torch.Tensor | None] | None, target_layer_ids: Sequence[int]
    ) -> None:
        assert store is not None, "the verify capture needs hidden-state capture enabled"
        hidden: list[torch.Tensor | None] = [None] * len(store)
        for layer_id in target_layer_ids:
            published = store[layer_id + 1]
            assert published is not None and published.shape[0] == self.rows, (
                f"layer {layer_id} published {None if published is None else tuple(published.shape)} "
                f"for a {self.rows}-row verify"
            )
            hidden[layer_id + 1] = torch.empty_like(published)
        self.hidden = hidden


class VerifyGraphRunner:
    """One captured verify forward (one request, ``rows`` rows) and its replay."""

    def __init__(
        self,
        *,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: FlashInferBackend,
        linear_state_pool: LinearStatePool | None,
        moe_offload_cache: OffloadMoeCache | None,
        rows: int,
        vocab_size: int,
        max_seq_len: int,
        capture_table_idx: int,
        sink_loc: int,
        target_layer_ids: Sequence[int],
    ) -> None:
        assert rows > 1, "a verify window is the pending token plus at least one candidate"
        self.stream = stream
        self.device = device
        self.model = model
        self.attn_backend = attn_backend
        self.pool = linear_state_pool
        self.moe_offload_cache = moe_offload_cache
        self.rows = rows
        self.replays = 0
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.capture_table_idx = capture_table_idx
        self.sink_loc = sink_loc
        self.target_layer_ids = tuple(target_layer_ids)
        self.bufs: VerifyCaptureBuffer | None = None
        self.graph: torch.cuda.CUDAGraph | None = None
        # Recorded at capture time and installed into the scheduler's open block after every
        # replay. Held for the graph's lifetime: the entries view into the graph's private
        # pool, and the references are what keep those allocations reserved.
        self.stash: dict[int, LayerStash] = {}
        self.fused: Callable[..., None] | None = None

    # ------------------------------------------------------------------ capture

    def _capture_req(self) -> Req:
        # Continues a one-token prefix so the linear layers take the continuing-state path:
        # a fresh request (cached_len 0) would bake a state zeroing into the graph, and the
        # GDN state of a request mid-decode is never fresh.
        req = Req(
            input_ids=torch.zeros(self.rows + 1, dtype=torch.int32, device="cpu"),
            table_idx=self.capture_table_idx,
            cached_len=1,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore[arg-type]
            cache_handle=None,  # type: ignore[arg-type]
        )
        if self.pool is not None:
            req.linear_slot_idx = self.pool.padding_slot
        return req

    def _reset_moe_offload_cache(self) -> None:
        if self.moe_offload_cache is not None:
            self.moe_offload_cache.reset()

    def capture(self) -> None:
        assert self.graph is None, "verify graph already captured"
        bufs = VerifyCaptureBuffer.init(
            self.rows,
            self.vocab_size,
            self.max_seq_len,
            self.device,
            with_state_pool=self.pool is not None,
        )
        req = self._capture_req()
        batch = Batch(reqs=[req], phase="decode")
        batch.padded_reqs = batch.reqs
        batch.all_logits = True
        bufs.set_batch(batch, sink_loc=self.sink_loc, gdn_slot=req.linear_slot_idx)
        # Plans outside the graph, and this is the wrapper's first plan: it freezes the row
        # count the graph is captured for.
        self.attn_backend.prepare_verify_capture(batch, bufs.fi)

        free_before = _free_memory(self.device)
        logger.info_rank0(
            f"Capturing the {self.rows}-row verify graph; free memory {mem_GB(free_before)}"
        )
        self._reset_moe_offload_cache()
        # A capture-only recorder: the scheduler's own instance is not open here, and the
        # entries this one records are what every replay hands to it.
        rb = GDNRollback(self.pool) if self.pool is not None else None
        ctx = get_global_ctx()
        graph = torch.cuda.CUDAGraph()
        # Warm-up and capture share one forward_batch (nesting is refused) and one open
        # rollback each: open() twice raises, so the block is closed and reopened between.
        with ctx.forward_batch(batch):
            try:
                if rb is not None:
                    rb.open(req.linear_slot_idx)
                    ctx.gdn_rollback = rb
                # Eager first, on the same static batch: autotuning, lazy buffers and the
                # kernels' identity caches must all be settled before the capture.
                bufs.logits.copy_(self.model.forward())
                bufs.alloc_hidden(self.model.last_hidden_states, self.target_layer_ids)
                if rb is not None:
                    rb.close()
                    rb.open(req.linear_slot_idx)
                torch.cuda.synchronize(self.device)
                with torch.cuda.graph(graph, pool=None, stream=self.stream):
                    bufs.logits.copy_(self.model.forward())
                    store = self.model.last_hidden_states
                    for layer_id in self.target_layer_ids:
                        bufs.hidden[layer_id + 1].copy_(store[layer_id + 1])
                if rb is not None:
                    # Snapshot before close(): close() empties the recorder's dict.
                    self.stash = dict(rb._stash)
                    self.fused = rb._fused
            finally:
                ctx.gdn_rollback = None
                if rb is not None:
                    rb.release()
        self._reset_moe_offload_cache()

        if self.pool is not None:
            assert len(self.stash) == self.pool.num_linear_layers, (
                f"the capture stashed {len(self.stash)} of "
                f"{self.pool.num_linear_layers} linear layers"
            )
            for layer_id, entry in self.stash.items():
                assert entry.q.shape[1] == self.rows, (
                    f"layer {layer_id} stashed a {entry.q.shape[1]}-row window"
                )
        self.graph = graph
        self.bufs = bufs
        free_after = _free_memory(self.device)
        logger.info_rank0(
            f"Verify graph captured; free memory {mem_GB(free_after)} "
            f"({mem_GB(free_before - free_after)} used)"
        )

    # ------------------------------------------------------------------ replay

    def can_replay(self, batch: Batch) -> bool:
        """Exactly the batches the captured graph was made for; anything else stays eager.

        The common miss is a block truncated near the request's output budget (fewer rows);
        the rest guard against a batch that would silently replay with the wrong geometry.
        """
        if self.graph is None:
            return False
        if not (batch.is_decode and batch.size == 1 and batch.padded_size == 1):
            return False
        req = batch.reqs[0]
        if req.extend_len != self.rows or req.cached_len <= 0 or not batch.all_logits:
            return False
        metadata = getattr(batch, "attn_metadata", None)
        if not isinstance(metadata, FIMetadata) or metadata.initialized:
            return False
        if self.pool is not None:
            fla = batch.fla_metadata
            if batch.linear_table_idx is None or batch.linear_table_idx.numel() != 1:
                return False
            if fla is None or fla.fresh_state_indices is not None or fla.track_dst is not None:
                return False
        return True

    def replay(self, batch: Batch) -> torch.Tensor:
        # A mismatch past can_replay is a bug, not a case to fall back on: the buffers would
        # be filled with the wrong rows and the forward would run anyway.
        assert self.can_replay(batch)
        assert self.graph is not None and self.bufs is not None
        bufs = self.bufs
        bufs.copy_from(batch)
        # Plan and replay back to back: every wrapper plans into the one int workspace.
        self.attn_backend.prepare_verify_replay(batch)
        self.graph.replay()
        # Counted and announced once, because a graph that never replays looks exactly like
        # a working one from the outside (same text, same timing): the first run of this
        # feature was captured for the wrong row count and no log line said so.
        self.replays += 1
        if self.replays == 1:
            logger.info_rank0("Verify graph replay active (first replay)")
        # The replay rewrote the static hidden buffers, but the model's store still names
        # the list the last eager forward built.
        self.model.model._captured_hidden_states = bufs.hidden
        rb = get_global_ctx().gdn_rollback
        if rb is not None and rb.recording:
            rb.adopt(self.stash, self.fused)
        return bufs.logits

    # ------------------------------------------------------------------ teardown

    def destroy(self) -> None:
        """Drop the graph, its private pool and the static buffers (before a pool rebuild)."""
        self.graph = None
        self.stash = {}
        self.fused = None
        inner = getattr(self.model, "model", None)
        if (
            inner is not None
            and self.bufs is not None
            and self.bufs.hidden is not None
            and getattr(inner, "_captured_hidden_states", None) is self.bufs.hidden
        ):
            # Through the store one small tensor would keep a freed buffer alive; the next
            # forward publishes a fresh list.
            inner._captured_hidden_states = None
        self.bufs = None
        gc.collect()


def _free_memory(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    return torch.cuda.mem_get_info(device)[0]


__all__ = ["VerifyCaptureBuffer", "VerifyGraphRunner"]
