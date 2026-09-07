"""Replaying the DFlash2 draft forward as one CUDA graph per context-row count.

A draft block is one forward of the five-layer draft over ``c`` context rows (the target's
hidden states for the tokens accepted last block) plus eight noise rows. Eagerly that is a
few hundred kernel launches for about four milliseconds of GPU work, and the launch gaps are
a third of its span. This runner captures the forward once per ``c`` in ``1..block`` over
static buffers and replays the matching graph.

What makes it cheap to keep correct is that the captured body is the eager body: the runner's
``_run_block`` runs the same ops in the same shapes on the same ``StaticDraftCache`` whether
called eagerly, for warm-up, under capture or for the shadow comparison. Two consequences
shape this file:

- one graph per ``c`` rather than one graph with ``c`` padded to ``block``: the GGUF matmuls
  pick their reduction order from the row count and the hidden norm reduces over the rows it
  is given, so a padded forward is not bitwise the eager one, and bitwise is what shadow mode
  asserts;
- the graph bakes no hidden-state address. The target publishes its hidden states from
  whichever store ran last (the verify graph's static buffers, a decode graph's, an eager
  list), so every replay copies the ``c`` rows into a buffer this runner owns.

The eight graphs share one memory pool: nothing allocated inside the body outlives a replay
(the outputs are copied into static buffers, the K/V ring lives in the runner), so a later
graph reusing an earlier graph's blocks is safe.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch
from freetoken.utils import init_logger, mem_GB

if TYPE_CHECKING:
    from freetoken.engine.draft_runner import DFlashRunner

logger = init_logger(__name__)

# Block start the graphs are captured at. Every capture stages positions
# [CAPTURE_SEQ_LEN - c, CAPTURE_SEQ_LEN + block) and the ring refuses nothing there; it only
# has to keep them non-negative for the largest c.
CAPTURE_SEQ_LEN = 16


@dataclass
class DraftCaptureBuffer:
    """Static inputs and outputs of the captured draft forwards, one request, one block.

    ``th`` holds the concatenated target hidden states of the context rows; graph ``c`` reads
    the view ``th[:, :c]``, which shares the buffer's base address. ``seq_len`` is the
    runner's own block-start tensor, so the eager and captured bodies read one buffer. The
    outputs are allocated from the first warm-up forward, the first time their shapes are
    known.
    """

    block: int
    hidden: int  # width of one target layer's hidden state, one column block of ``th``
    th: torch.Tensor  # [1, block, layers * hidden] in the draft's dtype
    ids: torch.Tensor  # int64 [1, block]: the anchor token then mask tokens
    seq_len: torch.Tensor  # int64 [1], shared with the runner
    mask_token_id: int
    tokens: torch.Tensor | None = None  # int64 [1, block - 1]
    probs: torch.Tensor | None = None  # [1, block - 1, vocab]

    @classmethod
    def init(
        cls,
        *,
        block: int,
        features: int,
        num_target_layers: int,
        mask_token_id: int,
        seq_len: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> DraftCaptureBuffer:
        assert block > 1, "a draft block is the anchor plus at least one candidate"
        assert num_target_layers > 0 and features % num_target_layers == 0, (
            f"the draft consumes {features} features, not a multiple of "
            f"{num_target_layers} target layers"
        )
        assert seq_len.dtype == torch.int64 and seq_len.shape == (1,), (
            f"the block start must be int64 [1], got {seq_len.dtype} {tuple(seq_len.shape)}"
        )
        return cls(
            block=block,
            hidden=features // num_target_layers,
            th=torch.zeros(1, block, features, dtype=dtype, device=device),
            ids=torch.full((1, block), mask_token_id, dtype=torch.int64, device=device),
            seq_len=seq_len,
            mask_token_id=mask_token_id,
        )

    def set_capture_inputs(self) -> None:
        """Values the warm-ups and captures run on; every replay overwrites what it reads."""
        self.seq_len.fill_(CAPTURE_SEQ_LEN)
        # Zero context is finite through the whole body (fc(0) -> RMSNorm(0) = 0), which is
        # what matters: the captured values are never replayed, but a NaN would be.
        self.th.zero_()
        self.ids.fill_(self.mask_token_id)
        self.ids[0, 0] = 0

    def alloc_outputs(self, tokens: torch.Tensor, probs: torch.Tensor) -> None:
        """Size the output buffers from the first warm-up's outputs."""
        candidates = self.block - 1
        assert tokens.shape == (1, candidates) and probs.dim() == 3, (
            f"the draft forward returned tokens {tuple(tokens.shape)} and probs "
            f"{tuple(probs.shape)} for a block of {self.block}"
        )
        assert probs.shape[:2] == (1, candidates), (
            f"probs {tuple(probs.shape)} do not cover the {candidates} candidates"
        )
        # Allocated as ordinary tensors even though capture runs under inference_mode: an
        # inference tensor refuses in-place updates outside that mode, and the scheduler
        # keeps these buffers after draft() returns.
        with torch.inference_mode(False):
            self.tokens = torch.empty(tokens.shape, dtype=tokens.dtype, device=tokens.device)
            self.probs = torch.empty(probs.shape, dtype=probs.dtype, device=probs.device)

    def copy_outputs(self, tokens: torch.Tensor, probs: torch.Tensor) -> None:
        """The copies recorded into each graph, so a replay lands in the static buffers."""
        assert self.tokens is not None and self.probs is not None
        # copy_ broadcasts: a [1, 1] result would silently fill every candidate slot.
        assert tokens.shape == self.tokens.shape and probs.shape == self.probs.shape, (
            f"a forward returned tokens {tuple(tokens.shape)} probs {tuple(probs.shape)}, "
            f"the buffers hold {tuple(self.tokens.shape)} {tuple(self.probs.shape)}"
        )
        self.tokens.copy_(tokens)
        self.probs.copy_(probs)


class DraftGraphRunner:
    """The captured draft forwards of one ``DFlashRunner``, one graph per context-row count."""

    def __init__(
        self, *, stream: torch.cuda.Stream, device: torch.device, runner: DFlashRunner
    ) -> None:
        self.stream = stream
        self.device = device
        self.runner = runner
        self.block = int(runner.block_size)
        self.target_layer_ids: tuple[int, ...] = tuple(runner.target_layer_ids)
        self.replays = 0
        self.bufs: DraftCaptureBuffer | None = None
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.pool = None

    # ------------------------------------------------------------------ capture

    def capture(self) -> None:
        assert not self.graphs, "draft graphs already captured"
        runner = self.runner
        # A DynamicCache does not fail under capture, it freezes: the recorded cat keeps the
        # capture-time length and every replay drafts against that context.
        assert runner._static_cache is not None, (
            "the draft graph needs the static ring cache; this draft runs on a DynamicCache"
        )
        bufs = DraftCaptureBuffer.init(
            block=self.block,
            features=int(runner.draft_model.fc.in_features),
            num_target_layers=len(self.target_layer_ids),
            mask_token_id=int(runner.mask_token_id),
            seq_len=runner._seq_len_t,
            device=self.device,
            dtype=runner.dtype,
        )
        free_before = _free_memory(self.device)
        logger.info_rank0(
            f"Capturing {self.block} draft graphs (1..{self.block} context rows); "
            f"free memory {mem_GB(free_before)}"
        )
        graphs: dict[int, torch.cuda.CUDAGraph] = {}
        pool = None
        # The eager draft runs under inference_mode (draft()); the captured body must too, or
        # the replayed ops carry autograd bookkeeping the eager ones never had.
        with torch.inference_mode():
            self._reset_ring()
            bufs.set_capture_inputs()
            try:
                for c in range(1, self.block + 1):
                    th = bufs.th[:, :c]
                    # Eager first, on the same buffers and shapes: lazy buffers (the embedding
                    # scale, cuBLAS workspaces, the SDPA kernel for this shape) must exist
                    # before the capture, which refuses allocations it cannot replay.
                    tokens, probs, _ = runner._run_block(th, bufs.ids, c, 0.0, 1.0, 0)
                    if bufs.tokens is None:
                        bufs.alloc_outputs(tokens, probs)
                    torch.cuda.synchronize(self.device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                        tokens, probs, _ = runner._run_block(th, bufs.ids, c, 0.0, 1.0, 0)
                        bufs.copy_outputs(tokens, probs)
                    if pool is None:
                        pool = graph.pool()
                    graphs[c] = graph
            finally:
                # The warm-ups and captures staged rows into the ring; they are finite and the
                # reset masks them, and the cleared owner makes the next draft() reset again.
                self._reset_ring()
        self.graphs = graphs
        self.pool = pool
        self.bufs = bufs
        free_after = _free_memory(self.device)
        logger.info_rank0(
            f"Draft graphs captured; free memory {mem_GB(free_after)} "
            f"({mem_GB(free_before - free_after)} used)"
        )

    def _reset_ring(self) -> None:
        self.runner.reset_cache()
        self.runner._cache_owner = None

    # ------------------------------------------------------------------ replay

    def can_replay(
        self, target_hidden_states: Sequence[torch.Tensor | None], k: int, temperature: float
    ) -> int | None:
        """The context-row count ``c`` of the graph this block replays through, or None.

        Host-side checks only. A block of another size has no graph; a sampled block must
        stay eager because a greedy graph returns one-hot probs, and the stochastic rejection
        sampler would treat those as the draft distribution (a correctness bug, not a slower
        path); a first block after a prefill has more rows than any graph.
        """
        if not self.graphs or self.bufs is None:
            return None
        if k != self.block or temperature > 0:
            return None
        bufs = self.bufs
        c: int | None = None
        for layer_id in self.target_layer_ids:
            index = layer_id + 1
            if index >= len(target_hidden_states):
                return None
            t = target_hidden_states[index]
            if not isinstance(t, torch.Tensor) or t.dim() != 3:
                return None
            if t.dtype != bufs.th.dtype or t.device != bufs.th.device:
                return None
            if t.shape[0] != 1 or t.shape[2] != bufs.hidden:
                return None
            rows = int(t.shape[1])
            if c is None:
                c = rows
            elif rows != c:
                return None
        if c is None or c < 1 or c > self.block:
            return None
        return c

    def replay(
        self,
        target_hidden_states: Sequence[torch.Tensor | None],
        current_token_id: torch.Tensor,
        seq_len: int,
        c: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert torch.cuda.current_stream() == self.stream
        # A ``c`` past can_replay with no graph is a bug, not a case to fall back on: the
        # buffers would be filled and nothing would run over them.
        graph = self.graphs.get(c)
        assert graph is not None and self.bufs is not None, f"no draft graph for c={c}"
        bufs = self.bufs
        bufs.seq_len.fill_(seq_len)
        # Only the c rows the graph reads; rows past c keep stale data no graph c touches.
        for i, layer_id in enumerate(self.target_layer_ids):
            published = target_hidden_states[layer_id + 1]
            assert published is not None
            cols = slice(i * bufs.hidden, (i + 1) * bufs.hidden)
            bufs.th[0, :c, cols].copy_(published[0])
        # The anchor arrives as the token pool's int32 scalar; the copy casts to the int64
        # the embedding gather was captured with.
        bufs.ids[0, 0].copy_(current_token_id.view(()))
        graph.replay()
        # Counted and announced once: a graph that never replays looks exactly like a working
        # one from the outside (same text, same acceptance), only slower.
        self.replays += 1
        if self.replays == 1:
            logger.info_rank0("Draft graph replay active (first replay)")
        return bufs.tokens, bufs.probs  # type: ignore[return-value]

    # ------------------------------------------------------------------ teardown

    def destroy(self) -> None:
        """Drop the graphs, their shared pool and the buffers; the ring stays with the runner."""
        self.graphs = {}
        self.pool = None
        self.bufs = None
        if getattr(self.runner, "graph", None) is self:
            # The runner dispatches on this attribute; a dropped graph must not stay wired.
            self.runner.graph = None
        gc.collect()


def _free_memory(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    return torch.cuda.mem_get_info(device)[0]


__all__ = ["CAPTURE_SEQ_LEN", "DraftCaptureBuffer", "DraftGraphRunner"]
