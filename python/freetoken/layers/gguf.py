"""Native-GGUF quantized layers: weights stay in their packed block layout and are
dequantized *inside* the borrowed llama.cpp CUDA kernels -- either fused into the matmul
(MMVQ/MMQ) or, for types with no MMQ kernel, by an explicit ``ggml_dequantize`` pass.

Mirrors vLLM/sglang's ``GGUFLinearMethod`` / ``GGUFEmbeddingMethod`` dispatch, ported
onto FreeToken's ``BaseOP``. FreeToken keeps fused projections (qkv, gate_up) as a
single tensor: because Q4_0/K-quants pack each *output row* independently over the
input dim, the loader can concatenate the per-shard packed rows along dim 0 (they
share an input dim, hence the same ``row_bytes``), so a fused layer is still one
``[out, row_bytes]`` qweight -- no per-shard padding bookkeeping needed.

**Merged vs. plain fused projections**:

When all output parts share the same quant type (the common case in gemma4), a plain
``GGUFLinear`` with concatenated packed rows is valid and efficient -- one kernel launch
dequantizes and multiplies. When parts use different quant types (as in Ornith's IQ3_M
checkpoint, where qkv_proj mixes IQ3_S and Q4_K), row_bytes differs per part, so torch.cat
would produce garbage. ``GGUFMergedLinear`` instead materializes the output of each part
separately via ``fused_mul_mat_gguf`` and concatenates the results along dim=-1 (equivalent
to the GEMM because all parts read the same input: ``cat([x @ W1.T, x @ W2.T]) == x @ cat([W1, W2], 0).T``).

**Matmul dispatch strategy** (4-tier, per fused_mul_mat_gguf):

1. **Unquantized (F32, F16, BF16)**: straight torch matmul ``x @ qweight.T``.
2. **Small-batch quantized (batch <= 6, MMVQ types)**: GEMV kernel via ``ggml_mul_mat_vec_a8``.
3. **Large-batch standard quants (MMQ types: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, K-quants)**: MMQ kernel
   via ``ggml_mul_mat_a8``.
4. **Large-batch I-quants (IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS, IQ1_M)**:
   I-quants have MMVQ and dequant kernels but NO MMQ kernel. Prefill therefore falls back to
   ``ggml_dequantize`` + plain torch matmul. This materializes a transient BF16 copy of the weight
   (cost: ``out_features * in_features * 2 bytes``), which is a real tradeoff for memory-bound
   prefill on large I-quant weights.

TP is assumed to be 1 (the gemma4 GGUF path restricts to TP=1, like the HF path).
"""

from __future__ import annotations

import torch

from freetoken.models.gguf.dequant import (
    BLOCK_SHAPE,
    DEQUANT_TYPES,
    GGML_BF16,
    GGML_F16,
    GGML_F32,
    GGML_IQ2_S,
    GGML_IQ2_XS,
    GGML_IQ3_S,
    GGML_IQ3_XXS,
    GGML_IQ4_NL,
    GGML_IQ4_XS,
    GGML_NAME,
    GGML_Q3_K,
    GGML_Q4_K,
    GGML_Q5_K,
    GGML_Q6_K,
    GGML_Q8_0,
    GGML_UNQUANTIZED,
    MMQ_TYPES,
    MMVQ_TYPES,
    row_bytes,
)
from dataclasses import dataclass

# ggml type -> the dtype its raw bytes represent. Only the unquantized types appear here;
# everything else goes through a dequant kernel.
_UNQUANTIZED_DTYPE = {
    GGML_F32: torch.float32,
    GGML_F16: torch.float16,
    GGML_BF16: torch.bfloat16,
}

from .base import BaseOP

# Up to this many activation rows the quantized GEMV (MMVQ, or the planar kernel at 2..8
# columns) is used; above it the weight is dequantized once and multiplied dense on the tensor
# cores. The two thresholds differ only because the types split into two cost curves, measured
# DRAM-cold on the real tensors of Qwen3.8-27B-UD-Q4_K_S (scripts/spec/kernel_rows.py in the
# Agents repo, 2026-09-08): the GEMV grows linearly with the row count (it re-reads the weight
# once per group of 8 columns) while dequantize+GEMM is nearly flat (0.46 ms for a 17408x5120
# tensor from 8 to 128 rows, the dequantization dominating), so they cross at ~52 rows for the
# K-quants and ~72 for the I-quants, whose GEMV is cheaper per column.
_MMVQ_SAFE = 48
_MMVQ_NO_MMQ_LIMIT = 72

# The vendored MMQ (kernel/csrc/gguf/mmq.cuh) is llama.cpp b2899's dp4a tile kernel: no tensor
# cores. Measured against both alternatives at 8, 16, 24, 32, 48, 64, 128 and 256 rows on every
# type and shape of this model, it never won a single case -- 1.2x to 2x slower than the GEMV
# below the crossover (Q3_K 17408x5120 at 48 rows: 0.833 vs 0.413 ms) and 2x to 5x slower than
# dequantize+GEMM above it (the same tensor at 256 rows: 4.385 vs 0.563). It stays in the tree
# because the MoE path still calls it, and because a port of upstream's tensor-core MMQ would
# replace it; it is simply not on this dispatch any more.


# The types the tensor-core MMQ covers (kernel/csrc/gguf/mma/mmq_entry.cuh: keep the two lists
# in step, the C++ side raises for anything else) and the conditions under which it was measured
# to win. Two thresholds, both from the A/B on this model's tensors at 9, 16, 24, 32, 40, 64 and
# 2048 rows (scripts/spec/ab_mma.py in the Agents repo):
#   - rows: the crossover against the quantized GEVM is ~24 rows on every large tensor (IQ4_XS
#     17408x5120 at 16 rows 0.080 vs 0.128 ms for the GEVM, at 24 rows 0.090 vs 0.164, at 64
#     0.109 vs 0.421). Below that the GEVM stays, which also leaves the 1- and 8-row decode and
#     speculative-verify paths (and the CUDA graphs captured over them) untouched.
#   - out_features: the kernel tiles 128 output rows per block, so a 1024-row tensor gives 8
#     blocks on 128 SMs and loses until ~1500 activation rows (0.096 vs 0.048 ms at 512 rows).
#     Requiring 32 tiles keeps those four tensors on the dense path.
# The kernel also needs whole tiles: out_features % 128 and in_features % 256.
MMA_TYPES = frozenset(
    {GGML_Q3_K, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0,
     GGML_IQ2_S, GGML_IQ2_XS, GGML_IQ3_S, GGML_IQ3_XXS, GGML_IQ4_NL, GGML_IQ4_XS}
)
_MMA_MIN_ROWS = 24
_MMA_MIN_OUT_TILES = 32  # out_features >= 32 * 128


def _use_mma(qweight_type: int, rows: int, out_features: int, in_features: int) -> bool:
    return (
        qweight_type in MMA_TYPES
        and rows >= _MMA_MIN_ROWS
        and out_features >= _MMA_MIN_OUT_TILES * 128
        and out_features % 128 == 0
        and in_features % 256 == 0
    )


def fused_mul_mat_gguf(x: torch.Tensor, qweight: torch.Tensor, qweight_type: int) -> torch.Tensor:
    """y = x @ dequant(qweight).T, dispatched by batch size and quant type.

    Dispatch order:
    1. Unquantized (F32/F16/BF16): plain torch matmul
    2. Enough rows, a ported type and a wide enough tensor (see _use_mma): the tensor-core MMQ
    3. Few rows (see _MMVQ_SAFE / _MMVQ_NO_MMQ_LIMIT), in MMVQ_TYPES: quantized GEVM kernel
    4. Otherwise, in DEQUANT_TYPES: dequantize the weight once, then a dense tensor-core matmul
    4. Otherwise MMQ, which today only a type outside DEQUANT_TYPES can reach
    """
    from freetoken.kernel.gguf import (
        ggml_dequantize,
        ggml_mul_mat_a8,
        ggml_mul_mat_vec_a8,
    )

    out_features = qweight.shape[0]
    if x.shape[0] == 0:
        return x.new_empty((0, out_features))
    if qweight_type in GGML_UNQUANTIZED:
        # GGUFLinear/GGUFEmbedding store every type in a uint8 buffer of row_bytes width,
        # including the unquantized ones, where "packed" just means the raw F32/F16/BF16
        # bytes. Those must be reinterpreted before the matmul: multiplying the byte view
        # directly gives an in_features of row_bytes (2x too wide for F16) and fails with
        # "mat1 and mat2 shapes cannot be multiplied". A checkpoint only reaches this path
        # when it stores a projection unquantized -- Apodex-1.1-mini ships output.weight as
        # F16, which is how this surfaced; models whose lm_head is Q6_K never hit it.
        w = qweight
        if w.dtype == torch.uint8:
            w = w.view(_UNQUANTIZED_DTYPE[qweight_type])
        # Cast the ACTIVATION, not the weight. Converting the weight would copy the whole
        # matrix on every call -- about 1 GB per forward for a 248k-vocab lm_head -- and
        # allocating that during CUDA graph capture fails outright. x is [tokens, hidden],
        # so casting it is negligible, and computing in the stored precision is what
        # llama.cpp does for these tensors anyway.
        return (x.to(w.dtype) @ w.T).to(x.dtype)
    block, type_size = BLOCK_SHAPE.get(qweight_type, (0, 0))
    in_features = qweight.shape[1] // type_size * block if type_size else 0
    if _use_mma(qweight_type, x.shape[0], out_features, in_features):
        # Ahead of the GEVM branch on purpose: the crossover is ~24 rows, below the GEVM's own
        # limit. Imported here, not with the others, because the kernel module is monkeypatched
        # in the dispatch tests and only the tests exercising this branch need to provide it.
        from freetoken.kernel.gguf import ggml_mul_mat_mma

        return ggml_mul_mat_mma(qweight, x, qweight_type, out_features)
    if qweight_type in MMVQ_TYPES and x.shape[0] <= (
        _MMVQ_SAFE if qweight_type in MMQ_TYPES else _MMVQ_NO_MMQ_LIMIT
    ):
        return ggml_mul_mat_vec_a8(qweight, x, qweight_type, out_features)
    if qweight_type in DEQUANT_TYPES:
        weight = ggml_dequantize(qweight, qweight_type, out_features, in_features, x.dtype)
        return x @ weight.T
    if qweight_type in MMQ_TYPES:
        return ggml_mul_mat_a8(qweight, x, qweight_type, out_features)
    raise NotImplementedError(f"unsupported GGUF type {GGML_NAME.get(qweight_type, qweight_type)}")


class GGUFLinear(BaseOP):
    """Linear whose weight is a native GGUF block-quantized ``[out, row_bytes]`` tensor."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        quant_type: int,
        has_bias: bool = False,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self._quant_type = quant_type
        self.qweight = torch.empty(out_features, row_bytes(in_features, quant_type), dtype=torch.uint8)
        self.bias = torch.empty(out_features) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = fused_mul_mat_gguf(x, self.qweight, self._quant_type)
        if self.bias is not None:
            out = out + self.bias
        return out

    def forward_swiglu(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """``forward(silu(gate) * up)`` with the activation folded into the matmul.

        The tensor-core MMQ quantizes its activation to q8_1 before multiplying, so it can do
        the SwiGLU while it is there. Written out separately, that activation is 74 MB per
        layer of the 27B, written by one kernel and read straight back by the next.

        Only the MMQ path can fuse; every other branch of the dispatch falls back to the
        two-step form, which is what the caller would have done anyway.
        """
        block, type_size = BLOCK_SHAPE.get(self._quant_type, (0, 0))
        in_features = self.qweight.shape[1] // type_size * block if type_size else 0
        if self.bias is None and gate.stride(-1) == 1 and up.stride(-1) == 1 and _use_mma(
            self._quant_type, gate.shape[0], self.out_features, in_features
        ):
            from freetoken.kernel.gguf import ggml_mul_mat_mma_swiglu

            return ggml_mul_mat_mma_swiglu(
                self.qweight, gate, up, self._quant_type, self.out_features
            )
        from freetoken.layers.activation import silu_and_mul_pair

        # The two-step form needs packed rows; on this path (few rows, or a type the MMQ does
        # not carry) the tensors are small and the copy is not what the time goes on.
        return self.forward(silu_and_mul_pair(gate.contiguous(), up.contiguous()))


class GGUFLMHead(GGUFLinear):
    """LM head over a native GGUF ``output.weight`` (untied embeddings).

    Identical to ``GGUFLinear`` except that during prefill it keeps only the last position
    of each sequence, exactly as ``ParallelLMHead`` (layers/embedding.py) and
    ``GGUFTiedLMHead`` (models/gemma4/gguf.py) already do.

    This is not an optimization, it is a memory correctness issue. Logits are
    [tokens, vocab], so on a large-vocabulary model the full-prefill tensor is enormous:
    Ornith-1.5's vocab is 248,320, which in bf16 is 486 KiB of logits PER TOKEN. A
    1,800-token prompt therefore asks for a single 894 MB allocation, which is more than the
    free VRAM left on an 8 GB card after weights and caches, and prefill dies with
    "CUDA driver error: device not ready" while decode is completely unaffected. Only the
    last position of each sequence is ever sampled, so every other row was computed and
    thrown away.

    The dense path never hit this because ``ParallelLMHead`` slices; the bug appears only
    when a GGUF checkpoint has untied embeddings and the head is swapped for a generic
    quantized Linear, which has no reason to know it is the head.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.core import get_global_ctx

        batch = get_global_ctx().batch
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(batch.size)
            x = x[indices].contiguous()
        return super().forward(x)


# How a merged projection's parts are regrouped for the forward. Consecutive parts of one quant
# type share a launch (their packed rows concatenate, row_bytes being equal), and a tiny Q8_0
# group is kept dense in bf16, where one cuBLAS call costs a few microseconds. On Qwen3.8-27B
# the GDN in_proj's ssm_beta and ssm_alpha are 48-row Q8_0 tensors next to an IQ4_XS qkv and
# gate: as two MMVQ launches they cost 2 x 24 us per layer, 2.3 ms per 8-row forward, for
# 23 MB of weights (nsys, 2026-09-07). The loader applies the same plan when it packs the
# parts, so the two sides agree by construction.
_TINY_DENSE_ROWS = 256


@dataclass(frozen=True)
class MergedPart:
    out_size: int
    quant_type: int  # the type the forward computes with (BF16 for a densified tiny group)
    source_type: int  # the type the members are stored as in the file
    members: tuple[int, ...]  # indices into the original parts, in order


def plan_merged_parts(output_sizes: list[int], quant_types: list[int]) -> list[MergedPart]:
    groups: list[tuple[int, int, tuple[int, ...]]] = []
    for i, (n, qt) in enumerate(zip(output_sizes, quant_types)):
        if groups and groups[-1][1] == qt:
            size, _, members = groups[-1]
            groups[-1] = (size + n, qt, members + (i,))
        else:
            groups.append((n, qt, (i,)))
    parts = []
    for n, qt, members in groups:
        compute = GGML_BF16 if (qt == GGML_Q8_0 and n <= _TINY_DENSE_ROWS) else qt
        parts.append(MergedPart(n, compute, qt, members))
    return parts


class GGUFMergedLinear(BaseOP):
    """Merged linear projection with parts that have different quant types.

    Used when fusing output-parallel projections (qkv, gate_up) whose parts use different
    quantization types. Unlike GGUFLinear (which concatenates packed rows along dim 0 and
    requires all parts to share row_bytes), GGUFMergedLinear materializes the output of
    each part separately via fused_mul_mat_gguf, then concatenates the results.

    Mathematically equivalent to a single GEMM, since all parts read the same input x:
    cat([x @ W1.T, x @ W2.T]) == x @ cat([W1, W2], 0).T
    (source: llama.cpp's iq*_m mixed-quant strategy).
    """

    def __init__(
        self,
        in_features: int,
        output_sizes: list[int],
        quant_types: list[int],
        has_bias: bool = False,
    ):
        """Initialize a merged linear projection.

        Args:
            in_features: Input feature dimension (shared by all parts).
            output_sizes: List of output sizes for each part; must all be > 0.
            quant_types: List of GGML quant types, one per part; must match output_sizes length.
            has_bias: Whether to allocate a bias term.

        Raises:
            ValueError: If output_sizes and quant_types lengths do not match, or if any output_size <= 0.
            NotImplementedError: If any quant_type is not supported (not in MMVQ_TYPES or GGML_UNQUANTIZED).
        """
        if len(output_sizes) != len(quant_types):
            raise ValueError(
                f"output_sizes length {len(output_sizes)} != quant_types length {len(quant_types)}"
            )
        if not all(o > 0 for o in output_sizes):
            raise ValueError(f"all output_sizes must be > 0, got {output_sizes}")

        # Validate each quant type is supported.
        for qt in quant_types:
            if qt not in MMVQ_TYPES and qt not in GGML_UNQUANTIZED:
                raise NotImplementedError(
                    f"quant type {GGML_NAME.get(qt, qt)} not in MMVQ_TYPES or GGML_UNQUANTIZED"
                )

        self.in_features = in_features
        self.output_sizes = output_sizes
        self.out_features = sum(output_sizes)
        # The forward runs one launch per regrouped part, not per original part.
        self.parts = plan_merged_parts(output_sizes, quant_types)
        self._quant_types = [part.quant_type for part in self.parts]
        self.part_names = []

        # Allocate packed weight buffers: one named tensor per part (qweight_0, qweight_1, ...).
        # Named (not underscore-prefixed) so they are discovered by state_dict.
        for i, part in enumerate(self.parts):
            name = f"qweight_{i}"
            self.part_names.append(name)
            setattr(
                self,
                name,
                torch.empty(
                    part.out_size, row_bytes(in_features, part.quant_type), dtype=torch.uint8
                ),
            )

        self.bias = torch.empty(self.out_features) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: compute each part's output and concatenate along dim=-1.

        Args:
            x: Input tensor of shape [..., in_features].

        Returns:
            Tensor of shape [..., out_features] with parts concatenated along dim=-1.
        """
        out = torch.cat(self._parts(x), dim=-1)
        if self.bias is not None:
            out = out + self.bias
        return out

    def _parts(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [
            fused_mul_mat_gguf(x, getattr(self, name), qt)
            for name, qt in zip(self.part_names, self._quant_types)
        ]

    def forward_parts(self, x: torch.Tensor) -> list[torch.Tensor]:
        """The per-part outputs, unconcatenated.

        ``forward`` concatenates because most callers want one tensor. A caller that splits it
        straight back -- a SwiGLU reading gate and up, say -- pays a full DRAM round trip for
        the concatenation and gets nothing for it: 17.3 ms per 2129-token prefill of the 27B,
        over 91 concatenations. Such callers should take the parts here instead.

        Refuses when a bias is set rather than silently dropping it (no GGUF checkpoint in this
        fork carries one on a merged projection).
        """
        assert self.bias is None, "forward_parts does not apply a bias"
        return self._parts(x)


class GGUFEmbedding(BaseOP):
    """Vocab embedding stored as a native GGUF block-quantized table.

    The full table is never dequantized: only the looked-up rows are gathered (in
    packed form) and dequantized per lookup, matching vLLM's ``_apply_gguf_embedding``.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        quant_type: int,
        embed_scale: float | None = None,
    ):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self._quant_type = quant_type
        self.qweight = torch.empty(
            num_embeddings, row_bytes(embedding_dim, quant_type), dtype=torch.uint8
        )
        self._embed_scale = embed_scale
        self._embed_scale_t: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.gguf import ggml_dequantize

        flat = x.flatten()
        rows = self.qweight.index_select(0, flat)  # [n, row_bytes] packed
        if self._quant_type in GGML_UNQUANTIZED:
            # Raw value bytes, not blocks: there is no dequant kernel for the unquantized
            # types (ggml_dequantize rejects type 1), so reinterpret the gathered rows.
            y = rows.view(_UNQUANTIZED_DTYPE[self._quant_type]).to(torch.bfloat16)
        else:
            y = ggml_dequantize(rows, self._quant_type, flat.shape[0], self.embedding_dim, torch.bfloat16)
        y = y.view(*x.shape, self.embedding_dim)
        if self._embed_scale is not None:
            if self._embed_scale_t is None:
                self._embed_scale_t = torch.tensor(self._embed_scale, dtype=y.dtype, device=y.device)
            y = y * self._embed_scale_t
        return y


def gguf_merged_or_plain(
    in_features: int,
    output_sizes: list[int],
    quant_types: list[int],
    has_bias: bool = False,
) -> GGUFLinear | GGUFMergedLinear:
    """Choose between GGUFLinear (uniform quant types) and GGUFMergedLinear (mixed types).

    When all output parts share the same quant type (the uniform case, common in gemma4),
    return a GGUFLinear with concatenated packed rows -- valid and cheaper since row_bytes
    is identical per part (one kernel launch instead of N).

    When quant types differ (the mixed case, produced by llama.cpp's IQ*_M / Q*_K_M),
    return a GGUFMergedLinear to avoid torch.cat garbage from misaligned row_bytes.

    Args:
        in_features: Input feature dimension.
        output_sizes: List of output sizes for each part.
        quant_types: List of GGML quant types, one per part.
        has_bias: Whether to allocate a bias term.

    Returns:
        GGUFLinear if all quant types are identical, else GGUFMergedLinear.
    """
    if len(set(quant_types)) == 1:
        # Uniform case: all parts use the same quant type.
        # Concatenate packed rows (they share row_bytes) into a single [sum(output_sizes), row_bytes] weight.
        out_features = sum(output_sizes)
        qt = quant_types[0]
        lin = GGUFLinear(in_features, out_features, qt, has_bias=has_bias)
        return lin
    else:
        # Mixed case: parts use different quant types.
        return GGUFMergedLinear(in_features, output_sizes, quant_types, has_bias=has_bias)


__all__ = [
    "GGUFLinear",
    "GGUFMergedLinear",
    "GGUFEmbedding",
    "fused_mul_mat_gguf",
    "gguf_merged_or_plain",
]
