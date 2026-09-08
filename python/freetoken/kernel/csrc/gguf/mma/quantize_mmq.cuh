#pragma once

// MMQ activation quantizer -- port of upstream llama.cpp
//   ggml/src/ggml-cuda/mmq.cuh:17-56         (mmq_q8_1_ds_layout, QK8_1_MMQ, block_q8_1_mmq)
//   ggml/src/ggml-cuda/mmq.cuh:60-101        (mmq_get_q8_1_ds_layout)
//   ggml/src/ggml-cuda/quantize.cu:456-556   (the quantize_mmq_q8_1 kernel)
//   ggml/src/ggml-cuda/quantize.cu:575-612   (quantize_mmq_q8_1_cuda, the launcher)
// (upstream checkout /home/danielamadori/.llama.cpp_zlab, Aug 2026).
//
// WHY THIS FILE CARRIES ITS OWN CONSTANTS. The vendored csrc/gguf/ggml-common.h defines QR*/QI*
// for the I-quants with DIFFERENT values than upstream (QR4_XS 8 vs upstream 2, ...), and the
// decode path (mmvq.cuh, mmvq_planar*.cuh, vecdotq.cuh) depends on the vendored ones. Sharing a
// header would mis-address nibbles silently instead of failing to compile, so everything here
// lives in `namespace ftmma`, includes NOTHING from the vendored headers, and touches no file on
// the decode path. This quantizer needs no QR/QI at all: activations are plain q8_1.
//
// DELIBERATE DIVERGENCES FROM UPSTREAM, each marked again at its site:
//   1. Activations are scalar_t (bf16 for this model) instead of float -- see the load.
//   2. The quantization arithmetic is the VENDORED one (d = amax/127, q = roundf(x/d)) rather
//      than upstream's reciprocal form -- see the note above the d/q computation.
//   3. No `ids` and no scatter arm: MoE never reaches this path (the vendored MoE kernels in
//      moe.cuh/moe_vec.cuh are untouched and separate).
//   4. No ggml_cuda_pdl_sync / ggml_cuda_kernel_launch: those need upstream's launch plumbing,
//      and PDL is Hopper-only anyway.
//   5. MMQ_Q8_1_DS_LAYOUT_D2S6 is kept in the code but NOT instantiated: it serves Q2_K only and
//      this checkpoint has no Q2_K (plan section 1.1). Enabling it is one line in the launcher.
//   6. float -> half is spelled with __float2half because the extension is built with
//      -D__CUDA_NO_HALF_CONVERSIONS__ (torch.utils.cpp_extension's COMMON_NVCC_FLAGS).

#include "common_shim.cuh"  // S1: FTMMA_ASSERT / FTMMA_PAD / FTMMA_MATRIX_ROW_PADDING / ftmma_type

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ftmma {

// ---------------------------------------------------------------------------------------------
// local constants (upstream ggml-common.h QK8_1, quantize.cuh:9)
// ---------------------------------------------------------------------------------------------

// FTMMA_QK8_1 (= 32, common_shim.cuh) is upstream's QK8_1: values per plain block_q8_1.
static constexpr int QK8_1_MMQ = 4 * FTMMA_QK8_1;  // 128 values per block_q8_1_mmq

static constexpr int CUDA_QUANTIZE_BLOCK_SIZE_MMQ = 128;  // upstream quantize.cuh:9

// upstream quantize.cuh:12. FTMMA_MATRIX_ROW_PADDING is 512 (common_shim.cuh, upstream common.cuh:176);
// the vendored dispatch pads to the same 512 (gguf_kernel.cu:62). Every in_features of this model
// (1024/5120/6144/10240/12288/17408) is already a multiple of 512, so the padding is a no-op
// here -- the general code is kept anyway.
static_assert(FTMMA_MATRIX_ROW_PADDING % (4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ) == 0,
              "Risk of out-of-bounds access.");

// ---------------------------------------------------------------------------------------------
// block_q8_1_mmq -- upstream mmq.cuh:26-56, byte for byte
// ---------------------------------------------------------------------------------------------

enum mmq_q8_1_ds_layout {
    MMQ_Q8_1_DS_LAYOUT_D4,
    MMQ_Q8_1_DS_LAYOUT_DS4,
    MMQ_Q8_1_DS_LAYOUT_D2S6,
};

struct block_q8_1_mmq {
    // The y float data is converted to a data layout that can simply be copied to shared memory as
    // a contiguous block.
    // The y float data is first grouped as blocks of 128 values.
    // These blocks are then treated as individual data values and transposed.
    //
    // To avoid shared memory bank conflicts each block is padded with 16 bytes.
    // This padding is also used to store block scales/partial sums.
    // The scales multiplied with the quantized data are equal to the unquantized values.
    // The partial sums are obtained by summing up a subgroup of the contained values (prior to
    //     quantization) and are only needed for performance reasons.
    //
    // The exact data stored depends on the x data type.
    union {
        float d4[4];    // 1 32 bit scale per 32 values, stored as d0,d1,d2,d3
        half2 ds4[4];   // 1 16 bit scale + 1 16 bit partial sum per 32 values, stored as
                        //     d0,s0,d1,s1,d2,s2,d3,s3
        half  d2s6[8];  // 1 16 bit scale per 64 values + 1 16 bit partial sum per 16 values for
                        //     the first 96 values, stored as d0,d1,s1,s2,s3,s4,s5
    };
    int8_t qs[QK8_1_MMQ];
};

// Local replica of upstream's plain block_q8_1, used only to restate upstream's static_assert and
// to derive the buffer strides below. It is NOT the vendored ::block_q8_1 (same 36 bytes, but this
// subdirectory depends on nothing in ggml-common.h).
struct block_q8_1_ref {
    half2  ds;
    int8_t qs[FTMMA_QK8_1];
};

static_assert(sizeof(block_q8_1_mmq) == QK8_1_MMQ + 4 * sizeof(half2),
              "Unexpected block_q8_1_mmq size");                      // upstream mmq.cuh:55
static_assert(sizeof(block_q8_1_mmq) == 4 * sizeof(block_q8_1_ref),
              "Unexpected block_q8_1_mmq size");                      // upstream mmq.cuh:56
static_assert(sizeof(block_q8_1_mmq) == 144, "block_q8_1_mmq must be 144 B");

// upstream mmq.cuh:60-101, trimmed to the types that exist in this checkpoint plus the
// Q4_0/Q4_1/Q5_0/Q5_1/Q2_K/IQ2_XXS/IQ1_S rows, kept so the table still reads like upstream.
// (Q1_0/Q2_0/MXFP4/NVFP4 dropped: not in the ftmma_type enum and not in any GGUF we load.)
static inline mmq_q8_1_ds_layout mmq_get_q8_1_ds_layout(const ftmma_type type_x) {
    switch (type_x) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
            return MMQ_Q8_1_DS_LAYOUT_DS4;
        case GGML_TYPE_Q5_0:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_Q5_1:
            return MMQ_Q8_1_DS_LAYOUT_DS4;
        case GGML_TYPE_Q8_0:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_Q2_K:
            return MMQ_Q8_1_DS_LAYOUT_D2S6;
        case GGML_TYPE_Q3_K:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
            return MMQ_Q8_1_DS_LAYOUT_DS4;
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_IQ1_S:
            return MMQ_Q8_1_DS_LAYOUT_DS4;
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        default:
            FTMMA_ABORT("ftmma: no q8_1 ds layout for GGUF quant type %d", (int) type_x);
    }
}

// ---------------------------------------------------------------------------------------------
// sizes and strides of the quantized activation buffer. Upstream computes these inline in
// mmq.cu:120,135-137,164-165; the caller here allocates with torch::empty instead of
// ggml_cuda_pool_alloc (as the vendored dispatch already does, gguf_kernel.cu:190-193), so the
// arithmetic is exposed as helpers rather than buried in a ggml_row_size call.
// ---------------------------------------------------------------------------------------------

// ne00 rounded up to MATRIX_ROW_PADDING -- upstream GGML_PAD(ne10, MATRIX_ROW_PADDING), mmq.cu:120.
static inline int64_t mmq_q8_1_padded_row_size(const int64_t ne00) {
    return FTMMA_PAD(ne00, (int64_t) FTMMA_MATRIX_ROW_PADDING);
}

// Bytes of the quantized activation buffer. `j_max_slack_blocks` reproduces upstream's
//   + ggml_cuda_mmq_get_J_max(...) * sizeof(block_q8_1_mmq)                        (mmq.cu:136)
// which the MMQ kernel reads past the last activation row when ne1 is not a multiple of J. Those
// values are never written by this kernel and are masked out by j_max in write_back, but the
// allocation has to cover them. THE CALLER (S3) MUST PASS ITS J.
static inline size_t mmq_q8_1_nbytes(
        const int64_t ne0_padded, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const int64_t j_max_slack_blocks) {
    return (size_t) ne3 * ne2 * ne1 * ne0_padded / QK8_1_MMQ * sizeof(block_q8_1_mmq) +
           (size_t) j_max_slack_blocks * sizeof(block_q8_1_mmq);
}

// Channel stride of the quantized buffer, in 32-bit words -- upstream mmq.cu:164-165
//   s12 = ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int))
static inline int64_t mmq_q8_1_channel_stride_ints(const int64_t ne0_padded, const int64_t ne1) {
    return ne1 * ne0_padded * (int64_t) sizeof(block_q8_1_ref) /
           (FTMMA_QK8_1 * (int64_t) sizeof(int));
}

// ---------------------------------------------------------------------------------------------
// the kernel -- upstream quantize.cu:456-556
// ---------------------------------------------------------------------------------------------

// DIVERGENCE 1 (activation dtype). Upstream reads `const float4 * x4` and takes 4 floats in one
// 16 B transaction. The vendored wrapper hands us bf16 (at::BFloat16; the dispatch also admits
// fp16/fp32 -- gguf_kernel.cu:182 DISPATCH_FLOAT_TYPES), so this is templated on scalar_t and the
// 4-wide load becomes an 8 B transaction for the 2-byte types. bf16 -> float is EXACT (bf16 is a
// strict subset of fp32: same exponent field, mantissa zero-extended), so the conversion adds no
// error of its own and the quantization below sees exactly the values the vendored quantize_q8_1
// sees for the same tensor -- gguf_kernel.cu:37 does the same `static_cast<float>(x[...])`.
template <typename scalar_t>
struct alignas(4 * sizeof(scalar_t)) ftmma_vec4 {
    scalar_t v[4];
};

template <typename scalar_t, mmq_q8_1_ds_layout ds_layout>
static __global__ void quantize_mmq_q8_1(
        const scalar_t * __restrict__ x, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2) {
    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const int64_t i0 = ((int64_t) blockDim.x * blockIdx.y + threadIdx.x) * 4;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i00 = i0;

    // DIVERGENCE 3: no `ids`, no scatter arm.
    const int64_t i2  = blockIdx.z % ne2;
    const int64_t i3  = blockIdx.z / ne2;
    const int64_t i01 = blockIdx.x;
    const int64_t base_idx = i3 * s03 + i2 * s02 + i01 * s01;

    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t k_block = i0 / QK8_1_MMQ;  // column block in the channel
    const int64_t iqs     = i0 % QK8_1_MMQ;  // quant index in block

    // Load 4 values per thread and calculate max. abs. value between them. Values past ne00 read as
    // 0, which is what makes the row padding a memset-free no-op (upstream does the same).
    float x0 = 0.0f, x1 = 0.0f, x2 = 0.0f, x3 = 0.0f;
    if (i0 < ne00) {
        const ftmma_vec4<scalar_t> xi = *((const ftmma_vec4<scalar_t> *) (x + (base_idx + i00)));
        x0 = static_cast<float>(xi.v[0]);
        x1 = static_cast<float>(xi.v[1]);
        x2 = static_cast<float>(xi.v[2]);
        x3 = static_cast<float>(xi.v[3]);
    }
    float amax = fabsf(x0);
    amax = fmaxf(amax, fabsf(x1));
    amax = fmaxf(amax, fabsf(x2));
    amax = fmaxf(amax, fabsf(x3));

    // Exchange max. abs. value between vals_per_scale/4 threads.
    // (fmaxf is associative and exact, so this 4-values-per-lane tree gives bit-identical amax to
    //  the vendored 1-value-per-lane reduction over the same 32 values, gguf_kernel.cu:42-45.)
#pragma unroll
    for (int offset = vals_per_scale / 8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, 32));  // 32 = warp size
    }

    float sum = 0.0f;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = x0 + x1 + x2 + x3;

        // Calculate sums across vals_per_sum/4 threads.
#pragma unroll
        for (int offset = vals_per_sum / 8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, 32);
        }
    }

    // DIVERGENCE 2 (quantization arithmetic). Upstream (quantize.cu:516-522) computes
    //     d_inv = 127.0f/amax;  q = roundf(x*d_inv);  d = 1.0f/d_inv;
    // This port uses the VENDORED form instead (gguf_kernel.cu:47-48, and the identical pair in
    // quantize_planar, :97-98):
    //     d = amax/127.0f;      q = roundf(x/d)       with q = 0 when amax == 0
    // Two reasons. (a) The activations this kernel writes are then bit-identical to the ones the
    // MMVQ/planar decode path builds from the same tensor, so an A/B of MMQ against the GEVM
    // compares the matmul and not the quantizer. (b) upstream's reciprocal is inf on an all-zero
    // group, so roundf(0*inf) = NaN and the float -> int8 conversion is undefined; the explicit
    // amax == 0 guard removes that, and it fires on every padded tail (see the load above).
    const float d = amax / 127.0f;
    char4 q = make_char4(0, 0, 0, 0);
    if (amax != 0.0f) {
        q.x = (int8_t) roundf(x0 / d);
        q.y = (int8_t) roundf(x1 / d);
        q.z = (int8_t) roundf(x2 / d);
        q.w = (int8_t) roundf(x3 / d);
    }

    // Block order, unchanged from upstream: ib = k_block*ne1 + row, i.e. the blocks of one column
    // group are contiguous ACROSS rows (column-major over (row, k_block)). The mma MMQ kernel
    // relies on this; it is NOT the row-major layout of the vendored block_q8_1 buffer.
    const int64_t ib0 = blockIdx.z * ((int64_t) gridDim.x * gridDim.y * blockDim.x / FTMMA_QK8_1);
    const int64_t ib  = ib0 + k_block * ne1 + blockIdx.x;

    // Write back 4 int8 values as a single 32 bit value for better memory bandwidth:
    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[iqs / 4] = q;

    // DIVERGENCE 6: the implicit float -> half of upstream's `make_half2(d, sum)` and
    // `d2s6[...] = sum` does not compile under -D__CUDA_NO_HALF_CONVERSIONS__; __float2half is the
    // same instruction, spelled out. (The vendored quantize_q8_1 spells it out for the same reason,
    // gguf_kernel.cu:57-58.)
    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 == 0 && iqs < 96) {
            y[ib].d2s6[2 + iqs / 16] = __float2half(sum);
            if (iqs % 64 == 0) {
                y[ib].d2s6[iqs / 64] = __float2half(d);
            }
        }
    } else if (iqs % 32 == 0) {
        if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
            y[ib].ds4[iqs / 32] = make_half2(__float2half(d), __float2half(sum));
        } else {
            y[ib].d4[iqs / 32] = d;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// the launcher -- upstream quantize.cu:575-612 (quantize_mmq_q8_1_cuda), minus `ids`
// ---------------------------------------------------------------------------------------------
//
// x       activations, scalar_t, logical shape [ne3][ne2][ne1][ne00] with ELEMENT strides
//         s03/s02/s01. For the dense prefill matmul this is just [R][K]: ne1 = R, ne00 = K,
//         s01 = K, ne2 = ne3 = 1, s02 = s03 = 0. x must be 4*sizeof(scalar_t)-aligned (torch
//         tensor storage is 256 B aligned) and s01 must be a multiple of 4.
// vy      output, mmq_q8_1_nbytes(ne0, ne1, ne2, ne3, J) bytes, 16 B aligned (torch::empty is).
// type_x  the WEIGHT type: it selects the ds layout, not the activation format.
// ne0     padded row length = mmq_q8_1_padded_row_size(ne00); a multiple of 512.
template <typename scalar_t>
static void quantize_mmq_q8_1_cuda(
        const scalar_t * x, void * vy, const ftmma_type type_x,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        cudaStream_t stream) {
    FTMMA_ASSERT(ne00 % 4 == 0);
    FTMMA_ASSERT(s01 % 4 == 0);  // the 4-wide vector load needs 4*sizeof(scalar_t) alignment
    FTMMA_ASSERT(ne0 % QK8_1_MMQ == 0);
    // ib0 above derives the per-channel block count from the grid, which is only exact when the
    // grid covers ne0 with no slack; FTMMA_MATRIX_ROW_PADDING is what guarantees that.
    FTMMA_ASSERT(ne0 % (4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ) == 0);

    // ne1 tends to assume the highest values, therefore use it as the "x" dimension of the CUDA grid:
    const int64_t block_num_y =
        (ne0 + 4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks((unsigned) ne1, (unsigned) block_num_y, (unsigned) (ne2 * ne3));
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    switch (mmq_get_q8_1_ds_layout(type_x)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            quantize_mmq_q8_1<scalar_t, MMQ_Q8_1_DS_LAYOUT_D4><<<num_blocks, block_size, 0, stream>>>(
                x, vy, ne00, s01, s02, s03, ne0, (int) ne1, (int) ne2);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            quantize_mmq_q8_1<scalar_t, MMQ_Q8_1_DS_LAYOUT_DS4><<<num_blocks, block_size, 0, stream>>>(
                x, vy, ne00, s01, s02, s03, ne0, (int) ne1, (int) ne2);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            // DIVERGENCE 5: not instantiated -- Q2_K only, and this checkpoint has no Q2_K.
            FTMMA_ABORT("ftmma: MMQ_Q8_1_DS_LAYOUT_D2S6 (Q2_K) is not instantiated");
            break;
        default:
            FTMMA_ABORT("fatal error");
            break;
    }
}

}  // namespace ftmma
