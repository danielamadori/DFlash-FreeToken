// mma/mmq_q5_K.cuh -- Q5_K for the mma MMQ kernel. Plan step S4.
//
// PORTED FROM llama.cpp @ f7aadef09:
//   ggml-common.h:344-356      block_q5_K (layout only; see "SAFE TYPE" below)
//   mmq-load-tiles.cuh:814-931 ggml_cuda_mmq_load_tiles_q5_K, the mma arm
//                              (:821-823 pointers, :831-864 quants, :866-902 scales)
//   mmq.cuh:766-771            the util_funcs row: load_tiles_q5_K + vec_dot_q8_1_q8_1_mma
//                              + write_back_mma
//   mmq-config-ampere.cuh:174-189   GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1 for every J
//   mmq.cuh:82-84              Q5_K uses MMQ_Q8_1_DS_LAYOUT_DS4 for the activations
//
// The vec-dot is NOT re-ported here. Upstream uses the very same
// ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma (mmq-vec-dot.cuh:315-443) for Q4_K and Q5_K
// (mmq.cuh:760-771), and mma/mmq_q4_K.cuh's header comment instructs the Q5_K agent to INCLUDE
// that header rather than copy the function, so the two cannot drift. get_int_b4
// (vecdotq.cuh:27-29) and unpack_scales_q45_K (mmq-load-tiles.cuh:693-701) come from there too;
// upstream shares both between Q4_K and Q5_K in exactly the same way.
//
// WHY Q5_K IS AS SAFE AS Q4_K (the per-type conflict table in plan S4):
// QR5_K == 2 and QI5_K == 32 in BOTH the vendored ggml-common.h (:93-94) and upstream
// (:136-137), and the two block_q5_K structs are byte-identical (half2 dm; uint8_t scales[12];
// uint8_t qh[32]; uint8_t qs[128]; 176 B). Note the field ORDER -- qh comes BEFORE qs in both --
// which matters here because this loader indexes both arrays. So nothing here can mis-address a
// nibble the way IQ4_XS would. The struct and the constants are still declared LOCALLY, in
// namespace ftmma, because the rule for the whole port is that mma/ includes nothing from the
// vendored headers -- and because QR5_K/QI5_K/QK_K/K_SCALE_SIZE are MACROS over there, so they
// cannot be spelled as C++ identifiers here at all.
//
// DIVERGENCES FROM UPSTREAM:
//  1. The dp4a arm of load_tiles is dropped (mmq-load-tiles.cuh:824-828, :860-862, :903-930,
//     and the whole of ggml_cuda_mmq_vec_dot_q5_K_q8_1_dp4a at mmq-vec-dot.cuh:948-983). This
//     port is mma-only; on sm_89 the dp4a arm is dead. The AMD (MFMA/WMMA) arms are dropped too
//     -- mma_int.cuh instantiates no J_MAJOR tile, and the `#if defined(AMD_MFMA_AVAILABLE)`
//     variant of the scales loop (:870-875, the `if (i < I)` spelling) goes with them; what is
//     kept is the NVIDIA `#else` spelling at :877-878, `i = (...) % I`.
//  2. half2 arithmetic is spelled with intrinsics (__hmul2 / __floats2half2_rn) instead of
//     upstream's `dm * make_half2(...)` at :895 and :899, because the torch extension is compiled
//     with -D__CUDA_NO_HALF2_OPERATORS__ and -D__CUDA_NO_HALF_CONVERSIONS__. The arithmetic is
//     unchanged. Same divergence as mma/mmq_q4_K.cuh's divergence 2.
//  3. The functions are members of mmq_type_traits<GGML_TYPE_Q5_K> rather than free templates
//     selected by the ggml_cuda_mmq_get_util_funcs switch (upstream mmq.cuh:535-843). See
//     mma/mmq_core.cuh divergence 3.

#pragma once

#include <cuda_fp16.h>

#include "mmq_core.cuh"
#include "mmq_q4_K.cuh"   // get_int_b4, unpack_scales_q45_K, ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma

namespace ftmma {

using namespace ggml_cuda_mma;

// ---------------------------------------------------------------------------------------------
// Local Q5_K constants and block layout. See the "SAFE TYPE" note above: these AGREE with the
// vendored ggml-common.h:93-100, they are re-declared only because that header's spellings are
// macros and because mma/ must not include it. FT_K_SCALE_SIZE (== 12) is already declared by
// mma/mmq_q4_K.cuh and is deliberately NOT redeclared here.
// ---------------------------------------------------------------------------------------------

static constexpr int FT_QR5_K = 2;
static constexpr int FT_QI5_K = FTMMA_QK_K / (4 * FT_QR5_K);  // 32

// upstream ggml-common.h:344-355 / vendored ggml-common.h:95-100 (identical layout AND order).
struct block_q5_K {
    half2   dm;                             // super-block scale for quantized scales/mins
    uint8_t scales[FT_K_SCALE_SIZE];        // scales and mins, quantized with 6 bits
    uint8_t qh[FTMMA_QK_K / 8];             // quants, high bit
    uint8_t qs[FTMMA_QK_K / 2];             // quants, low 4 bits
};
static_assert(sizeof(block_q5_K) == 2*sizeof(half) + FT_K_SCALE_SIZE + FTMMA_QK_K/2 + FTMMA_QK_K/8,
              "wrong q5_K block size/padding");
static_assert(sizeof(block_q5_K) == 176, "block_q5_K must be 176 B");

// ---------------------------------------------------------------------------------------------
// upstream mmq-load-tiles.cuh:814-931, the mma arm only (:821-823 for the pointers, :831-864 for
// the quants, :866-902 for the scales).
// ---------------------------------------------------------------------------------------------

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q5_K(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = FTMMA_WARP_SIZE;
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads<type, J, fallback>() / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I<type, J, fallback>();
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride<type, J, fallback>();

    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2*MMQ_TILE_NE_K);

    constexpr int threads_per_row = MMQ_ITER_K / (4 * FT_QR5_K);   // 32
    constexpr int nrows = warp_size / threads_per_row;             // 1
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = (const block_q5_K *) x + kbx0 + i*stride;
        const int ky = FT_QR5_K*txi;

        const int ql = get_int_b4(bxi->qs, txi);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_b4(bxi->qh, txi % (FT_QI5_K/4));
        const int qh0 = ((qh >> (2 * (txi / (FT_QI5_K/4)) + 0)) << 4) & 0x10101010;
        const int qh1 = ((qh >> (2 * (txi / (FT_QI5_K/4)) + 1)) << 4) & 0x10101010;

        const int kq0 = ky - ky % (FT_QI5_K/2) + txi % (FT_QI5_K/4) + 0;
        const int kq1 = ky - ky % (FT_QI5_K/2) + txi % (FT_QI5_K/4) + FT_QI5_K/4;

        x_qs[i*sram_stride + kq0] = ql0 | qh0;
        x_qs[i*sram_stride + kq1] = ql1 | qh1;
    }

    constexpr int rows_per_warp = warp_size / 2;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*rows_per_warp) {
        int i = (i0 + threadIdx.y*rows_per_warp + threadIdx.x/2) % I;
        {
            if (fallback) {
                i = min(i, i_max);
            }

            const block_q5_K * bxi = (const block_q5_K *) x + kbx0 + i*stride;

            const int * scales = (const int *) bxi->scales;
            const int ksc = threadIdx.x % 2;

            const int sc32 = unpack_scales_q45_K(scales, ksc + 0);
            const int  m32 = unpack_scales_q45_K(scales, ksc + 2);

            const uint8_t * sc8 = (const uint8_t *) &sc32;
            const uint8_t *  m8 = (const uint8_t *)  &m32;

            // divergence 2: upstream writes `bxi->dm * make_half2(1.0f, -1.0f)` and
            // `dm*make_half2(sc8[l], m8[l])`. Same arithmetic, intrinsic spelling.
            const half2 dm = __hmul2(bxi->dm, __floats2half2_rn(1.0f, -1.0f));

#pragma unroll
            for (int l = 0; l < (int) sizeof(int); ++l) {
                x_dm[i*sram_stride + sizeof(int)*ksc + l] =
                    __hmul2(dm, __floats2half2_rn((float) sc8[l], (float) m8[l]));
            }
        }
    }

    FTMMA_UNUSED(i_max);
}

// ---------------------------------------------------------------------------------------------
// The per-type hook mma/mmq_core.cuh declares. upstream mmq.cuh:766-771 selects exactly this
// pair for Q5_K (load_tiles_q5_K + vec_dot_q8_1_q8_1_mma + write_back_mma), and
// mmq-config-ampere.cuh:174-189 gives it GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1 -- the same layout as
// Q4_K, because the SRAM tile holds the unpacked 5-bit quants in the same 2*MMQ_TILE_NE_K ints
// (the fifth bit is merged into the byte by the loader above) plus the same per-32 dm pair.
// The activations must be quantized with MMQ_Q8_1_DS_LAYOUT_DS4 (mmq.cuh:82-84) because the
// vec-dot consumes both halves of y_dm; mma/quantize_mmq.cuh:120-122 already maps Q5_K to DS4,
// so nothing is needed there.
// ---------------------------------------------------------------------------------------------

template <> struct mmq_type_traits<GGML_TYPE_Q5_K> {
    static constexpr ggml_cuda_mmq_sram_layout sram_layout = GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1;

    template <int J, bool fallback>
    static __device__ __forceinline__ void load_tiles(
            const char * __restrict__ x, int * __restrict__ x_tile,
            const int kbx0, const int i_max, const int stride) {
        ggml_cuda_mmq_load_tiles_q5_K<GGML_TYPE_Q5_K, J, fallback>(x, x_tile, kbx0, i_max, stride);
    }

    template <int J, bool fallback>
    static __device__ __forceinline__ void vec_dot(
            const int * __restrict__ x, const int * __restrict__ y,
            float * __restrict__ sum, const int k00) {
        ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma<GGML_TYPE_Q5_K, J, fallback>(x, y, sum, k00);
    }
};

}  // namespace ftmma
