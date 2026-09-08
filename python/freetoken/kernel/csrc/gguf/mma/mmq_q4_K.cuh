// mma/mmq_q4_K.cuh -- Q4_K for the mma MMQ kernel. Plan step S3, owners C (tile loader) and
// D (vec-dot), merged into one file because S3 ships a single type end to end.
//
// PORTED FROM llama.cpp @ f7aadef09:
//   ggml-common.h:326-338      block_q4_K (layout only; see "SAFE TYPE" below)
//   vecdotq.cuh:27-29          get_int_b4
//   mmq-load-tiles.cuh:693-701 unpack_scales_q45_K
//   mmq-load-tiles.cuh:703-812 ggml_cuda_mmq_load_tiles_q4_K, the mma arm (:710-712, :735-779)
//   mmq-vec-dot.cuh:315-443    ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma, the NVIDIA arm (:369-442)
//
// WHY Q4_K IS THE SAFE TYPE TO START ON (plan S3/C and the per-type conflict table in S4):
// QR4_K == 2 and QI4_K == 32 in BOTH the vendored ggml-common.h (:85-86) and upstream (:137-138),
// and the two block_q4_K structs are byte-identical (half2 dm; uint8_t scales[12]; uint8_t
// qs[128]; 144 B). So nothing here can mis-address a nibble the way IQ4_XS would. The struct and
// the constants are still declared LOCALLY, in namespace ftmma, because the rule for the whole
// port is that mma/ includes nothing from the vendored headers -- and because QR4_K/QI4_K/QK_K
// are MACROS over there, so they cannot be spelled as C++ identifiers here at all.
//
// DIVERGENCES FROM UPSTREAM:
//  1. The dp4a arm of both functions is dropped (mmq-load-tiles.cuh:713-717, :780-810 and
//     mmq-vec-dot.cuh's q4_K dp4a path). This port is mma-only; on sm_89 the dp4a arm is dead.
//     The AMD (MFMA/WMMA) arms are dropped too -- mma_int.cuh instantiates no J_MAJOR tile.
//  2. half2 arithmetic is spelled with intrinsics (__hmul2 / __floats2half2_rn) instead of
//     upstream's `dm * make_half2(...)`, because the torch extension is compiled with
//     -D__CUDA_NO_HALF2_OPERATORS__ and -D__CUDA_NO_HALF_CONVERSIONS__ (the same reason S2 spells
//     its float->half with __float2half). The arithmetic is unchanged.
//  3. The functions are members of mmq_type_traits<GGML_TYPE_Q4_K> rather than free templates
//     selected by the ggml_cuda_mmq_get_util_funcs switch (upstream mmq.cuh:535-843). See
//     mma/mmq_core.cuh divergence 3: this is what lets the ten S4 agents add a type without
//     touching a shared file.
//
// NOTE FOR S4: ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma below is NOT Q4_K-specific -- upstream uses
// the same vec-dot for Q4_K, Q5_K and IQ1_S (mmq.cuh:760-771, :779-784). The plan puts it in
// mma/vec_dot_q8_1.cuh owned by D. Whoever lands Q5_K should either include this header or move
// the function into vec_dot_q8_1.cuh and have this file include that instead; it must not be
// copied, or the two copies will drift.

#pragma once

#include <cuda_fp16.h>

#include "mmq_core.cuh"

namespace ftmma {

using namespace ggml_cuda_mma;

// ---------------------------------------------------------------------------------------------
// Local Q4_K constants and block layout. See the "SAFE TYPE" note above: these AGREE with the
// vendored ggml-common.h:85-91, they are re-declared only because that header's spellings are
// macros and because mma/ must not include it.
// ---------------------------------------------------------------------------------------------

static constexpr int FT_QR4_K = 2;
static constexpr int FT_QI4_K = FTMMA_QK_K / (4 * FT_QR4_K);  // 32
static constexpr int FT_K_SCALE_SIZE = 12;

// upstream ggml-common.h:326-337 / vendored ggml-common.h:87-91 (identical layout).
struct block_q4_K {
    half2   dm;                             // super-block scale for quantized scales/mins
    uint8_t scales[FT_K_SCALE_SIZE];        // scales and mins, quantized with 6 bits
    uint8_t qs[FTMMA_QK_K / 2];             // 4-bit quants
};
static_assert(sizeof(block_q4_K) == 2*sizeof(half) + FT_K_SCALE_SIZE + FTMMA_QK_K/2,
              "wrong q4_K block size/padding");
static_assert(sizeof(block_q4_K) == 144, "block_q4_K must be 144 B");

// upstream vecdotq.cuh:27-29
static __device__ __forceinline__ int get_int_b4(const void * x, const int & i32) {
    return ((const int *) x)[i32]; // assume at least 4 byte alignment
}

// upstream mmq-load-tiles.cuh:693-701
static __device__ __forceinline__ int unpack_scales_q45_K(const int * scales, const int ksc) {
    // scale arrangement after the following two lines:
    //   - ksc == 0: sc0, sc1, sc2, sc3
    //   - ksc == 1: sc4, sc5, sc6, sc7
    //   - ksc == 2:  m0,  m1,  m2,  m3
    //   - ksc == 3:  m4,  m5,  m6,  m7
    return ((scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F) | // lower 4 bits
           ((scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030);  // upper 2 bits
}

// ---------------------------------------------------------------------------------------------
// upstream mmq-vec-dot.cuh:315-443, the `#else` (NVIDIA) arm at :369-442, verbatim except for the
// FT_ spelling of QI8_1 and the mmq_core accessor templates.
// Used by Q4_K (and, in S4, by Q5_K and IQ1_S).
// ---------------------------------------------------------------------------------------------

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    typedef tile<16,  8, int> tile_A;
    typedef tile< 8,  8, int> tile_B;
    typedef tile<16,  8, int> tile_C;

    constexpr int I             = ggml_cuda_mmq_get_I<type, J, fallback>();
    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride<type, J, fallback>();
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp<type, J, fallback>();
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + 2*MMQ_TILE_NE_K;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_dm = (const half2 *) y;

    tile_A   A[ntx][MMQ_TILE_NE_K/FT_QI8_1];
    float2 dmA[ntx][tile_C::ne/2][MMQ_TILE_NE_K/FT_QI8_1];

    const int i0 = (threadIdx.y/ntx)*rows_per_warp;

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += FT_QI8_1) {
            const int k0 = k00 + k01;

            load_ldmatrix(A[n][k01/FT_QI8_1], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
        }

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_A::I + tile_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += FT_QI8_1) {
                const int k0 = k00 + k01;

                dmA[n][l][k01/FT_QI8_1] = __half22float2(x_dm[i*sram_stride + k0/FT_QI8_1]);
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += FT_QI8_1) {
            tile_B   B;
            float2 dsB[tile_C::ne/2];

            load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K); // faster than load_ldmatrix

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);

                dsB[l] = __half22float2(y_dm[j*MMQ_TILE_Y_K + k01/FT_QI8_1]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n][k01/FT_QI8_1], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dmA[n][l/2][k01/FT_QI8_1].x*dsB[l%2].x*C.x[l];
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dmA[n][l/2][k01/FT_QI8_1].y*dsB[l%2].y;
                }
            }
        }
    }

    FTMMA_UNUSED(I);
}

// ---------------------------------------------------------------------------------------------
// upstream mmq-load-tiles.cuh:703-812, the mma arm only (:710-712 for the pointers, :735-737 for
// the quants, :743-779 for the scales).
// ---------------------------------------------------------------------------------------------

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q4_K(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = FTMMA_WARP_SIZE;
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads<type, J, fallback>() / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I<type, J, fallback>();
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride<type, J, fallback>();

    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2*MMQ_TILE_NE_K);

    constexpr int threads_per_row = MMQ_ITER_K / (4 * FT_QR4_K);   // 32
    constexpr int nrows = warp_size / threads_per_row;             // 1
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride;
        const int qs0 = get_int_b4(bxi->qs, txi);

        x_qs[i*sram_stride + 16*(txi/8) + txi % 8 + 0] = (qs0 >> 0) & 0x0F0F0F0F;
        x_qs[i*sram_stride + 16*(txi/8) + txi % 8 + 8] = (qs0 >> 4) & 0x0F0F0F0F;
    }

    constexpr int rows_per_warp = warp_size / 2;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*rows_per_warp) {
        int i = (i0 + threadIdx.y*rows_per_warp + threadIdx.x/2) % I;
        {
            if (fallback) {
                i = min(i, i_max);
            }

            const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride;

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
// The per-type hook mma/mmq_core.cuh declares. upstream mmq.cuh:760-765 selects exactly this
// pair for Q4_K (load_tiles_q4_K + vec_dot_q8_1_q8_1_mma + write_back_mma), and
// mmq-config-ampere.cuh:157-172 gives it GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1.
// ---------------------------------------------------------------------------------------------

template <> struct mmq_type_traits<GGML_TYPE_Q4_K> {
    static constexpr ggml_cuda_mmq_sram_layout sram_layout = GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1;

    template <int J, bool fallback>
    static __device__ __forceinline__ void load_tiles(
            const char * __restrict__ x, int * __restrict__ x_tile,
            const int kbx0, const int i_max, const int stride) {
        ggml_cuda_mmq_load_tiles_q4_K<GGML_TYPE_Q4_K, J, fallback>(x, x_tile, kbx0, i_max, stride);
    }

    template <int J, bool fallback>
    static __device__ __forceinline__ void vec_dot(
            const int * __restrict__ x, const int * __restrict__ y,
            float * __restrict__ sum, const int k00) {
        ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma<GGML_TYPE_Q4_K, J, fallback>(x, y, sum, k00);
    }
};

}  // namespace ftmma
