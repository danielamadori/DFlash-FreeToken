// mma/mmq_q8_0.cuh -- Q8_0 for the mma MMQ kernel. Plan step S4, one file per type; this one is
// the type agent's tile loader plus the vec-dot that upstream shares between Q8_0 and the
// D4-scaled types (plan S4, owner D's mma/vec_dot_q8_0.cuh -- see "NOTE FOR S4" below).
//
// PORTED FROM llama.cpp @ f7aadef09:
//   ggml-common.h:251-256      block_q8_0 (layout only; see "SAFE TYPE" below)
//   vecdotq.cuh:18-23          get_int_b2
//   mmq-load-tiles.cuh:463-525 ggml_cuda_mmq_load_tiles_q8_0, the mma arm
//                              (:471-472 pointers, :496-498 quants, :505-522 scales)
//   mmq-vec-dot.cuh:142-280    ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma, the NVIDIA arm (:203-278)
//   mmq.cuh:741-746            the (load_tiles_q8_0, vec_dot_q8_0_q8_1_mma<..., D4>) pairing
//   mmq.cuh:72-73              Q8_0 -> MMQ_Q8_1_DS_LAYOUT_D4
//   mmq-config-ampere.cuh:104-119  Q8_0 -> GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0
//
// WHY Q8_0 IS A SAFE TYPE (plan S4's per-type conflict table):
// QR8_0 == 1 and QI8_0 == 8 in BOTH the vendored ggml-common.h (:52-54) and upstream (:121-122,
// :251), and the two block_q8_0 structs are byte-identical (half d; int8_t qs[32]; 34 B). Q8_0 is
// not a nibble type at all -- there is no packing to mis-address. The struct and the constants are
// still declared LOCALLY, in namespace ftmma, because the rule for the whole port is that mma/
// includes nothing from the vendored headers -- and because QK8_0/QR8_0/QI8_0 are MACROS over
// there, so they cannot be spelled as C++ identifiers here at all.
//
// The 34-byte block is why the quants are read with get_int_b2 (two 16-bit loads) and not
// get_int_b4: `qs` sits at byte offset 2 of a 34-byte block, so a block_q8_0 array gives no
// 4-byte alignment guarantee on `qs`. This is upstream's reason too (vecdotq.cuh:18).
//
// DIVERGENCES FROM UPSTREAM:
//  1. The dp4a arm of both functions is dropped (mmq-load-tiles.cuh:473-476, :499-501, :521-522
//     and mmq-vec-dot.cuh:110-140's ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a). This port is mma-only;
//     on sm_89 the dp4a arm is dead. The AMD (MFMA/WMMA) arms -- mmq-vec-dot.cuh:145-201, which
//     uses tile<16,16,int,DATA_LAYOUT_J_MAJOR> -- are dropped too: mma_int.cuh instantiates no
//     J_MAJOR tile.
//  2. half -> float is spelled __half2float(bxi->d) instead of upstream's implicit
//     `x_df[...] = bxi->d;` (mmq-load-tiles.cuh:520), because the torch extension is compiled
//     with -D__CUDA_NO_HALF_CONVERSIONS__ (the same reason mmq_q4_K.cuh spells its half2
//     arithmetic with intrinsics). The arithmetic is unchanged.
//  3. The functions are members of mmq_type_traits<GGML_TYPE_Q8_0> rather than free templates
//     selected by the ggml_cuda_mmq_get_util_funcs switch (upstream mmq.cuh:535-843). See
//     mma/mmq_core.cuh divergence 3: this is what lets the S4 agents add a type without touching
//     a shared file.
//  4. FT_QI8_0 is NOT redeclared here: mma/mmq_core.cuh already defines it in namespace ftmma
//     (:80, value 8, same as upstream's QI8_0). Only FT_QR8_0 and the block struct are new.
//  5. get_int_b2 and ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma are wrapped in #ifndef guards. Several
//     S4 types need both verbatim (Q5_0, Q8_0, IQ4_NL, IQ4_XS, IQ3_XXS, IQ3_S ...); without the
//     guards, two per-type headers in the same translation unit would be a redefinition error.
//     The bodies are upstream unchanged, so whichever header defines them first defines the same
//     function. The guard SPELLINGS are the ones the sibling S4 headers already picked --
//     FTMMA_HAVE_GET_INT_B2 (mmq_iq4_nl.cuh:92, mmq_iq3_xxs.cuh:149),
//     FTMMA_GET_INT_B2_DEFINED (mmq_iq2_s.cuh:360) and
//     FTMMA_HAVE_VEC_DOT_Q8_0_Q8_1_MMA (mmq_iq4_nl.cuh:155, mmq_iq3_xxs.cuh:182) -- and this file
//     tests and defines ALL of them, so it composes with every sibling whichever way round they
//     are included. Siblings that define either symbol UNGUARDED still collide; see the report's
//     NEEDED_ELSEWHERE, and plan S4 owner D's mma/vec_dot_q8_0.cuh, which is the real fix.
//
// NOTE FOR S4: ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma below is NOT Q8_0-specific -- upstream uses the
// same vec-dot for Q5_0, Q8_0, MXFP4, IQ2_XXS, IQ3_XXS, IQ3_S, IQ4_NL and IQ4_XS (mmq.cuh:736-746,
// :786-836), which is why it keeps upstream's `ds_layout` template parameter even though Q8_0 only
// ever instantiates MMQ_Q8_1_DS_LAYOUT_D4. The plan puts it in mma/vec_dot_q8_0.cuh owned by D.
// Whoever lands the next D4 type should either include this header or move the function into
// vec_dot_q8_0.cuh and have this file include that instead; it must not be copied, or the two
// copies will drift (and will not link, being two definitions of the same template).

#pragma once

#include <cuda_fp16.h>

#include "mmq_core.cuh"

namespace ftmma {

using namespace ggml_cuda_mma;

// ---------------------------------------------------------------------------------------------
// Local Q8_0 constants and block layout. See the "SAFE TYPE" note above: these AGREE with the
// vendored ggml-common.h:52-58, they are re-declared only because that header's spellings are
// macros and because mma/ must not include it. FT_QI8_0 (== 8) comes from mmq_core.cuh:80
// (divergence 4).
// ---------------------------------------------------------------------------------------------

static constexpr int FT_QR8_0 = 1;
static_assert(FT_QI8_0 == FTMMA_QK8_0 / (4 * FT_QR8_0), "FT_QI8_0 must be QK8_0/(4*QR8_0)");

// upstream ggml-common.h:251-256 / vendored ggml-common.h:55-58 (identical layout).
struct block_q8_0 {
    half   d;                  // delta
    int8_t qs[FTMMA_QK8_0];    // quants
};
static_assert(sizeof(block_q8_0) == sizeof(half) + FTMMA_QK8_0,
              "wrong q8_0 block size/padding");
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 must be 34 B");

// upstream vecdotq.cuh:18-23. See divergence 5 for the guard.
#if !defined(FTMMA_HAVE_GET_INT_B2) && !defined(FTMMA_GET_INT_B2_DEFINED)
#define FTMMA_HAVE_GET_INT_B2
#define FTMMA_GET_INT_B2_DEFINED
static __device__ __forceinline__ int get_int_b2(const void * x, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) x; // assume at least 2 byte alignment

    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;

    return x32;
}
#endif // FTMMA_HAVE_GET_INT_B2

// ---------------------------------------------------------------------------------------------
// upstream mmq-vec-dot.cuh:142-280, the `#else` (NVIDIA) arm at :203-278, verbatim except for the
// FT_ spelling of QI8_0/QI8_1 and the mmq_core accessor templates.
// Used by Q8_0 (and, in the rest of S4, by every other D4-scaled type -- see "NOTE FOR S4").
// ---------------------------------------------------------------------------------------------

#ifndef FTMMA_HAVE_VEC_DOT_Q8_0_Q8_1_MMA
#define FTMMA_HAVE_VEC_DOT_Q8_0_Q8_1_MMA
template <ggml_type type, int J, bool fallback, mmq_q8_1_ds_layout ds_layout>
static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    typedef tile<16, 8, int> tile_A;
    typedef tile< 8, 8, int> tile_B;
    typedef tile<16, 8, int> tile_C;

    constexpr int I             = ggml_cuda_mmq_get_I<type, J, fallback>();
    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride<type, J, fallback>();
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp<type, J, fallback>();
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + 2*MMQ_TILE_NE_K;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;
    const half2 * y_ds = (const half2 *) y;

    tile_A A[ntx][MMQ_TILE_NE_K/FT_QI8_0];
    float dA[ntx][tile_C::ne/2][MMQ_TILE_NE_K/FT_QI8_0];

    const int i0 = (threadIdx.y/ntx)*rows_per_warp;

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += FT_QI8_0) {
            const int k0 = k00 + k01;

            load_ldmatrix(A[n][k01/FT_QI8_0], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
        }

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_A::I + tile_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += FT_QI8_0) {
                const int k0 = k00 + k01;

                dA[n][l][k01/FT_QI8_0] = x_df[i*sram_stride + k0/FT_QI8_0];
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += FT_QI8_0) {
            tile_B B;
            float dB[tile_C::ne/2];

            load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K); // faster than load_ldmatrix

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);

                if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D4) {
                    dB[l] =             y_df[j*MMQ_TILE_Y_K + k01/FT_QI8_1];
                } else {
                    dB[l] = __low2float(y_ds[j*MMQ_TILE_Y_K + k01/FT_QI8_1]);
                }
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n][k01/FT_QI8_0], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += C.x[l]*dA[n][l/2][k01/FT_QI8_0]*dB[l%2];
                }
            }
        }
    }

    FTMMA_UNUSED(I);
}
#endif // FTMMA_HAVE_VEC_DOT_Q8_0_Q8_1_MMA

// ---------------------------------------------------------------------------------------------
// upstream mmq-load-tiles.cuh:463-525, the mma arm only (:471-472 for the pointers, :496-498 for
// the quants, :505-522 for the scales).
//
// The K extent of one call is 2*MMQ_TILE_NE_K == 64 32-bit elements == 256 int8 == 8 Q8_0 blocks,
// which is MMQ_ITER_K/qk = 256/32 = 8 -- i.e. mmq_core.cuh's blocks_per_iter (mmq.cuh:880). The
// first loop writes those 8 blocks' quants as 64 ints per row; the second writes their 8 deltas.
// ---------------------------------------------------------------------------------------------

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q8_0(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = FTMMA_WARP_SIZE;
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads<type, J, fallback>() / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I<type, J, fallback>();
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride<type, J, fallback>();

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_tile + 2*MMQ_TILE_NE_K);

    // MMQ_ITER_K / (4 * QR8_0) == 64 required. but NV has only 32 threads per warp
    constexpr int threads_per_row = 32;
    constexpr int nrows = warp_size / threads_per_row;             // 1
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int kbx  = txi / FT_QI8_0;
    const int kqsx = txi % FT_QI8_0;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q8_0 * bxi = (const block_q8_0 *) x + kbx0 + i*stride + kbx;

        x_qs[i*sram_stride + 0             + txi] = get_int_b2(bxi[0].qs,                      kqsx);
        x_qs[i*sram_stride + MMQ_TILE_NE_K + txi] = get_int_b2(bxi[MMQ_TILE_NE_K/FT_QI8_0].qs, kqsx);
    }

    constexpr int blocks_per_tile_x_row = 2*MMQ_TILE_NE_K / FT_QI8_0;   // 8
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;    // 4
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q8_0 * bxi = (const block_q8_0 *) x + kbx0 + i*stride + kbxd;

        // divergence 2: upstream writes `x_df[...] = bxi->d;`. Same value, intrinsic spelling.
        x_df[i*sram_stride + kbxd] = __half2float(bxi->d);
    }

    FTMMA_UNUSED(i_max);
}

// ---------------------------------------------------------------------------------------------
// The per-type hook mma/mmq_core.cuh declares. upstream mmq.cuh:741-746 selects exactly this pair
// for Q8_0 (load_tiles_q8_0 + vec_dot_q8_0_q8_1_mma<..., MMQ_Q8_1_DS_LAYOUT_D4> +
// write_back_mma), and mmq-config-ampere.cuh:104-119 gives it GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0.
//
// The D4 ds layout is what mma/quantize_mmq.cuh already produces for Q8_0
// (mmq_get_q8_1_ds_layout, quantize_mmq.cuh:114-115, matching upstream mmq.cuh:72-73), and the
// MMQ_Q8_1_DS_LAYOUT_D4 arm of quantize_mmq_q8_1_cuda is instantiated (quantize_mmq.cuh:327-330),
// so nothing new is needed from the activation quantizer.
// ---------------------------------------------------------------------------------------------

template <> struct mmq_type_traits<GGML_TYPE_Q8_0> {
    static constexpr ggml_cuda_mmq_sram_layout sram_layout = GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0;

    template <int J, bool fallback>
    static __device__ __forceinline__ void load_tiles(
            const char * __restrict__ x, int * __restrict__ x_tile,
            const int kbx0, const int i_max, const int stride) {
        ggml_cuda_mmq_load_tiles_q8_0<GGML_TYPE_Q8_0, J, fallback>(x, x_tile, kbx0, i_max, stride);
    }

    template <int J, bool fallback>
    static __device__ __forceinline__ void vec_dot(
            const int * __restrict__ x, const int * __restrict__ y,
            float * __restrict__ sum, const int k00) {
        ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<GGML_TYPE_Q8_0, J, fallback, MMQ_Q8_1_DS_LAYOUT_D4>(
            x, y, sum, k00);
    }
};

}  // namespace ftmma
