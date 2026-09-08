// mma/mmq_q3_K.cuh -- Q3_K for the mma MMQ kernel. Plan step S4, one type end to end
// (tile loader + vec-dot), following mma/mmq_q4_K.cuh exactly in structure and comment style.
//
// PORTED FROM llama.cpp @ f7aadef09:
//   ggml-common.h:130-131      QI3_K / QR3_K (layout only; see "SAFE TYPE" below)
//   ggml-common.h:311-321      block_q3_K (layout only)
//   vecdotq.cuh:18-25          get_int_b2
//   vecdotq.cuh:447            VDR_Q3_K_Q8_1_MMQ
//   mmq-load-tiles.cuh:590-691 ggml_cuda_mmq_load_tiles_q3_K, the mma arm
//                              (:597-599 pointers, :607-639 quants, :641-675 scales)
//   mmq-vec-dot.cuh:481-614    ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma, the TURING arm (:533-609)
//   mmq.cuh:754-759            the (load_tiles_q3_K, vec_dot_q8_0_16_q8_1_mma) pairing
//   mmq-config-ampere.cuh:140-155  GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K for every Q3_K row
//
// WHY Q3_K IS SAFE UNDER THE PORT'S CONSTANT RULE (plan S4 conflict table):
// QR3_K == 4 and QI3_K == QK_K/(4*QR3_K) == 16 in BOTH the vendored ggml-common.h (:76-77) and
// upstream (:130-131), and the two block_q3_K structs are byte-identical (uint8_t hmask[32];
// uint8_t qs[64]; uint8_t scales[12]; half d; 110 B, vendored :78-83 vs upstream :315-320). So
// nothing here can mis-address a nibble or a high-bit the way IQ4_XS would. The struct and the
// constants are still declared LOCALLY, in namespace ftmma, because the rule for the whole port
// is that mma/ includes nothing from the vendored headers -- and because QR3_K/QI3_K/QK_K are
// MACROS over there, so they cannot be spelled as C++ identifiers here at all.
//
// WHAT IS SPECIFIC TO Q3_K (the "high-bit plane" and the "scale sign trick"):
//   * quants: the low 2 bits come from qs[], the 3rd bit from the hmask[] plane, and the value is
//     re-centred with __vsubss4(low2 | high<<2, 0x04040404) -- i.e. q = (0..7) - 4, signed.
//     That is why the tile holds ready-made SIGNED int8 quants and the vec-dot is the plain
//     q8_0_16 one (no per-block min term).
//   * scales: 6 bits split 4-low + 2-high across scales[12], re-centred the same way with
//     __vsubss4(sc_low | sc_high, 0x20202020) -- i.e. sc = (0..63) - 32, signed. The mma arm then
//     folds the super-block d into it right there (x_df[...] = d*sc8[l]), so the vec-dot needs
//     only a float scale per group of 16 -- the reason Q3_K uses SRAM layout Q3_K
//     (2*MMQ_TILE_NE_K + MMQ_TILE_NE_K/2 + 4 ints per row: 64 for qs, 16 floats for the scales)
//     and NOT the Q8_1 layout Q4_K uses.
//
// q8_1 ACTIVATION LAYOUT: upstream mmq.cuh:79-80 puts Q3_K on MMQ_Q8_1_DS_LAYOUT_D4 (four floats
// d per block_q8_1_mmq, no sum term), and mma/quantize_mmq.cuh:118-119 already returns D4 for
// Q3_K and instantiates that arm (:327-330). So the vec-dot below reads `const float * y_df =
// (const float *) y` and needs nothing new from the quantizer.
//
// DIVERGENCES FROM UPSTREAM:
//  1. The dp4a arm of both functions is dropped (mmq-load-tiles.cuh:600-604, :636, :673,
//     :677-690 and mmq-vec-dot.cuh:446-479). This port is mma-only; on sm_89 the dp4a arm is
//     dead. The AMD (MFMA/WMMA) arms are dropped too (mmq-vec-dot.cuh:483-532) -- mma_int.cuh
//     instantiates no J_MAJOR tile. Note that dropping the dp4a arm also drops upstream's third
//     loader loop (:677-690, `x_df[i] = bxi->d`), which exists only for the dp4a tile layout:
//     the mma arm has already multiplied d into every scale.
//  2. `const float d = bxi->d;` is spelled `__half2float(bxi->d)`, because the torch extension is
//     compiled with -D__CUDA_NO_HALF_CONVERSIONS__ (same reason mmq_q4_K.cuh divergence 2 spells
//     its half2 arithmetic with intrinsics). The arithmetic is unchanged.
//  3. The functions are members of mmq_type_traits<GGML_TYPE_Q3_K> rather than free templates
//     selected by the ggml_cuda_mmq_get_util_funcs switch (upstream mmq.cuh:535-843). See
//     mma/mmq_core.cuh divergence 3: this is what lets the S4 agents add a type without touching
//     a shared file.
//  4. QI8_1 in the vec-dot is spelled FT_QI8_1 (mma/mmq_core.cuh), and the k01 step
//     `QR3_K*VDR_Q3_K_Q8_1_MMQ` is spelled FT_QR3_K*FT_VDR_Q3_K_Q8_1_MMQ. Same values (8).
//  5. get_int_b2, FT_QR3_K/FT_VDR_Q3_K_Q8_1_MMQ and the vec-dot sit behind one-shot #ifndef
//     guards, which upstream has no need for. See the NOTE below: several S4 headers legitimately
//     need the same symbols in namespace ftmma, and mmq_q4_K.cuh's unguarded style would make two
//     of them un-includable together. The guarded bodies are byte-identical across the files that
//     share them, so which copy wins does not change the arithmetic.
//
// NOTE FOR THE REST OF S4: ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma below is NOT Q3_K-specific --
// upstream uses the same vec-dot for Q3_K, IQ2_XS, IQ2_S and NVFP4 (mmq.cuh:754-759, :791-796,
// :797-802, :834-839), and get_int_b2 is needed by half the S4 types. Until they are hoisted into
// a shared mma/vec_dot_q8_0_16.cuh, this file coexists with the sibling headers by way of the
// guards of divergence 5: FTMMA_VEC_DOT_Q8_0_16_Q8_1_MMA_DEFINED (matching mma/mmq_iq2_s.cuh:375)
// and FTMMA_HAVE_GET_INT_B2 / FTMMA_GET_INT_B2_DEFINED (matching mma/mmq_q8_0.cuh:93 and
// mma/mmq_q6_K.cuh:97). mma/mmq_iq2_xs.cuh instead nests its copies in namespace ftmma::iq2_xs
// and never collides. HOWEVER: mma/mmq_iq3_s.cuh, mma/mmq_iq3_xxs.cuh and mma/mmq_iq4_nl.cuh
// currently define get_int_b2 UNGUARDED in namespace ftmma, so those three still conflict with
// each other and with this file -- see RISKS in the S4 report. The same warning mmq_q4_K.cuh
// carries for ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma (Q4_K / Q5_K / IQ1_S).

#pragma once

#include <cuda_fp16.h>

#include "mmq_core.cuh"

namespace ftmma {

using namespace ggml_cuda_mma;

// ---------------------------------------------------------------------------------------------
// SHARED WITH THE OTHER S4 PER-TYPE HEADERS -- see "NOTE FOR THE REST OF S4" above. get_int_b2 is
// also needed by Q6_K, Q8_0, IQ3_S, IQ3_XXS, IQ4_NL and IQ2_S; the one-shot guard below is the
// convention those files converged on (both macro spellings are tested and both are defined, so
// it composes with either half of the convention). mma/mmq_iq2_xs.cuh sidesteps the whole issue
// with a nested namespace and never collides either way.
// ---------------------------------------------------------------------------------------------
#if !defined(FTMMA_HAVE_GET_INT_B2) && !defined(FTMMA_GET_INT_B2_DEFINED)
#define FTMMA_HAVE_GET_INT_B2
#define FTMMA_GET_INT_B2_DEFINED

// upstream vecdotq.cuh:18-25
static __device__ __forceinline__ int get_int_b2(const void * x, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) x; // assume at least 2 byte alignment

    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;

    return x32;
}

#endif // !FTMMA_HAVE_GET_INT_B2 && !FTMMA_GET_INT_B2_DEFINED

// ---------------------------------------------------------------------------------------------
// upstream mmq-vec-dot.cuh:481-614, the `#elif defined(TURING_MMA_AVAILABLE)` arm at :533-609,
// verbatim except for the FT_ spelling of QI8_1 / QR3_K / VDR_Q3_K_Q8_1_MMQ and the mmq_core
// accessor templates.
// Used by Q3_K (and, in the rest of S4, by IQ2_XS and IQ2_S -- see the NOTE above), hence the
// same one-shot guard: mma/mmq_iq2_s.cuh:375 ships a byte-identical copy under this exact macro,
// so whichever header the wiring agent includes first wins and the other skips. FT_QR3_K and
// FT_VDR_Q3_K_Q8_1_MMQ live INSIDE the guard because they are the k01 step of this loop
// (mmq-vec-dot.cuh:582) and mmq_iq2_s.cuh declares them in the same block.
// ---------------------------------------------------------------------------------------------
#ifndef FTMMA_VEC_DOT_Q8_0_16_Q8_1_MMA_DEFINED
#define FTMMA_VEC_DOT_Q8_0_16_Q8_1_MMA_DEFINED

// upstream ggml-common.h:131 (QR3_K, 4 in the vendored header too -- :76) and vecdotq.cuh:447.
static constexpr int FT_QR3_K              = 4;
static constexpr int FT_VDR_Q3_K_Q8_1_MMQ  = 2;

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    typedef tile<16, 4, int> tile_A;
    typedef tile<16, 8, int> tile_A_8;
    typedef tile< 8, 4, int> tile_B;
    typedef tile<16, 8, int> tile_C;

    constexpr int I             = ggml_cuda_mmq_get_I<type, J, fallback>();
    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride<type, J, fallback>();
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp<type, J, fallback>();
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + MMQ_TILE_NE_K*2;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = (threadIdx.y / ntx) * (ntx*tile_A::I);

    tile_A  A[ntx][8];
    float  dA[ntx][tile_C::ne/2][8];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 8) {
            const int k0 = k00 + k01;

            load_ldmatrix(((tile_A_8 *) A[n])[k01/8], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
        }

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_C::I + tile_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 4) {
                const int k0 = k00 + k01;

                dA[n][l][k01/4] = x_df[i*sram_stride + k0/4];
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += FT_QR3_K*FT_VDR_Q3_K_Q8_1_MMQ) {
            tile_B B[2];
            float dB[tile_C::ne/2];

            // Here load_generic is faster than load_ldmatrix.
            load_generic(B[0], y_qs + j0*MMQ_TILE_Y_K + (k01 + 0),         MMQ_TILE_Y_K);
            load_generic(B[1], y_qs + j0*MMQ_TILE_Y_K + (k01 + tile_B::J), MMQ_TILE_Y_K);

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);

                dB[l] = y_df[j*MMQ_TILE_Y_K + k01/FT_QI8_1];
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C[2];
                mma(C[0], A[n][k01/4 + 0], B[0]);
                mma(C[1], A[n][k01/4 + 1], B[1]);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dB[l%2]*(C[0].x[l]*dA[n][l/2][k01/4 + 0] + C[1].x[l]*dA[n][l/2][k01/4 + 1]);
                }
            }
        }
    }

    FTMMA_UNUSED(I);
}

#endif // FTMMA_VEC_DOT_Q8_0_16_Q8_1_MMA_DEFINED

// ---------------------------------------------------------------------------------------------
// Q3_K-only constants and block layout. See the "SAFE TYPE" note above: these AGREE with the
// vendored ggml-common.h:76-83, they are re-declared only because that header's spellings are
// macros and because mma/ must not include it. FT_QR3_K comes from the guarded block above (it is
// 4 in every copy). FT_K_SCALE_SIZE is deliberately NOT re-declared here -- mma/mmq_q4_K.cuh
// already owns that name in namespace ftmma, and upstream's block_q3_K (:318) spells the same 12
// as a literal anyway.
// ---------------------------------------------------------------------------------------------

static constexpr int FT_QI3_K = FTMMA_QK_K / (4 * FT_QR3_K);  // 16

// upstream ggml-common.h:315-320 / vendored ggml-common.h:78-83 (identical layout).
struct block_q3_K {
    uint8_t hmask[FTMMA_QK_K / 8];  // quants - high bit
    uint8_t qs[FTMMA_QK_K / 4];     // quants - low 2 bits
    uint8_t scales[12];             // scales, quantized with 6 bits
    half    d;                      // super-block scale
};
static_assert(sizeof(block_q3_K) == sizeof(half) + FTMMA_QK_K/4 + FTMMA_QK_K/8 + 12,
              "wrong q3_K block size/padding");
static_assert(sizeof(block_q3_K) == 110, "block_q3_K must be 110 B");

// ---------------------------------------------------------------------------------------------
// upstream mmq-load-tiles.cuh:590-691, the mma arm only (:597-599 for the pointers, :607-639 for
// the quants + high-bit plane, :641-675 for the sign-corrected scales folded with d).
// ---------------------------------------------------------------------------------------------

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q3_K(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = FTMMA_WARP_SIZE;
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads<type, J, fallback>() / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I<type, J, fallback>();
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride<type, J, fallback>();

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);

    constexpr int threads_per_row = MMQ_ITER_K / (4 * FT_QR3_K);   // 16
    constexpr int nrows = warp_size / threads_per_row;             // 2
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = (const block_q3_K *) x + kbx0 + i*stride;

        const int x_ql_0 = get_int_b2(bxi->qs,    kqsx);
        const int x_qh_0 = get_int_b2(bxi->hmask, kqsx % (FT_QI3_K/2)) >> (4 * (kqsx / (FT_QI3_K/2)));

#pragma unroll
        for (int l = 0; l < FT_QR3_K; ++l) {
            const int k = (kqsx/8)*32 + l*8 + kqsx % 8;

            const int x_ql_k =  (x_ql_0 >> (2*l))       & 0x03030303;
            const int x_qh_k = ((x_qh_0 >>    l)  << 2) & 0x04040404;

            const int x_qs_k = __vsubss4(x_ql_k | x_qh_k, 0x04040404);

            x_qs[i*sram_stride + k] = x_qs_k;
        }
    }

    constexpr int rows_per_warp = warp_size / 4;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*rows_per_warp) {
        int i = i0 + threadIdx.y*rows_per_warp + threadIdx.x/4;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = (const block_q3_K *) x + kbx0 + i*stride;

        const int ksc = threadIdx.x % 4;

        const int ksc_low = ksc % (FT_QI3_K/8);
        const int shift_low = 4 * (ksc / (FT_QI3_K/8));
        const int sc_low = (get_int_b2(bxi->scales, ksc_low) >> shift_low) & 0x0F0F0F0F;

        const int ksc_high = FT_QI3_K/8;
        const int shift_high = 2 * ksc;
        const int sc_high = ((get_int_b2(bxi->scales, ksc_high) >> shift_high) << 4) & 0x30303030;

        const int sc = __vsubss4(sc_low | sc_high, 0x20202020);

        const int8_t * sc8 = (const int8_t *) &sc;
        // divergence 2: upstream writes `const float d = bxi->d;`. Same value, intrinsic spelling.
        const float d = __half2float(bxi->d);

#pragma unroll
        for (int l = 0; l < int(sizeof(int)); ++l) {
            x_df[i*sram_stride + sizeof(int)*ksc + l] = d*sc8[l];
        }
    }

    FTMMA_UNUSED(i_max);
}

// ---------------------------------------------------------------------------------------------
// The per-type hook mma/mmq_core.cuh declares. upstream mmq.cuh:754-759 selects exactly this pair
// for Q3_K (load_tiles_q3_K + vec_dot_q8_0_16_q8_1_mma + write_back_mma), and
// mmq-config-ampere.cuh:140-155 gives it GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K.
// ---------------------------------------------------------------------------------------------

template <> struct mmq_type_traits<GGML_TYPE_Q3_K> {
    static constexpr ggml_cuda_mmq_sram_layout sram_layout = GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K;

    template <int J, bool fallback>
    static __device__ __forceinline__ void load_tiles(
            const char * __restrict__ x, int * __restrict__ x_tile,
            const int kbx0, const int i_max, const int stride) {
        ggml_cuda_mmq_load_tiles_q3_K<GGML_TYPE_Q3_K, J, fallback>(x, x_tile, kbx0, i_max, stride);
    }

    template <int J, bool fallback>
    static __device__ __forceinline__ void vec_dot(
            const int * __restrict__ x, const int * __restrict__ y,
            float * __restrict__ sum, const int k00) {
        ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma<GGML_TYPE_Q3_K, J, fallback>(x, y, sum, k00);
    }
};

}  // namespace ftmma
