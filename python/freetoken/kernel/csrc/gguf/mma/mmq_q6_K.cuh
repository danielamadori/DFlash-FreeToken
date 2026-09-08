// mma/mmq_q6_K.cuh -- Q6_K for the mma MMQ kernel. Plan step S4, one type end to end
// (tile loader + vec-dot), structured exactly like mma/mmq_q4_K.cuh.
//
// PORTED FROM llama.cpp @ f7aadef09:
//   ggml-common.h:362-368       block_q6_K (layout only; see "SAFE TYPE" below)
//   vecdotq.cuh:17-24           get_int_b2
//   mmq-load-tiles.cuh:938-1024 ggml_cuda_mmq_load_tiles_q6_K, the mma arm
//                               (:945-948 pointers, :980-982 quants, :1004 the d, :1015-1016 sc)
//   mmq-vec-dot.cuh:1018-1176   ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma, the TURING_MMA arm
//                               (:1074-1175)
//   mmq.cuh:772-777             the (load_tiles_q6_K, vec_dot_q6_K_q8_1_mma) pairing
//   mmq-config-ampere.cuh:191-195  GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K for every Ada Q6_K row
//
// WHY Q6_K IS SAFE (the CRITICAL SAFETY RULE of S4, and the per-type conflict table in the plan):
// QR6_K == 2 and QI6_K == 32 in BOTH the vendored ggml-common.h (:102-103) and upstream
// (:139-140), and the two block_q6_K structs are byte-identical (uint8_t ql[128]; uint8_t qh[64];
// int8_t scales[16]; half d; 210 B). So no nibble can be mis-addressed the way IQ4_XS would be.
// The struct is still declared LOCALLY, in namespace ftmma, because the rule for the whole port is
// that mma/ includes nothing from the vendored headers -- and because QR6_K/QI6_K/QK_K are MACROS
// over there, so they cannot be spelled as C++ identifiers here at all. FT_QI6_K is NOT redeclared
// here: mma/mmq_core.cuh:82 already defines it (== FTMMA_QK_K/8 == 32, i.e. QK_K/(4*QR6_K)), and a
// second namespace-scope definition in the same TU would be a redefinition.
//
// THE q8_1 ds LAYOUT: Q6_K has no min/partial-sum term, so upstream quantizes the activations for
// it with MMQ_Q8_1_DS_LAYOUT_D4 (mmq.cuh:85-91) and the vec-dot below reads `const float * y_df =
// (const float *) y`, i.e. block_q8_1_mmq::d4. mma/quantize_mmq.cuh:123-129 already returns
// MMQ_Q8_1_DS_LAYOUT_D4 for GGML_TYPE_Q6_K and instantiates that arm, so nothing is needed from
// the quantizer for this type.
//
// DIVERGENCES FROM UPSTREAM:
//  1. The dp4a arm of both functions is dropped (the `#else` branches of the four
//     AMD_MFMA_AVAILABLE || TURING_MMA_AVAILABLE || AMD_WMMA_AVAILABLE guards in
//     mmq-load-tiles.cuh:938-1024, and ggml_cuda_mmq_vec_dot_q6_K_q8_1_dp4a,
//     mmq-vec-dot.cuh:983-1016). This port is mma-only; on sm_89 the dp4a arm is dead.
//     The AMD (MFMA/WMMA) arms are dropped too -- mma_int.cuh instantiates no J_MAJOR tile, which
//     is what the AMD arm of the vec-dot (mmq-vec-dot.cuh:1020-1073) needs.
//  2. half -> float is spelled with __half2float instead of upstream's implicit conversion
//     (`x_df[...] = bxi->d`), because the torch extension is compiled with
//     -D__CUDA_NO_HALF_CONVERSIONS__, which deletes exactly that conversion. Same value; this is
//     the read direction of the same constraint that made mma/quantize_mmq.cuh:283 spell its
//     float -> half with __float2half and mma/mmq_q4_K.cuh use __hmul2/__floats2half2_rn.
//  3. `sizeof(int)` in the two loop bounds is cast to int (upstream mmq-vec-dot.cuh:1119 writes
//     `ksc < sizeof(int)`), to keep the signed/unsigned comparison warning out of the build.
//     mma/mmq_q4_K.cuh made the same cast for the same reason.
//  4. The functions are members of mmq_type_traits<GGML_TYPE_Q6_K> rather than free templates
//     selected by the ggml_cuda_mmq_get_util_funcs switch (upstream mmq.cuh:535-843). See
//     mma/mmq_core.cuh divergence 3: this is what lets the S4 agents add a type without touching
//     a shared file.
//  5. get_int_b2 is wrapped in a two-spelling include guard; see its comment at the definition.
//
// NOTE: unlike mma/mmq_q4_K.cuh's vec-dot (which upstream shares between Q4_K, Q5_K and IQ1_S),
// ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma is used by Q6_K alone (mmq.cuh:772-777 is its only caller),
// so it lives here and nothing else should include it.

#pragma once

#include <cuda_fp16.h>

#include "mmq_core.cuh"

namespace ftmma {

using namespace ggml_cuda_mma;

// ---------------------------------------------------------------------------------------------
// Local Q6_K constants and block layout. See the "SAFE TYPE" note above: these AGREE with the
// vendored ggml-common.h:102-109, they are re-declared only because that header's spellings are
// macros and because mma/ must not include it. FT_QI6_K comes from mma/mmq_core.cuh:82.
// ---------------------------------------------------------------------------------------------

static constexpr int FT_QR6_K = 2;
static_assert(FT_QI6_K == FTMMA_QK_K / (4 * FT_QR6_K), "FT_QI6_K must be QK_K/(4*QR6_K) == 32");

// upstream ggml-common.h:362-367 / vendored ggml-common.h:104-109 (identical layout).
struct block_q6_K {
    uint8_t ql[FTMMA_QK_K / 2];      // quants, lower 4 bits
    uint8_t qh[FTMMA_QK_K / 4];      // quants, upper 2 bits
    int8_t  scales[FTMMA_QK_K / 16]; // scales, quantized with 8 bits
    half    d;                       // super-block scale
};
// upstream ggml-common.h:368
static_assert(sizeof(block_q6_K) == sizeof(half) + FTMMA_QK_K/16 + 3*FTMMA_QK_K/4,
              "wrong q6_K block size/padding");
static_assert(sizeof(block_q6_K) == 210, "block_q6_K must be 210 B");

// upstream vecdotq.cuh:17-24. Q6_K blocks are 210 B, so a block field is NOT 4-byte aligned in
// general and the 2 x 16 bit load is required (this is why Q6_K uses get_int_b2 where Q4_K, whose
// blocks are 144 B, uses get_int_b4).
//
// divergence 5: the definition is wrapped in an include guard. get_int_b2 is not Q6_K-specific --
// upstream keeps it in vecdotq.cuh and seven of the S4 types need it -- but this port has no
// shared vecdotq header, so every per-type file carries its own copy and the second one included
// would be a redefinition. Two guard spellings are in flight among the S4 files
// (FTMMA_HAVE_GET_INT_B2 and FTMMA_GET_INT_B2_DEFINED), so this block tests and defines BOTH and
// therefore composes with either. See RISKS: three S4 files still define it unguarded, which the
// wiring agent has to resolve centrally (hoisting one copy into mma/mmq_core.cuh is the fix).
#if !defined(FTMMA_HAVE_GET_INT_B2) && !defined(FTMMA_GET_INT_B2_DEFINED)
#define FTMMA_HAVE_GET_INT_B2
#define FTMMA_GET_INT_B2_DEFINED
static __device__ __forceinline__ int get_int_b2(const void * x, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) x; // assume at least 2 byte alignment

    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;

    return x32;
}
#endif // !FTMMA_HAVE_GET_INT_B2 && !FTMMA_GET_INT_B2_DEFINED

// ---------------------------------------------------------------------------------------------
// upstream mmq-load-tiles.cuh:938-1024, the mma arm only (:945-948 for the pointers, :956-989 for
// the quants, :991-1005 for the super-block scale, :1007-1023 for the 16 int8 sub-scales).
//
// SRAM row layout (sram_stride == 76 ints, mmq_core.cuh's GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K):
//   [ 0 .. 63] x_qs  64 ints  = 256 six-bit quants, already biased by -32 into signed int8
//   [64      ] x_df   1 float = the super-block scale d
//   [65 .. 68] x_sc   4 ints  = the 16 int8 sub-scales
//   [69 .. 75]        7 ints  = bank-conflict padding
// ---------------------------------------------------------------------------------------------

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q6_K(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = FTMMA_WARP_SIZE;
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads<type, J, fallback>() / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I<type, J, fallback>();
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride<type, J, fallback>();

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
    int   * x_sc = (int   *) (x_df + MMQ_TILE_NE_K/FT_QI6_K);

    constexpr int threads_per_row = MMQ_ITER_K / (4 * FT_QR6_K);   // 32
    constexpr int nrows = warp_size / threads_per_row;             // 1
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = (const block_q6_K *) x + kbx0 + i*stride;

        const int ql = get_int_b2(bxi->ql, txi);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_b2(bxi->qh, (FT_QI6_K/4) * (txi / (FT_QI6_K/2)) + txi % (FT_QI6_K/4));
        const int qh0 = ((qh >> ((txi & 0x08) >> 2)) << 4) & 0x30303030;
        const int qh1 =  (qh >> ((txi & 0x08) >> 2))       & 0x30303030;

        const int kq0 = 2*txi - txi % (FT_QI6_K/2) + 0;
        const int kq1 = 2*txi - txi % (FT_QI6_K/2) + FT_QI6_K/2;

        x_qs[i*sram_stride + kq0] = __vsubss4(ql0 | qh0, 0x20202020);
        x_qs[i*sram_stride + kq1] = __vsubss4(ql1 | qh1, 0x20202020);
    }

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*warp_size) {
        int i = (i0 + threadIdx.y*warp_size + threadIdx.x) % I;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = (const block_q6_K *) x + kbx0 + i*stride;

        // divergence 2: upstream writes `x_df[i*sram_stride] = bxi->d`.
        x_df[i*sram_stride] = __half2float(bxi->d);
    }

    constexpr int rows_per_warp = warp_size / 4;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*rows_per_warp) {
        int i = (i0 + threadIdx.y*rows_per_warp + threadIdx.x/(MMQ_TILE_NE_K/8)) % I;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = (const block_q6_K *) x + kbx0 + i*stride + (threadIdx.x % (MMQ_TILE_NE_K/8)) / 4;

        x_sc[i*sram_stride + threadIdx.x%4] = get_int_b2(bxi->scales, threadIdx.x % (MMQ_TILE_NE_K/8));
    }

    FTMMA_UNUSED(i_max);
}

// ---------------------------------------------------------------------------------------------
// upstream mmq-vec-dot.cuh:1018-1176, the `#elif defined(TURING_MMA_AVAILABLE)` arm at :1074-1175,
// verbatim except for the FT_ spelling of QI8_1 and the mmq_core accessor templates.
//
// Q6_K carries one int8 sub-scale per 16 logical values, so the mma is the m16n8k16 shape
// (tile_A 16x4, tile_B 8x4) rather than Q4_K's m16n8k32: the k loop steps 8 SRAM ints == 32
// values == 2 sub-scale groups, and the accumulator tmp[][] is kept in float per sub-scale before
// the single multiplication by the super-block scale dA at the end of each j0 iteration.
// y is read through `y_df` (block_q8_1_mmq::d4, MMQ_Q8_1_DS_LAYOUT_D4) -- see the header note.
// ---------------------------------------------------------------------------------------------

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    typedef tile<16, 4, int> tile_A;
    typedef tile< 8, 4, int> tile_B;
    typedef tile<16, 8, int> tile_C;

    constexpr int I             = ggml_cuda_mmq_get_I<type, J, fallback>();
    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride<type, J, fallback>();
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp<type, J, fallback>();
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + MMQ_TILE_NE_K*2;
    const int   * x_sc = (const int   *) x_df + MMQ_TILE_NE_K/FT_QI6_K;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = (threadIdx.y / ntx) * (ntx*tile_A::I);

    tile_A   A[ntx][8];
    int    scA[ntx][tile_C::ne/2][8];
    float   dA[ntx][tile_C::ne/2];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 8) {
            const int k0 = k00 + k01;

            load_ldmatrix(A[n][k01/4 + 0], x_qs + (i0 + n*tile_A::I)*sram_stride + (k0 + 0),         sram_stride);
            load_ldmatrix(A[n][k01/4 + 1], x_qs + (i0 + n*tile_A::I)*sram_stride + (k0 + tile_A::J), sram_stride);
        }

#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 16) {
            const int k0 = k00 + k01;

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int i = i0 + n*tile_C::I + tile_C::get_i(2*l);

                const int      sc_packed = x_sc[i*sram_stride + k0/16];
                const int8_t * sc        = (const int8_t *) &sc_packed;

                // divergence 3: upstream writes `ksc < sizeof(int)`.
#pragma unroll
                for (int ksc = 0; ksc < (int) sizeof(int); ++ksc) {
                    scA[n][l][k01/4 + ksc] = sc[ksc];
                }
            }
        }

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_C::I + tile_C::get_i(2*l);

            dA[n][l] = x_df[i*sram_stride];
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
        float tmp[ntx][tile_C::ne] = {{0.0f}};

#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 8) {
            tile_B B[2];
            float dB[tile_C::ne/2];

            // Here load_generic is faster than load_ldmatrix.
            load_generic(B[0], y_qs + j0*MMQ_TILE_Y_K + 0         + k01, MMQ_TILE_Y_K);
            load_generic(B[1], y_qs + j0*MMQ_TILE_Y_K + tile_B::J + k01, MMQ_TILE_Y_K);

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
                    tmp[n][l] += (C[0].x[l]*scA[n][l/2][k01/4 + 0] + C[1].x[l]*scA[n][l/2][k01/4 + 1])*dB[l%2];
                }
            }
        }

#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                sum[(j0/tile_C::J + n)*tile_C::ne + l] += tmp[n][l]*dA[n][l/2];
            }
        }
    }

    FTMMA_UNUSED(I);
}

// ---------------------------------------------------------------------------------------------
// The per-type hook mma/mmq_core.cuh declares. upstream mmq.cuh:772-777 selects exactly this pair
// for Q6_K (load_tiles_q6_K + vec_dot_q6_K_q8_1_mma + write_back_mma), and
// mmq-config-ampere.cuh:191-195 gives it GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K.
// ---------------------------------------------------------------------------------------------

template <> struct mmq_type_traits<GGML_TYPE_Q6_K> {
    static constexpr ggml_cuda_mmq_sram_layout sram_layout = GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K;

    template <int J, bool fallback>
    static __device__ __forceinline__ void load_tiles(
            const char * __restrict__ x, int * __restrict__ x_tile,
            const int kbx0, const int i_max, const int stride) {
        ggml_cuda_mmq_load_tiles_q6_K<GGML_TYPE_Q6_K, J, fallback>(x, x_tile, kbx0, i_max, stride);
    }

    template <int J, bool fallback>
    static __device__ __forceinline__ void vec_dot(
            const int * __restrict__ x, const int * __restrict__ y,
            float * __restrict__ sum, const int k00) {
        ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma<GGML_TYPE_Q6_K, J, fallback>(x, y, sum, k00);
    }
};

}  // namespace ftmma
