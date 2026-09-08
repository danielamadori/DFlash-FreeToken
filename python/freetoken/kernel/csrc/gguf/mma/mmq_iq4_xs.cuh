// mma/mmq_iq4_xs.cuh -- IQ4_XS for the mma MMQ kernel. Plan step S4, owner C5 (the iq4_xs row of
// the S4 file list). Structured exactly like mma/mmq_q4_K.cuh: local constants, local block
// struct, the ported mma-arm loader, the ported mma-arm vec-dot, then the mmq_type_traits
// specialisation.
//
// PORTED FROM llama.cpp @ f7aadef09:
//   ggml-common.h:454-459        block_iq4_xs (layout only; see "THE QR TRAP" below)
//   ggml-common.h:166-167        QI4_XS / QR4_XS
//   ggml-common.h:1119-1121      kvalues_iq4nl, the 16-entry int8 lookup table
//   vecdotq.cuh:26-28            get_int_b4
//   vecdotq.cuh:30-95            get_int_from_table_16, the CUDA (__byte_perm) arm at :59-80
//   mmq-load-tiles.cuh:1420-1485 ggml_cuda_mmq_load_tiles_iq4_xs, the mma arm
//                                (:1427-1429 pointers, :1436-1461 quants, :1463-1484 scales)
//   mmq-vec-dot.cuh:142-280      ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma, the NVIDIA arm (:203-278)
//   mmq.cuh:815-820              the (loader, vec-dot, write-back) triple upstream picks for
//                                IQ4_XS, and the MMQ_Q8_1_DS_LAYOUT_D4 it hands the vec-dot
//   mmq-config-ampere.cuh:312-327  GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0 for every IQ4_XS row
//
// THE QR TRAP LIVES HERE (plan S4's per-type conflict table, first row).
//   vendored csrc/gguf/ggml-common.h:185-186   QR4_XS = 8  ->  QI4_XS = 256/(4*8)  =  8
//   upstream ggml/src/ggml-common.h:166-167    QR4_XS = 2  ->  QI4_XS = 256/(4*2)  = 32
// The two block_iq4_xs structs are byte-identical (half d; uint16_t scales_h; uint8_t
// scales_l[4]; uint8_t qs[128]; 136 B) and kvalues_iq4nl holds the same 16 values on both sides,
// so the ONLY thing that differs is QR4_XS -- and it is exactly the constant this loader uses to
// size its thread grid:
//     threads_per_row = MMQ_ITER_K / (4 * QR4_XS)
// With upstream's 2 that is 32 threads per row, i.e. one thread per int of the 128-byte qs array,
// which is what the k0 = 8*(kqsx/4) + kqsx%4 addressing below assumes. With the vendored 8 it
// would be 8 threads per row: three quarters of every weight block would never be read, and the
// three quarters that were read would land at the wrong k0. That failure is SILENT -- it compiles,
// it runs, it returns plausible-looking numbers. Hence the rule for the whole port: this file
// includes NOTHING from the vendored headers and declares every constant, struct and table it
// needs LOCALLY, inside namespace ftmma. The vendored files stay byte-identical; the decode path
// (mmvq.cuh / mmvq_planar*.cuh / vecdotq.cuh) keeps using the vendored QR4_XS == 8, which is
// correct for the decode path's own addressing and must not be touched.
//
// DIVERGENCES FROM UPSTREAM:
//  1. The dp4a arm of both functions is dropped (mmq-load-tiles.cuh:1430-1433, :1457-1459,
//     :1481-1482 and the whole ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a). This port is mma-only; on
//     sm_89 the dp4a arm is dead. The AMD (MFMA/WMMA) arms are dropped too -- mma_int.cuh
//     instantiates no J_MAJOR tile, so mmq-vec-dot.cuh:145-202 has nothing to compile against.
//     get_int_from_table_16 likewise keeps only its CUDA arm (vecdotq.cuh:59-80); the
//     GGML_USE_HIP (__builtin_amdgcn_perm) and GGML_USE_MUSA (generic) arms are dropped.
//  2. NESTED NAMESPACE. Everything except the mmq_type_traits specialisation lives in
//     `namespace ftmma::iq4_xs`. mma/mmq_q4_K.cuh already defines get_int_b4 at ftmma scope, and
//     the ten parallel S4 agents would otherwise collide on get_int_b4,
//     get_int_from_table_16, kvalues_iq4nl and ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma -- all of
//     which are `static __device__ __forceinline__`, so a second definition in the same
//     translation unit is a hard redefinition error, whatever the include order. Nesting keeps
//     upstream's spelling of every name and makes the file safe to include next to any other
//     mma/mmq_<type>.cuh. See NOTE FOR THE WIRING AGENT below for the proper fix.
//  3. The functions are members of mmq_type_traits<GGML_TYPE_IQ4_XS> rather than free templates
//     selected by the ggml_cuda_mmq_get_util_funcs switch (upstream mmq.cuh:790-843). See
//     mma/mmq_core.cuh divergence 3: this is what lets the ten S4 agents add a type without
//     touching a shared file.
//  4. QI8_0/QI8_1 are spelled FT_QI8_0/FT_QI8_1 (mma/mmq_core.cuh), because the vendored
//     ggml-common.h makes them macros. Same values (8), spelling only.
//
// NOTE FOR THE WIRING AGENT / OWNER D: ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma below is NOT
// IQ4_XS-specific -- upstream uses the same vec-dot for Q8_0, IQ2_XXS, IQ3_XXS, IQ3_S, IQ4_XS and
// IQ4_NL (mmq.cuh:775-826), and the plan puts it in mma/vec_dot_q8_0.cuh owned by D. Six S4 files
// will each carry a private copy in their own nested namespace. Once they have all landed, hoist
// ONE copy (this one, or any -- they are ports of the same function) into mma/vec_dot_q8_0.cuh,
// hoist get_int_b4 / get_int_from_table_16 / kvalues_iq4nl next to it, and have the six files
// include that header instead. Copies must not be allowed to drift.

#pragma once

#include <cuda_fp16.h>

#include "mmq_core.cuh"

namespace ftmma {
namespace iq4_xs {

using namespace ggml_cuda_mma;

// ---------------------------------------------------------------------------------------------
// Local IQ4_XS constants, block layout and lookup table. See "THE QR TRAP" above: FT_QR4_XS is
// UPSTREAM's 2, NOT the vendored ggml-common.h:185's 8. The struct and the table agree with the
// vendored header (:187-192, :927-928); they are re-declared only because mma/ must not include
// it and because QR4_XS/QI4_XS/QK_K are macros over there.
// ---------------------------------------------------------------------------------------------

static constexpr int FT_QR4_XS = 2;                                 // upstream ggml-common.h:167
static constexpr int FT_QI4_XS = FTMMA_QK_K / (4 * FT_QR4_XS);      // upstream ggml-common.h:166 -> 32

static_assert(FT_QI4_XS == 32, "IQ4_XS: upstream QI4_XS is 32; the vendored header's is 8");

// upstream ggml-common.h:454-459 / vendored ggml-common.h:187-192 (identical layout).
struct block_iq4_xs {
    half     d;
    uint16_t scales_h;
    uint8_t  scales_l[FTMMA_QK_K/64];
    uint8_t  qs[FTMMA_QK_K/2];
};
static_assert(sizeof(block_iq4_xs) == sizeof(half) + sizeof(uint16_t) + FTMMA_QK_K/64 + FTMMA_QK_K/2,
              "wrong iq4_xs block size/padding");
static_assert(sizeof(block_iq4_xs) == 136, "block_iq4_xs must be 136 B");

// upstream ggml-common.h:1119-1121 (GGML_TABLE_BEGIN expands to `static const __device__ ...`
// under GGML_COMMON_IMPL_CUDA, ggml-common.h:493). Same 16 values as vendored :927-928.
static const __device__ int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

// upstream vecdotq.cuh:26-28
static __device__ __forceinline__ int get_int_b4(const void * x, const int & i32) {
    return ((const int *) x)[i32]; // assume at least 4 byte alignment
}

// upstream vecdotq.cuh:30-95, the CUDA arm (:59-80) only -- divergence 1.
// q4 contains 8 indices with 4 bit each.
// This function selects those bytes from table that are at those indices and returns them as int2.
// The first int contains the bytes with even indices in q4, the second int contains the bytes with odd indices in q4.
static __device__ __forceinline__ int2 get_int_from_table_16(const int & q4, const int8_t * table) {
    // CUDA does not have an instruction for selecting bytes with 4 bit indices.
    // However, __byte_perm is an instruction that selects bytes with 3 bit indices that can be used instead.
    const uint32_t * table32 = (const uint32_t *) table;

    // __byte_perm selects bytes based on the lower 16 bits in its third argument.
    // Therefore, do 2 iterations over the 32 bits in q4 with 0 and 16 shift.
    // To handle the fourth bit, first call _byte_perm both for the low and the high 64 bit of table, using the low 3 bits.
    // Then, call __byte_perm again to select from the low and high bytes based on the fourth bit.
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = (0x32103210 | ((q4 & 0x88888888) >> 1));
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;

        const uint32_t low  = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }

    // tmp contains the bytes from tyble in the same order as the 4 bit indices in q4.
    // However, for the result we need ints with all even/odd 4 bit indices in q4.
    // Therefore, 2 more calls to __byte_perm to put the bytes in the correct order.
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

// ---------------------------------------------------------------------------------------------
// upstream mmq-vec-dot.cuh:142-280, the `#else` (NVIDIA) arm at :203-278, verbatim except for the
// FT_ spelling of QI8_0/QI8_1 and the mmq_core accessor templates.
// Used by IQ4_XS (and, upstream, by Q8_0, IQ2_XXS, IQ3_XXS, IQ3_S and IQ4_NL -- see the NOTE FOR
// THE WIRING AGENT in the header comment).
//
// x_df is FLOAT here, not half2 as in the Q4_K vec-dot: the Q8_0 SRAM layout carries one 32-bit
// scale per 32 weights and no min term, which is also why upstream hands this vec-dot
// MMQ_Q8_1_DS_LAYOUT_D4 for IQ4_XS (mmq.cuh:819) -- the activation side then needs only d, and
// mma/quantize_mmq.cuh:132-134 already maps GGML_TYPE_IQ4_XS to MMQ_Q8_1_DS_LAYOUT_D4 and
// instantiates that quantizer (:327-330). The `ds_layout` template parameter is kept, and the
// MMQ_Q8_1_DS_LAYOUT_DS4 branch with it, so the function stays a verbatim port and can be hoisted
// unchanged for the types that do want DS4.
// ---------------------------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------------------------
// upstream mmq-load-tiles.cuh:1420-1485, the mma arm only (:1427-1429 for the pointers,
// :1436-1461 for the quants, :1463-1484 for the scales).
//
// The IQ4_XS block is 256 weights: 128 bytes of 4-bit indices into kvalues_iq4nl, plus one
// 6-bit scale per 32 weights split across scales_l (low 4 bits, two scales per byte) and
// scales_h (high 2 bits, eight scales in one uint16). The loader turns both into the Q8_0 SRAM
// layout -- 2*MMQ_TILE_NE_K == 64 ints of int8 weights followed by MMQ_TILE_NE_K/FT_QI8_0 * 2 == 8
// floats of scale, per row of the tile, with sram_stride == 76 (mmq_core.cuh's
// GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0) -- so the vec-dot above never sees the 4-bit form.
// ---------------------------------------------------------------------------------------------

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq4_xs(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = FTMMA_WARP_SIZE;                                    // mmq_core.cuh divergence 8
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads<type, J, fallback>() / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I<type, J, fallback>();
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride<type, J, fallback>();

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);

    // THE QR TRAP: FT_QR4_XS is upstream's 2, so this is 32 -- one thread per int of the 128-byte
    // qs array. The vendored QR4_XS (8) would make it 8 and read a quarter of every block.
    constexpr int threads_per_row = MMQ_ITER_K / (4 * FT_QR4_XS);   // 32
    constexpr int nrows = warp_size / threads_per_row;              // 1
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq4_xs * bxi = (const block_iq4_xs *) x + kbx0 + i*stride;

        const int aux_q4 = get_int_b4(bxi->qs, kqsx);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        const int k0 = 8 * (kqsx / 4) + kqsx % 4;

        x_qs[i*sram_stride + k0 + 0] = v.x;
        x_qs[i*sram_stride + k0 + 4] = v.y;
    }

    constexpr int rows_per_warp = warp_size / 8;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / (MMQ_TILE_NE_K/4);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq4_xs * bxi = (const block_iq4_xs *) x + kbx0 + i*stride;

        const float d = __half2float(bxi->d);

        const int ls = ((bxi->scales_l[(threadIdx.x % 8)/2] >> (4*(threadIdx.x % 2))) & 0x0F)
            | (((bxi->scales_h >> (2*(threadIdx.x % 8))) & 0x03) << 4);

        x_df[i*sram_stride             + threadIdx.x % 8] = d * (ls - 32);
    }

    FTMMA_UNUSED(i_max);
}

}  // namespace iq4_xs

// ---------------------------------------------------------------------------------------------
// The per-type hook mma/mmq_core.cuh declares. upstream mmq.cuh:815-820 selects exactly this pair
// for IQ4_XS (load_tiles_iq4_xs + vec_dot_q8_0_q8_1_mma<..., MMQ_Q8_1_DS_LAYOUT_D4> +
// write_back_mma), and mmq-config-ampere.cuh:312-327 gives it GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0 in
// every one of its rows.
// ---------------------------------------------------------------------------------------------

template <> struct mmq_type_traits<GGML_TYPE_IQ4_XS> {
    static constexpr ggml_cuda_mmq_sram_layout sram_layout = GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0;

    template <int J, bool fallback>
    static __device__ __forceinline__ void load_tiles(
            const char * __restrict__ x, int * __restrict__ x_tile,
            const int kbx0, const int i_max, const int stride) {
        iq4_xs::ggml_cuda_mmq_load_tiles_iq4_xs<GGML_TYPE_IQ4_XS, J, fallback>(x, x_tile, kbx0, i_max, stride);
    }

    template <int J, bool fallback>
    static __device__ __forceinline__ void vec_dot(
            const int * __restrict__ x, const int * __restrict__ y,
            float * __restrict__ sum, const int k00) {
        iq4_xs::ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<GGML_TYPE_IQ4_XS, J, fallback, MMQ_Q8_1_DS_LAYOUT_D4>(
            x, y, sum, k00);
    }
};

}  // namespace ftmma
