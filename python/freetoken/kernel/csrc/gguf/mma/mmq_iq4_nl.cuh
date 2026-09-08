// mma/mmq_iq4_nl.cuh -- IQ4_NL for the mma MMQ kernel. Plan step S4, one type end to end,
// structured exactly like mma/mmq_q4_K.cuh (which S3 shipped and whose header comment states the
// porting rules this file follows).
//
// PORTED FROM llama.cpp @ f7aadef09:
//   ggml-common.h:163-164        QR4_NL / QI4_NL
//   ggml-common.h:447-452        QK4_NL, block_iq4_nl (layout only; see "SAFE TYPE" below)
//   ggml-common.h:1120-1122      kvalues_iq4nl, the 16-entry non-linear table
//   vecdotq.cuh:18-25            get_int_b2
//   vecdotq.cuh:31-95            get_int_from_table_16, the `#elif !defined(GGML_USE_MUSA)`
//                                (CUDA __byte_perm) arm at :57-80
//   mmq-load-tiles.cuh:1487-1552 ggml_cuda_mmq_load_tiles_iq4_nl, the mma arm
//                                (:1494-1496 pointers, :1523-1525 quants, :1546-1547 scales)
//   mmq-vec-dot.cuh:142-280      ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma, the `#else` (NVIDIA) arm
//                                at :202-279
//   mmq.cuh:820-826              the (load_tiles_iq4_nl, vec_dot_q8_0_q8_1_mma<D4>,
//                                write_back_mma) triple that selects this pair for IQ4_NL
//   mmq-config-ampere.cuh:329-344 IQ4_NL rows -> GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0
//
// WHY IQ4_NL IS A SAFE TYPE (the CRITICAL SAFETY RULE of S4, and mmq_q4_K.cuh's "SAFE TYPE" note):
// the vendored ggml-common.h and upstream AGREE on every IQ4_NL constant --
//   QK4_NL == 32   vendored :177 / upstream :447
//   QR4_NL ==  2   vendored :178 / upstream :164
//   QI4_NL ==  4   vendored :179 / upstream :163   (both spelled QK4_NL/(4*QR4_NL))
// the two block_iq4_nl structs are byte-identical (half d; uint8_t qs[16]; 18 B), and the two
// kvalues_iq4nl tables hold the same 16 int8 values in the same order (vendored :927-928 /
// upstream :1120-1122).  So nothing here can mis-address a nibble or mis-decode a table entry the
// way IQ4_XS (vendored QR4_XS == 8, upstream == 2) would.  The constants, the struct and the table
// are STILL declared LOCALLY, in namespace ftmma, because the rule for the whole port is that
// mma/ includes nothing from the vendored headers -- and because QK4_NL/QR4_NL/QI4_NL are MACROS
// over there, so they cannot be spelled as C++ identifiers here at all.
//
// DIVERGENCES FROM UPSTREAM:
//  1. The dp4a arm of both functions is dropped (mmq-load-tiles.cuh:1497-1500, :1526-1529,
//     :1548-1549 and the whole mmq-vec-dot.cuh dp4a path). This port is mma-only; on sm_89 the
//     dp4a arm is dead. The AMD (MFMA/WMMA) arms are dropped too -- mma_int.cuh instantiates no
//     J_MAJOR tile, so mmq-vec-dot.cuh:145-201 has nothing to compile against.
//  2. The HIP (:35-56) and MUSA (:81-94) arms of get_int_from_table_16 are dropped; only the CUDA
//     __byte_perm arm (:57-80) is kept. Its arithmetic is byte-for-byte upstream's.
//  3. The functions are members of mmq_type_traits<GGML_TYPE_IQ4_NL> rather than free templates
//     selected by the ggml_cuda_mmq_get_util_funcs switch (upstream mmq.cuh:535-843). See
//     mma/mmq_core.cuh divergence 3: this is what lets the S4 agents add a type without touching
//     a shared file.
//  4. `constexpr int warp_size = ggml_cuda_get_physical_warp_size()` becomes FTMMA_WARP_SIZE
//     (mmq_core.cuh divergence 8: the shim's version is a host `static inline int`).
//  5. The three helper blocks below that are NOT IQ4_NL-specific are wrapped in one-shot include
//     guards (FTMMA_HAVE_*). They are shared with other S4 types -- see the NOTE FOR S4 -- and the
//     guards make this header safe to co-include with a sibling that adopts the same guard names.
//     Upstream has no such guards because upstream has one big translation unit per type.
//
// NOTE FOR S4 (shared code -- do NOT copy it, include this header instead, or the copies drift):
//   * ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma is NOT IQ4_NL-specific. Upstream uses it for Q8_0,
//     Q4_0/Q4_1/Q5_0/Q5_1, IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S, IQ4_XS, IQ4_NL and MXFP4
//     (mmq.cuh:786-830). The plan puts it in mma/vec_dot_q8_0.cuh owned by D; until that file
//     exists it lives here, under FTMMA_HAVE_VEC_DOT_Q8_0_Q8_1_MMA.
//   * get_int_from_table_16 is shared with IQ4_XS and MXFP4 (they differ only in the table),
//     under FTMMA_HAVE_GET_INT_FROM_TABLE_16.
//   * kvalues_iq4nl is shared with IQ4_XS, under FTMMA_HAVE_KVALUES_IQ4NL.
//   * get_int_b2 is shared with Q4_0/Q5_0/Q8_0/Q3_K, under FTMMA_HAVE_GET_INT_B2.

#pragma once

#include <cuda_fp16.h>

#include "mmq_core.cuh"

namespace ftmma {

using namespace ggml_cuda_mma;

// ---------------------------------------------------------------------------------------------
// Local IQ4_NL constants and block layout. See the "SAFE TYPE" note above: these AGREE with the
// vendored ggml-common.h:177-183, they are re-declared only because that header's spellings are
// macros and because mma/ must not include it.
// ---------------------------------------------------------------------------------------------

static constexpr int FT_QR4_NL = 2;
static constexpr int FT_QI4_NL = FTMMA_QK4_NL / (4 * FT_QR4_NL);  // 4

static_assert(FT_QI4_NL == 4, "QI4_NL must be 4");

// upstream ggml-common.h:448-452 / vendored ggml-common.h:180-183 (identical layout).
struct block_iq4_nl {
    half    d;                        // delta
    uint8_t qs[FTMMA_QK4_NL / 2];     // nibbles / quants
};
static_assert(sizeof(block_iq4_nl) == sizeof(half) + FTMMA_QK4_NL/2,
              "wrong iq4_nl block size/padding");
static_assert(sizeof(block_iq4_nl) == 18, "block_iq4_nl must be 18 B");

// upstream vecdotq.cuh:18-25
#ifndef FTMMA_HAVE_GET_INT_B2
#define FTMMA_HAVE_GET_INT_B2
static __device__ __forceinline__ int get_int_b2(const void * x, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) x; // assume at least 2 byte alignment

    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;

    return x32;
}
#endif  // FTMMA_HAVE_GET_INT_B2

// upstream ggml-common.h:1120-1122 (GGML_TABLE_BEGIN/END expands to exactly this on CUDA).
// Values verified identical to the vendored ggml-common.h:927-928.
#ifndef FTMMA_HAVE_KVALUES_IQ4NL
#define FTMMA_HAVE_KVALUES_IQ4NL
static const __device__ int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
#endif  // FTMMA_HAVE_KVALUES_IQ4NL

// upstream vecdotq.cuh:31-95, the CUDA `#elif !defined(GGML_USE_MUSA)` arm at :57-80 only
// (divergence 2).
#ifndef FTMMA_HAVE_GET_INT_FROM_TABLE_16
#define FTMMA_HAVE_GET_INT_FROM_TABLE_16
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
#endif  // FTMMA_HAVE_GET_INT_FROM_TABLE_16

// ---------------------------------------------------------------------------------------------
// upstream mmq-vec-dot.cuh:142-280, the `#else` (NVIDIA) arm at :202-279, verbatim except for the
// FT_ spelling of QI8_0/QI8_1 and the mmq_core accessor templates.
// IQ4_NL instantiates it with ds_layout == MMQ_Q8_1_DS_LAYOUT_D4 (upstream mmq.cuh:820-826), which
// is what mma/quantize_mmq.cuh:133-134 produces for GGML_TYPE_IQ4_NL -- the D4 branch reads only
// y_df (the float scale), never the DS4 half2 (scale, sum) pair, because IQ4_NL has no min term.
// Used by IQ4_NL (and, in the rest of S4, by Q8_0 / IQ4_XS / the IQ2-IQ3 family / MXFP4).
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
#endif  // FTMMA_HAVE_VEC_DOT_Q8_0_Q8_1_MMA

// ---------------------------------------------------------------------------------------------
// upstream mmq-load-tiles.cuh:1487-1552, the mma arm only (:1494-1496 for the pointers,
// :1523-1525 for the quants, :1546-1547 for the scales).
//
// Shape check for this port's single config row (mmq_core.cuh: nthreads 256, I 128, warp_size 32,
// MMQ_ITER_K 256, sram_layout Q8_0 -> sram_stride 76):
//   threads_per_row       = MMQ_ITER_K/(4*QR4_NL) = 32, nrows = 1
//   kbx = txi/QI4_NL in [0,8), kqsx = txi%QI4_NL in [0,4)
//   k0  = kbx*(2*QI4_NL) + kqsx in [0,60), + QI4_NL -> [0,64) = 2*MMQ_TILE_NE_K ints per row
//   blocks_per_tile_x_row = MMQ_TILE_NE_K/QI4_NL = 8 scales per tile row, one per 8-int block,
//   which is exactly the k0/QI8_0 the vec-dot above indexes (QI8_0 == 8 ints == 32 quants ==
//   one IQ4_NL block). rows_per_warp = 32/8 = 4.
// ---------------------------------------------------------------------------------------------

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq4_nl(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = FTMMA_WARP_SIZE;                                 // divergence 4
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads<type, J, fallback>() / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I<type, J, fallback>();
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride<type, J, fallback>();

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);

    constexpr int threads_per_row = MMQ_ITER_K / (4 * FT_QR4_NL);   // 32
    constexpr int nrows = warp_size / threads_per_row;              // 1
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int kbx  = txi / FT_QI4_NL;
    const int kqsx = txi % FT_QI4_NL;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq4_nl * bxi = (const block_iq4_nl *) x + kbx0 + i*stride + kbx;

        const int aux_q4 = get_int_b2(bxi->qs, kqsx);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        const int k0 = kbx * (2 * FT_QI4_NL) + kqsx;

        x_qs[i*sram_stride + k0 + 0]         = v.x;
        x_qs[i*sram_stride + k0 + FT_QI4_NL] = v.y;
    }

    constexpr int blocks_per_tile_x_row = MMQ_TILE_NE_K / FT_QI4_NL;  // 8
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;  // 4
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq4_nl * bxi = (const block_iq4_nl *) x + kbx0 + i*stride + kbxd;

        x_df[i*sram_stride + kbxd] = __half2float(bxi->d);
    }

    FTMMA_UNUSED(i_max);
}

// ---------------------------------------------------------------------------------------------
// The per-type hook mma/mmq_core.cuh declares. upstream mmq.cuh:820-826 selects exactly this pair
// for IQ4_NL (load_tiles_iq4_nl + vec_dot_q8_0_q8_1_mma<D4> + write_back_mma), and
// mmq-config-ampere.cuh:329-344 gives it GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0.
// ---------------------------------------------------------------------------------------------

template <> struct mmq_type_traits<GGML_TYPE_IQ4_NL> {
    static constexpr ggml_cuda_mmq_sram_layout sram_layout = GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0;

    template <int J, bool fallback>
    static __device__ __forceinline__ void load_tiles(
            const char * __restrict__ x, int * __restrict__ x_tile,
            const int kbx0, const int i_max, const int stride) {
        ggml_cuda_mmq_load_tiles_iq4_nl<GGML_TYPE_IQ4_NL, J, fallback>(x, x_tile, kbx0, i_max, stride);
    }

    template <int J, bool fallback>
    static __device__ __forceinline__ void vec_dot(
            const int * __restrict__ x, const int * __restrict__ y,
            float * __restrict__ sum, const int k00) {
        ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<GGML_TYPE_IQ4_NL, J, fallback, MMQ_Q8_1_DS_LAYOUT_D4>(
            x, y, sum, k00);
    }
};

}  // namespace ftmma
