// mma/mmq_iq3_xxs.cuh -- IQ3_XXS for the mma MMQ kernel. Plan step S4, one type end to end,
// written to the shape mma/mmq_q4_K.cuh established in S3 (read that file first).
//
// PORTED FROM llama.cpp @ f7aadef09:
//   ggml-common.h:406-411        block_iq3_xxs (layout only; see "THE TRAP" below)
//   ggml-common.h:151-152        QI3_XXS / QR3_XXS   <-- THE TRAP, see below
//   ggml-common.h:1017-1050      iq3xxs_grid[256]
//   vecdotq.cuh:18-25            get_int_b2
//   vecdotq.cuh:97-104           unpack_ksigns
//   mmq-load-tiles.cuh:1287-1348 ggml_cuda_mmq_load_tiles_iq3_xxs, the mma arm
//                                (:1295-1296 pointers, :1303-1345 the body)
//   mmq-vec-dot.cuh:142-280      ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma, the `#else` (NVIDIA) arm
//                                at :202-278
//   mmq.cuh:803-808              the (load_tiles, vec_dot, write_back) triple for IQ3_XXS
//   mmq.cuh:88-91                MMQ_Q8_1_DS_LAYOUT_D4 for IQ3_XXS
//   mmq-config-ampere.cuh:278-293  GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0 for IQ3_XXS
//
// THE TRAP (the safety rule in the S4 brief, and why every constant below is local):
// the VENDORED ggml-common.h:135-136 says
//     #define QR3_XXS 8            #define QI3_XXS (QK_K / (4 * QR3_XXS))   // == 8
// while UPSTREAM ggml-common.h:151-152 says
//     #define QR3_XXS 4            #define QI3_XXS (QK_K / (4*QR3_XXS))     // == 16
// The vendored value is the one the vendored dequantize/mmvq path is built around; it is NOT the
// value this loader wants. QR3_XXS appears twice in the loader and both uses would break silently
// with 8: `threads_per_row = (MMQ_ITER_K / (4*QR3_XXS)) / 2` would be 4 instead of 8, so half the
// 32 quant ints of each block would never be written to SRAM, and the `for (l < QR3_XXS)` grid
// loop would run 8 times instead of 4 and write 16 ints per thread instead of 8, past the end of
// its 8-int slot. Neither is a compile error and neither traps. So FT_QR3_XXS below is 4 (the
// UPSTREAM value, because the code around it is upstream's), this file includes nothing from the
// vendored headers, and the vendored headers are unmodified. Same reasoning as mmq_q4_K.cuh's
// "SAFE TYPE" note, except that for Q4_K the two headers agreed and here they do not.
// block_iq3_xxs itself IS byte-identical between the two (half d; uint8_t qs[3*QK_K/8]; 98 B) and
// iq3xxs_grid is element-for-element identical (verified by diffing the two tables), so only the
// QR/QI spelling is dangerous -- but the struct and the table are still declared locally, because
// the rule for all of mma/ is that it includes nothing from the vendored headers.
//
// HOW IQ3_XXS DECODES (upstream mmq-load-tiles.cuh:1315-1345, for the reader of the loop below):
// one 256-element block is 32 bytes of scale/sign aux + 64 bytes of grid indices. Each of the 8
// threads of a row takes 8 grid indices (q3[0..7], read as two ints) and one aux32. aux32's low
// 28 bits are four 7-bit sign packs, one per pair of grid entries; its top 4 bits are the block
// scale ls. Each grid index selects a uint32 of four 4-bit-ish signed magnitudes; unpack_ksigns
// turns a 7-bit pack into a byte-broadcast selector, __vcmpne4 turns the selected bits into a
// per-byte mask, and `__vsub4(g ^ mask, mask)` negates the masked bytes. 4 iterations x 2 ints =
// 8 ints = 32 quants per thread, 8 threads = the 256 quants of the block. That is the "5 matmuls
// plus packed signs" shape: the whole cost is in this table lookup, and the vec-dot afterwards is
// the plain q8_0 one.
//
// DIVERGENCES FROM UPSTREAM:
//  1. The dp4a arm of both functions is dropped (mmq-load-tiles.cuh:1298-1300, :1338-1339,
//     :1343-1344 and mmq-vec-dot.cuh's non-mma paths), as is the AMD MFMA/WMMA arm of the vec-dot
//     (mmq-vec-dot.cuh:145-201): this port is mma-only and mma_int.cuh instantiates no J_MAJOR
//     tile. Same divergence 1 as mmq_q4_K.cuh.
//  2. `const float d = bxi->d;` becomes `__half2float(bxi->d)`. The torch extension is compiled
//     with -D__CUDA_NO_HALF_CONVERSIONS__, so the implicit half->float conversion upstream relies
//     on does not exist here. The arithmetic is unchanged. (mmq_q4_K.cuh's divergence 2 is the
//     same problem in its half2 spelling.)
//  3. The functions are members of mmq_type_traits<GGML_TYPE_IQ3_XXS> rather than free templates
//     selected by ggml_cuda_mmq_get_util_funcs (upstream mmq.cuh:535-843). mmq_core.cuh
//     divergence 3.
//  4. QK_K, QR3_XXS, QI8_0 are spelled FTMMA_QK_K / FT_QR3_XXS / FT_QI8_0: they are MACROS in the
//     vendored header and cannot be C++ identifiers here (mmq_core.cuh divergence 1).
//
// SHARED WITH OTHER S4 TYPES -- READ BEFORE ADDING ANOTHER TYPE.
// Three things below are NOT IQ3_XXS-specific and upstream shares them across types:
//   get_int_b2                          (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_S, Q4_1/Q5_1 ... )
//   unpack_ksigns                       (IQ2_XXS, IQ2_XS, IQ3_XXS)
//   ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma (Q4_0, Q5_0, Q8_0, IQ2_XXS, IQ3_XXS, IQ3_S, IQ4_XS,
//                                        IQ4_NL -- mmq.cuh:715-843)
// Because every mma/mmq_<type>.cuh ends up in ONE translation unit (mmq_entry.cuh includes them
// all), two files defining the same symbol in namespace ftmma is a redefinition error. Each of
// the three is therefore wrapped in an FTMMA_HAVE_* include guard: whichever type header is
// included first defines it, the rest skip it, and the definitions are identical because they are
// all upstream's. A type that needs one of these must use the SAME guard macro and the SAME text,
// or move all three into a shared mma/vec_dot_q8_0.cuh and have every user include that -- what
// mmq_q4_K.cuh's "NOTE FOR S4" asks for on the q8_1 side. Do not copy them under a new name.
//
// NOTE ON THE ACTIVATION LAYOUT: upstream mmq.cuh:89-90 gives IQ3_XXS MMQ_Q8_1_DS_LAYOUT_D4 (a
// float d per 32-value chunk, no min term), and mma/quantize_mmq.cuh:127-129 already produces D4
// for GGML_TYPE_IQ3_XXS. So no quantizer change is needed; the ds_layout template parameter is
// kept in upstream's spelling and instantiated as MMQ_Q8_1_DS_LAYOUT_D4.

#pragma once

#include <cuda_fp16.h>

#include "mmq_core.cuh"

namespace ftmma {

using namespace ggml_cuda_mma;

// ---------------------------------------------------------------------------------------------
// Local IQ3_XXS constants and block layout. See "THE TRAP" above: FT_QR3_XXS is UPSTREAM's 4, not
// the vendored header's 8, and nothing here is included from the vendored headers.
// ---------------------------------------------------------------------------------------------

// upstream ggml-common.h:151-152. Vendored ggml-common.h:135-136 disagrees (QR3_XXS 8, QI3_XXS 8).
static constexpr int FT_QR3_XXS = 4;
static constexpr int FT_QI3_XXS = FTMMA_QK_K / (4 * FT_QR3_XXS);  // 16

// upstream ggml-common.h:406-410 / vendored ggml-common.h:137-141 (identical layout).
struct block_iq3_xxs {
    half    d;
    uint8_t qs[3 * (FTMMA_QK_K / 8)];   // 64 B of grid indices then 32 B of sign/scale aux
};
static_assert(sizeof(block_iq3_xxs) == sizeof(half) + 3*(FTMMA_QK_K/8),
              "wrong iq3_xxs block size/padding");
static_assert(sizeof(block_iq3_xxs) == 98, "block_iq3_xxs must be 98 B");

// upstream ggml-common.h:1017-1050, GGML_TABLE_BEGIN(uint32_t, iq3xxs_grid, 256).
// Verified element-for-element identical to the vendored ggml-common.h:563 copy; duplicated here
// only because mma/ includes nothing from that header. 1 KiB of __constant__-cached data.
static const __device__ uint32_t iq3xxs_grid[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

// upstream vecdotq.cuh:18-25 -- see "SHARED WITH OTHER S4 TYPES" above for the guard.
#ifndef FTMMA_HAVE_GET_INT_B2
#define FTMMA_HAVE_GET_INT_B2
static __device__ __forceinline__ int get_int_b2(const void * x, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) x; // assume at least 2 byte alignment

    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;

    return x32;
}
#endif // FTMMA_HAVE_GET_INT_B2

// upstream vecdotq.cuh:97-104 -- see "SHARED WITH OTHER S4 TYPES" above for the guard.
#ifndef FTMMA_HAVE_UNPACK_KSIGNS
#define FTMMA_HAVE_UNPACK_KSIGNS
static __device__ __forceinline__ uint32_t unpack_ksigns(const uint8_t v) {
    // v is a 7 bit int, with the 8th sign being encodable as popcnt
    // with xor we can "correct" the bit instead of having to mask
    const uint32_t p = __popc(v) & 1;
    const uint32_t s = v ^ p << 7;
    // broadcast over uint to allow for 0x08040201 / 0x80402010 as selectors
    return s * 0x01010101;
}
#endif // FTMMA_HAVE_UNPACK_KSIGNS

// ---------------------------------------------------------------------------------------------
// upstream mmq-vec-dot.cuh:142-280, the `#else` (NVIDIA, TURING_MMA) arm at :202-278, verbatim
// except for the FT_ spelling of QI8_0/QI8_1 and the mmq_core accessor templates.
// Used by IQ3_XXS with ds_layout == MMQ_Q8_1_DS_LAYOUT_D4 (upstream mmq.cuh:807); the ds_layout
// parameter is kept in upstream's spelling because the same function serves the DS4 types too.
// See "SHARED WITH OTHER S4 TYPES" above for the guard.
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
// upstream mmq-load-tiles.cuh:1287-1348, the mma arm only (:1295-1296 for the pointers,
// :1303-1345 for the body). FT_QR3_XXS is 4 here -- see "THE TRAP" in the header comment.
// ---------------------------------------------------------------------------------------------

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq3_xxs(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = FTMMA_WARP_SIZE;                                 // mmq_core divergence 8
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads<type, J, fallback>() / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I<type, J, fallback>();
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride<type, J, fallback>();

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);

    constexpr int threads_per_row = (MMQ_ITER_K / (4 * FT_QR3_XXS)) / 2;   // 8
    constexpr int nrows = warp_size / threads_per_row;                     // 4
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq3_xxs * bxi = (const block_iq3_xxs *) x + kbx0 + i*stride;

        const int2 q3_packed = make_int2(get_int_b2(bxi->qs, 2*kqsx+0), get_int_b2(bxi->qs, 2*kqsx+1));
        const uint8_t * q3 = (const uint8_t *) &q3_packed;
        const uint32_t aux32 = get_int_b2(bxi->qs, FTMMA_QK_K/16 + kqsx);

#pragma unroll
        for (int l = 0; l < FT_QR3_XXS; ++l) {
            const int2 grid_pos = make_int2(iq3xxs_grid[q3[2*l+0]], iq3xxs_grid[q3[2*l+1]]);
            const uint32_t signs = unpack_ksigns(aux32 >> (7*l));

            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);

            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

            x_qs[i*sram_stride + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*sram_stride + 8*kqsx + (2*l + 1)] = grid_h;
        }

        const int ls = aux32 >> 28;
        // divergence 2: upstream writes `const float d = bxi->d;`. Same value, explicit intrinsic.
        const float d = __half2float(bxi->d);
        x_df[i*sram_stride + kqsx] = (ls*d + d/2)/2;
    }

    FTMMA_UNUSED(i_max);
}

// ---------------------------------------------------------------------------------------------
// The per-type hook mma/mmq_core.cuh declares. upstream mmq.cuh:803-808 selects exactly this pair
// for IQ3_XXS (load_tiles_iq3_xxs + vec_dot_q8_0_q8_1_mma<..., D4> + write_back_mma), and
// mmq-config-ampere.cuh:278-293 gives it GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0.
// ---------------------------------------------------------------------------------------------

template <> struct mmq_type_traits<GGML_TYPE_IQ3_XXS> {
    static constexpr ggml_cuda_mmq_sram_layout sram_layout = GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0;

    template <int J, bool fallback>
    static __device__ __forceinline__ void load_tiles(
            const char * __restrict__ x, int * __restrict__ x_tile,
            const int kbx0, const int i_max, const int stride) {
        ggml_cuda_mmq_load_tiles_iq3_xxs<GGML_TYPE_IQ3_XXS, J, fallback>(x, x_tile, kbx0, i_max, stride);
    }

    template <int J, bool fallback>
    static __device__ __forceinline__ void vec_dot(
            const int * __restrict__ x, const int * __restrict__ y,
            float * __restrict__ sum, const int k00) {
        ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<GGML_TYPE_IQ3_XXS, J, fallback, MMQ_Q8_1_DS_LAYOUT_D4>(
            x, y, sum, k00);
    }
};

}  // namespace ftmma
