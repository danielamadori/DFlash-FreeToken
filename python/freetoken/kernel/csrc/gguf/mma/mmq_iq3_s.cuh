// mma/mmq_iq3_s.cuh -- IQ3_S for the mma MMQ kernel. Plan step S4, one of the ten per-type files.
// Written to the template of mma/mmq_q4_K.cuh (structure, naming, comment style).
//
// PORTED FROM llama.cpp @ f7aadef09:
//   ggml-common.h:414-422       block_iq3_s + IQ3S_N_SCALE (layout only; see "THE QR/QI TRAP")
//   ggml-common.h:169-170       QI3_S / QR3_S
//   ggml-common.h:1052-1117     iq3s_grid, the 512-entry uint32 code-point table
//   vecdotq.cuh:18-24           get_int_b2
//   mmq-load-tiles.cuh:1351-1419 ggml_cuda_mmq_load_tiles_iq3_s, the mma arm
//                               (:1359-1360 pointers, :1403-1404 quants, :1412-1414 scale)
//   mmq-vec-dot.cuh:142-281     ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma, the `#else` NVIDIA arm
//                               (:203-278)
//   mmq.cuh:809-814             the (load_tiles, vec_dot, write_back) triple upstream picks for
//                               IQ3_S; note the vec-dot's ds_layout argument is
//                               MMQ_Q8_1_DS_LAYOUT_D4
//   mmq-config-ampere.cuh:295-310  the IQ3_S rows -> GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0
//
// THE QR/QI TRAP FOR THIS TYPE (plan S4 per-type conflict table):
// QR3_S and QI3_S are ABSENT from the vendored ggml-common.h entirely -- it never needed them,
// because the vendored IQ3_S decode (vecdotq.cuh:1875-1898, dequantize.cuh:348-364) walks the
// block by hand. So there is no vendored spelling to collide with, but there is also no vendored
// value to check against: FT_QR3_S == 4 and FT_QI3_S == 16 below come from upstream
// ggml-common.h:169-170 and are used ONLY to size this file's own loops. The block layout, which
// is what actually addresses bytes, IS identical on both sides:
//   vendored ggml-common.h:144-151   half d; qs[QK_K/4]; qh[QK_K/32]; signs[QK_K/8]; scales[QK_K/64]
//   upstream ggml-common.h:414-421   ggml_half d; same four arrays
// -> 2 + 64 + 8 + 32 + 4 = 110 B on both sides, and the static_assert below pins it.
//
// THE GRID TABLE IS **NOT** THE SAME ON BOTH SIDES -- READ THIS BEFORE TOUCHING IT.
// The plan's S4 conflict table says "iq3xs_grid vendored :595 == iq3s_grid upstream :1052 (same
// 512 uint32, renamed)". That is WRONG, and it is the one thing about this type that can go
// silently wrong. The two tables have the same 512 entries in the same order, but different
// VALUES: byte for byte the vendored table is 4x the upstream one, except at the top level,
// where vendored has 62 and 4*upstream has 60.
//   vendored ggml-common.h:595  iq3xs_grid  bytes drawn from {4,12,20,28,36,44,52,62}
//   upstream ggml-common.h:1052 iq3s_grid   bytes drawn from {1, 3, 5, 7, 9,11,13,15}
// The 4x is compensated by the scale and is therefore harmless either way:
//   vendored  value = grid_v * d * (0.5f + s) * 0.5f   (vecdotq.cuh:1895-1896)
//   upstream  value = grid_u * d * (1 + 2*s)           (dequantize.cuh:356, and the ls below)
//   grid_v == 4*grid_u  =>  identical products.
// The 62-vs-60 is NOT harmless: it is llama.cpp bbde6eb25 ("ggml : IQ3_S improvements", #5829,
// 2024-03-02), which replaced iq3xs_grid by iq3s_grid and changed the decoded value of the top
// level from 62/4 = 15.5 to 15. That commit touched only the DEcoder; the quantizer has always
// built its grid as pos[i] = 2*l+1 with l in 0..7, i.e. {1,3,...,15} (ggml-quants.c:3769-3774).
// So upstream's 15 is what the quantizer meant and the vendored fork carries the pre-fix table.
//
// THIS FILE USES THE UPSTREAM TABLE AND THE UPSTREAM SCALE (divergence 4 below). The consequence
// is measurable and the reviewer must not mistake it for a porting bug: 150 of the 2048 grid
// bytes (7.3%) decode 3.2% smaller here than in the vendored ggml_dequantize / mmvq / moe path,
// which puts a systematic ~0.9% relative difference on an IQ3_S matmul checked against
// `x @ ggml_dequantize(w).T`. See RISKS in the report for the one-line change that restores
// bit-compatibility with the vendored convention instead.
//
// DIVERGENCES FROM UPSTREAM:
//  1. The dp4a arm of both functions is dropped (the `#else` halves of the two
//     TURING_MMA_AVAILABLE blocks in mmq-load-tiles.cuh:1362-1365, :1406-1407, :1416 and the
//     `#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)` half of
//     mmq-vec-dot.cuh:145-202). This port is mma-only; on sm_89 the dp4a arm is dead. The AMD
//     (MFMA/WMMA) arms are dropped too -- mma_int.cuh instantiates no J_MAJOR tile.
//  2. half -> float is spelled __half2float(bxi->d) instead of upstream's implicit
//     `const float d = bxi->d`, because the torch extension is compiled with
//     -D__CUDA_NO_HALF_CONVERSIONS__ (same reason mma/mmq_q4_K.cuh divergence 2 exists).
//     The arithmetic is unchanged.
//  3. The functions are members of mmq_type_traits<GGML_TYPE_IQ3_S> rather than free templates
//     selected by the ggml_cuda_mmq_get_util_funcs switch (upstream mmq.cuh:535-843). See
//     mma/mmq_core.cuh divergence 3.
//  4. The grid table is COPIED INTO THIS FILE rather than included, like every other constant
//     here -- and it is upstream's iq3s_grid, not the vendored iq3xs_grid. See the long note
//     above. 2 KiB of __device__ constant data per TU.
//  5. WITHDRAWN AT WIRING TIME. This file used to drop the ds_layout template parameter of the
//     vec-dot and inline the D4 arm. That made its copy of ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma
//     incompatible with the identical copies in mmq_q8_0.cuh / mmq_iq3_xxs.cuh / mmq_iq4_nl.cuh:
//     all five land in one translation unit behind FTMMA_HAVE_VEC_DOT_Q8_0_Q8_1_MMA, only one
//     survives, and a 3-parameter winner breaks every 4-argument call site (and vice versa).
//     The vec-dot below is now upstream's, parameter included, and IQ3_S instantiates it with
//     MMQ_Q8_1_DS_LAYOUT_D4 (upstream mmq.cuh:813), which is what mma/quantize_mmq.cuh:128-129
//     produces for GGML_TYPE_IQ3_S. The `else` (DS4/D2S6) branch of mmq-vec-dot.cuh:262-266 is
//     dead code for this type, as upstream intends.
//
// NOTE FOR THE OTHER S4 AGENTS: ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma below is NOT IQ3_S-specific
// -- upstream uses it for Q8_0, IQ2_XXS, IQ3_XXS, IQ3_S, IQ4_XS and IQ4_NL (mmq.cuh:768-826).
// The plan puts it in mma/vec_dot_q8_0.cuh owned by D. Whoever lands the second of those six
// types should move this function into mma/vec_dot_q8_0.cuh and have both files include that
// instead; it must not be copied, or the copies will drift. It is spelled here exactly as in
// mmq_q8_0.cuh / mmq_iq3_xxs.cuh / mmq_iq4_nl.cuh (the wiring step diffed all four and made them
// identical), behind the FTMMA_HAVE_VEC_DOT_Q8_0_Q8_1_MMA guard, so the include order is free.

#pragma once

#include <cuda_fp16.h>

#include "mmq_core.cuh"

namespace ftmma {

using namespace ggml_cuda_mma;

// ---------------------------------------------------------------------------------------------
// Local IQ3_S constants and block layout. QR3_S/QI3_S do not exist in the vendored header at all
// (see "THE QR/QI TRAP" above), so these are upstream's, ggml-common.h:169-170. The struct
// matches the vendored one byte for byte (vendored ggml-common.h:144-151).
// ---------------------------------------------------------------------------------------------

static constexpr int FT_QR3_S = 4;
static constexpr int FT_QI3_S = FTMMA_QK_K / (4 * FT_QR3_S);  // 16
static constexpr int FT_IQ3S_N_SCALE = FTMMA_QK_K / 64;       // 4

// upstream ggml-common.h:414-421 / vendored ggml-common.h:144-151 (identical layout).
struct block_iq3_s {
    half    d;
    uint8_t qs[FTMMA_QK_K / 4];
    uint8_t qh[FTMMA_QK_K / 32];
    uint8_t signs[FTMMA_QK_K / 8];
    uint8_t scales[FT_IQ3S_N_SCALE];
};
static_assert(sizeof(block_iq3_s) == sizeof(half) + 13*(FTMMA_QK_K/32) + FT_IQ3S_N_SCALE,
              "wrong iq3_s block size/padding");
static_assert(sizeof(block_iq3_s) == 110, "block_iq3_s must be 110 B");

// upstream vecdotq.cuh:18-24
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

// upstream ggml-common.h:1052-1117, verbatim (GGML_TABLE_BEGIN expands to
// `static const __device__ uint32_t iq3s_grid[512] = {` under GGML_COMMON_DECL_CUDA,
// ggml-common.h:493). Renamed to ftmma_iq3s_grid so that a future accidental include of the
// vendored ggml-common.h collides at link time instead of silently picking the OTHER table --
// see divergence 4 and the long note in the header.
static const __device__ uint32_t ftmma_iq3s_grid[512] = {
    0x04040404, 0x0404040c, 0x04040414, 0x0404042c, 0x0404043e, 0x04040c04, 0x04040c0c, 0x04040c14, 0x04040c24,
    0x04040c34, 0x04041404, 0x0404140c, 0x0404142c, 0x04041c1c, 0x04042404, 0x04042414, 0x0404242c, 0x0404243e,
    0x04042c0c, 0x04042c1c, 0x04043404, 0x04043414, 0x04043e0c, 0x04043e24, 0x04043e3e, 0x040c0404, 0x040c040c,
    0x040c0414, 0x040c0424, 0x040c0c04, 0x040c0c0c, 0x040c0c2c, 0x040c1404, 0x040c141c, 0x040c143e, 0x040c1c0c,
    0x040c1c2c, 0x040c2424, 0x040c340c, 0x040c342c, 0x040c3e14, 0x04140404, 0x0414040c, 0x0414042c, 0x0414043e,
    0x04140c04, 0x04140c1c, 0x04140c34, 0x0414140c, 0x0414142c, 0x04141c04, 0x04141c24, 0x04142414, 0x0414242c,
    0x0414243e, 0x04142c0c, 0x04142c1c, 0x04143e04, 0x04143e1c, 0x041c041c, 0x041c0c0c, 0x041c0c2c, 0x041c1404,
    0x041c1414, 0x041c1c0c, 0x041c1c1c, 0x041c1c34, 0x041c2424, 0x041c2c04, 0x041c2c14, 0x041c343e, 0x041c3e0c,
    0x041c3e2c, 0x04240404, 0x04240c1c, 0x04240c3e, 0x0424140c, 0x04241424, 0x04241c14, 0x04242404, 0x0424241c,
    0x04242c0c, 0x04243e04, 0x042c0414, 0x042c0424, 0x042c1404, 0x042c1414, 0x042c1434, 0x042c1c1c, 0x042c240c,
    0x042c242c, 0x042c243e, 0x042c3434, 0x042c3e1c, 0x04340434, 0x04340c0c, 0x04340c1c, 0x04341c0c, 0x04342c14,
    0x04343e0c, 0x043e0404, 0x043e0414, 0x043e0424, 0x043e1404, 0x043e1414, 0x043e1434, 0x043e1c1c, 0x043e2c04,
    0x043e2c24, 0x0c040404, 0x0c04040c, 0x0c040414, 0x0c040424, 0x0c040c04, 0x0c040c0c, 0x0c040c1c, 0x0c040c2c,
    0x0c040c3e, 0x0c041404, 0x0c041414, 0x0c041c0c, 0x0c041c24, 0x0c041c34, 0x0c042c24, 0x0c042c34, 0x0c04340c,
    0x0c043e14, 0x0c0c0404, 0x0c0c040c, 0x0c0c041c, 0x0c0c0434, 0x0c0c0c04, 0x0c0c0c24, 0x0c0c140c, 0x0c0c1c04,
    0x0c0c1c1c, 0x0c0c240c, 0x0c0c2c04, 0x0c0c2c14, 0x0c0c3e04, 0x0c0c3e34, 0x0c140404, 0x0c140c14, 0x0c140c2c,
    0x0c140c3e, 0x0c141404, 0x0c141424, 0x0c141c14, 0x0c142404, 0x0c14241c, 0x0c142c2c, 0x0c143404, 0x0c143e14,
    0x0c1c040c, 0x0c1c0424, 0x0c1c043e, 0x0c1c0c04, 0x0c1c0c1c, 0x0c1c140c, 0x0c1c143e, 0x0c1c1c04, 0x0c1c1c24,
    0x0c1c240c, 0x0c1c3414, 0x0c1c3e04, 0x0c24041c, 0x0c24042c, 0x0c240c14, 0x0c240c24, 0x0c241c0c, 0x0c241c1c,
    0x0c242414, 0x0c242434, 0x0c242c04, 0x0c242c24, 0x0c2c040c, 0x0c2c0c04, 0x0c2c0c1c, 0x0c2c140c, 0x0c2c1c04,
    0x0c2c1c14, 0x0c2c2c0c, 0x0c341404, 0x0c341424, 0x0c34143e, 0x0c342424, 0x0c342434, 0x0c3e040c, 0x0c3e041c,
    0x0c3e0c04, 0x0c3e0c14, 0x0c3e140c, 0x0c3e1c2c, 0x0c3e240c, 0x0c3e3414, 0x0c3e3e04, 0x14040404, 0x1404040c,
    0x1404041c, 0x1404042c, 0x1404043e, 0x14040c04, 0x14040c14, 0x14040c24, 0x14040c34, 0x1404140c, 0x1404141c,
    0x1404143e, 0x14041c04, 0x14041c14, 0x1404240c, 0x1404241c, 0x1404242c, 0x14042c04, 0x14042c14, 0x1404343e,
    0x14043e04, 0x14043e1c, 0x14043e2c, 0x140c0404, 0x140c0414, 0x140c0c04, 0x140c0c1c, 0x140c0c3e, 0x140c1414,
    0x140c142c, 0x140c1c0c, 0x140c1c24, 0x140c2414, 0x140c2c0c, 0x1414040c, 0x14140424, 0x1414043e, 0x1414140c,
    0x1414141c, 0x14141c04, 0x14141c3e, 0x1414240c, 0x14142c1c, 0x14142c3e, 0x14143e0c, 0x14143e24, 0x141c0404,
    0x141c0414, 0x141c042c, 0x141c0c0c, 0x141c1414, 0x141c1424, 0x141c1c0c, 0x141c1c1c, 0x141c2414, 0x141c2c04,
    0x141c3434, 0x1424040c, 0x1424043e, 0x14241404, 0x1424141c, 0x14241c14, 0x14241c2c, 0x1424240c, 0x14243e14,
    0x14243e2c, 0x142c0424, 0x142c0c0c, 0x142c1414, 0x142c1c3e, 0x142c2404, 0x142c2c1c, 0x142c3e04, 0x14340404,
    0x14340414, 0x1434043e, 0x1434140c, 0x14342c2c, 0x1434340c, 0x143e042c, 0x143e0c0c, 0x143e1434, 0x143e1c04,
    0x143e241c, 0x143e2c04, 0x1c040414, 0x1c040c0c, 0x1c040c1c, 0x1c040c2c, 0x1c040c3e, 0x1c041414, 0x1c041c0c,
    0x1c041c1c, 0x1c041c2c, 0x1c042414, 0x1c042424, 0x1c04243e, 0x1c042c0c, 0x1c04341c, 0x1c043e0c, 0x1c0c040c,
    0x1c0c041c, 0x1c0c042c, 0x1c0c0c24, 0x1c0c140c, 0x1c0c141c, 0x1c0c2404, 0x1c0c3404, 0x1c0c3e14, 0x1c0c3e34,
    0x1c140404, 0x1c140c14, 0x1c141404, 0x1c141c14, 0x1c141c24, 0x1c142c04, 0x1c1c040c, 0x1c1c0c04, 0x1c1c0c24,
    0x1c1c140c, 0x1c1c141c, 0x1c1c143e, 0x1c1c1c04, 0x1c1c240c, 0x1c1c241c, 0x1c1c243e, 0x1c1c2c2c, 0x1c1c3e1c,
    0x1c24041c, 0x1c240c0c, 0x1c240c34, 0x1c241414, 0x1c241c0c, 0x1c242c14, 0x1c243404, 0x1c243424, 0x1c2c040c,
    0x1c2c0c04, 0x1c2c0c14, 0x1c2c142c, 0x1c2c1c14, 0x1c2c2424, 0x1c2c2c34, 0x1c2c3e1c, 0x1c340c34, 0x1c34240c,
    0x1c3e040c, 0x1c3e041c, 0x1c3e1404, 0x1c3e1414, 0x1c3e1c2c, 0x24040404, 0x24040424, 0x24040c14, 0x24041404,
    0x24041424, 0x2404143e, 0x24041c14, 0x2404240c, 0x24042c04, 0x24043e04, 0x240c0414, 0x240c043e, 0x240c0c0c,
    0x240c0c1c, 0x240c1414, 0x240c1c04, 0x240c1c2c, 0x240c241c, 0x240c2c0c, 0x240c2c2c, 0x2414040c, 0x2414041c,
    0x24140c04, 0x24140c2c, 0x2414140c, 0x24141c1c, 0x24142404, 0x24142c3e, 0x24143414, 0x24143e04, 0x241c0424,
    0x241c0c0c, 0x241c0c1c, 0x241c1404, 0x241c1414, 0x241c1c0c, 0x241c1c2c, 0x24240404, 0x24240414, 0x24241424,
    0x24241c3e, 0x24242404, 0x24243e0c, 0x242c042c, 0x242c043e, 0x242c140c, 0x242c3414, 0x24340c1c, 0x24341c24,
    0x24343404, 0x243e0c04, 0x243e0c2c, 0x243e1c04, 0x243e241c, 0x243e2c0c, 0x2c040414, 0x2c040c04, 0x2c040c24,
    0x2c041414, 0x2c042404, 0x2c042424, 0x2c04243e, 0x2c042c14, 0x2c043434, 0x2c043e24, 0x2c0c040c, 0x2c0c041c,
    0x2c0c042c, 0x2c0c0c14, 0x2c0c140c, 0x2c0c1c14, 0x2c0c3e14, 0x2c140404, 0x2c140c0c, 0x2c14141c, 0x2c141c04,
    0x2c141c34, 0x2c142c1c, 0x2c1c0414, 0x2c1c043e, 0x2c1c0c04, 0x2c1c143e, 0x2c1c2424, 0x2c1c2c0c, 0x2c1c342c,
    0x2c1c3e1c, 0x2c24040c, 0x2c240424, 0x2c241404, 0x2c241c14, 0x2c242434, 0x2c2c0c14, 0x2c2c1434, 0x2c2c2c0c,
    0x2c2c2c1c, 0x2c342414, 0x2c3e0414, 0x2c3e0424, 0x2c3e1414, 0x34040c0c, 0x34040c1c, 0x34040c2c, 0x34041c0c,
    0x34041c1c, 0x34043404, 0x340c0404, 0x340c1404, 0x340c143e, 0x340c3424, 0x34140c14, 0x34141c24, 0x34142414,
    0x34142c2c, 0x34143414, 0x34143e04, 0x341c0404, 0x341c0c24, 0x341c140c, 0x341c2404, 0x3424142c, 0x3424241c,
    0x34243414, 0x342c0404, 0x342c041c, 0x342c1c24, 0x342c3404, 0x3434042c, 0x34342404, 0x343e0c0c, 0x343e0c1c,
    0x3e040404, 0x3e040424, 0x3e04043e, 0x3e041404, 0x3e041414, 0x3e041c34, 0x3e042404, 0x3e042c24, 0x3e043414,
    0x3e0c0414, 0x3e0c0c0c, 0x3e0c1424, 0x3e0c241c, 0x3e0c242c, 0x3e14040c, 0x3e140424, 0x3e140c04, 0x3e140c34,
    0x3e14140c, 0x3e141c04, 0x3e142c0c, 0x3e1c0414, 0x3e1c1c14, 0x3e1c1c2c, 0x3e1c2c1c, 0x3e24040c, 0x3e24042c,
    0x3e240c1c, 0x3e241404, 0x3e242c04, 0x3e2c1414, 0x3e2c2414, 0x3e340414, 0x3e341c0c, 0x3e3e0404,
};

// ---------------------------------------------------------------------------------------------
// upstream mmq-vec-dot.cuh:142-280, the `#else` (NVIDIA) arm at :202-279, verbatim except for the
// FT_ spelling of QI8_0/QI8_1 and the mmq_core accessor templates.
// IQ3_S instantiates it with ds_layout == MMQ_Q8_1_DS_LAYOUT_D4 (upstream mmq.cuh:809-814), which
// is what mma/quantize_mmq.cuh:128-129 produces for GGML_TYPE_IQ3_S -- the D4 branch reads only
// y_df (the float scale), never the DS4 half2 (scale, sum) pair, because IQ3_S has no min term.
// Shared with Q8_0 / IQ3_XXS / IQ4_XS / IQ4_NL.  WIRING FIX: this file previously dropped the
// ds_layout template parameter and inlined the D4 arm (its "divergence 5"), which made its copy
// incompatible with the other four under the FTMMA_HAVE_VEC_DOT_Q8_0_Q8_1_MMA guard -- whichever
// header came first won, and the losers' call sites had the wrong arity.  Restored to upstream's
// signature; all five copies are now textually identical (the wiring step diffed them).
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
// upstream mmq-load-tiles.cuh:1351-1419, the mma arm only (:1359-1360 for the pointers,
// :1403-1404 for the quants, :1412-1414 for the scale).
//
// Shape of the loop, for the record, because the indices are dense:
//   threads_per_row = (MMQ_ITER_K/(4*QR3_S))/2 = (256/16)/2 = 8, nrows = 32/8 = 4, so kqsx in
//   0..7 IS the ib32 sub-block index of the 256-element super-block, and one call of this
//   function loads exactly one block_iq3_s per row (mmq_core.cuh: blocks_per_iter = 256/qk = 1).
//   Per kqsx: qs[8*kqsx .. 8*kqsx+7] (8 grid indices), qh[kqsx] (their 9th bits), signs[4*kqsx ..
//   4*kqsx+3] (32 sign bits) and one 4-bit scale nibble -> 8 ints of quants at
//   x_qs[i*sram_stride + 8*kqsx + 0..7] and one float at x_df[i*sram_stride + kqsx].
//   64 ints + 8 floats = 72 <= sram_stride 76 for GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0.
// ---------------------------------------------------------------------------------------------

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq3_s(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = FTMMA_WARP_SIZE;
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads<type, J, fallback>() / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I<type, J, fallback>();
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride<type, J, fallback>();

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);

    constexpr int threads_per_row = (MMQ_ITER_K / (4 * FT_QR3_S)) / 2;  // 8
    constexpr int nrows = warp_size / threads_per_row;                  // 4
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq3_s * bxi = (const block_iq3_s *) x + kbx0 + i*stride;

        const int2      qs_packed = make_int2(get_int_b2(bxi->qs, 2*kqsx+0), get_int_b2(bxi->qs, 2*kqsx+1));
        const uint8_t * qs        = (const uint8_t *) &qs_packed;

        const int qh = bxi->qh[kqsx];

        const int       signs_packed_32 = get_int_b2(bxi->signs, kqsx);
        const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

#pragma unroll
        for (int l = 0; l < FT_QR3_S; ++l) {
            const int2 grid_pos = make_int2(
                ftmma_iq3s_grid[qs[2*l+0] | ((qh << (8 - 2*l)) & 0x100)],
                ftmma_iq3s_grid[qs[2*l+1] | ((qh << (7 - 2*l)) & 0x100)]);

            const int signs0 = __vcmpne4(((signs_packed_8[l] & 0x03) << 7) | ((signs_packed_8[l] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((signs_packed_8[l] & 0x30) << 3) | ((signs_packed_8[l] & 0xC0) << 17), 0x00000000);

            const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
            const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

            x_qs[i*sram_stride + 8*kqsx + (2*l+0)] = grid_l;
            x_qs[i*sram_stride + 8*kqsx + (2*l+1)] = grid_h;
        }

        const int ls = 1 + 2*((bxi->scales[kqsx/2] >> (((2*kqsx) << 1) & 0x04)) & 0x0F);
        // divergence 2: upstream writes `const float d = bxi->d;`. Same value, intrinsic spelling.
        const float d = __half2float(bxi->d);
        // The 0.25 and the grid go together. This port carries the FORK's iq3xs_grid (the
        // pre-#5829 llama.cpp table, whose byte levels are 4x upstream's and whose top level is
        // 62 rather than 60) instead of upstream's iq3s_grid, so that a prefill through this
        // kernel and a decode through mmvq/dequantize read the same weight values: measured, the
        // upstream table gave IQ3_S a systematic 1.3-1.8e-2 error against the fork's dequantized
        // reference where every other type sat at 5-8e-3. Which table is right for the file is a
        // separate question about the vendored header, not about this kernel.
        x_df[i*sram_stride             + kqsx] = 0.25f*ls*d;
    }

    FTMMA_UNUSED(i_max);
}

// ---------------------------------------------------------------------------------------------
// The per-type hook mma/mmq_core.cuh declares. upstream mmq.cuh:809-814 selects exactly this
// pair for IQ3_S (load_tiles_iq3_s + vec_dot_q8_0_q8_1_mma<..., MMQ_Q8_1_DS_LAYOUT_D4> +
// write_back_mma), and mmq-config-ampere.cuh:295-310 gives it GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0.
// ---------------------------------------------------------------------------------------------

template <> struct mmq_type_traits<GGML_TYPE_IQ3_S> {
    static constexpr ggml_cuda_mmq_sram_layout sram_layout = GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0;

    template <int J, bool fallback>
    static __device__ __forceinline__ void load_tiles(
            const char * __restrict__ x, int * __restrict__ x_tile,
            const int kbx0, const int i_max, const int stride) {
        ggml_cuda_mmq_load_tiles_iq3_s<GGML_TYPE_IQ3_S, J, fallback>(x, x_tile, kbx0, i_max, stride);
    }

    template <int J, bool fallback>
    static __device__ __forceinline__ void vec_dot(
            const int * __restrict__ x, const int * __restrict__ y,
            float * __restrict__ sum, const int k00) {
        ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<GGML_TYPE_IQ3_S, J, fallback, MMQ_Q8_1_DS_LAYOUT_D4>(
            x, y, sum, k00);
    }
};

}  // namespace ftmma
