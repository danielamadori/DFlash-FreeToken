// mma/mmq_iq2_xs.cuh -- IQ2_XS for the mma MMQ kernel. Plan step S4.
//
// PORTED FROM llama.cpp @ f7aadef09:
//   ggml-common.h:387-392        block_iq2_xs (layout only; see "THE QR2_XS TRAP" below)
//   ggml-common.h:627-756        iq2xs_grid[512], the 2-bit codebook (verbatim, see below)
//   vecdotq.cuh:18-24            get_int_b2
//   vecdotq.cuh:97-104           unpack_ksigns
//   mmq-load-tiles.cuh:1154-1217 ggml_cuda_mmq_load_tiles_iq2_xs, the mma arm
//                                (:1162-1163 pointers, :1170-1172 the thread map,
//                                 :1184-1205 grid + ksigns decode, :1207-1211 the scales)
//   mmq-vec-dot.cuh:481-613      ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma, the TURING_MMA arm
//                                (:534-608)
//   mmq.cuh:791-796              the util_funcs row: load_tiles_iq2_xs + vec_dot_q8_0_16_q8_1_mma
//                                + write_back_mma
//   mmq-config-ampere.cuh:244-259  GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K for every J
//   mmq.cuh:60-101               IQ2_XS uses MMQ_Q8_1_DS_LAYOUT_D4 for the activations
//                                (mma/quantize_mmq.cuh:124-129 already agrees, and D4 is one of
//                                 the two layouts that quantizer instantiates -- nothing needed)
//
// THE QR2_XS TRAP (the reason this type is on the CRITICAL list of the plan's S4 conflict table):
// the vendored ggml-common.h:118 defines QR2_XS as 8, upstream ggml-common.h:146 defines it as 4.
// This loader's thread map is
//     threads_per_row = (MMQ_ITER_K / (4 * QR2_XS)) / 2
// which is 8 with upstream's 4 and 4 with the vendored 8 -- i.e. half the threads, each reading
// the WRONG uint16 pair out of bxi->qs, SILENTLY. Likewise QR2_XS is the trip count of the grid
// decode loop (:1188): 4 upstream, 8 vendored, which would run off the end of q2_packed. So
// FT_QR2_XS below is 4, upstream's value, and this file includes NOTHING from the vendored
// headers -- neither ggml-common.h nor vecdotq.cuh -- exactly as mma/mmq_q4_K.cuh's header
// comment lays down for the whole port. The block struct is byte-identical between the two
// headers (half d; uint16_t qs[32]; uint8_t scales[8]; 74 B) and iq2xs_grid is identical in both
// (verified value by value); only the QR/QI macros disagree.
//
// DIVERGENCES FROM UPSTREAM:
//  1. The dp4a arm of both functions is dropped (mmq-load-tiles.cuh:1164-1167, :1201-1203,
//     :1212-1214, and ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_dp4a). This port is mma-only; on sm_89
//     the dp4a arm is dead. The AMD (MFMA/WMMA) arms are dropped too -- mma_int.cuh instantiates
//     no J_MAJOR tile, so mmq-vec-dot.cuh:483-532 has nothing to compile against.
//  2. `const float d = bxi->d;` (:1208) is spelled `__half2float(bxi->d)`, because the torch
//     extension is compiled with -D__CUDA_NO_HALF_CONVERSIONS__ (the same reason mma/mmq_q4_K.cuh
//     spells its half2 arithmetic with intrinsics -- its divergence 2). The arithmetic is
//     unchanged: the scale expression ((ls & 0x0F)*d + d/2)/4 is byte-for-byte upstream's.
//  3. The functions are members of mmq_type_traits<GGML_TYPE_IQ2_XS> rather than free templates
//     selected by the ggml_cuda_mmq_get_util_funcs switch (upstream mmq.cuh:535-843). See
//     mma/mmq_core.cuh divergence 3.
//  4. EVERYTHING IS IN A NESTED namespace ftmma::iq2_xs, which mma/mmq_q4_K.cuh does not do.
//     Reason: unlike vec_dot_q8_1_q8_1_mma (Q4_K's, which Q5_K reuses by INCLUDING mmq_q4_K.cuh),
//     ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma is shared by Q3_K, IQ2_XS and IQ2_S
//     (mmq.cuh:748-758, :791-802), and so are get_int_b2 and unpack_ksigns -- and those three S4
//     files are written in parallel, with no file any one of them may assume exists. Two of them
//     defining the same name at ftmma scope is a redefinition error the moment mmq_entry.cuh
//     includes both. The nested namespace makes that impossible without any cross-agent
//     agreement; the cost is one extra copy of a function TEMPLATE, which is instantiated per
//     `type` anyway, so no device code is duplicated. See NEEDED_ELSEWHERE: the follow-up is to
//     lift the shared three into mma/vec_dot_q8_0_16.cuh, exactly as the plan intends.

#pragma once

#include <cuda_fp16.h>

#include "mmq_core.cuh"

namespace ftmma {
namespace iq2_xs {

using namespace ggml_cuda_mma;

// ---------------------------------------------------------------------------------------------
// Local IQ2_XS constants and block layout. FT_QR2_XS is UPSTREAM's 4, NOT the vendored 8 -- see
// "THE QR2_XS TRAP" above. These are re-declared here (and not included) both because the
// vendored spellings are macros and because one of them is wrong for this code.
// ---------------------------------------------------------------------------------------------

static constexpr int FT_QR2_XS = 4;                              // upstream ggml-common.h:146
static constexpr int FT_QI2_XS = FTMMA_QK_K / (4 * FT_QR2_XS);   // 16, upstream ggml-common.h:145

// The k01 step of the q8_0_16 vec-dot. Upstream spells it QR3_K*VDR_Q3_K_Q8_1_MMQ
// (mmq-vec-dot.cuh:581) because that vec-dot is Q3_K's; QR3_K == 4 (upstream ggml-common.h:131,
// vendored :131 -- they agree) and VDR_Q3_K_Q8_1_MMQ == 2 (vecdotq.cuh:447).
static constexpr int FT_QR3_K              = 4;
static constexpr int FT_VDR_Q3_K_Q8_1_MMQ  = 2;

// upstream ggml-common.h:387-392 / vendored ggml-common.h:120-124 (identical layout).
struct block_iq2_xs {
    half     d;
    uint16_t qs[FTMMA_QK_K/8];
    uint8_t  scales[FTMMA_QK_K/32];
};
static_assert(sizeof(block_iq2_xs) == sizeof(half) + FTMMA_QK_K/8*sizeof(uint16_t) + FTMMA_QK_K/32,
              "wrong iq2_xs block size/padding");
static_assert(sizeof(block_iq2_xs) == 74, "block_iq2_xs must be 74 B");

// upstream vecdotq.cuh:18-24
static __device__ __forceinline__ int get_int_b2(const void * x, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) x; // assume at least 2 byte alignment

    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;

    return x32;
}

// upstream vecdotq.cuh:97-104
static __device__ __forceinline__ uint32_t unpack_ksigns(const uint8_t v) {
    // v is a 7 bit int, with the 8th sign being encodable as popcnt
    // with xor we can "correct" the bit instead of having to mask
    const uint32_t p = __popc(v) & 1;
    const uint32_t s = v ^ p << 7;
    // broadcast over uint to allow for 0x08040201 / 0x80402010 as selectors
    return s * 0x01010101;
}

// ---------------------------------------------------------------------------------------------
// upstream ggml-common.h:627-756, GGML_TABLE_BEGIN(uint64_t, iq2xs_grid, 512) ... GGML_TABLE_END.
// The 512 values below are byte-for-byte upstream's (and byte-for-byte the vendored copy's, which
// was diffed value by value before this file was written); only the declaration is local, because
// mma/ includes nothing from the vendored headers. Read as (const uint2 *) at :1189, so the two
// halves of each entry are the 8 grid bytes of one 16-quant group.
// ---------------------------------------------------------------------------------------------

static const __device__ uint64_t iq2xs_grid[512] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919,
    0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919,
    0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x080808082b080808,
    0x080808082b08082b, 0x080808082b081919, 0x080808082b082b08, 0x080808082b190819,
    0x080808082b191908, 0x080808082b192b19, 0x080808082b2b0808, 0x0808081908080819,
    0x0808081908081908, 0x080808190808192b, 0x0808081908082b19, 0x0808081908190808,
    0x080808190819082b, 0x0808081908191919, 0x0808081908192b08, 0x0808081908192b2b,
    0x08080819082b0819, 0x08080819082b1908, 0x0808081919080808, 0x080808191908082b,
    0x0808081919081919, 0x0808081919082b08, 0x0808081919190819, 0x0808081919191908,
    0x08080819192b0808, 0x08080819192b2b08, 0x080808192b080819, 0x080808192b081908,
    0x080808192b190808, 0x0808082b08080808, 0x0808082b0808082b, 0x0808082b08081919,
    0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908, 0x0808082b082b0808,
    0x0808082b19080819, 0x0808082b19081908, 0x0808082b19190808, 0x0808082b19191919,
    0x0808082b2b080808, 0x0808082b2b082b2b, 0x0808190808080819, 0x0808190808081908,
    0x080819080808192b, 0x0808190808082b19, 0x0808190808190808, 0x080819080819082b,
    0x0808190808191919, 0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908,
    0x0808190819080808, 0x080819081908082b, 0x0808190819081919, 0x0808190819082b08,
    0x0808190819190819, 0x0808190819191908, 0x080819081919192b, 0x08081908192b0808,
    0x080819082b080819, 0x080819082b081908, 0x080819082b190808, 0x0808191908080808,
    0x080819190808082b, 0x0808191908081919, 0x0808191908082b08, 0x0808191908190819,
    0x0808191908191908, 0x08081919082b0808, 0x0808191919080819, 0x0808191919081908,
    0x0808191919190808, 0x08081919192b0819, 0x080819192b080808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b08190808, 0x0808192b082b192b, 0x0808192b19080808,
    0x0808192b1908082b, 0x0808192b2b081908, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808082b2b, 0x08082b0808190819,
    0x08082b0808191908, 0x08082b08082b0808, 0x08082b08082b1919, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b0819192b08, 0x08082b082b080808,
    0x08082b082b2b0808, 0x08082b082b2b2b2b, 0x08082b1908080819, 0x08082b1908081908,
    0x08082b1908190808, 0x08082b1919080808, 0x08082b192b080819, 0x08082b192b082b19,
    0x08082b2b08080808, 0x08082b2b082b0808, 0x08082b2b082b2b08, 0x08082b2b2b19192b,
    0x08082b2b2b2b0808, 0x0819080808080819, 0x0819080808081908, 0x081908080808192b,
    0x0819080808082b19, 0x0819080808190808, 0x081908080819082b, 0x0819080808191919,
    0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908, 0x0819080819080808,
    0x081908081908082b, 0x0819080819081919, 0x0819080819082b08, 0x0819080819190819,
    0x0819080819191908, 0x08190808192b0808, 0x08190808192b2b2b, 0x081908082b080819,
    0x081908082b081908, 0x081908082b190808, 0x0819081908080808, 0x081908190808082b,
    0x0819081908081919, 0x0819081908082b08, 0x0819081908190819, 0x0819081908191908,
    0x08190819082b0808, 0x0819081919080819, 0x0819081919081908, 0x0819081919190808,
    0x081908192b080808, 0x081908192b191908, 0x081908192b19192b, 0x0819082b08080819,
    0x0819082b08081908, 0x0819082b0808192b, 0x0819082b08190808, 0x0819082b19080808,
    0x0819082b192b0808, 0x0819190808080808, 0x081919080808082b, 0x0819190808081919,
    0x0819190808082b08, 0x0819190808190819, 0x0819190808191908, 0x08191908082b0808,
    0x0819190819080819, 0x0819190819081908, 0x0819190819082b19, 0x0819190819190808,
    0x08191908192b1908, 0x081919082b080808, 0x0819191908080819, 0x0819191908081908,
    0x0819191908190808, 0x0819191919080808, 0x0819192b08080808, 0x0819192b08191908,
    0x0819192b19082b19, 0x08192b0808080819, 0x08192b0808081908, 0x08192b0808190808,
    0x08192b080819082b, 0x08192b0819080808, 0x08192b0819191908, 0x08192b082b08192b,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b19192b192b, 0x08192b2b19190819,
    0x08192b2b2b2b2b19, 0x082b080808080808, 0x082b08080808082b, 0x082b080808081919,
    0x082b080808082b08, 0x082b080808082b2b, 0x082b080808190819, 0x082b080808191908,
    0x082b0808082b0808, 0x082b080819080819, 0x082b080819081908, 0x082b080819190808,
    0x082b08082b080808, 0x082b08082b2b0808, 0x082b081908080819, 0x082b081908081908,
    0x082b081908190808, 0x082b081919080808, 0x082b081919082b08, 0x082b0819192b1919,
    0x082b082b08080808, 0x082b082b082b082b, 0x082b082b2b080808, 0x082b082b2b2b2b08,
    0x082b190808080819, 0x082b190808081908, 0x082b190808190808, 0x082b1908082b2b19,
    0x082b190819080808, 0x082b191908080808, 0x082b191919080819, 0x082b19191919082b,
    0x082b19192b192b19, 0x082b192b08080819, 0x082b192b08192b2b, 0x082b192b2b2b192b,
    0x082b2b0808080808, 0x082b2b0808082b08, 0x082b2b0808082b2b, 0x082b2b08082b0808,
    0x082b2b0819191919, 0x082b2b082b082b08, 0x082b2b082b2b082b, 0x082b2b19192b2b08,
    0x082b2b192b190808, 0x082b2b2b08082b08, 0x082b2b2b082b0808, 0x082b2b2b2b08082b,
    0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819, 0x1908080808081908,
    0x190808080808192b, 0x1908080808082b19, 0x1908080808190808, 0x190808080819082b,
    0x1908080808191919, 0x1908080808192b08, 0x19080808082b0819, 0x19080808082b1908,
    0x1908080819080808, 0x190808081908082b, 0x1908080819081919, 0x1908080819082b08,
    0x1908080819082b2b, 0x1908080819190819, 0x1908080819191908, 0x19080808192b0808,
    0x19080808192b1919, 0x190808082b080819, 0x190808082b081908, 0x190808082b190808,
    0x1908081908080808, 0x190808190808082b, 0x1908081908081919, 0x1908081908082b08,
    0x1908081908190819, 0x1908081908191908, 0x19080819082b0808, 0x1908081919080819,
    0x1908081919081908, 0x1908081919190808, 0x190808192b080808, 0x190808192b081919,
    0x190808192b2b082b, 0x1908082b08080819, 0x1908082b08081908, 0x1908082b08190808,
    0x1908082b0819082b, 0x1908082b082b2b19, 0x1908082b19080808, 0x1908190808080808,
    0x190819080808082b, 0x1908190808081919, 0x1908190808082b08, 0x1908190808190819,
    0x1908190808191908, 0x1908190808192b19, 0x19081908082b0808, 0x1908190819080819,
    0x1908190819081908, 0x1908190819190808, 0x190819082b080808, 0x190819082b191908,
    0x1908191908080819, 0x1908191908081908, 0x1908191908190808, 0x19081919082b1908,
    0x1908191919080808, 0x190819192b192b2b, 0x1908192b08080808, 0x1908192b08082b2b,
    0x1908192b19081908, 0x1908192b19190808, 0x19082b0808080819, 0x19082b0808081908,
    0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919, 0x19082b0819191908,
    0x19082b08192b082b, 0x19082b1908080808, 0x19082b1908190819, 0x19082b1919081908,
    0x19082b1919190808, 0x19082b19192b2b19, 0x19082b2b08081908, 0x1919080808080808,
    0x191908080808082b, 0x1919080808081919, 0x1919080808082b08, 0x1919080808190819,
    0x1919080808191908, 0x19190808082b0808, 0x19190808082b2b08, 0x1919080819080819,
    0x1919080819081908, 0x1919080819190808, 0x191908082b080808, 0x1919081908080819,
    0x1919081908081908, 0x1919081908190808, 0x1919081908191919, 0x1919081919080808,
    0x191908191908082b, 0x1919082b08080808, 0x1919082b19081908, 0x1919082b2b2b2b2b,
    0x1919190808080819, 0x1919190808081908, 0x1919190808190808, 0x19191908082b0819,
    0x1919190819080808, 0x19191908192b0808, 0x191919082b080819, 0x191919082b2b0819,
    0x1919191908080808, 0x1919191908082b08, 0x191919192b080808, 0x191919192b082b08,
    0x1919192b082b0819, 0x1919192b192b2b08, 0x1919192b2b2b0819, 0x19192b0808080808,
    0x19192b0808191908, 0x19192b0819080819, 0x19192b0819190808, 0x19192b082b192b19,
    0x19192b1908192b2b, 0x19192b1919080808, 0x19192b191908082b, 0x19192b2b2b081919,
    0x192b080808080819, 0x192b080808081908, 0x192b080808190808, 0x192b080819080808,
    0x192b080819191908, 0x192b0808192b082b, 0x192b08082b08192b, 0x192b08082b2b2b19,
    0x192b081908080808, 0x192b082b082b1908, 0x192b082b19082b2b, 0x192b082b2b19082b,
    0x192b190808080808, 0x192b19080819192b, 0x192b191908190808, 0x192b191919080808,
    0x192b191919081919, 0x192b19192b2b1908, 0x192b2b0808080819, 0x192b2b08192b2b2b,
    0x192b2b19082b1919, 0x192b2b2b0808192b, 0x192b2b2b19191908, 0x192b2b2b192b082b,
    0x2b08080808080808, 0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08,
    0x2b08080808190819, 0x2b08080808191908, 0x2b080808082b0808, 0x2b080808082b2b2b,
    0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808082b080808,
    0x2b0808082b08082b, 0x2b0808082b2b2b08, 0x2b0808082b2b2b2b, 0x2b08081908080819,
    0x2b08081908081908, 0x2b0808190808192b, 0x2b08081908190808, 0x2b08081919080808,
    0x2b08081919190819, 0x2b08081919192b19, 0x2b08082b08080808, 0x2b08082b082b0808,
    0x2b08082b2b080808, 0x2b08082b2b08082b, 0x2b08082b2b2b0808, 0x2b08082b2b2b2b08,
    0x2b08190808080819, 0x2b08190808081908, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190819080808, 0x2b081908192b0808, 0x2b0819082b082b19,
    0x2b08191908080808, 0x2b08191919081908, 0x2b0819192b2b1919, 0x2b08192b08192b08,
    0x2b08192b192b2b2b, 0x2b082b0808080808, 0x2b082b0808082b08, 0x2b082b08082b1919,
    0x2b082b0819192b2b, 0x2b082b082b080808, 0x2b082b082b08082b, 0x2b082b082b2b2b08,
    0x2b082b190808192b, 0x2b082b2b082b082b, 0x2b082b2b2b080808, 0x2b082b2b2b082b08,
    0x2b082b2b2b19192b, 0x2b082b2b2b2b2b08, 0x2b19080808080819, 0x2b19080808081908,
    0x2b19080808190808, 0x2b19080819080808, 0x2b1908081919192b, 0x2b1908082b081908,
    0x2b19081908080808, 0x2b190819082b082b, 0x2b190819192b1908, 0x2b19082b1919192b,
    0x2b19082b2b082b19, 0x2b19190808080808, 0x2b19190808081919, 0x2b19190819081908,
    0x2b19190819190808, 0x2b19190819192b08, 0x2b191919082b2b19, 0x2b1919192b190808,
    0x2b1919192b19082b, 0x2b19192b19080819, 0x2b192b0819190819, 0x2b192b082b2b192b,
    0x2b192b1919082b19, 0x2b192b2b08191919, 0x2b192b2b192b0808, 0x2b2b080808080808,
    0x2b2b08080808082b, 0x2b2b080808082b08, 0x2b2b080808082b2b, 0x2b2b0808082b0808,
    0x2b2b0808082b2b2b, 0x2b2b08082b2b0808, 0x2b2b081919190819, 0x2b2b081919192b19,
    0x2b2b08192b2b192b, 0x2b2b082b08080808, 0x2b2b082b0808082b, 0x2b2b082b08082b08,
    0x2b2b082b082b2b2b, 0x2b2b082b2b080808, 0x2b2b082b2b2b0808, 0x2b2b190819080808,
    0x2b2b19082b191919, 0x2b2b192b192b1919, 0x2b2b192b2b192b08, 0x2b2b2b0808082b2b,
    0x2b2b2b08082b0808, 0x2b2b2b08082b082b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b0808,
    0x2b2b2b082b2b2b08, 0x2b2b2b1908081908, 0x2b2b2b192b081908, 0x2b2b2b192b08192b,
    0x2b2b2b2b082b2b08, 0x2b2b2b2b082b2b2b, 0x2b2b2b2b2b190819, 0x2b2b2b2b2b2b2b2b,
};

// ---------------------------------------------------------------------------------------------
// upstream mmq-vec-dot.cuh:481-613, the `#elif defined(TURING_MMA_AVAILABLE)` arm at :534-608,
// verbatim except for the FT_ spelling of the QI/QR/VDR constants and the mmq_core accessor
// templates. Upstream uses this same vec-dot for Q3_K, IQ2_XS and IQ2_S (mmq.cuh:748-758,
// :791-802); see divergence 4 for why it lives in a nested namespace here.
//
// It reads y_df as a FLAT float, i.e. it assumes the D4 activation ds layout -- which is what
// mmq_get_q8_1_ds_layout returns for IQ2_XS (mma/quantize_mmq.cuh:124-129) and one of the two
// layouts quantize_mmq_q8_1_cuda instantiates. Nothing extra is needed from the quantizer.
// ---------------------------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------------------------
// upstream mmq-load-tiles.cuh:1154-1217, the mma arm only (:1162-1163 for the pointers,
// :1198-1200 for the quants, :1209-1211 for the scales).
//
// The x tile is GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K (mmq-config-ampere.cuh:244-259): 2*MMQ_TILE_NE_K
// ints of quants followed by MMQ_TILE_NE_K/2 floats of scales, +4 of bank-conflict padding. The
// writes below stay inside it: 8*kqsx + 2*l+1 <= 63 with kqsx < 8 and l < 4, and 2*kqsx+1 <= 15
// against the 16 float slots at offset 2*MMQ_TILE_NE_K.
// ---------------------------------------------------------------------------------------------

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq2_xs(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = FTMMA_WARP_SIZE;
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads<type, J, fallback>() / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I<type, J, fallback>();
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride<type, J, fallback>();

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);

    constexpr int threads_per_row = (MMQ_ITER_K / (4 * FT_QR2_XS)) / 2;  // 8
    constexpr int nrows = warp_size / threads_per_row;                   // 4
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq2_xs * bxi = (const block_iq2_xs *) x + kbx0 + i*stride;

        const int2 q2_packed = make_int2(get_int_b2(bxi->qs, 2*kqsx+0), get_int_b2(bxi->qs, 2*kqsx+1));
        const uint16_t * q2 = (const uint16_t *) &q2_packed;

    #pragma unroll
        for (int l = 0; l < FT_QR2_XS; ++l) {
            const uint2 grid_pos = ((const uint2*)iq2xs_grid)[q2[l] & 0x1FF];
            const uint32_t signs = unpack_ksigns(q2[l] >> 9);

            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);

            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

            x_qs[i*sram_stride + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*sram_stride + 8*kqsx + (2*l + 1)] = grid_h;
        }

        const int ls = bxi->scales[kqsx];
        // divergence 2: upstream writes `const float d = bxi->d;`. Same value, intrinsic spelling.
        const float d = __half2float(bxi->d);
        x_df[i*sram_stride + 2*kqsx+0] = ((ls &  0x0F)*d + d/2)/4;
        x_df[i*sram_stride + 2*kqsx+1] = ((ls >>    4)*d + d/2)/4;
    }

    FTMMA_UNUSED(i_max);
}

}  // namespace iq2_xs

// ---------------------------------------------------------------------------------------------
// The per-type hook mma/mmq_core.cuh declares. upstream mmq.cuh:791-796 selects exactly this pair
// for IQ2_XS (load_tiles_iq2_xs + vec_dot_q8_0_16_q8_1_mma + write_back_mma), and
// mmq-config-ampere.cuh:244-259 gives it GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K.
// ---------------------------------------------------------------------------------------------

template <> struct mmq_type_traits<GGML_TYPE_IQ2_XS> {
    static constexpr ggml_cuda_mmq_sram_layout sram_layout = GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K;

    template <int J, bool fallback>
    static __device__ __forceinline__ void load_tiles(
            const char * __restrict__ x, int * __restrict__ x_tile,
            const int kbx0, const int i_max, const int stride) {
        iq2_xs::ggml_cuda_mmq_load_tiles_iq2_xs<GGML_TYPE_IQ2_XS, J, fallback>(x, x_tile, kbx0, i_max, stride);
    }

    template <int J, bool fallback>
    static __device__ __forceinline__ void vec_dot(
            const int * __restrict__ x, const int * __restrict__ y,
            float * __restrict__ sum, const int k00) {
        iq2_xs::ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma<GGML_TYPE_IQ2_XS, J, fallback>(x, y, sum, k00);
    }
};

}  // namespace ftmma
