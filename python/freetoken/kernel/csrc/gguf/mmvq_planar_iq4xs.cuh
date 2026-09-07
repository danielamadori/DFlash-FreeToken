// IQ4_XS planar-activation MMVQ kernel ("design A", docs/plans/mmvq-8row-kernel-plan.md R3) for
// 2..8 activation rows. Ported from scratch/sass/proto_iq4.cu iq4xs_hoisted (measured 0.0584 ms
// on IQ4_XS 17408x5120 at 8 columns vs 0.062 / 0.067 ms for the block_q8_1 kernel).
//
// Plugs into mmvq_planar.cuh: this header must be included AFTER the shared helpers of that file
// (mmvq_planar_row / mmvq_planar_store / MMVQ_PLANAR_WARPS) and BEFORE its mmvq_planar_type_ok()
// and mmvq_planar_dispatch() switches, which activate their pre-wired `case 23:` slots when
// MMVQ_PLANAR_HAS_IQ4_XS is defined to 1. mmvq_planar.cuh auto-includes a file named
// "mmvq_planar_iq4_xs.cuh" at exactly that point; this file is named mmvq_planar_iq4xs.cuh, so
// either add `#include "mmvq_planar_iq4xs.cuh"` next to that __has_include block or copy this
// file to the auto-included name.
//
// Activation layout contract (see mmvq_planar.cuh):
//   yq[c * Kp + k]           int8    q8_1 quants of activation row c (zero for K <= k < Kp)
//   ds[c * (Kp/32) + k/32]   float2  {d, d * sum_q}; only .x (the per-32 scale) is used here:
//                                    IQ4_XS has no min term, so d * sum_q is not needed
//   dst[c * nrows + row]     scalar_t
//
// Block format (ggml-common.h block_iq4_xs, 136 bytes => 8-byte alignment only, hence uint2 loads):
//   half d | uint16 scales_h | uint8 scales_l[4] | uint8 qs[128]
//   sub-block s (0..7) = 32 weights: qs[16s + k] low nibble -> k = 32s + k, high nibble -> k = 32s + 16 + k
//   6-bit sub-block scale ls = scales_l nibble s | (scales_h 2-bit field s) << 4; weight = d * (ls - 32) * kvalues_iq4nl[nibble]
//
// Kernel shape: one warp per weight row, lane s = lane & 7 owns sub-block s of a K-block, 4
// K-blocks per warp iteration (i = lane >> 3, step 4). Per lane and K-block the 16 bytes of
// nibbles are looked up ONCE through the 16-entry kvalues_iq4nl table (byte-permute lookup, the
// table lives in 4 immediate 32-bit constants: no memory traffic) into 8 int32 of ordered int8
// values; then per activation column: 2 LDG.128 of yq + 1 LDG.32 of ds.x, 8 IDP4A, 1 I2F, 1 FMUL,
// 1 FFMA. Same products as vecdotq.cuh vec_dot_iq4_xs_q8_1 (int8 x table value, times
// d * (ls - 32) * d8); only the fp32 summation order differs.
#pragma once

#include <stdint.h>

#define MMVQ_PLANAR_HAS_IQ4_XS 1

// kvalues_iq4nl = {-127,-104,-83,-65, -49,-35,-22,-10, 1,13,25,38, 53,69,89,113} packed
// little-endian, 4 int8 per word (word w = entries 4w..4w+3). Verified against ggml-common.h.
#define MMVQ_PLANAR_IQ4NL_W0 0xbfad9881u
#define MMVQ_PLANAR_IQ4NL_W1 0xf6eaddcfu
#define MMVQ_PLANAR_IQ4NL_W2 0x26190d01u
#define MMVQ_PLANAR_IQ4NL_W3 0x71594535u

// Table lookup of 8 nibbles (one 32-bit word of qs) WITHOUT reordering: returns
//   r.x = {v(b0.lo), v(b0.hi), v(b1.lo), v(b1.hi)}, r.y = {v(b2.lo), v(b2.hi), v(b3.lo), v(b3.hi)}
// (b0..b3 = the 4 bytes of q4). __byte_perm uses bits 2:0 of each selector nibble (nvcc masks
// the selector with 0x7777), so `lo` picks table entry (nibble & 7) from words 0/1 and `hi`
// entry (nibble & 7) + 8 from words 2/3; `sel` then chooses lo or hi per byte by nibble bit 3.
static __device__ __forceinline__ int2 mmvq_planar_iq4nl_lookup_interleaved(const uint32_t q4) {
  const uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
  int2 r;
  {
    const uint32_t lo = __byte_perm(MMVQ_PLANAR_IQ4NL_W0, MMVQ_PLANAR_IQ4NL_W1, q4);
    const uint32_t hi = __byte_perm(MMVQ_PLANAR_IQ4NL_W2, MMVQ_PLANAR_IQ4NL_W3, q4);
    r.x = (int)__byte_perm(lo, hi, sel);
  }
  {
    const uint32_t lo = __byte_perm(MMVQ_PLANAR_IQ4NL_W0, MMVQ_PLANAR_IQ4NL_W1, q4 >> 16);
    const uint32_t hi = __byte_perm(MMVQ_PLANAR_IQ4NL_W2, MMVQ_PLANAR_IQ4NL_W3, q4 >> 16);
    r.y = (int)__byte_perm(lo, hi, sel >> 16);
  }
  return r;
}

// Table lookup with the vec_dot_iq4_xs_q8_1 / get_int_from_table_16 semantics:
//   .x = low-nibble values of the 4 bytes in order (weights k .. k+3),
//   .y = high-nibble values in order (weights k+16 .. k+19).
static __device__ __forceinline__ int2 mmvq_planar_iq4nl_lookup(const uint32_t q4) {
  const int2 t = mmvq_planar_iq4nl_lookup_interleaved(q4);
  return make_int2((int)__byte_perm((uint32_t)t.x, (uint32_t)t.y, 0x6420u),
                   (int)__byte_perm((uint32_t)t.x, (uint32_t)t.y, 0x7531u));
}

// ---------------------------------------------------------------- IQ4_XS
template <typename scalar_t, int NCOLS>
__global__ void __launch_bounds__(MMVQ_PLANAR_WARPS * WARP_SIZE)
mmvq_planar_iq4_xs(const void* __restrict__ vx, const int8_t* __restrict__ yq, const float2* __restrict__ ds,
                   scalar_t* __restrict__ dst, const int ncols, const int nrows, const int Kp) {
  const int row = mmvq_planar_row();
  if (row >= nrows) {
    return;  // whole warp: the shuffles in mmvq_planar_store are never reached by a partial warp
  }
  const int nb = ncols / QK_K;
  const int nb8 = Kp / 32;
  const int lane = threadIdx.x & 31, s = lane & 7;

  float acc[NCOLS];
#pragma unroll
  for (int c = 0; c < NCOLS; ++c) {
    acc[c] = 0.0f;
  }

  const block_iq4_xs* xr = (const block_iq4_xs*)vx + (size_t)row * nb;
  for (int i = lane >> 3; i < nb; i += 4) {
    const block_iq4_xs* b = xr + i;
    // 136-byte blocks: only 8-byte alignment is guaranteed, so three LDG.64 per lane.
    const uint2 hdr = *(const uint2*)b;                         // .x = d | scales_h << 16, .y = scales_l[4]
    const uint2 qa = *(const uint2*)(b->qs + 16 * s);           // 16 bytes of nibbles = 32 weights of sub-block s
    const uint2 qb = *(const uint2*)(b->qs + 16 * s + 8);
    // 6-bit sub-block scale: low 4 bits from scales_l nibble s, high 2 bits from scales_h field s.
    const int ls = (int)(((hdr.y >> (8 * (s >> 1))) >> (4 * (s & 1))) & 0xFu) |
                   (int)((((hdr.x >> 16) >> (2 * s)) & 3u) << 4);
    const float dls = __half2float(__ushort_as_half((unsigned short)(hdr.x & 0xffffu))) * (float)(ls - 32);
    // Weight values, looked up once per block: v[w].x = weights 32s + 4w .. +3, v[w].y = 32s + 16 + 4w .. +3
    const int2 v0 = mmvq_planar_iq4nl_lookup(qa.x);
    const int2 v1 = mmvq_planar_iq4nl_lookup(qa.y);
    const int2 v2 = mmvq_planar_iq4nl_lookup(qb.x);
    const int2 v3 = mmvq_planar_iq4nl_lookup(qb.y);
    const int8_t* y0 = yq + i * QK_K + 32 * s;
    const float* d0 = (const float*)(ds + i * 8 + s);  // .x of ds[c * nb8 + i * 8 + s]
#pragma unroll
    for (int c = 0; c < NCOLS; ++c) {
      const int4 ulo = *(const int4*)(y0 + c * Kp);        // activation k = 32s .. 32s+15
      const int4 uhi = *(const int4*)(y0 + c * Kp + 16);   // activation k = 32s+16 .. 32s+31
      const float d8 = d0[(size_t)c * nb8 * 2];            // float2 stride: 2 floats per ds entry
      int sumi = __dp4a(v0.x, ulo.x, 0);
      sumi = __dp4a(v1.x, ulo.y, sumi);
      sumi = __dp4a(v2.x, ulo.z, sumi);
      sumi = __dp4a(v3.x, ulo.w, sumi);
      sumi = __dp4a(v0.y, uhi.x, sumi);
      sumi = __dp4a(v1.y, uhi.y, sumi);
      sumi = __dp4a(v2.y, uhi.z, sumi);
      sumi = __dp4a(v3.y, uhi.w, sumi);
      acc[c] = fmaf((float)sumi * dls, d8, acc[c]);
    }
  }
  mmvq_planar_store<scalar_t, NCOLS>(acc, dst, row, nrows);
}
