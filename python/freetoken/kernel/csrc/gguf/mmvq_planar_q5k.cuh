// Planar-activation MMVQ kernel for Q5_K (design A, docs/plans/mmvq-8row-kernel-plan.md R3):
// the Q5_K sibling of mmvq_planar_q4_K in mmvq_planar.cuh, for 2..8 activation rows.
//
// block_q5_K (176 B = 11 x 16 B, so every field below is 16 B aligned when the weight is):
//   byte   0: half2   dm          super-block scale / min           } the same 16-byte header as
//   byte   4: uint8   scales[12]  6-bit sub-block scales and mins   } block_q4_K -> mmvq_planar_k_scales
//   byte  16: uint8   qh[32]      5th bit of every weight: bit s of qh[l] belongs to sub-block s, value l
//   byte  48: uint8   qs[128]     low 4 bits: bytes 32j..32j+31 hold sub-blocks 2j (low nibble)
//                                 and 2j+1 (high nibble), j = 0..3
// (ggml dequantize_row_q5_K / vecdotq.cuh vec_dot_q5_K_q8_1: vl = qs[32j + ..], vh = qh[..] >> 2j,
//  value = (vl >> 4i & 0x0f) | ((vh >> i) << 4 & 0x10) for sub-block 2j + i).
//
// Lane assignment, activation indexing, scales, the min term and the accumulation are exactly
// those of mmvq_planar_q4_K: lane l = lane & 7 of a K-block owns j = l >> 1, h = l & 1, i.e. the
// 16 qs bytes at 32j + 16h (32 weights: k = 64j + 16h + 0..15 in sub-block 2j, k = 64j + 32 +
// 16h + 0..15 in sub-block 2j+1) plus the 16 qh bytes at 16h that carry their 5th bits. The
// weight is unpacked once per lane (vlo / vhi, 5-bit values 0..31 as int8 lanes) and dotted
// against all NCOLS activation rows with 2 LDG.128 of yq + 1 LDG.128 of ds per column.
// Products are int8 x int5 dp4a times the same 6-bit scale and the same q8_1 scale d as the
// block_q8_1 path; the min term uses ds.y = d * sum_q of the QUANTIZED activation, so it equals
// the dot2 * m * d8 term of vec_dot_q5_K_q8_1_impl_vmmq. Only the fp32 summation order differs.
//
// Wiring: mmvq_planar.cuh includes this header after its shared helpers and before its
// mmvq_planar_type_ok / mmvq_planar_dispatch switches (the pre-wired case 13 slots). The forward
// declaration + self-include below also let it be included from gguf_kernel.cu BEFORE
// mmvq_planar.cuh; including it AFTER mmvq_planar.cuh from there would NOT register the type
// (the switches would already be compiled without MMVQ_PLANAR_HAS_Q5_K).
#pragma once

#include <stdint.h>

#define MMVQ_PLANAR_HAS_Q5_K 1

#ifndef MMVQ_PLANAR_WARPS
#define MMVQ_PLANAR_WARPS 1  // must agree with mmvq_planar.cuh (it uses the same default)
#endif

template <typename scalar_t, int NCOLS>
__global__ void __launch_bounds__(MMVQ_PLANAR_WARPS * WARP_SIZE)
mmvq_planar_q5_K(const void* __restrict__ vx, const int8_t* __restrict__ yq, const float2* __restrict__ ds,
                 scalar_t* __restrict__ dst, const int ncols, const int nrows, const int Kp);

#include "mmvq_planar.cuh"  // helpers (mmvq_planar_row / _k_scales / _store); no-op when included from it

// ---------------------------------------------------------------- Q5_K
// Per lane and column: 2 LDG.128 (yq) + 1 LDG.128 (ds) + 8 IDP4A + 2 IMUL + 2 I2F + 6 FFMA/FMUL,
// as for Q4_K; per lane and K-block one extra LDG.128 (qh) and 4 x (SHR + 2 LOP3) of unpacking.
template <typename scalar_t, int NCOLS>
__global__ void __launch_bounds__(MMVQ_PLANAR_WARPS * WARP_SIZE)
mmvq_planar_q5_K(const void* __restrict__ vx, const int8_t* __restrict__ yq, const float2* __restrict__ ds,
                 scalar_t* __restrict__ dst, const int ncols, const int nrows, const int Kp) {
  const int row = mmvq_planar_row();
  if (row >= nrows) {
    return;  // whole warp: the shuffles in mmvq_planar_store are never reached by a partial warp
  }
  const int nb = ncols / QK_K;
  const int nb8 = Kp / 32;
  const int lane = threadIdx.x & 31, l = lane & 7, j = l >> 1, h = l & 1;

  float acc[NCOLS];
#pragma unroll
  for (int c = 0; c < NCOLS; ++c) {
    acc[c] = 0.0f;
  }

  const block_q5_K* xr = (const block_q5_K*)vx + (size_t)row * nb;
  for (int i = lane >> 3; i < nb; i += 4) {
    const block_q5_K* b = xr + i;
    const uint4 hdr = *(const uint4*)b;                         // dm + 12 scale bytes, one 16 B load
    const uint4 q = *(const uint4*)(b->qs + 32 * j + 16 * h);   // 16 B of low nibbles = 32 weights
    const uint4 qh = *(const uint4*)(b->qh + 16 * h);           // their 5th bits (bits 2j / 2j+1)
    int sc0, sc1, m0, m1;
    mmvq_planar_k_scales(hdr, j, sc0, sc1, m0, m1);
    const float m0f = h ? 0.0f : (float)m0;
    const float m1f = h ? 0.0f : (float)m1;
    const float2 dm = __half22float2(*(const half2*)&hdr.x);
    // qh >> 2j puts sub-block 2j's bit at bit 0 of every byte and sub-block 2j+1's at bit 1;
    // << 4 / << 3 move them to bit 4 (the 0x10 of a 5-bit value), the mask drops the rest.
    const uint32_t s = 2 * j;
    const uint32_t hx = qh.x >> s, hy = qh.y >> s, hz = qh.z >> s, hw = qh.w >> s;
    const int vlo[4] = {(int)((q.x & 0x0f0f0f0fu) | ((hx << 4) & 0x10101010u)),
                        (int)((q.y & 0x0f0f0f0fu) | ((hy << 4) & 0x10101010u)),
                        (int)((q.z & 0x0f0f0f0fu) | ((hz << 4) & 0x10101010u)),
                        (int)((q.w & 0x0f0f0f0fu) | ((hw << 4) & 0x10101010u))};
    const int vhi[4] = {(int)(((q.x >> 4) & 0x0f0f0f0fu) | ((hx << 3) & 0x10101010u)),
                        (int)(((q.y >> 4) & 0x0f0f0f0fu) | ((hy << 3) & 0x10101010u)),
                        (int)(((q.z >> 4) & 0x0f0f0f0fu) | ((hz << 3) & 0x10101010u)),
                        (int)(((q.w >> 4) & 0x0f0f0f0fu) | ((hw << 3) & 0x10101010u))};
    const int8_t* y0 = yq + i * QK_K + 64 * j + 16 * h;
    const float2* d0 = ds + i * 8 + 2 * j;
#pragma unroll
    for (int c = 0; c < NCOLS; ++c) {
      const int4 ulo = *(const int4*)(y0 + c * Kp);
      const int4 uhi = *(const int4*)(y0 + c * Kp + 32);
      const float4 dsc = *(const float4*)(d0 + c * nb8);  // {d_lo, d*sum_lo, d_hi, d*sum_hi}
      int dlo = __dp4a(vlo[0], ulo.x, 0);
      dlo = __dp4a(vlo[1], ulo.y, dlo);
      dlo = __dp4a(vlo[2], ulo.z, dlo);
      dlo = __dp4a(vlo[3], ulo.w, dlo);
      int dhi = __dp4a(vhi[0], uhi.x, 0);
      dhi = __dp4a(vhi[1], uhi.y, dhi);
      dhi = __dp4a(vhi[2], uhi.z, dhi);
      dhi = __dp4a(vhi[3], uhi.w, dhi);
      float f = (float)(dlo * sc0) * dsc.x;
      f = fmaf((float)(dhi * sc1), dsc.z, f);
      float mn = m0f * dsc.y;
      mn = fmaf(m1f, dsc.w, mn);
      acc[c] = fmaf(dm.x, f, acc[c]);
      acc[c] = fmaf(-dm.y, mn, acc[c]);
    }
  }
  mmvq_planar_store<scalar_t, NCOLS>(acc, dst, row, nrows);
}
