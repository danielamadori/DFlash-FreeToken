// Planar-activation MMVQ ("design A", docs/plans/mmvq-8row-kernel-plan.md R3) for 2..8 activation rows.
//
// Activation layout (built by quantize_row_planar_cuda in gguf_kernel.cu):
//   yq[ncols_y][Kp]      int8    the q8_1 quants of activation row c, k-contiguous, zero past K
//   ds[ncols_y][Kp/32]   float2  {d, d * sum_q}: the per-32-block scale and that scale times the
//                                sum of the 32 QUANTIZED int8 values (what the block_q8_1 path's
//                                "dot2" term computes: vecdotq.cuh vec_dot_q4_K_q8_1_impl_vmmq)
//   Kp = K rounded up to 512 (same padding as quantize_row_q8_1_cuda).
// The quants are the same as block_q8_1's (amax / 127, roundf); d is stored HALF-ROUNDED
// (MMVQ_PLANAR_HALF_D=1) so every product uses the same scale the installed kernel reads back
// through __low2float(ds), and the two paths differ only in fp32 summation order.
//
// Kernel shape: one warp per weight row, 8 lanes per K-block (QK_K = 256), 4 K-blocks per
// warp iteration; the weight block is unpacked ONCE per lane and dotted against all NCOLS
// activation rows via 16-byte loads of yq and float4 loads of ds. NCOLS is a template
// parameter (2..8): an odd row count runs the exact instantiation rather than an 8-wide loop on
// zero-padded columns, so the y-side loads and dp4a scale with the real row count and the
// quantize buffer never needs 8-column zero fill.
//
// Registration of further types (the Q5_K and IQ4_XS steps): see "PLANAR_TYPES: add here".
#pragma once

#include <stdint.h>
#include <stdlib.h>

#ifndef MMVQ_PLANAR
#define MMVQ_PLANAR 1  // 0 = compile the planar path out; the block_q8_1 kernels take every call
#endif
#ifndef MMVQ_PLANAR_HALF_D
#define MMVQ_PLANAR_HALF_D 1  // 1 = store d as half2float(float2half(d)), the installed kernel's scale
#endif
#ifndef MMVQ_PLANAR_WARPS
#define MMVQ_PLANAR_WARPS 1  // warps (= weight rows) per thread block; the prototype measured with 1
#endif
#ifndef MMVQ_PLANAR_MIN_COLS
#define MMVQ_PLANAR_MIN_COLS 2  // 1 column keeps the (bit-identical) one-warp block_q8_1 path
#endif
#define MMVQ_PLANAR_MAX_COLS 8

// ---------------------------------------------------------------- shared helpers

// Weight row of this warp: MMVQ_PLANAR_WARPS warps per block, one row each.
static __device__ __forceinline__ int mmvq_planar_row() {
  return blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
}

// Warp-reduce NCOLS per-lane partial sums and store dst[c * nrows + row] (dst is [ncols_y, nrows]).
template <typename scalar_t, int NCOLS>
static __device__ __forceinline__ void mmvq_planar_store(float (&acc)[NCOLS], scalar_t* __restrict__ dst,
                                                         const int row, const int nrows) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int c = 0; c < NCOLS; ++c) {
    float s = acc[c];
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      s += SGLANG_SHFL_XOR_SYNC(uint32_t(-1), s, mask);
    }
    if (lane == 0) {
      dst[(size_t)c * nrows + row] = s;
    }
  }
}

// One halfword (16 bits) of the 12 K-quant scale bytes that follow ``dm`` in block_q4_K /
// block_q5_K: with the 16-byte block header in ``hdr``, halfword o (0..5) lives in hdr.y (0,1),
// hdr.z (2,3), hdr.w (4,5). ``o`` is a per-lane constant, so this is a pair of selects and a
// shift on registers; it deliberately avoids indexing a local array with a runtime lane value,
// which nvcc turns into a stack frame (the prototype's A2 variant).
static __device__ __forceinline__ uint32_t mmvq_planar_k_halfword(const uint4 hdr, const int o) {
  const uint32_t w = (o < 2) ? hdr.y : ((o < 4) ? hdr.z : hdr.w);
  return (w >> ((o & 1) * 16)) & 0xffffu;
}

// Scales sc and mins m of sub-blocks 2j and 2j+1 (j = 0..3) of a Q4_K / Q5_K block, from the
// 16-byte header {dm, scales[12]}: the 6-bit unpacking of vecdotq.cuh vec_dot_q4_K_q8_1
// (aux[0] = sc pair, aux[1] = m pair) with the halfword indices resolved per lane.
static __device__ __forceinline__ void mmvq_planar_k_scales(const uint4 hdr, const int j,
                                                            int& sc0, int& sc1, int& m0, int& m1) {
  const uint32_t sA = mmvq_planar_k_halfword(hdr, j < 2 ? j : j + 2);
  const uint32_t sB = mmvq_planar_k_halfword(hdr, j < 2 ? j + 2 : j - 2);
  const uint32_t sC = mmvq_planar_k_halfword(hdr, j < 2 ? j + 2 : j);
  uint32_t aux0, aux1;
  if (j < 2) {
    aux0 = sA & 0x3f3f;
    aux1 = sB & 0x3f3f;
  } else {
    aux0 = (sA & 0x0f0f) | ((sB & 0xc0c0) >> 2);
    aux1 = ((sA >> 4) & 0x0f0f) | ((sC & 0xc0c0) >> 2);
  }
  sc0 = aux0 & 0xff;
  sc1 = (aux0 >> 8) & 0xff;
  m0 = aux1 & 0xff;
  m1 = (aux1 >> 8) & 0xff;
}

// ---------------------------------------------------------------- Q4_K
// Ported from scratch/sass/proto.cu q4k_hoisted (0.0617 ms on Q4_K 17408x5120 at 8 columns,
// 1.13x the DRAM floor, vs 0.085 for the block_q8_1 kernel). Lane l = lane & 7 of a K-block owns
// 16 bytes of qs at 32*j + 16*h (j = l >> 1, h = l & 1): the low nibbles are 16 values of
// sub-block 2j (activation k = 64j + 16h ..), the high nibbles 16 values of sub-block 2j+1
// (k = 64j + 32 + 16h ..). The min term of a sub-block pair is added by the h = 0 lane only.
// Per lane and column: 2 LDG.128 (yq) + 1 LDG.128 (ds) + 8 IDP4A + 2 IMUL + 2 I2F + 6 FFMA/FMUL.
template <typename scalar_t, int NCOLS>
__global__ void __launch_bounds__(MMVQ_PLANAR_WARPS * WARP_SIZE)
mmvq_planar_q4_K(const void* __restrict__ vx, const int8_t* __restrict__ yq, const float2* __restrict__ ds,
                 scalar_t* __restrict__ dst, const int ncols, const int nrows, const int Kp) {
  const int row = mmvq_planar_row();
  if (row >= nrows) {
    return;  // whole warp: the shuffles below are never reached by a partial warp
  }
  const int nb = ncols / QK_K;
  const int nb8 = Kp / 32;
  const int lane = threadIdx.x & 31, l = lane & 7, j = l >> 1, h = l & 1;

  float acc[NCOLS];
#pragma unroll
  for (int c = 0; c < NCOLS; ++c) {
    acc[c] = 0.0f;
  }

  const block_q4_K* xr = (const block_q4_K*)vx + (size_t)row * nb;
  for (int i = lane >> 3; i < nb; i += 4) {
    const block_q4_K* b = xr + i;
    const uint4 hdr = *(const uint4*)b;                         // dm + 12 scale bytes, one 16 B load
    const uint4 q = *(const uint4*)(b->qs + 32 * j + 16 * h);   // 16 B of nibbles = 32 weights
    int sc0, sc1, m0, m1;
    mmvq_planar_k_scales(hdr, j, sc0, sc1, m0, m1);
    const float m0f = h ? 0.0f : (float)m0;
    const float m1f = h ? 0.0f : (float)m1;
    const float2 dm = __half22float2(*(const half2*)&hdr.x);
    const int vlo[4] = {(int)(q.x & 0x0f0f0f0fu), (int)(q.y & 0x0f0f0f0fu),
                        (int)(q.z & 0x0f0f0f0fu), (int)(q.w & 0x0f0f0f0fu)};
    const int vhi[4] = {(int)((q.x >> 4) & 0x0f0f0f0fu), (int)((q.y >> 4) & 0x0f0f0f0fu),
                        (int)((q.z >> 4) & 0x0f0f0f0fu), (int)((q.w >> 4) & 0x0f0f0f0fu)};
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

// ---------------------------------------------------------------- launch + dispatch

// Launch KERNEL_<scalar_t, nvecs> for nvecs in 2..8 (call inside a function where scalar_t, vx,
// yq, ds, dst, ncols, nrows, Kp, nvecs, stream are in scope). Grid: one warp per weight row.
#define MMVQ_PLANAR_CASE_(KERNEL_, N_)                                                       \
  case N_:                                                                                   \
    KERNEL_<scalar_t, N_><<<grid_, block_, 0, stream>>>(vx, yq, ds, dst, ncols, nrows, Kp);  \
    break
#define MMVQ_PLANAR_RUN(KERNEL_)                                                        \
  do {                                                                                  \
    const dim3 grid_((nrows + MMVQ_PLANAR_WARPS - 1) / MMVQ_PLANAR_WARPS, 1, 1);         \
    const dim3 block_(MMVQ_PLANAR_WARPS * WARP_SIZE, 1, 1);                             \
    switch (nvecs) {                                                                    \
      MMVQ_PLANAR_CASE_(KERNEL_, 2);                                                    \
      MMVQ_PLANAR_CASE_(KERNEL_, 3);                                                    \
      MMVQ_PLANAR_CASE_(KERNEL_, 4);                                                    \
      MMVQ_PLANAR_CASE_(KERNEL_, 5);                                                    \
      MMVQ_PLANAR_CASE_(KERNEL_, 6);                                                    \
      MMVQ_PLANAR_CASE_(KERNEL_, 7);                                                    \
      MMVQ_PLANAR_CASE_(KERNEL_, 8);                                                    \
      default:                                                                          \
        break; /* unreachable: mmvq_planar_ok gates 2..8 */                             \
    }                                                                                   \
  } while (0)

// Other types plug in here: a per-type header included at this point (after the helpers, before
// the two switches below) defines MMVQ_PLANAR_HAS_<TYPE> 1 and the kernel template
//   template <typename scalar_t, int NCOLS> __global__ void
//   mmvq_planar_<type>(const void* vx, const int8_t* yq, const float2* ds, scalar_t* dst,
//                      int ncols /* K */, int nrows /* out_features */, int Kp);
// with the same layout contract as mmvq_planar_q4_K above (row = mmvq_planar_row(), 32-lane
// warp per row, dst[c * nrows + row], ds.x = d, ds.y = d*sum_q), and gets a case in
// mmvq_planar_type_ok / mmvq_planar_dispatch. Q5_K and IQ4_XS:
#include "mmvq_planar_q5k.cuh"    // defines MMVQ_PLANAR_HAS_Q5_K, mmvq_planar_q5_K (type 13)
#include "mmvq_planar_iq4xs.cuh"  // defines MMVQ_PLANAR_HAS_IQ4_XS, mmvq_planar_iq4_xs (type 23)

// Types with a planar kernel (GGUF ggml_type ids, as in the ggml_mul_mat_vec_a8 switch).
static inline bool mmvq_planar_type_ok(const int type) {
  switch (type) {
    case 12:  // Q4_K
      return true;
    // PLANAR_TYPES: add here (or define MMVQ_PLANAR_HAS_<TYPE> in your header, see above)
#if defined(MMVQ_PLANAR_HAS_Q5_K) && MMVQ_PLANAR_HAS_Q5_K
    case 13:  // Q5_K
      return true;
#endif
#if defined(MMVQ_PLANAR_HAS_IQ4_XS) && MMVQ_PLANAR_HAS_IQ4_XS
    case 23:  // IQ4_XS
      return true;
#endif
    default:
      return false;
  }
}

// Whether ggml_mul_mat_vec_a8(type, nvecs activation rows) takes the planar path: compile switch,
// runtime switch (FREETOKEN_MMVQ_PLANAR=0 in the environment turns it off for an A/B without a
// rebuild), type, and 2 <= nvecs <= 8 (1 row keeps the bit-identical one-warp path; above 8 the
// group-of-8 block_q8_1 path stays).
static inline bool mmvq_planar_ok(const int type, const int nvecs) {
#if !MMVQ_PLANAR
  return false;
#else
  static const bool env_on = [] {
    const char* e = getenv("FREETOKEN_MMVQ_PLANAR");
    return e == nullptr || e[0] == '\0' || (e[0] != '0' && e[0] != 'f' && e[0] != 'F' && e[0] != 'n' && e[0] != 'N');
  }();
  return env_on && nvecs >= MMVQ_PLANAR_MIN_COLS && nvecs <= MMVQ_PLANAR_MAX_COLS && mmvq_planar_type_ok(type);
#endif
}

// Run the planar kernel of ``type``; only valid when mmvq_planar_ok(type, nvecs).
// vx: packed weight [nrows][ncols / QK_K blocks]; yq / ds: the planar activation (above);
// dst: [nvecs][nrows]; ncols = K (in_features); Kp = K padded to 512.
template <typename scalar_t>
static void mmvq_planar_dispatch(const int type, const void* vx, const int8_t* yq, const float2* ds, scalar_t* dst,
                                 const int ncols, const int nrows, const int nvecs, const int Kp,
                                 cudaStream_t stream) {
  switch (type) {
    case 12:
      MMVQ_PLANAR_RUN(mmvq_planar_q4_K);
      break;
    // PLANAR_TYPES: add here (the guarded slots below are pre-wired for the Q5_K / IQ4_XS steps)
#if defined(MMVQ_PLANAR_HAS_Q5_K) && MMVQ_PLANAR_HAS_Q5_K
    case 13:
      MMVQ_PLANAR_RUN(mmvq_planar_q5_K);
      break;
#endif
#if defined(MMVQ_PLANAR_HAS_IQ4_XS) && MMVQ_PLANAR_HAS_IQ4_XS
    case 23:
      MMVQ_PLANAR_RUN(mmvq_planar_iq4_xs);
      break;
#endif
    default:
      break;  // unreachable: the wrapper checks mmvq_planar_ok first
  }
}
