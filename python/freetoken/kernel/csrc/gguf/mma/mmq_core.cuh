// mma/mmq_core.cuh -- the MMQ (matrix multiplication quantized) machinery, ported from
// llama.cpp @ f7aadef09 ggml/src/ggml-cuda/mmq.cuh.  Plan step S3, owner B.
//
// WHAT IS HERE (upstream line numbers on every block):
//   mmq.cuh:8-11     MMQ_ITER_K / MMQ_NWARPS
//   mmq.cuh:109-161  MMQ_TILE_NE_K / MMQ_TILE_Y_K, ggml_cuda_mmq_sram_layout + its stride
//   mmq.cuh:163-226  struct ggml_cuda_mmq_config + the CASE macro
//   mmq-config-ampere.cuh  the CASE rows (see divergence 3)
//   mmq.cuh:278-373  the config accessors
//   mmq.cuh:467-519  ggml_cuda_mmq_write_back_mma
//   mmq.cuh:867-941  mul_mat_q_process_tile
//   mmq.cuh:946-1054 mul_mat_q, NON-stream-k arm only
//   mmq.cuh:1371-1385 struct mmq_args + mmq_get_nbytes_shared
//   mmq.cuh:1387-1428 launch_mul_mat_q, non-stream-k arm only
//   mmq.cuh:1469-1561 mul_mat_q_switch_J / mul_mat_q_case
//
// DIVERGENCES FROM UPSTREAM (each is also commented at its site):
//  1. NAMESPACE. Everything is in `namespace ftmma`, and this file includes NOTHING from the
//     vendored ggml-common.h / vecdotq.cuh. See mma/common_shim.cuh's header comment for why
//     (vendored QR4_XS == 8, upstream == 2: a shared header would mis-address nibbles silently).
//     Consequence: names that are MACROS in the vendored header (QK8_1, QI8_1, QR4_K, QK_K, ...)
//     cannot be reused as C++ identifiers here. They are spelled FT_QI8_1 etc.
//  2. NO STREAM-K. Plan S3: the kernel is shipped with stream_k = false in every config row, so
//     mul_mat_q keeps only the (it = blockIdx.x, jt = blockIdx.y) tiling arm (mmq.cuh:986-1054)
//     and mul_mat_q_stream_k_fixup / tmp_fixup / ggml_cuda_pool_alloc are not ported at all.
//     Plan Q3 says this is the single most likely reason an R=3000 measurement could miss;
//     the fix, if the S4 GB/s column shows >2 DRAM passes, is to swap blockIdx.x/y here or to
//     port mmq.cuh:1233-1330 after all.
//  3. THE CONFIG TABLE IS TEMPLATED ON THE TYPE INSTEAD OF LISTING 11 x 5 ROWS. Every Ada row of
//     mmq-config-ampere.cuh is identical for every type this checkpoint uses -- 256 threads,
//     occupancy 1, I = 128, K_vram = MMQ_ITER_K -- and differs only in sram_layout, which is a
//     per-type property. So the sram_layout comes from mmq_type_traits<type> (which the per-type
//     header owns) and the table below is five CASE rows over J. This is what lets the ten S4
//     agents add a type without editing this file. The CASE macro keeps upstream's shape and
//     static_asserts.
//  4. NO `fallback` INSTANTIATIONS. Plan section 1.1: every out_features of this checkpoint is a
//     multiple of 128, so `fallback` stays a template parameter (upstream's spelling is intact)
//     but only `false` is ever instantiated; mmq_entry.cuh refuses the matmul otherwise.
//  5. NO MoE (ids_dst / expert_bounds) AND NO NVFP4 y_scale. This entry point is the dense
//     prefill matmul only; the vendored moe.cuh keeps the MoE path and is untouched.
//  6. dst IS TEMPLATED ON dst_t, not hard-wired to float. The vendored dispatch hands out
//     bf16/fp16/fp32 activations and expects the same dtype back (gguf_kernel.cu:288-290); a
//     float dst plus a conversion pass would cost R*out_features*4 B of extra traffic per
//     projection (~0.4 ms at R=3000, out=17408). write_back therefore converts float -> dst_t at
//     the single point where the value is stored, exactly as the vendored MMQ does
//     (mmq.cuh:127).
//  7. NO ggml_backend_cuda_context. launch_mul_mat_q loses its `ctx` parameter; the non-stream-k
//     arm needs no pool allocation at all (upstream's only pool use is tmp_fixup, mmq.cuh:1442).
//     The quantized-activation scratch is a torch::empty made by the caller (mmq_entry.cuh).
//  8. ggml_cuda_get_physical_warp_size() is `static inline int` in the shim (host), so device
//     code here uses FTMMA_WARP_SIZE directly where upstream calls the constexpr function.

#pragma once

#include <climits>

#include <cuda_fp16.h>

#include "common_shim.cuh"
#include "mma_int.cuh"
#include "quantize_mmq.cuh"

namespace ftmma {

using namespace ggml_cuda_mma;

// ---------------------------------------------------------------------------------------------
// upstream mmq.cuh:8-11 and :109-120
// ---------------------------------------------------------------------------------------------

static constexpr int MMQ_ITER_K = 256;
static constexpr int MMQ_NWARPS = 8;

// Local copies of the QI constants this file needs. Deliberately NOT the vendored macros:
// QI8_0/QI8_1/QI6_K are #defines in the vendored ggml-common.h and would textually clobber any
// C++ identifier of the same name (this bit S2 once already, see mma/common_shim.cuh divergence 2).
// Values: QIx = QKx / (4*QRx), with QK8_0 = QK8_1 = 32, QR8_0 = QR8_1 = 1, QK_K = 256, QR6_K = 2.
// QR6_K/QI6_K agree between the vendored header (:102-103) and upstream (:140), so this is only a
// spelling divergence, not a semantic one.
static constexpr int FT_QI8_0 = FTMMA_QK8_0 / 4;   // 8
static constexpr int FT_QI8_1 = FTMMA_QK8_1 / 4;   // 8
static constexpr int FT_QI6_K = FTMMA_QK_K / 8;    // 32

// Decouple shared memory tile sizes from WARP_SIZE to allow for different warp sizes.
// The K dimension of the tiles has either,
// 1*MMQ_TILE_NE_K==32 (always for TILE_Y_K) or 2*MMQ_TILE_NE_K==64 (typically for TILE_X_K),
// 32 bit elements for the quantized data (does not include scales).
// The final tile size in K direction is padded to avoid shared memory bank conflicts,
// in terms of 32 bit elements that means K % 8 == 4 for mma.
static constexpr int MMQ_TILE_NE_K = 32;

// block_q8_1_mmq has (128 8-bit ints == 32 32-bit ints + 4 32-bit scales)
static constexpr int MMQ_TILE_Y_K = MMQ_TILE_NE_K + MMQ_TILE_NE_K / FT_QI8_1;  // 36

static_assert(MMQ_TILE_Y_K * sizeof(int) == sizeof(block_q8_1_mmq),
              "MMQ_TILE_Y_K must describe one block_q8_1_mmq");

// The largest J this port instantiates. Used to size the read-past-the-end slack of the quantized
// activation buffer (see mmq_entry.cuh) -- DIVERGENCE from upstream's ggml_cuda_mmq_get_J_max
// (mmq.cuh:360-369), which walks J downwards in steps of 8 and would return a J SMALLER than the
// one switch_J picks, because this port's J set is {8,16,32,64,128} and not every multiple of 8.
// Over-allocating the slack at the maximum J is 128*144 = 18432 B and is always sufficient.
static constexpr int MMQ_J_MAX = 128;

// ---------------------------------------------------------------------------------------------
// upstream mmq.cuh:122-161 -- SRAM layout of the x tile and its (bank-conflict padded) stride.
// The FP4 layouts (Blackwell) are dropped; the rest are kept so the S4 agents find their row.
// ---------------------------------------------------------------------------------------------

enum ggml_cuda_mmq_sram_layout {
    GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0,
    GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1,
    GGML_CUDA_MMQ_SRAM_LAYOUT_Q2_K,
    GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K,
    GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K,
};

static constexpr __host__ __device__ int ggml_cuda_mmq_get_sram_stride(ggml_cuda_mmq_sram_layout sram_layout) {
    switch (sram_layout) {
        case GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0:
            return 2*MMQ_TILE_NE_K + 2*MMQ_TILE_NE_K/FT_QI8_0 + 4;
        case GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1:
            return 2*MMQ_TILE_NE_K + 2*MMQ_TILE_NE_K/FT_QI8_1 + 4;
        case GGML_CUDA_MMQ_SRAM_LAYOUT_Q2_K:
            return 2*MMQ_TILE_NE_K + MMQ_TILE_NE_K          + 4;
        case GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K:
            return 2*MMQ_TILE_NE_K + MMQ_TILE_NE_K/2        + 4;
        case GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K:
            return 2*MMQ_TILE_NE_K + MMQ_TILE_NE_K/FT_QI6_K + MMQ_TILE_NE_K/8 + 7;
        default:
            return -1;
    }
}

static_assert(ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0) % 8 == 4, "Wrong padding.");
static_assert(ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1) % 8 == 4, "Wrong padding.");
static_assert(ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q2_K) % 8 == 4, "Wrong padding.");
static_assert(ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K) % 8 == 4, "Wrong padding.");
static_assert(ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K) % 8 == 4, "Wrong padding.");

// ---------------------------------------------------------------------------------------------
// PER-TYPE HOOK (divergence 3). Every mma/mmq_<type>.cuh specialises this for its own type and
// nothing else; this file never mentions a concrete quant type. A specialisation provides:
//
//   static constexpr ggml_cuda_mmq_sram_layout sram_layout;
//   template <int J, bool fallback> static __device__ __forceinline__
//       void load_tiles(const char * __restrict__ x, int * __restrict__ x_tile,
//                       const int kbx0, const int i_max, const int stride);
//   template <int J, bool fallback> static __device__ __forceinline__
//       void vec_dot(const int * __restrict__ x, const int * __restrict__ y,
//                    float * __restrict__ sum, const int k00);
//
// The primary template is declared and NOT defined on purpose: naming an unported type in
// mul_mat_q_switch_J is then a hard compile error rather than a silently wrong kernel.
// ---------------------------------------------------------------------------------------------

template <ggml_type type> struct mmq_type_traits;

// ---------------------------------------------------------------------------------------------
// upstream mmq.cuh:163-226 -- the config struct and the CASE macro.
// ---------------------------------------------------------------------------------------------

struct ggml_cuda_mmq_config {
    ggml_type                 type;        // src0->type
    int                       nthreads;    // Number of threads per CUDA block.
    int                       occupancy;   // Targeted occupancy for the MMA kernel.
    int                       I;           // SRAM tile width in src0->ne[1]/dst->ne[0] direction.
    int                       J;           // SRAM tile width in src1->ne[1]/dst->ne[1] direction.
    ggml_cuda_mmq_sram_layout sram_layout; // SRAM tile length in the K direction (32 bit elements).
    int                       K_vram;      // VRAM tile length in the K direction (logical elements).
    bool                      stream_k;    // Whether or not to use stream-k decomposition.
    bool                      fallback;    // Whether an out-of-bounds check in the I direction is needed.

    constexpr __host__ __device__ ggml_cuda_mmq_config(
            ggml_type type, int nthreads, int occupancy, int I, int J,
            ggml_cuda_mmq_sram_layout sram_layout, int K_vram, bool stream_k, bool fallback) :
        type(type), nthreads(nthreads), occupancy(occupancy), I(I), J(J),
        sram_layout(sram_layout), K_vram(K_vram), stream_k(stream_k), fallback(fallback) {}

    constexpr __device__ int rows_per_warp() const {
        // Upstream's AMD arm (return 16) is dropped: no MFMA/WMMA in this port.
        return J >= 48 && J % 16 == 0 ? 32 : 16;
    }

    // Upstream also carries use_mma_data_layout() to select between the dp4a and the mma SRAM
    // layout. This port has only the mma layout (turing_mma_available is true on every device it
    // will run on -- mmq_entry.cuh asserts cc >= GGML_CUDA_CC_TURING), so the predicate is gone
    // and mmq_get_nbytes_shared uses the mma formula unconditionally.
};

#define CASE(type_, nthreads_, occupancy_, I_, J_, sram_layout_, K_vram_, stream_k_, fallback_)                                            \
    if (type == (type_) && J == (J_) && fallback == (fallback_)) {                                                                        \
        static_assert((nthreads_) %  32 == 0 && (nthreads_)       <= 512, "bad nthreads");                                                \
        static_assert(                          (occupancy_)      <=   8, "bad occupancy");                                               \
        static_assert((I_)        %  32 == 0,                             "bad I");                                                       \
        static_assert((J_)        %   8 == 0,                             "bad J");                                                       \
        static_assert((K_vram_)   % 256 == 0,                             "bad K_vram");                                                  \
        return ggml_cuda_mmq_config((type_), (nthreads_), (occupancy_), (I_), (J_), (sram_layout_), (K_vram_), (stream_k_), (fallback_)); \
    }                                                                                                                                     \

// The Ada Lovelace table. Upstream reaches this through ggml_cuda_mmq_get_config_ampere
// (mmq-config-ampere.cuh), which every NVIDIA arch >= Volta uses. Every row of it that concerns
// this checkpoint reads
//     CASE(<type>, 256, 1, 128, <J>, <layout>, MMQ_ITER_K, true, <fallback>)
// so the type dimension is folded into the template parameter (divergence 3), the J set is
// trimmed to {8,16,32,64,128} (plan S3; Q5 is the experiment that decides whether 40/80 pay), the
// fallback rows are dropped (divergence 4) and stream_k is false (divergence 2).
template <ggml_type type>
static constexpr __host__ __device__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config_ada(int J, bool fallback) {
    constexpr ggml_cuda_mmq_sram_layout layout = mmq_type_traits<type>::sram_layout;

    CASE(type, 256, 1, 128,   8, layout, MMQ_ITER_K, false, false);
    CASE(type, 256, 1, 128,  16, layout, MMQ_ITER_K, false, false);
    CASE(type, 256, 1, 128,  32, layout, MMQ_ITER_K, false, false);
    CASE(type, 256, 1, 128,  64, layout, MMQ_ITER_K, false, false);
    CASE(type, 256, 1, 128, 128, layout, MMQ_ITER_K, false, false);

    return ggml_cuda_mmq_config(GGML_TYPE_COUNT, 256, 1, 128, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, 256, false, true);
}

#undef CASE

// upstream mmq.cuh:228-276, collapsed: this port compiles for one arch family only, so the host
// and device selectors are the same function.
template <ggml_type type>
static constexpr __host__ __device__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config(int J, bool fallback) {
    return ggml_cuda_mmq_get_config_ada<type>(J, fallback);
}

// upstream mmq.cuh:278-373 -- the accessors, as templates because J/fallback are compile-time here.
template <ggml_type type, int J, bool fallback>
static constexpr __host__ __device__ int ggml_cuda_mmq_get_nthreads() {
    return ggml_cuda_mmq_get_config<type>(J, fallback).nthreads;
}

template <ggml_type type, int J, bool fallback>
static constexpr __host__ __device__ int ggml_cuda_mmq_get_occupancy() {
    return ggml_cuda_mmq_get_config<type>(J, fallback).occupancy;
}

template <ggml_type type, int J, bool fallback>
static constexpr __host__ __device__ int ggml_cuda_mmq_get_I() {
    return ggml_cuda_mmq_get_config<type>(J, fallback).I;
}

template <ggml_type type, int J, bool fallback>
static constexpr __host__ __device__ int ggml_cuda_mmq_get_K_vram() {
    return ggml_cuda_mmq_get_config<type>(J, fallback).K_vram;
}

template <ggml_type type, int J, bool fallback>
static constexpr __host__ __device__ bool ggml_cuda_mmq_get_stream_k() {
    return ggml_cuda_mmq_get_config<type>(J, fallback).stream_k;
}

template <ggml_type type, int J, bool fallback>
static constexpr __host__ __device__ int ggml_cuda_mmq_get_sram_stride() {
    return ggml_cuda_mmq_get_sram_stride(ggml_cuda_mmq_get_config<type>(J, fallback).sram_layout);
}

template <ggml_type type, int J, bool fallback>
static constexpr __device__ int ggml_cuda_mmq_get_rows_per_warp() {
    return ggml_cuda_mmq_get_config<type>(J, fallback).rows_per_warp();
}

// ---------------------------------------------------------------------------------------------
// upstream mmq.cuh:467-519 -- write back the mma accumulators.
// The AMD arm (tile<16,16,int,DATA_LAYOUT_J_MAJOR>, mmq.cuh:473) is dropped: mma_int.cuh
// instantiates no J_MAJOR tile by design. The NVFP4 y_scale arm and the MoE `ids_dst` indirection
// are dropped too; ids_dst is still passed and is always the identity map (see mul_mat_q), which
// keeps the shared-memory layout byte-identical to upstream's.
// ---------------------------------------------------------------------------------------------

template <ggml_type type, int J, bool fallback, typename dst_t>
static __device__ __forceinline__ void ggml_cuda_mmq_write_back_mma(
        const float * __restrict__ sum, const int * __restrict__ ids_dst, dst_t * __restrict__ dst,
        const int stride, const int i_max, const int j_max) {

    typedef tile<16, 8, int> tile_C;

    constexpr int I             = ggml_cuda_mmq_get_I<type, J, fallback>();
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp<type, J, fallback>();
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    const int i0 = (threadIdx.y / ntx) * (ntx*tile_C::I);

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);

                if (j > j_max) {
                    continue;
                }

                const int i = i0 + n*tile_C::I + tile_C::get_i(l);

                if (fallback && i > i_max) {
                    continue;
                }

                // divergence 6: float -> dst_t at the store, not in a second pass.
                dst[ids_dst[j]*stride + i] = sum[(j0/tile_C::J + n)*tile_C::ne + l];
            }
        }
    }
    FTMMA_UNUSED(i_max);
}

// ---------------------------------------------------------------------------------------------
// upstream mmq.cuh:867-941 -- one output tile.
// The `fixup` template parameter and the tmp_fixup write-back are dropped with stream-k
// (divergence 2); the Blackwell FP4 ne_block selection (mmq.cuh:887-892) is dropped too.
// ---------------------------------------------------------------------------------------------

template <ggml_type type, int J, bool fallback, typename dst_t>
static __device__ __forceinline__ void mul_mat_q_process_tile(
        const char * __restrict__ x, const int offset_x, const int * __restrict__ y,
        const int * __restrict__ ids_dst, dst_t * __restrict__ dst,
        const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int tile_x_max_i, const int tile_y_max_j, const int kb0_start, const int kb0_stop) {

    constexpr int warp_size = FTMMA_WARP_SIZE;                                   // divergence 8
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads<type, J, fallback>() / warp_size;
    constexpr int qk        = ftmma_type_traits<type>::qk;
    constexpr int I         = ggml_cuda_mmq_get_I<type, J, fallback>();

    extern __shared__ int data_mul_mat_q[];
    int * tile_y = data_mul_mat_q + J;
    int * tile_x = tile_y + FTMMA_PAD(J*MMQ_TILE_Y_K, nwarps*warp_size);

    constexpr int ne_block = QK8_1_MMQ;

    constexpr int ITER_K          = ggml_cuda_mmq_get_K_vram<type, J, fallback>();
    constexpr int blocks_per_iter = ITER_K / qk;

    float sum[J*I / (nwarps*warp_size)] = {0.0f};

    constexpr int sz = sizeof(block_q8_1_mmq) / sizeof(int);

    for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        mmq_type_traits<type>::template load_tiles<J, fallback>(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
        {
            const int * by0 = y + ncols_y * (kb0 * qk / ne_block) * sz;
#pragma unroll
            for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                int l = l0 + threadIdx.y*warp_size + threadIdx.x;

                tile_y[l] = by0[l];
            }
        }

        __syncthreads();

        mmq_type_traits<type>::template vec_dot<J, fallback>(tile_x, tile_y, sum, 0);

        __syncthreads();

        {
            const int * by0 = y + ncols_y * ((kb0 * qk / ne_block) * sz + sz);
#pragma unroll
            for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                int l = l0 + threadIdx.y*warp_size + threadIdx.x;

                tile_y[l] = by0[l];
            }
        }

        __syncthreads();

        mmq_type_traits<type>::template vec_dot<J, fallback>(tile_x, tile_y, sum, MMQ_TILE_NE_K);

        __syncthreads();
    }

    ggml_cuda_mmq_write_back_mma<type, J, fallback, dst_t>(
        sum, ids_dst, dst, stride_col_dst, tile_x_max_i, tile_y_max_j);
}

// ---------------------------------------------------------------------------------------------
// upstream mmq.cuh:946-1054 -- the kernel, NON-stream-k arm only (divergence 2).
// ---------------------------------------------------------------------------------------------

template <ggml_type type, int J, bool fallback, typename dst_t>
__launch_bounds__(ggml_cuda_mmq_get_nthreads<type, J, fallback>(), ggml_cuda_mmq_get_occupancy<type, J, fallback>())
static __global__ void mul_mat_q(
        const char * __restrict__ x, const int * __restrict__ y, dst_t * __restrict__ dst,
        const uint3 blocks_per_ne00, const int nrows_x, const int ncols_dst, const int stride_row_x,
        const int ncols_y, const int stride_col_dst,
        const uint3 channel_ratio, const uint3 nchannels_y,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio, const uint3 nsamples_y,
        const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst) {

    // Skip unused template specializations for faster compilation:
    if (ggml_cuda_mmq_get_config<type>(J, fallback).type == GGML_TYPE_COUNT) {
        NO_DEVICE_CODE;
        return;
    }

    constexpr int warp_size = FTMMA_WARP_SIZE;                                   // divergence 8
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads<type, J, fallback>() / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I<type, J, fallback>();

    // Initialize the ids for writing back data with just the index.
    // For regular matrix multiplications this is never changed (this port has no MoE path, so it
    // is ALWAYS the identity -- it is kept because the J ints it occupies are what makes tile_y
    // start at data + J, i.e. it is part of upstream's shared-memory layout).
    extern __shared__ int ids_dst_shared[]; // Stored at beginning of shared memory.
#pragma unroll
    for (int j0 = 0; j0 < J; j0 += nwarps*warp_size) {
        const int j = j0 + threadIdx.y*warp_size + threadIdx.x;

        if (j0 + nwarps*warp_size > J && j >= J) {
            break;
        }

        ids_dst_shared[j] = j;
    }
    __syncthreads();

    const uint2 tmp2 = fast_div_modulo(blockIdx.z, nchannels_y);
    const int wt = tmp2.x;
    const int zt = tmp2.y;
    const int jt = blockIdx.y;
    const int it = blockIdx.x;

    // Defaults for regular matrix multiplication (upstream's `if (ids_dst)` MoE arm is dropped):
    const int col_diff   = ncols_dst;
    int offset_y   = wt*stride_sample_y   + zt*stride_channel_y;
    int offset_dst = wt*stride_sample_dst + zt*stride_channel_dst + jt*J*stride_col_dst;

    offset_y   += (jt*J)*(sizeof(block_q8_1_mmq)/sizeof(int));
    offset_dst += it*I;

    const int tile_x_max_i = nrows_x  - it*I - 1;
    const int tile_y_max_j = col_diff - jt*J - 1;

    const int offset_x = fastdiv(wt, sample_ratio)*stride_sample_x
                       + fastdiv(zt, channel_ratio)*stride_channel_x + it*I*stride_row_x;

    mul_mat_q_process_tile<type, J, fallback, dst_t>
        (x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst,
         stride_row_x, ncols_y, stride_col_dst,
         tile_x_max_i, tile_y_max_j, 0, blocks_per_ne00.z);
}

// ---------------------------------------------------------------------------------------------
// upstream mmq.cuh:1371-1385 -- the argument bundle and the shared-memory size.
// Dropped fields: ids_dst, expert_bounds (MoE) and y_scale (NVFP4).
// ---------------------------------------------------------------------------------------------

template <typename dst_t>
struct mmq_args {
    const char * x; ggml_type type_x; const int * y; dst_t * dst;
    int64_t ncols_x; int64_t nrows_x; int64_t ncols_dst; int64_t stride_row_x; int64_t ncols_y; int64_t nrows_dst;
    int64_t nchannels_x; int64_t nchannels_y; int64_t stride_channel_x; int64_t stride_channel_y; int64_t stride_channel_dst;
    int64_t nsamples_x; int64_t nsamples_y; int64_t stride_sample_x; int64_t stride_sample_y; int64_t stride_sample_dst;
    int64_t ncols_max;
};

static inline size_t mmq_get_nbytes_shared(const ggml_cuda_mmq_config & config) {
    const size_t nbs_ids = config.J*sizeof(int);
    // upstream ggml_cuda_mmq_get_nbytes_shared_x (mmq.cuh:415-421), mma arm only.
    const size_t nbs_x   = (size_t) config.I * ggml_cuda_mmq_get_sram_stride(config.sram_layout) * 4;
    const size_t nbs_y   = config.J * sizeof(block_q8_1_mmq);
    return nbs_ids + nbs_x + FTMMA_PAD(nbs_y, config.nthreads*sizeof(int));
}

// ---------------------------------------------------------------------------------------------
// upstream mmq.cuh:1387-1428 -- the launcher. Everything from :1431 (stream-k) is dropped, and
// with it `ggml_backend_cuda_context & ctx` (divergence 7).
// ---------------------------------------------------------------------------------------------

template <ggml_type type, int J, bool fallback, typename dst_t>
static void launch_mul_mat_q(const mmq_args<dst_t> & args, cudaStream_t stream) {
    const int id        = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;

    const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config<type>(J, fallback);
    GGML_ASSERT(config.nthreads % warp_size == 0);
    const int    nwarps        = config.nthreads / warp_size;
    const size_t nbytes_shared = mmq_get_nbytes_shared(config);

    const dim3 block_dims(warp_size, nwarps, 1);

    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q<type, J, fallback, dst_t>), nbytes_shared);

    const int nty  = (args.nrows_x   + config.I - 1) / config.I;
    const int ntx  = (args.ncols_max + config.J - 1) / config.J;
    const int ntzw = args.nchannels_y * args.nsamples_y;
    const dim3 block_nums_xy_tiling(nty, ntx, ntzw);

    GGML_ASSERT(args.nchannels_y % args.nchannels_x == 0);
    GGML_ASSERT(args.nsamples_y  % args.nsamples_x  == 0);
    const int channel_ratio = args.nchannels_y / args.nchannels_x;
    const int sample_ratio  = args.nsamples_y  / args.nsamples_x;

    const uint3 blocks_per_ne00_fd = init_fastdiv_values(args.ncols_x / ftmma_type_traits<type>::qk);
    const uint3 nchannels_y_fd     = init_fastdiv_values(args.nchannels_y);
    const uint3 nsamples_y_fd      = init_fastdiv_values(args.nsamples_y);
    const uint3 channel_ratio_fd   = init_fastdiv_values(channel_ratio);
    const uint3 sample_ratio_fd    = init_fastdiv_values(sample_ratio);

    static_assert(!ggml_cuda_mmq_get_stream_k<type, J, fallback>(),
                  "this port ships the non-stream-k arm only (divergence 2)");

    mul_mat_q<type, J, fallback, dst_t><<<block_nums_xy_tiling, block_dims, nbytes_shared, stream>>>
        (args.x, args.y, args.dst,
         blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
         channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
         sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst);
}

// ---------------------------------------------------------------------------------------------
// upstream mmq.cuh:1469-1561 -- pick J, then instantiate.
// The J switch is trimmed to {8,16,32,64,128} (plan S3); upstream's 24/40/48/56/72/80/88/96/104/
// 112/120 rows are dropped, which is what keeps the instantiation count at 11 types x 5 J = 55.
// ---------------------------------------------------------------------------------------------

template <ggml_type type, bool fallback, typename dst_t>
static void mul_mat_q_switch_J(const mmq_args<dst_t> & args, cudaStream_t stream) {
    const int    id    = ggml_cuda_get_device();
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    int J_best        = 0;
    int ntiles_J_best = INT_MAX;

    // upstream steps J by 8 over its full table; this port has five J values, so the loop is
    // written over them explicitly. The selection rule (smallest number of column tiles, stop as
    // soon as one tile suffices) is upstream's, unchanged.
    constexpr int J_values[] = {8, 16, 32, 64, 128};
    for (int idx = 0; idx < 5 && ntiles_J_best > 1; ++idx) {
        const int J = J_values[idx];
        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config<type>(J, fallback);
        if (config.type == GGML_TYPE_COUNT) {
            continue;
        }

        if (mmq_get_nbytes_shared(config) > smpbo) {
            continue;
        }

        const int ntiles_x = (args.ncols_max + config.J - 1) / config.J;

        if (ntiles_x < ntiles_J_best) {
            J_best = J;
            ntiles_J_best = ntiles_x;
        }
    }

    switch (J_best) {
        case   8:
            launch_mul_mat_q<type,   8, fallback, dst_t>(args, stream);
            break;
        case  16:
            launch_mul_mat_q<type,  16, fallback, dst_t>(args, stream);
            break;
        case  32:
            launch_mul_mat_q<type,  32, fallback, dst_t>(args, stream);
            break;
        case  64:
            launch_mul_mat_q<type,  64, fallback, dst_t>(args, stream);
            break;
        case 128:
            launch_mul_mat_q<type, 128, fallback, dst_t>(args, stream);
            break;
        default:
            FTMMA_ABORT("ftmma: no MMQ config fits (J_best=%d)", J_best);
            break;
    }
}

// upstream mmq.cuh:1552-1561. Only the fallback == false arm is instantiated (divergence 4); the
// caller (mmq_entry.cuh) has already refused nrows_x % 128 != 0.
template <ggml_type type, typename dst_t>
static void mul_mat_q_case(const mmq_args<dst_t> & args, cudaStream_t stream) {
    GGML_ASSERT(args.nrows_x % 128 == 0);
    constexpr bool fallback = false;
    mul_mat_q_switch_J<type, fallback, dst_t>(args, stream);
}

}  // namespace ftmma
