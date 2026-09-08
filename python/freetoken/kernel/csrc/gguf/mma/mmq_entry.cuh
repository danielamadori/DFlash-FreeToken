// mma/mmq_entry.cuh -- the single host entry point of the mma MMQ port. Plan step S3, owner B/F.
//
// This is the only file of mma/ that gguf_kernel.cu includes, and the only one that knows about
// torch. It does what upstream's ggml_cuda_mul_mat_q does (mmq.cu:60-176, the `if (!ids)` arm):
// allocate the quantized-activation scratch, run the activation quantizer, build mmq_args and
// call mul_mat_q_case. The differences from upstream are all consequences of not having ggml:
//
//   * the scratch is a torch::empty instead of a ggml_cuda_pool_alloc (the vendored dispatch
//     already does this -- gguf_kernel.cu:190-193);
//   * the slack the MMQ kernel reads past the last activation row is sized at MMQ_J_MAX rather
//     than upstream's ggml_cuda_mmq_get_J_max (see mma/mmq_core.cuh, MMQ_J_MAX);
//   * there is no ggml_tensor, so the shapes arrive as the four ints the vendored dispatch
//     already passes around (gguf_kernel.cu:280-292): ncols = in_features, nrows = out_features,
//     nvecs = activation rows;
//   * dst is scalar_t, not float (mma/mmq_core.cuh divergence 6);
//   * ftmma_mul_mat RETURNS FALSE instead of aborting when the request is outside what is ported,
//     so the caller can fall back to ggml_dequantize + a dense matmul. Every precondition the
//     kernel needs is checked here, in one place.

#pragma once

#include <torch/all.h>

#include "mmq_core.cuh"
#include "mmq_q4_K.cuh"

namespace ftmma {

// Types with an mmq_type_traits specialisation in this build. S4 adds one line per type here and
// one #include above; nothing else in mma/ changes.
static inline bool ftmma_mul_mat_type_supported(const int type) {
    switch ((ftmma_type) type) {
        case GGML_TYPE_Q4_K:
            return true;
        default:
            return false;
    }
}

// Runtime spelling of ftmma_type_traits<type>::qk (which is a compile-time constant and cannot be
// indexed by a runtime type). Only the ported types need a row; the default is the K-quant block
// size, which is what every type of this checkpoint except Q8_0/IQ4_NL uses (plan section 1.1).
static inline int mmq_type_qk(const ftmma_type type) {
    switch (type) {
        case GGML_TYPE_Q8_0:   return FTMMA_QK8_0;
        case GGML_TYPE_IQ4_NL: return FTMMA_QK4_NL;
        default:               return FTMMA_QK_K;
    }
}

// Compile-time preconditions of the ported kernel, checked once on the host.
//   ncols % 256 == 0   the k loop advances MMQ_ITER_K logical elements at a time
//                      (mmq_core.cuh: blocks_per_iter = MMQ_ITER_K/qk)
//   nrows % 128 == 0   no `fallback` instantiation exists (mmq_core.cuh divergence 4). Plan
//                      section 1.1: every out_features of this checkpoint satisfies this.
//   cc >= TURING       the port has only the mma data layout; there is no dp4a arm to fall back
//                      to inside the kernel.
static inline bool ftmma_mul_mat_shape_supported(
        const int type, const int64_t ncols, const int64_t nrows, const int64_t nvecs) {
    if (!ftmma_mul_mat_type_supported(type)) {
        return false;
    }
    if (ncols <= 0 || nrows <= 0 || nvecs <= 0) {
        return false;
    }
    if (ncols % MMQ_ITER_K != 0 || nrows % 128 != 0) {
        return false;
    }
    const int id = ggml_cuda_get_device();
    if (id < 0 || id >= ggml_cuda_info().device_count) {
        return false;
    }
    return ggml_cuda_info().devices[id].cc >= GGML_CUDA_CC_TURING;
}

// The matmul. Y[nvecs][nrows] = X[nvecs][ncols] @ dequant(W)[nrows][ncols]^T, computed in int8.
//
//   type   GGUF quant type code of W (ftmma_type / the vendored `int64_t type`)
//   vx     packed weight, contiguous, nrows rows of ncols/qk blocks
//   x      activations, contiguous [nvecs][ncols], scalar_t
//   dst    output, contiguous [nvecs][nrows], scalar_t
//
// Returns false, having done nothing, if this (type, shape, device) is not ported; the caller
// must then fall back. Aborts only on a genuine internal inconsistency.
template <typename scalar_t>
bool ftmma_mul_mat(
        const int type, const void * vx, const scalar_t * x, scalar_t * dst,
        const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {

    if (!ftmma_mul_mat_shape_supported(type, ncols, nrows, nvecs)) {
        return false;
    }

    const ftmma_type type_x = (ftmma_type) type;

    // ---- quantized activation scratch, upstream mmq.cu:120,136-138 -------------------------
    const int64_t ne00 = ncols;                                   // K
    const int64_t ne1  = nvecs;                                   // R
    const int64_t ne0  = mmq_q8_1_padded_row_size(ne00);          // K padded to 512
    const size_t  nbytes_y = mmq_q8_1_nbytes(ne0, ne1, 1, 1, MMQ_J_MAX);
    GGML_ASSERT(nbytes_y % sizeof(int) == 0);

    const auto opts = torch::TensorOptions()
                          .dtype(torch::kInt32)
                          .device(torch::kCUDA, ggml_cuda_get_device());
    at::Tensor y_q8_1 = torch::empty({(int64_t) (nbytes_y / sizeof(int))}, opts);

    quantize_mmq_q8_1_cuda<scalar_t>(
        x, y_q8_1.data_ptr(), type_x,
        /*ne00=*/ne00, /*s01=*/ne00, /*s02=*/ne00*ne1, /*s03=*/ne00*ne1,
        /*ne0=*/ne0, /*ne1=*/ne1, /*ne2=*/1, /*ne3=*/1, stream);
    CUDA_CHECK(cudaGetLastError());

    // ---- mmq_args, upstream mmq.cu:162-174 --------------------------------------------------
    // The channel/sample dimensions are all 1 on this path (one 2-D weight, one 2-D activation
    // block); their strides are the natural ones so that the arithmetic reads like upstream's
    // even though blockIdx.z is always 0 here.
    const int64_t stride_row_x = ne00 / (int64_t) mmq_type_qk(type_x);  // blocks per weight row

    mmq_args<scalar_t> args = {
        (const char *) vx, type_x, (const int *) y_q8_1.data_ptr(), dst,
        /*ncols_x=*/ne00, /*nrows_x=*/nrows, /*ncols_dst=*/ne1,
        /*stride_row_x=*/stride_row_x, /*ncols_y=*/ne1, /*nrows_dst=*/nrows,
        /*nchannels_x=*/1, /*nchannels_y=*/1,
        /*stride_channel_x=*/nrows*stride_row_x,
        /*stride_channel_y=*/mmq_q8_1_channel_stride_ints(ne0, ne1),
        /*stride_channel_dst=*/nrows*ne1,
        /*nsamples_x=*/1, /*nsamples_y=*/1,
        /*stride_sample_x=*/nrows*stride_row_x,
        /*stride_sample_y=*/mmq_q8_1_channel_stride_ints(ne0, ne1),
        /*stride_sample_dst=*/nrows*ne1,
        /*ncols_max=*/ne1};

    switch (type_x) {
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K, scalar_t>(args, stream);
            break;
        default:
            // unreachable: ftmma_mul_mat_type_supported() gates this switch.
            FTMMA_ABORT("ftmma_mul_mat: type %d passed the support check but has no case", type);
            return false;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

}  // namespace ftmma
