// copied from
// https://github.com/vllm-project/vllm/blob/4492e3a55428e161ca8db381edc28263e5da4c8d/csrc/quantization/gguf/mmvq.cuh
// copied and adapted from https://github.com/ggerganov/llama.cpp/blob/b2899/ggml-cuda/mmvq.cu
// ``ncols_y`` activation rows are handled by ONE block, so the weight blocks it walks are read
// once and reused across all of them. The original took one block per activation row, which
// made the weight traffic nvecs times the matrix: fine for a decode step, which has one row,
// and ruinous for a speculative verification, which has eight and re-read 6.82 GiB of
// MMQ-less weights seven extra times -- about 8 ms per token on a 24 GB card.
//
// The arithmetic per output is unchanged: the same vec_dot over the same blocks in the same
// order, reduced the same way. Only which thread block does it moved.
template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda, int ncols_y>
static __global__ void mul_mat_vec_q(
    const void* __restrict__ vx,
    const void* __restrict__ vy,
    scalar_t* __restrict__ dst,
    const int ncols,
    const int nrows,
    const int nvecs) {
  const auto row = blockIdx.x * blockDim.y + threadIdx.y;
  const int vec0 = blockIdx.y * ncols_y;

  if (row >= nrows || vec0 >= nvecs) {
    return;
  }

  const int blocks_per_row = ncols / qk;
  const int blocks_per_warp = vdr * WARP_SIZE / qi;
  const int nrows_y = (ncols + 512 - 1) / 512 * 512;

  // partial sum for each thread, one per activation row this block carries
  float tmp[ncols_y];
#pragma unroll
  for (int j = 0; j < ncols_y; ++j) {
    tmp[j] = 0.0f;
  }

  const block_q_t* x = (const block_q_t*)vx;
  const block_q8_1* y = (const block_q8_1*)vy;

  for (auto i = threadIdx.x / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
    const int ibx = row * blocks_per_row + i;  // x block index, independent of the activation row

    const int iqs = vdr * (threadIdx.x % (qi / vdr));  // x block quant index when casting the quants to int

#pragma unroll
    for (int j = 0; j < ncols_y; ++j) {
      if (vec0 + j >= nvecs) {
        break;
      }
      // y block index that aligns with ibx
      const int iby = (vec0 + j) * (nrows_y / QK8_1) + i * (qk / QK8_1);
      tmp[j] += vec_dot_q_cuda(&x[ibx], &y[iby], iqs);
    }
  }

  // sum up partial sums and write back result
#pragma unroll
  for (int j = 0; j < ncols_y; ++j) {
    if (vec0 + j >= nvecs) {
      break;
    }
    float sum = tmp[j];
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
      sum += SGLANG_SHFL_XOR_SYNC(uint32_t(-1), sum, mask);
    }
    if (threadIdx.x == 0) {
      dst[(vec0 + j) * nrows + row] = sum;
    }
  }
}

// How many activation rows one block carries. Bounded so the per-thread partial sums stay in
// registers; beyond it the weight is re-read once per group, which is still far better than
// once per row.
#define MMVQ_LAUNCH(QK_, QI_, BLOCK_, VDR_, VECDOT_)                                            \
  do {                                                                                          \
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;                     \
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);                                        \
    if (nvecs <= 1) {                                                                            \
      const dim3 block_nums(block_num_y, 1, 1);                                                   \
      mul_mat_vec_q<scalar_t, QK_, QI_, BLOCK_, VDR_, VECDOT_, 1>                                 \
          <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);              \
    } else if (nvecs <= 2) {                                                                      \
      const dim3 block_nums(block_num_y, 1, 1);                                                   \
      mul_mat_vec_q<scalar_t, QK_, QI_, BLOCK_, VDR_, VECDOT_, 2>                                 \
          <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);              \
    } else if (nvecs <= 4) {                                                                      \
      const dim3 block_nums(block_num_y, 1, 1);                                                   \
      mul_mat_vec_q<scalar_t, QK_, QI_, BLOCK_, VDR_, VECDOT_, 4>                                 \
          <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);              \
    } else if (nvecs <= 8) {                                                                      \
      const dim3 block_nums(block_num_y, 1, 1);                                                   \
      mul_mat_vec_q<scalar_t, QK_, QI_, BLOCK_, VDR_, VECDOT_, 8>                                 \
          <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);              \
    } else {                                                                                      \
      const dim3 block_nums(block_num_y, (nvecs + 8 - 1) / 8, 1);                                  \
      mul_mat_vec_q<scalar_t, QK_, QI_, BLOCK_, VDR_, VECDOT_, 8>                                  \
          <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);              \
    }                                                                                             \
  } while (0)

template <typename scalar_t>
static void mul_mat_vec_q4_0_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  MMVQ_LAUNCH(QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1);
}

template <typename scalar_t>
static void mul_mat_vec_q4_1_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  MMVQ_LAUNCH(QK4_0, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1);
}

template <typename scalar_t>
static void mul_mat_vec_q5_0_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  MMVQ_LAUNCH(QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1);
}

template <typename scalar_t>
static void mul_mat_vec_q5_1_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  MMVQ_LAUNCH(QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1);
}

template <typename scalar_t>
static void mul_mat_vec_q8_0_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  MMVQ_LAUNCH(QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1);
}

template <typename scalar_t>
static void mul_mat_vec_q2_K_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  MMVQ_LAUNCH(QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1);
}

template <typename scalar_t>
static void mul_mat_vec_q3_K_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  MMVQ_LAUNCH(QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1);
}

template <typename scalar_t>
static void mul_mat_vec_q4_K_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  MMVQ_LAUNCH(QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1);
}

template <typename scalar_t>
static void mul_mat_vec_q5_K_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  MMVQ_LAUNCH(QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1);
}

template <typename scalar_t>
static void mul_mat_vec_q6_K_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  MMVQ_LAUNCH(QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1);
}

template <typename scalar_t>
static void mul_mat_vec_iq2_xxs_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  MMVQ_LAUNCH(QK_K, QI2_XXS, block_iq2_xxs, 1, vec_dot_iq2_xxs_q8_1);
}

template <typename scalar_t>
static void mul_mat_vec_iq2_xs_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  MMVQ_LAUNCH(QK_K, QI2_XS, block_iq2_xs, 1, vec_dot_iq2_xs_q8_1);
}

template <typename scalar_t>
static void mul_mat_vec_iq2_s_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  MMVQ_LAUNCH(QK_K, QI2_S, block_iq2_s, 1, vec_dot_iq2_s_q8_1);
}

template <typename scalar_t>
static void mul_mat_vec_iq3_xxs_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  MMVQ_LAUNCH(QK_K, QI3_XXS, block_iq3_xxs, 1, vec_dot_iq3_xxs_q8_1);
}

template <typename scalar_t>
static void mul_mat_vec_iq1_s_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  MMVQ_LAUNCH(QK_K, QI1_S, block_iq1_s, 1, vec_dot_iq1_s_q8_1);
}

template <typename scalar_t>
static void mul_mat_vec_iq1_m_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  MMVQ_LAUNCH(QK_K, QI1_M, block_iq1_m, 1, vec_dot_iq1_m_q8_1);
}

template <typename scalar_t>
static void mul_mat_vec_iq4_nl_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  MMVQ_LAUNCH(QK4_NL, QI4_NL, block_iq4_nl, VDR_Q4_0_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1);
}

template <typename scalar_t>
static void mul_mat_vec_iq4_xs_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  MMVQ_LAUNCH(QK_K, QI4_XS, block_iq4_xs, 1, vec_dot_iq4_xs_q8_1);
}

template <typename scalar_t>
static void mul_mat_vec_iq3_s_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  MMVQ_LAUNCH(QK_K, QI3_XS, block_iq3_s, 1, vec_dot_iq3_s_q8_1);
}
