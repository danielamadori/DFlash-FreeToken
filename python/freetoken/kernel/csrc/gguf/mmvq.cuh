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
// Shape of the upstream llama.cpp kernel (mmvq.cu, GENERIC table): ``nwarps`` warps split the K
// dimension of the same rows and combine through shared memory, and every thread carries
// ``rows_per_block`` weight rows, so the activation-side loads and the activation-only
// partial sums inside vec_dot are shared by two outputs instead of being redone per row.
// The vendored shape was one warp and one row per block: at 8 activation columns that
// kernel spends twice the memory time in per-column instructions (measured, DRAM-cold).
template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda,
          int ncols_y, int nwarps, int rows_per_block, bool guard_cols>
__launch_bounds__(nwarps * WARP_SIZE, 1)
static __global__ void mul_mat_vec_q(
    const void* __restrict__ vx,
    const void* __restrict__ vy,
    scalar_t* __restrict__ dst,
    const int ncols,
    const int nrows,
    const int nvecs) {
  const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
  const int row0 = rows_per_block * blockIdx.x;
  const int vec0 = blockIdx.y * ncols_y;

  const int blocks_per_row = ncols / qk;
  constexpr int blocks_per_iter = vdr * nwarps * WARP_SIZE / qi;
  const int nrows_y = (ncols + 512 - 1) / 512 * 512;
  const int y_stride = nrows_y / QK8_1;  // q8_1 blocks per activation row

  // partial sums: one per (activation column, weight row) this thread carries
  float tmp[ncols_y][rows_per_block];
#pragma unroll
  for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
    for (int i = 0; i < rows_per_block; ++i) {
      tmp[j][i] = 0.0f;
    }
  }

  const block_q_t* x = (const block_q_t*)vx;
  const block_q8_1* y = (const block_q8_1*)vy;

  // A row past the end (odd nrows) is clamped to the last row: its dot products are computed
  // on valid memory and discarded at the write.
  int row_of[rows_per_block];
#pragma unroll
  for (int i = 0; i < rows_per_block; ++i) {
    row_of[i] = (row0 + i < nrows) ? (row0 + i) : (nrows - 1);
  }

  for (int kbx = tid / (qi / vdr); kbx < blocks_per_row; kbx += blocks_per_iter) {
    const int kby = kbx * (qk / QK8_1);          // y block index that aligns with kbx
    const int kqs = vdr * (tid % (qi / vdr));    // x block quant index when casting the quants to int

#pragma unroll
    for (int j = 0; j < ncols_y; ++j) {
      if (guard_cols && vec0 + j >= nvecs) {
        break;
      }
      const block_q8_1* yj = &y[(vec0 + j) * y_stride + kby];
#pragma unroll
      for (int i = 0; i < rows_per_block; ++i) {
        tmp[j][i] += vec_dot_q_cuda(&x[row_of[i] * blocks_per_row + kbx], yj, kqs);
      }
    }
  }

  __shared__ float tmp_shared[nwarps - 1 > 0 ? nwarps - 1 : 1][ncols_y][rows_per_block][WARP_SIZE];
  if (threadIdx.y > 0) {
#pragma unroll
    for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
      for (int i = 0; i < rows_per_block; ++i) {
        tmp_shared[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
      }
    }
  }
  __syncthreads();
  if (threadIdx.y > 0) {
    return;
  }

  // sum up partial sums and write back result
#pragma unroll
  for (int j = 0; j < ncols_y; ++j) {
    if (guard_cols && vec0 + j >= nvecs) {
      break;
    }
#pragma unroll
    for (int i = 0; i < rows_per_block; ++i) {
#pragma unroll
      for (int l = 0; l < nwarps - 1; ++l) {
        tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
      }
      float sum = tmp[j][i];
#pragma unroll
      for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        sum += SGLANG_SHFL_XOR_SYNC(uint32_t(-1), sum, mask);
      }
      if (threadIdx.x == i && row0 + i < nrows) {
        dst[(vec0 + j) * nrows + row0 + i] = sum;
      }
    }
  }
}

// One exact instantiation per column count up to 8 (no per-iteration column check), with the
// upstream GENERIC parameters: 4 warps below 5 columns, 2 above; 2 rows per thread except at
// 1 column. Above 8 columns the weight is re-read once per group of 8, with the column guard.
#define MMVQ_CASE(QK_, QI_, BLOCK_, VDR_, VECDOT_, N_, NW_, RPB_)                                                                \
  case N_: {                                                                                      \
    const dim3 block_nums((nrows + RPB_ - 1) / RPB_, 1, 1);                                       \
    const dim3 block_dims(WARP_SIZE, NW_, 1);                                                     \
    mul_mat_vec_q<scalar_t, QK_, QI_, BLOCK_, VDR_, VECDOT_, N_, NW_, RPB_, false>                \
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);                \
    break;                                                                                        \
  }
#define MMVQ_LAUNCH(QK_, QI_, BLOCK_, VDR_, VECDOT_)                                              \
  do {                                                                                            \
    switch (nvecs) {                                                                              \
      MMVQ_CASE(QK_, QI_, BLOCK_, VDR_, VECDOT_, 1, 4, 1)                                                                          \
      MMVQ_CASE(QK_, QI_, BLOCK_, VDR_, VECDOT_, 2, 4, 2)                                                                          \
      MMVQ_CASE(QK_, QI_, BLOCK_, VDR_, VECDOT_, 3, 4, 2)                                                                          \
      MMVQ_CASE(QK_, QI_, BLOCK_, VDR_, VECDOT_, 4, 4, 2)                                                                          \
      MMVQ_CASE(QK_, QI_, BLOCK_, VDR_, VECDOT_, 5, 2, 2)                                                                          \
      MMVQ_CASE(QK_, QI_, BLOCK_, VDR_, VECDOT_, 6, 2, 2)                                                                          \
      MMVQ_CASE(QK_, QI_, BLOCK_, VDR_, VECDOT_, 7, 2, 2)                                                                          \
      MMVQ_CASE(QK_, QI_, BLOCK_, VDR_, VECDOT_, 8, 2, 2)                                                                          \
      default: {                                                                                  \
        const dim3 block_nums((nrows + 2 - 1) / 2, (nvecs + 8 - 1) / 8, 1);                       \
        const dim3 block_dims(WARP_SIZE, 2, 1);                                                   \
        mul_mat_vec_q<scalar_t, QK_, QI_, BLOCK_, VDR_, VECDOT_, 8, 2, 2, true>                   \
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);            \
        break;                                                                                    \
      }                                                                                           \
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
