// copied from
// https://github.com/vllm-project/vllm/blob/4492e3a55428e161ca8db381edc28263e5da4c8d/csrc/quantization/gguf/mmvq.cuh
// copied and adapted from https://github.com/ggerganov/llama.cpp/blob/b2899/ggml-cuda/mmvq.cu
// Launch shape re-ported from llama.cpp ggml-cuda/mmvq.cu (2026-08, MMVQ_PARAMETERS_GENERIC):
// a block carries ``ncols_y`` activation rows and ``rows_per_block`` weight rows, and
// ``nwarps`` warps split the K dimension and meet in shared memory. The weight blocks a thread
// walks are read once and reused across every activation row it carries; the activation
// (q8_1) ints it loads are reused across the weight rows it carries.
//
// The arithmetic per output is unchanged: the same vec_dot over the same blocks. What moves is
// which thread accumulates which block, so at 2..8 columns the float32 summation order differs
// from the one-warp kernel; at 1 column the shape is one warp and one row per block, exactly the
// previous kernel, so a decode step is bit-identical.
template <
    typename scalar_t,
    int qk,
    int qi,
    typename block_q_t,
    int vdr,
    vec_dot_q_cuda_t vec_dot_q_cuda,
    int ncols_y,
    int nwarps,
    int rows_per_block>
__launch_bounds__(nwarps* WARP_SIZE, 1) static __global__ void mul_mat_vec_q(
    const void* __restrict__ vx,
    const void* __restrict__ vy,
    scalar_t* __restrict__ dst,
    const int ncols,
    const int nrows) {
  static_assert((vdr * nwarps * WARP_SIZE) % qi == 0, "K split must land on whole quant blocks");
  constexpr int blocks_per_iter = vdr * nwarps * WARP_SIZE / qi;
  static_assert(blocks_per_iter >= 1, "one warp must cover at least one quant block per iteration");

  const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
  const int row0 = rows_per_block * blockIdx.x;
  const int vec0 = ncols_y * blockIdx.y;

  const int blocks_per_row = ncols / qk;
  // q8_1 rows are padded to 512 values by quantize_row_q8_1_cuda (gguf_kernel.cu).
  const int blocks_per_y = ((ncols + 512 - 1) / 512 * 512) / QK8_1;

  const block_q_t* x = (const block_q_t*)vx;
  const block_q8_1* y = (const block_q8_1*)vy + (size_t)vec0 * blocks_per_y;

  // A block that hangs past the last row (odd nrows) re-reads the last row into its second
  // slot and drops it in the epilogue: no branch inside the K loop, no read past the tensor.
  int xrow[rows_per_block];
#pragma unroll
  for (int i = 0; i < rows_per_block; ++i) {
    xrow[i] = min(row0 + i, nrows - 1) * blocks_per_row;
  }

  float tmp[ncols_y][rows_per_block];
#pragma unroll
  for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
    for (int i = 0; i < rows_per_block; ++i) {
      tmp[j][i] = 0.0f;
    }
  }

  for (int kbx = tid / (qi / vdr); kbx < blocks_per_row; kbx += blocks_per_iter) {
    const int kby = kbx * (qk / QK8_1);         // y block index that aligns with kbx
    const int kqs = vdr * (tid % (qi / vdr));  // x block quant index when casting the quants to int
#pragma unroll
    for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
      for (int i = 0; i < rows_per_block; ++i) {
        tmp[j][i] += vec_dot_q_cuda(&x[xrow[i] + kbx], &y[j * blocks_per_y + kby], kqs);
      }
    }
  }

  if constexpr (nwarps > 1) {
    __shared__ float tmp_shared[nwarps - 1][ncols_y][rows_per_block][WARP_SIZE];
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
#pragma unroll
    for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
      for (int i = 0; i < rows_per_block; ++i) {
#pragma unroll
        for (int l = 0; l < nwarps - 1; ++l) {
          tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
        }
      }
    }
  }

  // sum up partial sums and write back result
#pragma unroll
  for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
    for (int i = 0; i < rows_per_block; ++i) {
      float sum = tmp[j][i];
#pragma unroll
      for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        sum += SGLANG_SHFL_XOR_SYNC(uint32_t(-1), sum, mask);
      }
      if (threadIdx.x == i && (rows_per_block == 1 || row0 + i < nrows)) {
        dst[(size_t)(vec0 + j) * nrows + row0 + i] = sum;
      }
    }
  }
}

// Warps per block. llama.cpp's MMVQ_PARAMETERS_GENERIC table (mmvq.cu calc_nwarps) says 4 warps
// up to 4 columns and 2 above, and that is kept where it measured best; but 1 column keeps one
// warp so the decode step stays bit-identical to the previous kernel, and types whose quant
// block is served by 8 threads or fewer (the I-quants: qi/vdr <= 8) or 16 (Q4_K, Q5_K, Q3_K
// at 5..8 columns) run faster with ONE warp per block -- more independent blocks in flight
// beat the K split (RTX 4090, DRAM-cold A/B on the 27B tensors, 2026-09-07: IQ4_XS 17408x5120
// at 8 columns 0.075 -> 0.062 ms, IQ3_S 0.066 -> 0.061, Q4_K 0.086 -> 0.083, Q5_K 0.091 ->
// 0.086; Q6_K (32 threads per block) and Q8_0 (4) lose with one warp and keep the table).
static constexpr int mmvq_nwarps(int ncols_y, int threads_per_block) {
  return ncols_y <= 1 ? 1
       : threads_per_block <= 8 ? 1
       : threads_per_block <= 16 ? (ncols_y <= 4 ? 4 : 1)
       : (ncols_y <= 4 ? 4 : 2);
}
static constexpr int mmvq_rows_per_block(int ncols_y) {
  return ncols_y <= 1 ? 1 : 2;
}

template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda, int ncols_y>
static void mmvq_launch(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int ngroups,
    cudaStream_t stream) {
  constexpr int nwarps = mmvq_nwarps(ncols_y, qi / vdr);
  constexpr int rows_per_block = mmvq_rows_per_block(ncols_y);
  const dim3 block_nums((nrows + rows_per_block - 1) / rows_per_block, ngroups, 1);
  const dim3 block_dims(WARP_SIZE, nwarps, 1);
  mul_mat_vec_q<scalar_t, qk, qi, block_q_t, vdr, vec_dot_q_cuda, ncols_y, nwarps, rows_per_block>
      <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

// ``ncols_y`` activation rows per block group, ``ngroups`` groups along grid.y.
template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
static void mmvq_switch(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int ncols_y,
    const int ngroups,
    cudaStream_t stream) {
#define MMVQ_CASE(N_)                                                                    \
  case N_:                                                                               \
    mmvq_launch<scalar_t, qk, qi, block_q_t, vdr, vec_dot_q_cuda, N_>(                   \
        vx, vy, dst, ncols, nrows, ngroups, stream);                                     \
    break
  switch (ncols_y) {
    MMVQ_CASE(1);
    MMVQ_CASE(2);
    MMVQ_CASE(3);
    MMVQ_CASE(4);
    MMVQ_CASE(5);
    MMVQ_CASE(6);
    MMVQ_CASE(7);
    MMVQ_CASE(8);
    default:
      break;  // unreachable: mmvq_dispatch only passes 1..8
  }
#undef MMVQ_CASE
}

// Types with a planar kernel (mmvq_planar.cuh: Q4_K, plus whatever registers there) do not reach
// this dispatch at 2..8 activation rows when MMVQ_PLANAR is on: ggml_mul_mat_vec_a8 routes them
// through mmvq_planar_ok / mmvq_planar_dispatch before quantizing to block_q8_1. Everything else
// (1 row, > 8 rows, other types, MMVQ_PLANAR=0) takes the path below, unchanged.
//
// Up to 8 activation rows go to the exact instantiation, so no column-count check runs inside
// the K loop. Beyond 8 the rows are cut into groups of 8 along grid.y (the weight is re-read
// once per group, still far better than once per row) and the remainder gets its own exact
// launch on the tail of the q8_1 buffer and of dst.
template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
static void mmvq_dispatch(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  if (nvecs <= 8) {
    mmvq_switch<scalar_t, qk, qi, block_q_t, vdr, vec_dot_q_cuda>(vx, vy, dst, ncols, nrows, nvecs, 1, stream);
    return;
  }
  const int full = nvecs / 8;
  const int rem = nvecs % 8;
  mmvq_switch<scalar_t, qk, qi, block_q_t, vdr, vec_dot_q_cuda>(vx, vy, dst, ncols, nrows, 8, full, stream);
  if (rem > 0) {
    const int base = full * 8;
    const size_t blocks_per_y = (size_t)((ncols + 512 - 1) / 512 * 512) / QK8_1;
    const block_q8_1* y_tail = (const block_q8_1*)vy + (size_t)base * blocks_per_y;
    mmvq_switch<scalar_t, qk, qi, block_q_t, vdr, vec_dot_q_cuda>(
        vx, y_tail, dst + (size_t)base * nrows, ncols, nrows, rem, 1, stream);
  }
}

#define MMVQ_LAUNCH(QK_, QI_, BLOCK_, VDR_, VECDOT_) \
  mmvq_dispatch<scalar_t, QK_, QI_, BLOCK_, VDR_, VECDOT_>(vx, vy, dst, ncols, nrows, nvecs, stream)

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
