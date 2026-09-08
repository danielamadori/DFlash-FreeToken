// ---------------------------------------------------------------------------------------------------------
// FreeToken MMQ-mma port, step S1: standalone compile check for the shim and the int mma primitives.
//
// The variant build script compiles ONLY gguf_kernel.cu (build_variant.sh: sources=[gguf_kernel.cu]),
// so this translation unit is never part of the extension. It exists to be compiled on its own:
//
//   nvcc -c -O3 -std=c++17 -arch=sm_89 --expt-relaxed-constexpr mma/selftest_compile.cu -o /dev/null
//
// which is exactly what tests/kernel/test_mma_headers_compile.py runs (no GPU needed -- nvcc never
// opens a CUDA context). It instantiates every tile shape and every primitive the MMQ port will
// use, so a signature that does not compile fails here rather than 1200 lines later.
// ---------------------------------------------------------------------------------------------------------

#include "common_shim.cuh"
#include "mma_int.cuh"

using namespace ftmma;
using namespace ftmma::ggml_cuda_mma;

// ---- tile shapes the port instantiates (mmq-vec-dot.cuh's NVIDIA arms) ----
typedef tile<16, 8, int> tile_A_k32;   // A for mma.m16n8k32, and tile_C everywhere
typedef tile< 8, 8, int> tile_B_k32;   // B for mma.m16n8k32
typedef tile<16, 4, int> tile_A_k16;   // A for mma.m16n8k16 (q8_0_16 / q6_K)
typedef tile< 8, 4, int> tile_B_k16;   // B for mma.m16n8k16
typedef tile<32, 8, int> tile_C_wide;  // the repeated-in-I layout

static_assert(tile_A_k32::ne == 4, "tile<16,8,int> must hold 4 ints per thread");
static_assert(tile_B_k32::ne == 2, "tile<8,8,int> must hold 2 ints per thread");
static_assert(tile_A_k16::ne == 2, "tile<16,4,int> must hold 2 ints per thread");
static_assert(tile_B_k16::ne == 1, "tile<8,4,int> must hold 1 int per thread");
static_assert(tile_C_wide::ne == 8, "tile<32,8,int> must hold 8 ints per thread");
static_assert(tile_A_k32::dl == DATA_LAYOUT_I_MAJOR, "default data layout must be I-major");
static_assert(get_input_data_layout() == DATA_LAYOUT_I_MAJOR, "sm_89 input layout must be I-major");

// The cast in mmq-vec-dot.cuh:563/783 -- `((tile_A_8 *) A[n])[k]` over an array of tile<16,4,int>
// -- is only valid if the two tiles have the same per-thread footprint per unit of J.
static_assert(sizeof(tile_A_k32) == 2 * sizeof(tile_A_k16), "tile<16,8,int> must be 2x tile<16,4,int>");

// ---- the qk traits, for the 11 types of this checkpoint (plan section 1.1) ----
static_assert(ftmma_type_traits<GGML_TYPE_Q8_0>::qk    ==  32, "");
static_assert(ftmma_type_traits<GGML_TYPE_IQ4_NL>::qk  ==  32, "");
static_assert(ftmma_type_traits<GGML_TYPE_Q3_K>::qk    == 256, "");
static_assert(ftmma_type_traits<GGML_TYPE_Q4_K>::qk    == 256, "");
static_assert(ftmma_type_traits<GGML_TYPE_Q5_K>::qk    == 256, "");
static_assert(ftmma_type_traits<GGML_TYPE_Q6_K>::qk    == 256, "");
static_assert(ftmma_type_traits<GGML_TYPE_IQ2_XS>::qk  == 256, "");
static_assert(ftmma_type_traits<GGML_TYPE_IQ2_S>::qk   == 256, "");
static_assert(ftmma_type_traits<GGML_TYPE_IQ3_XXS>::qk == 256, "");
static_assert(ftmma_type_traits<GGML_TYPE_IQ3_S>::qk   == 256, "");
static_assert(ftmma_type_traits<GGML_TYPE_IQ4_XS>::qk  == 256, "");
// the upstream spelling resolves to the same traits
static_assert(ggml_cuda_type_traits<GGML_TYPE_Q4_K>::qk == 256, "");
// the type codes are the ones the vendored dispatch already passes as `int64_t type`
static_assert((int) GGML_TYPE_Q4_K == 12 && (int) GGML_TYPE_IQ4_XS == 23, "ggml type codes");

// ---- a kernel that touches every primitive ----
__global__ void ftmma_selftest_kernel(const int * __restrict__ xs, int * __restrict__ dst,
                                      const int stride, const uint3 fd) {
    __shared__ int sram[16 * 64];
    for (int i = threadIdx.x; i < 16 * 64; i += blockDim.x) {
        sram[i] = xs[i];
    }
    __syncthreads();

    tile_A_k32 A32;
    tile_B_k32 B32;
    tile_A_k16 A16;
    tile_B_k16 B16;
    tile_A_k32 C;

    load_ldmatrix(A32, sram, stride);
    load_ldmatrix(B32, sram, stride);
    load_ldmatrix(A16, sram, stride);
    load_generic (B16, sram, stride);
    load_generic (B32, sram, stride);

    mma(C, A32, B32);
    mma(C, A16, B16);

    int acc = 0;
#pragma unroll
    for (int l = 0; l < C.ne; ++l) {
        acc += C.x[l] * (C.get_i(l) + C.get_j(l));
    }
    acc += ggml_cuda_movmatrix(acc);
    acc += (int) fastdiv((uint32_t) threadIdx.x, fd);
    acc += (int) fastmodulo((uint32_t) threadIdx.x, fd);

    int4 tmp;
    ggml_cuda_memcpy_1<sizeof(int4)>(&tmp, sram);
    acc += tmp.x + tmp.y + tmp.z + tmp.w;

    dst[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

// ---- a host entry point that touches the host side of the shim ----
void ftmma_selftest_launch(const int * xs, int * dst, const int stride, cudaStream_t stream) {
    const int id  = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    GGML_ASSERT(ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_AMPERE);
    GGML_ASSERT(turing_mma_available(cc));

    const uint3 fd = init_fastdiv_values(GGML_PAD(stride, 128) / 20);

    CUDA_SET_SHARED_MEMORY_LIMIT(ftmma_selftest_kernel, (int) smpbo);
    ftmma_selftest_kernel<<<nsm, WARP_SIZE, 0, stream>>>(xs, dst, stride, fd);
    CUDA_CHECK(cudaGetLastError());
}
