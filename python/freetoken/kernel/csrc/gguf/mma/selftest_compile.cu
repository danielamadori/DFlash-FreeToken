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

// ---------------------------------------------------------------------------------------------------------
// S4 ADDITION (wiring step): every ported type header, in one translation unit. This is the plan's
// S4 CPU test. It fails at compile time if two per-type headers define the same helper at ftmma
// scope without an FTMMA_HAVE_* guard -- get_int_b2 / get_int_b4 / unpack_ksigns / kvalues_iq4nl /
// get_int_from_table_16 and the shared vec-dots ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma /
// _q8_0_16_q8_1_mma / _q8_1_q8_1_mma -- or if a ported type has no mmq_type_traits specialisation.
//
// NOTE: the vendored ggml-common.h is deliberately NOT included here. It cannot be: it needs
// c10::BFloat16 (ggml-common.h:966), so it only compiles inside the torch extension. The
// coexistence check that matters -- the port's local constants next to the vendored QR/QI macros
// in one TU -- is therefore the FULL build of gguf_kernel.cu, which includes ggml-common.h at :12
// and mma/mmq_entry.cuh at :26. That build is clean, with no macro-redefinition diagnostic.
// What this file adds on top is that the port's own constants are upstream's, not the vendored
// ones, which is the half of the decode-safety gate a compiler can check.
// ---------------------------------------------------------------------------------------------------------

#include "mmq_core.cuh"
#include "mmq_q4_K.cuh"
#include "mmq_iq2_s.cuh"
#include "mmq_iq2_xs.cuh"
#include "mmq_iq3_s.cuh"
#include "mmq_iq3_xxs.cuh"
#include "mmq_iq4_nl.cuh"
#include "mmq_iq4_xs.cuh"
#include "mmq_q3_K.cuh"
#include "mmq_q5_K.cuh"
#include "mmq_q6_K.cuh"
#include "mmq_q8_0.cuh"

// The four constants the vendored ggml-common.h gets "wrong" for this port (it has 8 where
// upstream has 2 or 4, because the vendored decode path walks the blocks differently). A port
// that ever picked up the vendored value would mis-address nibbles SILENTLY, so pin them here.
static_assert(ftmma::iq4_xs::FT_QR4_XS == 2, "IQ4_XS must use upstream's QR4_XS (vendored is 8)");
static_assert(ftmma::iq2_xs::FT_QR2_XS == 4, "IQ2_XS must use upstream's QR2_XS (vendored is 8)");
static_assert(ftmma::iq2_s::FT_QR2_S   == 4, "IQ2_S must use upstream's QR2_S (vendored is 8)");
static_assert(ftmma::FT_QR3_XXS        == 4, "IQ3_XXS must use upstream's QR3_XXS (vendored is 8)");
static_assert(ftmma::FT_QR3_S          == 4, "IQ3_S must use upstream's QR3_S (vendored has none)");
// and the ones that do agree, pinned so a vendored change is caught too
static_assert(ftmma::FT_QR3_K == 4 && ftmma::FT_QR4_K == 2 && ftmma::FT_QR5_K == 2, "");
static_assert(ftmma::FT_QR6_K == 2 && ftmma::FT_QR8_0 == 1 && ftmma::FT_QR4_NL == 2, "");

// Every ported type must actually have a traits specialisation: the primary template is declared
// and not defined, so naming sram_layout on an unported type is a hard error rather than a
// silently wrong kernel.
template <ftmma::ggml_type t> struct ftmma_selftest_traits_present {
    static constexpr int layout = (int) ftmma::mmq_type_traits<t>::sram_layout;
};
static_assert(ftmma_selftest_traits_present<ftmma::GGML_TYPE_Q3_K>::layout    >= 0, "");
static_assert(ftmma_selftest_traits_present<ftmma::GGML_TYPE_Q4_K>::layout    >= 0, "");
static_assert(ftmma_selftest_traits_present<ftmma::GGML_TYPE_Q5_K>::layout    >= 0, "");
static_assert(ftmma_selftest_traits_present<ftmma::GGML_TYPE_Q6_K>::layout    >= 0, "");
static_assert(ftmma_selftest_traits_present<ftmma::GGML_TYPE_Q8_0>::layout    >= 0, "");
static_assert(ftmma_selftest_traits_present<ftmma::GGML_TYPE_IQ2_XS>::layout  >= 0, "");
static_assert(ftmma_selftest_traits_present<ftmma::GGML_TYPE_IQ2_S>::layout   >= 0, "");
static_assert(ftmma_selftest_traits_present<ftmma::GGML_TYPE_IQ3_XXS>::layout >= 0, "");
static_assert(ftmma_selftest_traits_present<ftmma::GGML_TYPE_IQ3_S>::layout   >= 0, "");
static_assert(ftmma_selftest_traits_present<ftmma::GGML_TYPE_IQ4_NL>::layout  >= 0, "");
static_assert(ftmma_selftest_traits_present<ftmma::GGML_TYPE_IQ4_XS>::layout  >= 0, "");

// The shared-memory budget is NOT asserted here: mmq_get_nbytes_shared is a host function, and
// making it constexpr would edit mmq_core.cuh. It was computed by hand at wiring time instead:
// I*stride*4 + J*4 + PAD(J*144, 1024), I=128, stride 76 (Q8_0/Q8_1/Q6_K layouts) or 84 (Q3_K),
// which is 40992..57856 B for the stride-76 types and 45088..61952 B for the stride-84 types
// (Q3_K, IQ2_XS, IQ2_S) over J in {8,16,32,64,128} -- all well inside Ada's 101376 B opt-in, and
// mul_mat_q_switch_J re-checks against the device's smpbo at runtime anyway.
