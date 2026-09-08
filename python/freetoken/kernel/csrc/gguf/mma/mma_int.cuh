#pragma once
// ---------------------------------------------------------------------------------------------------------
// FreeToken MMQ-mma port, step S1: the INT-ONLY subset of upstream's tensor-core primitives.
//
// PORTED FROM  llama.cpp @ f7aadef09, ggml/src/ggml-cuda/mma.cuh:
//                :1-18    the header comment describing the tile contract (kept verbatim below)
//                :69-97   namespace + enum data_layout + get_input_data_layout
//                :98-274  the tile template and its DATA_LAYOUT_I_MAJOR specialisation
//                :777-784 load_generic
//                :785-799 load_ldmatrix for tile<8, 8, T>
//                :800-828 load_ldmatrix for tile<16, 4, T, dl>
//                :829-859 load_ldmatrix for tile<16, 8, T, dl>
//                :919-941 mma(tile<16,8,int>, tile<16,4,int>, tile<8,4,int>)   m16n8k16.s8.s8
//                :942-968 mma(tile<16,8,int>, tile<16,8,int>, tile<8,8,int>)   m16n8k32.s8.s8
//
// WHAT WAS DROPPED, and why (the plan's S1 says "INT-ONLY subset ... drop every half2/bf16/tf32/fp4
// tile and every AMD layout"; target is sm_89 only):
//   - every half2 / nv_bfloat162 / float / tf32 / fp4 tile specialisation and every mma over them
//     (upstream mma.cuh:275-719, :970-1456). MMQ's NVIDIA arm multiplies int8 into int32 and
//     nothing else; the FA kernels that need the float tiles are not part of this port.
//   - get_half2 / get_transposed / load_ldmatrix_trans (:719-775, :884-918): half2-only.
//   - the AMD_MFMA / AMD_WMMA / RDNA3 / RDNA4 / CDNA arms of every construct, and the
//     DATA_LAYOUT_*_MIRRORED / _J_MAJOR / _SCRAMBLED tile specialisations that exist only for them.
//     The enum values survive (a ported `constexpr data_layout input_layout = get_input_data_layout()`
//     must still name them) but only DATA_LAYOUT_I_MAJOR is ever instantiated here.
//   - the VOLTA_MMA_AVAILABLE arms (:151-180 and friends) and GGML_CUDA_MMA_NO_VOLTA_PERM.
//   - the CUDART_VERSION < 11080 shuffle fallback for ggml_cuda_movmatrix -- that helper is in
//     common_shim.cuh, and this tree builds against CUDA 12.x.
//
// WHAT WAS KEPT even though sm_89 does not need it: the Turing (__CUDA_ARCH__ < 800) arm inside
// each int8 mma, which decomposes m16n8k16/m16n8k32 into 2x/4x m8n8k16. It is dead code at sm_89
// (890 >= GGML_CUDA_CC_AMPERE 800, so the Ampere arm is the live one, per plan section 3 S1) but
// it costs nothing, and keeping it means this file is a literal subset of upstream rather than an
// edit of it.
//
// NAMESPACE: ftmma::ggml_cuda_mma. The inner name is upstream's, so code ported in S3/S4 can keep
// its `using namespace ggml_cuda_mma;` unchanged while everything stays enclosed in ftmma.
// ---------------------------------------------------------------------------------------------------------
//
// (upstream mma.cuh:2-17, verbatim -- this is the contract the tile indices implement)
// This file contains primitives that expose the tensor core PTX instructions for CUDA code.
// The primitives can be used in a similar way as the nvcuda::wmma interface but with a well-defined memory layout.
// The documentation for the PTX instructions can be found under:
//   https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-multiply-accumulate-operation-using-mma-instruction
//
// Like with nvcuda::wmma there are three types of matrix tiles: A, B, and C with A @ B = C.
// A is a row-major matrix with shape M x K.
// B is a column-major matrix with shape K x N.
// C is a column-major matrix with shape M x N.
// A, B, and C are represented using the same fundamental data type: a row-major matrix with I rows and J columns.
// Note that J is measured in physical 32 bit elements instead of logical elements.
// The methods get_i and get_j can be used to get the physical 32 bit index of the lth element of a thread within a tile.
// All matrix tiles have ne physical 32 bit elements per warp.
//
// As described in the PTX documentation, all pointers for load_ldmatrix must be to shared memory and aligned to 16 bytes.
// The API in this file also assumes that the pointers for load_generic are aligned to 16 bytes, unaligned pointers are considered undefined behavior.

#include "common_shim.cuh"

namespace ftmma {
namespace ggml_cuda_mma {

    // Some architectures like Volta or CDNA3 perform multiple matrix multiplications per warp in parallel,
    //     effectively the warp is being split into subgroups of threads that each perform a single mma instruction.
    // In those cases the data can be split in different ways across the warp.
    // Only DATA_LAYOUT_I_MAJOR is instantiated in this port (sm_89); the other enumerators are kept
    // so that ported code naming them still compiles.
    enum data_layout {
        // By default the data uses the I direction as its major dimension and the J direction as its minor dimension.
        // For the A/C matrices this means I major == row major, J major == column major.
        // For the B matrix this means I major == column major, J major == row major.
        // MIRRORED == Each data value is held exactly once per thread subgroup.
        DATA_LAYOUT_I_MAJOR           =  0, // Always used for Turing, Ampere, Ada Lovelace, consumer Blackwell, matrix A&B for RDNA4 and CDNA.
        DATA_LAYOUT_J_MAJOR           = 10, // Matrix C for CDNA and RDNA4, int and float matrix C for RDNA3.
        DATA_LAYOUT_I_MAJOR_MIRRORED  = 20, // Volta, matrix A&B for RDNA3.
        DATA_LAYOUT_J_MAJOR_MIRRORED  = 30,
        DATA_LAYOUT_I_MAJOR_SCRAMBLED = 40, // Scrambled matrix C for faster transposition (RDNA4/CDNA), convert to float to unscramble.
    };
    // Implemented mma combinations are:
    //   - (I_MAJOR, I_MAJOR)          -> I_MAJOR
    // (upstream also implements the two MIRRORED input combinations; those are AMD/Volta only.)

    static constexpr __device__ data_layout get_input_data_layout() {
        // Divergence: upstream returns DATA_LAYOUT_I_MAJOR_MIRRORED under RDNA3/VOLTA_MMA_AVAILABLE.
        // Neither exists in this port.
        return DATA_LAYOUT_I_MAJOR;
    }

    template <int I_, int J_, typename T, data_layout ds_=DATA_LAYOUT_I_MAJOR>
    struct tile {};

    // upstream mma.cuh:101-274, the `#else` (NVIDIA Turing-and-newer) arm only.
    template <int I_, int J_, typename T>
    struct tile<I_, J_, T, DATA_LAYOUT_I_MAJOR> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR;

        static constexpr int ne = I * J / FTMMA_WARP_SIZE;
        T x[ne] = {0};

        static constexpr __device__ bool supported() {
            if (I ==  8 && J ==  4) return true;
            if (I ==  8 && J ==  8) return true;
            if (I == 16 && J ==  8) return true;
            if (I == 16 && J == 16) return true;
            if (I == 32 && J ==  8) return true;
            return false;
        }

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (I == 8 && J == 4) {
                return threadIdx.x / 4;
            } else if constexpr (I == 8 && J == 8) {
                return threadIdx.x / 4;
            } else if constexpr (I == 16 && J == 8) {
                return ((l / 2) * 8) + (threadIdx.x / 4);
            } else if constexpr (I == 16 && J == 16) {
                return (((l / 2) % 2) * 8) + (threadIdx.x / 4);
            } else if constexpr (I == 32 && J == 8) {
                return tile<16, 8, T>::get_i(l); // Memory layout simply repeated with same pattern in i direction.
            } else {
                FTMMA_NO_DEVICE_CODE;
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 8 && J == 4) {
                return threadIdx.x % 4;
            } else if constexpr (I == 8 && J == 8) {
                return (l * 4) + (threadIdx.x % 4);
            } else if constexpr (I == 16 && J == 8) {
                return ((threadIdx.x % 4) * 2) + (l % 2);
            } else if constexpr (I == 16 && J == 16) {
                return ((l / 4) * 8) + ((threadIdx.x % 4) * 2) + (l % 2);
            } else if constexpr (I == 32 && J == 8) {
                return tile<16, 8, T>::get_j(l); // Memory layout simply repeated with same pattern in i direction.
            } else {
                FTMMA_NO_DEVICE_CODE;
                return -1;
            }
        }
    };

    // upstream mma.cuh:777-784
    template <int I, int J, typename T, data_layout dl>
    static __device__ __forceinline__ void load_generic(tile<I, J, T, dl> & t, const T * __restrict__ xs0, const int stride) {
#pragma unroll
        for (int l = 0; l < t.ne; ++l) {
            t.x[l] = xs0[t.get_i(l)*stride + t.get_j(l)];
        }
    }

    // upstream mma.cuh:785-799
    template <typename T>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<8, 8, T> & t, const T * __restrict__ xs0, const int stride) {
#ifdef FTMMA_TURING_MMA_AVAILABLE
        int * xi = (int *) t.x;
        const int * xs = (const int *) xs0 + (threadIdx.x % t.I) * stride + ((threadIdx.x / t.I) * (t.J / 2)) % t.J;
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
            : "=r"(xi[0]), "=r"(xi[1])
            : "l"(xs));
#else
        FTMMA_UNUSED_VARS(t, xs0, stride);
        FTMMA_NO_DEVICE_CODE;
#endif // FTMMA_TURING_MMA_AVAILABLE
    }

    // upstream mma.cuh:800-828, Turing arm only (the RDNA3/RDNA4/MFMA arms are dropped).
    template <typename T, data_layout dl>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<16, 4, T, dl> & t, const T * __restrict__ xs0, const int stride) {
#ifdef FTMMA_TURING_MMA_AVAILABLE
        int * xi = (int *) t.x;
        const int * xs = (const int *) xs0 + (threadIdx.x % t.I) * stride;
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
            : "=r"(xi[0]), "=r"(xi[1])
            : "l"(xs));
#else
        FTMMA_UNUSED_VARS(t, xs0, stride);
        FTMMA_NO_DEVICE_CODE;
#endif // FTMMA_TURING_MMA_AVAILABLE
    }

    // upstream mma.cuh:829-859, Turing arm only (the Volta/RDNA3/RDNA4/MFMA arms are dropped).
    template <typename T, data_layout dl>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<16, 8, T, dl> & t, const T * __restrict__ xs0, const int stride) {
#ifdef FTMMA_TURING_MMA_AVAILABLE
        int * xi = (int * ) t.x;
        const int * xs = (const int *) xs0 + (threadIdx.x % t.I) * stride + (threadIdx.x / t.I) * (t.J / 2);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
            : "l"(xs));
#else
        FTMMA_UNUSED_VARS(t, xs0, stride);
        FTMMA_NO_DEVICE_CODE;
#endif // FTMMA_TURING_MMA_AVAILABLE
    }

    // upstream mma.cuh:919-941 -- int8 x int8 -> int32, m16n8k16.
    // Used by the q8_0_16 / q6_K vec_dots (tile_A 16x4, tile_B 8x4).
    static __device__ __forceinline__ void mma(
            tile<16, 8, int> & D, const tile<16, 4, int> & A, const tile<8, 4, int> & B) {
#ifdef FTMMA_TURING_MMA_AVAILABLE
#if __CUDA_ARCH__ >= FTMMA_CUDA_CC_AMPERE
        asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
            : "+r"(D.x[0]), "+r"(D.x[1]), "+r"(D.x[2]), "+r"(D.x[3])
            : "r"(A.x[0]), "r"(A.x[1]), "r"(B.x[0]));
#else
        // On Turing m16n8k16 mma is not available, use 2x m8n8k16 mma instead:
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(D.x[0]), "+r"(D.x[1])
            : "r"(A.x[0]), "r"(B.x[0]));
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(D.x[2]), "+r"(D.x[3])
            : "r"(A.x[1]), "r"(B.x[0]));
#endif // __CUDA_ARCH__ >= FTMMA_CUDA_CC_AMPERE
#else
        FTMMA_UNUSED_VARS(D, A, B);
        FTMMA_NO_DEVICE_CODE;
#endif // FTMMA_TURING_MMA_AVAILABLE
    }

    // upstream mma.cuh:942-968 -- int8 x int8 -> int32, m16n8k32.
    // This is the instruction the whole port exists for (plan section 1.3): 660.6 TOPS dense on
    // Ada against 165.2 TFLOP/s for bf16-with-fp32-accumulate.
    static __device__ __forceinline__ void mma(
            tile<16, 8, int> & D, const tile<16, 8, int> & A, const tile<8, 8, int> & B) {
#ifdef FTMMA_TURING_MMA_AVAILABLE
#if __CUDA_ARCH__ >= FTMMA_CUDA_CC_AMPERE
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+r"(D.x[0]), "+r"(D.x[1]), "+r"(D.x[2]), "+r"(D.x[3])
            : "r"(A.x[0]), "r"(A.x[1]), "r"(A.x[2]), "r"(A.x[3]), "r"(B.x[0]), "r"(B.x[1]));
#else
        // On Turing m16n8k32 mma is not available, use 4x m8n8k16 mma instead:
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(D.x[0]), "+r"(D.x[1])
            : "r"(A.x[0]), "r"(B.x[0]));
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(D.x[2]), "+r"(D.x[3])
            : "r"(A.x[1]), "r"(B.x[0]));
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(D.x[0]), "+r"(D.x[1])
            : "r"(A.x[2]), "r"(B.x[1]));
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(D.x[2]), "+r"(D.x[3])
            : "r"(A.x[3]), "r"(B.x[1]));
#endif // __CUDA_ARCH__ >= FTMMA_CUDA_CC_AMPERE
#else
        FTMMA_UNUSED_VARS(D, A, B);
        FTMMA_NO_DEVICE_CODE;
#endif // FTMMA_TURING_MMA_AVAILABLE
    }

}  // namespace ggml_cuda_mma
}  // namespace ftmma
