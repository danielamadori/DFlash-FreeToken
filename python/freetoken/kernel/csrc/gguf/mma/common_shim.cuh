#pragma once
// ---------------------------------------------------------------------------------------------------------
// FreeToken MMQ-mma port, step S1: the "common.cuh" shim.
//
// PORTED FROM  llama.cpp @ f7aadef09, ggml/src/ggml-cuda/common.cuh
//              (WARP_SIZE :46, the GGML_CUDA_CC_* block :50-110, CUDA_SET_SHARED_MEMORY_LIMIT :230-245,
//               the *_MMA_AVAILABLE feature macros :274-292, no_device_code/NO_DEVICE_CODE :400-421,
//               ggml_cuda_memcpy_1 :785-820, init_fastdiv_values/fastdiv/fastmodulo :909-946,
//               ggml_cuda_type_traits :980-1135, ggml_cuda_device_info :1137-1165)
//              plus GGML_PAD / GGML_ABORT / GGML_ASSERT / GGML_UNUSED / GGML_UNUSED_VARS from
//              llama.cpp ggml/include/ggml.h :258-288.
//
// WHY THIS FILE EXISTS AT ALL (the decode-safety gate of docs/plans/prefill-mmq-mma-plan.md):
//   The vendored csrc/gguf/ggml-common.h defines QR/QI constants for the I-quants with DIFFERENT
//   values than upstream (QR4_XS 8 vs 2, QR2_XXS/QR2_XS/QR2_S/QR3_XXS 8 vs 4, and QR3_S/QI3_S are
//   absent entirely). The decode path -- mmvq.cuh, mmvq_planar*.cuh, vecdotq.cuh -- reads the
//   VENDORED values and is correct with them. Sharing a header between the two worlds would
//   silently mis-address nibbles rather than fail to compile. So this port is a closed world:
//   everything lives under csrc/gguf/mma/ in `namespace ftmma`, includes NOTHING from
//   ggml-common.h / vecdotq.cuh, and carries its own constants.
//
// DIVERGENCES FROM UPSTREAM, and why:
//   1. NAMESPACE. Upstream puts these at global scope. Here every declaration is inside
//      `namespace ftmma`, so `ftmma::fastdiv` cannot be confused with anything the vendored tree
//      might grow later. The int mma primitives live in `namespace ftmma::ggml_cuda_mma`, so code
//      ported from upstream keeps its `using namespace ggml_cuda_mma;` verbatim.
//   2. MACROS CANNOT BE NAMESPACED. Every macro below is defined FTMMA_-prefixed and then aliased
//      to its upstream spelling under an `#ifndef` guard, so a port stays byte-for-byte readable
//      and an existing definition in the vendored tree always wins. Today only WARP_SIZE is
//      already defined (dispatch.h:10, value 32 -- identical to upstream common.cuh:46); a
//      static_assert below pins that.
//   3. NO ggml_backend / ggml_tensor / ggml_cuda_pool. This port allocates with torch::empty
//      (the vendored dispatch already does, gguf_kernel.cu:190-193), so the pool types are gone.
//   4. ggml_cuda_info() here is a small lazily-filled cache over cudaGetDeviceProperties instead
//      of the backend-owned singleton. Same field names (cc, nsm, smpb, smpbo, warp_size) so
//      `ggml_cuda_info().devices[id].cc` ports verbatim.
//   5. TARGET IS sm_89 ONLY. The AMD (HIP/RDNA/CDNA), Moore Threads and Volta arms of every
//      upstream construct are dropped. GGML_CUDA_CC_IS_AMD/_MTHREADS are still defined (as
//      always-false on the cc values this port can see) so that ported host dispatch code that
//      branches on them compiles unchanged and folds away.
//   6. ggml_cuda_highest_compiled_arch() is the identity. Upstream derives it from
//      __CUDA_ARCH_LIST__; we compile for exactly one arch, so there is nothing to derive.
//
// This header must compile standalone: `nvcc -c -arch=sm_89 -std=c++17` on a TU that includes
// only this file. It therefore depends on the CUDA runtime and the C++ standard library only --
// no torch, no ATen, no ggml.
// ---------------------------------------------------------------------------------------------------------

#include <cuda_runtime.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

// ---------------------------------------------------------------------------------------------------------
// 1. Warp size.  upstream common.cuh:46
// The vendored dispatch.h:10 already defines WARP_SIZE as 32, the same value upstream uses; we
// never redefine it, we only fill it in when this header is compiled standalone, and we pin it.
// ---------------------------------------------------------------------------------------------------------

#define FTMMA_WARP_SIZE 32

#ifndef WARP_SIZE
#define WARP_SIZE FTMMA_WARP_SIZE
#endif  // WARP_SIZE

static_assert(WARP_SIZE == FTMMA_WARP_SIZE,
              "ftmma: the enclosing translation unit defines WARP_SIZE != 32; the mma tile layouts in "
              "mma_int.cuh assume a 32-thread warp (I*J/32 elements per tile).");

// ---------------------------------------------------------------------------------------------------------
// 2. Compute-capability constants and predicates.  upstream common.cuh:50-110
// cc is encoded as 100*major + 10*minor for NVIDIA, so an RTX 4090 (8.9) is 890.
// ---------------------------------------------------------------------------------------------------------

#define FTMMA_CUDA_CC_PASCAL          600
#define FTMMA_CUDA_CC_DP4A            610
#define FTMMA_CUDA_CC_VOLTA           700
#define FTMMA_CUDA_CC_TURING          750
#define FTMMA_CUDA_CC_AMPERE          800
#define FTMMA_CUDA_CC_ADA_LOVELACE    890
#define FTMMA_CUDA_CC_HOPPER          900
#define FTMMA_CUDA_CC_BLACKWELL      1200
#define FTMMA_CUDA_CC_OFFSET_AMD      0x1000000
#define FTMMA_CUDA_CC_OFFSET_MTHREADS 0x0100000

#define FTMMA_CUDA_CC_IS_NVIDIA(cc)   ((cc) < FTMMA_CUDA_CC_OFFSET_MTHREADS)
#define FTMMA_CUDA_CC_IS_MTHREADS(cc) ((cc) >= FTMMA_CUDA_CC_OFFSET_MTHREADS && (cc) < FTMMA_CUDA_CC_OFFSET_AMD)
#define FTMMA_CUDA_CC_IS_AMD(cc)      ((cc) >= FTMMA_CUDA_CC_OFFSET_AMD)

// Divergence 5: this port never runs on AMD, but ported host dispatch code branches on these.
// Defining them keeps that code compiling verbatim; every branch folds away at -O3.
#define FTMMA_CUDA_CC_IS_RDNA(cc)     (false && (cc))
#define FTMMA_CUDA_CC_IS_RDNA1(cc)    (false && (cc))
#define FTMMA_CUDA_CC_IS_RDNA2(cc)    (false && (cc))
#define FTMMA_CUDA_CC_IS_RDNA3(cc)    (false && (cc))
#define FTMMA_CUDA_CC_IS_RDNA3_0(cc)  (false && (cc))
#define FTMMA_CUDA_CC_IS_RDNA3_5(cc)  (false && (cc))
#define FTMMA_CUDA_CC_IS_RDNA4(cc)    (false && (cc))
#define FTMMA_CUDA_CC_IS_GCN(cc)      (false && (cc))
#define FTMMA_CUDA_CC_IS_CDNA(cc)     (false && (cc))
#define FTMMA_CUDA_CC_IS_CDNA1(cc)    (false && (cc))
#define FTMMA_CUDA_CC_IS_CDNA2(cc)    (false && (cc))
#define FTMMA_CUDA_CC_IS_CDNA3(cc)    (false && (cc))
#define FTMMA_CUDA_CC_IS_CDNA4(cc)    (false && (cc))

#define FTMMA_CUDA_MAX_DEVICES 16

// upstream common.cuh:176 -- the padding of the last row of a quantized matrix. Every in_features
// of this checkpoint (1024/5120/6144/10240/12288/17408) is already a multiple of 512, so this is a
// no-op here; it is carried because the ported launcher computes with it.
#define FTMMA_MATRIX_ROW_PADDING 512

// ---------------------------------------------------------------------------------------------------------
// 3. Device-code feature macros.  upstream common.cuh:274-292
// On sm_89 both TURING_MMA_AVAILABLE and AMPERE_MMA_AVAILABLE are defined; on a host pass neither
// is (__CUDA_ARCH__ is undefined), which is what makes the NO_DEVICE_CODE arms compile.
// ---------------------------------------------------------------------------------------------------------

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= FTMMA_CUDA_CC_TURING
#define FTMMA_TURING_MMA_AVAILABLE
#endif  // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= FTMMA_CUDA_CC_TURING

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= FTMMA_CUDA_CC_AMPERE
#define FTMMA_AMPERE_MMA_AVAILABLE
#define FTMMA_CP_ASYNC_AVAILABLE
#endif  // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= FTMMA_CUDA_CC_AMPERE

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= FTMMA_CUDA_CC_DP4A
#define FTMMA_FP16_AVAILABLE
#endif  // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= FTMMA_CUDA_CC_DP4A

// ---------------------------------------------------------------------------------------------------------
// 4. GGML_UNUSED / GGML_UNUSED_VARS / GGML_PAD / GGML_ABORT / GGML_ASSERT.
//    upstream ggml/include/ggml.h:258-288
// ---------------------------------------------------------------------------------------------------------

#define FTMMA_UNUSED(x) (void) (x)

namespace ftmma {
template <typename... Args>
__host__ __device__ constexpr inline void unused_vars_impl(Args &&...) noexcept {}
}  // namespace ftmma

#define FTMMA_UNUSED_VARS(...) ::ftmma::unused_vars_impl(__VA_ARGS__)

#define FTMMA_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))

namespace ftmma {
// Divergence 3: upstream calls ggml_abort(), which lives in libggml. Here it is a local
// fprintf+abort so that this header has no link-time dependency at all.
[[noreturn]] static inline void abort_impl(const char * file, int line, const char * fmt, ...) {
    fflush(stdout);
    fprintf(stderr, "ftmma: %s:%d: ", file, line);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
    fflush(stderr);
    abort();
}
}  // namespace ftmma

#define FTMMA_ABORT(...)  ::ftmma::abort_impl(__FILE__, __LINE__, __VA_ARGS__)
#define FTMMA_ASSERT(x)                                       \
    do {                                                      \
        if (!(x)) {                                           \
            FTMMA_ABORT("FTMMA_ASSERT(%s) failed", #x);       \
        }                                                     \
    } while (0)

#if defined(__CUDACC__) && CUDART_VERSION >= 11010
#define FTMMA_CUDA_ASSUME(x) __builtin_assume(x)
#else
#define FTMMA_CUDA_ASSUME(x)
#endif

// upstream common.cuh:1644 -- PDL (programmatic dependent launch) is Hopper-only, so on sm_89
// __restrict__ is always the right expansion.
#define FTMMA_CUDA_RESTRICT __restrict__

// ---------------------------------------------------------------------------------------------------------
// 5. NO_DEVICE_CODE.  upstream common.cuh:400-421
// Traps at runtime if a kernel arm that has no implementation for the compiled arch is reached.
// ---------------------------------------------------------------------------------------------------------

namespace ftmma {
[[noreturn]] static __device__ void no_device_code(
        const char * file_name, const int line, const char * function_name, const int arch) {
    printf("%s:%d: ERROR: CUDA kernel %s has no device code compatible with CUDA arch %d. "
           "The ftmma MMQ port is compiled for sm_89 only.\n",
           file_name, line, function_name, arch);
    __trap();
}
}  // namespace ftmma

#ifdef __CUDA_ARCH__
#define FTMMA_NO_DEVICE_CODE ::ftmma::no_device_code(__FILE__, __LINE__, __FUNCTION__, __CUDA_ARCH__)
#else
#define FTMMA_NO_DEVICE_CODE  // not valid in host code
#endif  // __CUDA_ARCH__

// ---------------------------------------------------------------------------------------------------------
// 6. CUDA_CHECK / CUDA_SET_SHARED_MEMORY_LIMIT.  upstream common.cuh:186-245
// The mma MMQ kernels ask for more than the 48 KiB default of dynamic shared memory, so the
// launcher must opt in once per device with cudaFuncSetAttribute.
// ---------------------------------------------------------------------------------------------------------

namespace ftmma {
static inline int cuda_get_device() {
    int id = 0;
    cudaGetDevice(&id);
    return id;
}
}  // namespace ftmma

#define FTMMA_CUDA_CHECK(err)                                                                    \
    do {                                                                                         \
        const cudaError_t err_ = (err);                                                          \
        if (err_ != cudaSuccess) {                                                               \
            FTMMA_ABORT("CUDA error: %s (%s)", cudaGetErrorString(err_), #err);                  \
        }                                                                                        \
    } while (0)

#define FTMMA_CUDA_SET_SHARED_MEMORY_LIMIT(kernel, nbytes)                                             \
    do {                                                                                               \
        static bool shared_memory_limit_raised[FTMMA_CUDA_MAX_DEVICES] = { false };                    \
        const int   id                                                 = ::ftmma::cuda_get_device();   \
        if (!shared_memory_limit_raised[id]) {                                                         \
            FTMMA_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, \
                                                  nbytes));                                            \
            shared_memory_limit_raised[id] = true;                                                     \
        }                                                                                              \
    } while (0)

// ---------------------------------------------------------------------------------------------------------
// 7. Upstream-spelled aliases (divergence 2). Each is guarded, so if the enclosing TU (or a future
// vendored header) already defines the name, THAT definition wins and we never redefine it --
// which is what turns an accidental collision into a compile-visible difference instead of a
// silent one. If a guard ever fires you will see it as a semantic change, not a warning, so the
// per-type CPU compile test in the plan (S4) checks for macro-redefinition warnings too.
// ---------------------------------------------------------------------------------------------------------

#ifndef GGML_CUDA_CC_PASCAL
#define GGML_CUDA_CC_PASCAL          FTMMA_CUDA_CC_PASCAL
#define GGML_CUDA_CC_DP4A            FTMMA_CUDA_CC_DP4A
#define GGML_CUDA_CC_VOLTA           FTMMA_CUDA_CC_VOLTA
#define GGML_CUDA_CC_TURING          FTMMA_CUDA_CC_TURING
#define GGML_CUDA_CC_AMPERE          FTMMA_CUDA_CC_AMPERE
#define GGML_CUDA_CC_ADA_LOVELACE    FTMMA_CUDA_CC_ADA_LOVELACE
#define GGML_CUDA_CC_HOPPER          FTMMA_CUDA_CC_HOPPER
#define GGML_CUDA_CC_BLACKWELL       FTMMA_CUDA_CC_BLACKWELL
#define GGML_CUDA_CC_OFFSET_AMD      FTMMA_CUDA_CC_OFFSET_AMD
#define GGML_CUDA_CC_OFFSET_MTHREADS FTMMA_CUDA_CC_OFFSET_MTHREADS
#define GGML_CUDA_CC_IS_NVIDIA(cc)   FTMMA_CUDA_CC_IS_NVIDIA(cc)
#define GGML_CUDA_CC_IS_MTHREADS(cc) FTMMA_CUDA_CC_IS_MTHREADS(cc)
#define GGML_CUDA_CC_IS_AMD(cc)      FTMMA_CUDA_CC_IS_AMD(cc)
#define GGML_CUDA_CC_IS_RDNA(cc)     FTMMA_CUDA_CC_IS_RDNA(cc)
#define GGML_CUDA_CC_IS_RDNA1(cc)    FTMMA_CUDA_CC_IS_RDNA1(cc)
#define GGML_CUDA_CC_IS_RDNA2(cc)    FTMMA_CUDA_CC_IS_RDNA2(cc)
#define GGML_CUDA_CC_IS_RDNA3(cc)    FTMMA_CUDA_CC_IS_RDNA3(cc)
#define GGML_CUDA_CC_IS_RDNA3_0(cc)  FTMMA_CUDA_CC_IS_RDNA3_0(cc)
#define GGML_CUDA_CC_IS_RDNA3_5(cc)  FTMMA_CUDA_CC_IS_RDNA3_5(cc)
#define GGML_CUDA_CC_IS_RDNA4(cc)    FTMMA_CUDA_CC_IS_RDNA4(cc)
#define GGML_CUDA_CC_IS_GCN(cc)      FTMMA_CUDA_CC_IS_GCN(cc)
#define GGML_CUDA_CC_IS_CDNA(cc)     FTMMA_CUDA_CC_IS_CDNA(cc)
#define GGML_CUDA_CC_IS_CDNA1(cc)    FTMMA_CUDA_CC_IS_CDNA1(cc)
#define GGML_CUDA_CC_IS_CDNA2(cc)    FTMMA_CUDA_CC_IS_CDNA2(cc)
#define GGML_CUDA_CC_IS_CDNA3(cc)    FTMMA_CUDA_CC_IS_CDNA3(cc)
#define GGML_CUDA_CC_IS_CDNA4(cc)    FTMMA_CUDA_CC_IS_CDNA4(cc)
#endif  // GGML_CUDA_CC_PASCAL

// GGML_CUDA_MAX_DEVICES is deliberately NOT aliased: it is an object-like macro whose name reads
// like an ordinary constant, it is needed only inside this file's own macros, and a sibling header
// that wrote `static constexpr int GGML_CUDA_MAX_DEVICES = 16;` would be macro-clobbered into
// `static constexpr int 16 = 16;` purely by include order. Use FTMMA_CUDA_MAX_DEVICES.
//
// MATRIX_ROW_PADDING has exactly the same hazard but IS aliased, because upstream quantize.cu
// spells it that way and mma/quantize_mmq.cuh (S2) consumes it as a macro. A sibling that wants a
// constant of that name must use FTMMA_MATRIX_ROW_PADDING instead of declaring its own.
#ifndef MATRIX_ROW_PADDING
#define MATRIX_ROW_PADDING FTMMA_MATRIX_ROW_PADDING
#endif
#ifndef GGML_UNUSED
#define GGML_UNUSED(x) FTMMA_UNUSED(x)
#endif
#ifndef GGML_UNUSED_VARS
#define GGML_UNUSED_VARS(...) FTMMA_UNUSED_VARS(__VA_ARGS__)
#endif
#ifndef GGML_PAD
#define GGML_PAD(x, n) FTMMA_PAD(x, n)
#endif
#ifndef GGML_ABORT
#define GGML_ABORT(...) FTMMA_ABORT(__VA_ARGS__)
#endif
#ifndef GGML_ASSERT
#define GGML_ASSERT(x) FTMMA_ASSERT(x)
#endif
#ifndef GGML_CUDA_ASSUME
#define GGML_CUDA_ASSUME(x) FTMMA_CUDA_ASSUME(x)
#endif
#ifndef GGML_CUDA_RESTRICT
#define GGML_CUDA_RESTRICT FTMMA_CUDA_RESTRICT
#endif
#ifndef NO_DEVICE_CODE
#define NO_DEVICE_CODE FTMMA_NO_DEVICE_CODE
#endif
#ifndef CUDA_CHECK
#define CUDA_CHECK(err) FTMMA_CUDA_CHECK(err)
#endif
#ifndef CUDA_SET_SHARED_MEMORY_LIMIT
#define CUDA_SET_SHARED_MEMORY_LIMIT(kernel, nbytes) FTMMA_CUDA_SET_SHARED_MEMORY_LIMIT(kernel, nbytes)
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= FTMMA_CUDA_CC_TURING && !defined(TURING_MMA_AVAILABLE)
#define TURING_MMA_AVAILABLE
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= FTMMA_CUDA_CC_AMPERE && !defined(AMPERE_MMA_AVAILABLE)
#define AMPERE_MMA_AVAILABLE
#endif

// ===========================================================================================================
namespace ftmma {
// ===========================================================================================================

// -------------------------------------------------------------------------------------------------------
// 8. Device-property cache.  upstream common.cuh:1137-1165 + ggml-cuda.cu's ggml_cuda_init().
// Field names match upstream so `ggml_cuda_info().devices[id].cc` ports verbatim.
// Divergence 4: filled lazily from cudaGetDeviceProperties on first use instead of at backend init.
// NOTE the first call creates a CUDA context; the CPU-only compile test never calls it.
// -------------------------------------------------------------------------------------------------------

struct cuda_device_info {
    int    cc        = 0;  // compute capability, 100*major + 10*minor
    int    nsm       = 0;  // number of streaming multiprocessors
    size_t smpb      = 0;  // max. shared memory per block
    size_t smpbo     = 0;  // max. shared memory per block, with opt-in
    int    warp_size = 0;
};

struct cuda_device_info_table {
    int              device_count = 0;
    cuda_device_info devices[FTMMA_CUDA_MAX_DEVICES] = {};
};

static inline const cuda_device_info_table & cuda_info() {
    static cuda_device_info_table info = [] {
        cuda_device_info_table t;
        cudaError_t err = cudaGetDeviceCount(&t.device_count);
        if (err != cudaSuccess || t.device_count <= 0) {
            t.device_count = 0;
            return t;
        }
        if (t.device_count > FTMMA_CUDA_MAX_DEVICES) {
            t.device_count = FTMMA_CUDA_MAX_DEVICES;
        }
        for (int id = 0; id < t.device_count; ++id) {
            cudaDeviceProp prop;
            FTMMA_CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
            t.devices[id].cc        = 100 * prop.major + 10 * prop.minor;
            t.devices[id].nsm       = prop.multiProcessorCount;
            t.devices[id].smpb      = prop.sharedMemPerBlock;
            t.devices[id].warp_size = prop.warpSize;

            int smpbo = 0;
            FTMMA_CUDA_CHECK(cudaDeviceGetAttribute(&smpbo, cudaDevAttrMaxSharedMemoryPerBlockOptin, id));
            t.devices[id].smpbo = (size_t) smpbo;
        }
        return t;
    }();
    return info;
}

// Upstream spellings, so the ported launcher reads unchanged.
static inline const cuda_device_info_table & ggml_cuda_info() { return cuda_info(); }
static inline int  ggml_cuda_get_device()                     { return cuda_get_device(); }
static inline void ggml_cuda_set_device(const int device)     { FTMMA_CUDA_CHECK(cudaSetDevice(device)); }

// Divergence 6: identity. We compile for exactly one arch (sm_89), so there is no arch list to
// search; upstream's version picks the highest arch in __CUDA_ARCH_LIST__ that is <= arch.
static inline int ggml_cuda_highest_compiled_arch(const int arch) { return arch; }

static inline int  ggml_cuda_get_physical_warp_size()            { return FTMMA_WARP_SIZE; }
static inline bool turing_mma_available(const int cc)  { return FTMMA_CUDA_CC_IS_NVIDIA(cc) && cc >= FTMMA_CUDA_CC_TURING; }
static inline bool ampere_mma_available(const int cc)  { return FTMMA_CUDA_CC_IS_NVIDIA(cc) && cc >= FTMMA_CUDA_CC_AMPERE; }
static inline bool amd_mfma_available(const int)       { return false; }  // divergence 5
static inline bool amd_wmma_available(const int)       { return false; }  // divergence 5
static inline bool volta_mma_available(const int cc)   { return FTMMA_CUDA_CC_IS_NVIDIA(cc) && cc == FTMMA_CUDA_CC_VOLTA; }
static inline bool blackwell_mma_available(const int cc) {
    return FTMMA_CUDA_CC_IS_NVIDIA(cc) && cc >= FTMMA_CUDA_CC_BLACKWELL;
}

// -------------------------------------------------------------------------------------------------------
// 9. fastdiv / fastmodulo.  upstream common.cuh:903-946
// See https://gmplib.org/~tege/divcnst-pldi94.pdf figure 4.1. Precompute mp (m' in the paper) and
// L such that n/d == (mulhi(n, mp) + n) >> L. The divisor is packed into .z so that fastmodulo
// needs a single uint3.
// -------------------------------------------------------------------------------------------------------

static inline uint3 init_fastdiv_values(uint64_t d_64) {
    FTMMA_ASSERT(d_64 != 0);
    FTMMA_ASSERT(d_64 <= std::numeric_limits<uint32_t>::max());

    const uint32_t d = (uint32_t) d_64;

    // compute L = ceil(log2(d));
    uint32_t L = 0;
    while (L < 32 && (uint32_t{ 1 } << L) < d) {
        L++;
    }

    const uint32_t mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
    // pack divisor as well to reduce error surface
    return make_uint3(mp, L, d);
}

static __device__ __forceinline__ uint32_t fastdiv(uint32_t n, const uint3 fastdiv_values) {
    // expects fastdiv_values to contain <mp, L, divisor> in <x, y, z>
    // fastdiv_values.z is unused and optimized away by the compiler.
    // Compute high 32 bits of n * mp
    const uint32_t hi = __umulhi(n, fastdiv_values.x);
    // add n, apply bit shift
    return (hi + n) >> fastdiv_values.y;
}

static __device__ __forceinline__ uint32_t fastmodulo(uint32_t n, const uint3 fastdiv_values) {
    // expects fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    return n - fastdiv(n, fastdiv_values) * fastdiv_values.z;
}

// Calculate both division and modulo at once, returns <n/divisor, n%divisor>
static __device__ __forceinline__ uint2 fast_div_modulo(uint32_t n, const uint3 fastdiv_values) {
    const uint32_t div_val = fastdiv(n, fastdiv_values);
    const uint32_t mod_val = n - div_val * fastdiv_values.z;
    return make_uint2(div_val, mod_val);
}

// -------------------------------------------------------------------------------------------------------
// 10. ggml_cuda_memcpy_1.  upstream common.cuh:383-395 and :785-820
// Aligned 8/16-byte transfers between registers and SRAM/VRAM. The tile loaders in S3/S4 use this
// for every non-ldmatrix path; it is carried here because it belongs to common.cuh, not mma.cuh.
// -------------------------------------------------------------------------------------------------------

static constexpr __device__ int ggml_cuda_get_max_cpy_bytes() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= FTMMA_CUDA_CC_VOLTA
    return 16;
#else
    return 8;
#endif
}

// Important: do not use this function if dst and src both point at registers.
//     Due to the strict aliasing rule the compiler can do incorrect optimizations if src and dst
//     have different types. The function is intended for copies between registers and SRAM/VRAM to
//     make the compiler emit the right instructions.
template <int nbytes, int alignment = 0>
static __device__ __forceinline__ void ggml_cuda_memcpy_1(void * __restrict__ dst, const void * __restrict__ src) {
    static_assert(nbytes <= ggml_cuda_get_max_cpy_bytes() || alignment == 0,
                  "You are misusing the alignment parameter for ggml_cuda_memcpy_1. "
                  "Call ggml_cuda_memcpy_1 in a loop instead.");
    if constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr int nb_per_cpy = alignment == 0 ? nbytes : alignment;

#pragma unroll
    for (int i = 0; i < nbytes / nb_per_cpy; ++i) {
        if constexpr (nb_per_cpy == 1) {
            ((char *) dst)[i] = ((const char *) src)[i];
        } else if constexpr (nb_per_cpy == 2) {
            ((short *) dst)[i] = ((const short *) src)[i];
        } else if constexpr (nb_per_cpy == 4) {
            ((int *) dst)[i] = ((const int *) src)[i];
        } else if constexpr (nb_per_cpy == 8) {
            ((int2 *) dst)[i] = ((const int2 *) src)[i];
        } else if constexpr (nb_per_cpy == 16) {
            ((int4 *) dst)[i] = ((const int4 *) src)[i];
        } else {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
}

// -------------------------------------------------------------------------------------------------------
// 11. ggml_cuda_movmatrix.  upstream mma.cuh:26-67
// Transposes an 8x8 b16 matrix held one 32-bit register per thread. Lives in the shim (not in
// mma_int.cuh) because the plan puts it here; it is an int-typed helper either way. The
// CUDART_VERSION < 11.8 shuffle fallback is dropped -- this tree builds against CUDA 12.x.
// -------------------------------------------------------------------------------------------------------

static __device__ __forceinline__ int ggml_cuda_movmatrix(const int x) {
    int ret = 0;

#ifdef FTMMA_TURING_MMA_AVAILABLE
    asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(ret) : "r"(x));
#else
    FTMMA_UNUSED(x);
    FTMMA_NO_DEVICE_CODE;
#endif  // FTMMA_TURING_MMA_AVAILABLE
    return ret;
}

// -------------------------------------------------------------------------------------------------------
// 12. Quant-type enum and traits.  upstream ggml.h:390-433 and common.cuh:980-1135
//
// The numeric values are the GGUF/ggml type codes, which is exactly what the vendored dispatch
// already passes around as `int64_t type` (gguf_kernel.cu:194-270). Enumerators keep their
// upstream spelling but live in `namespace ftmma`, so a port writes GGML_TYPE_Q4_K unchanged and
// there is no global symbol to collide with.
//
// ONLY qk IS PROVIDED, DELIBERATELY. qr and qi are the constants the vendored ggml-common.h
// disagrees with upstream about (QR4_XS 8 vs 2, QR2_XXS/QR2_XS/QR2_S/QR3_XXS 8 vs 4, QR3_S/QI3_S
// missing); putting them here would recreate exactly the hazard this subdirectory exists to
// avoid. Per the plan (S4), each per-type loader header defines its own QR/QI locally.
// qk has no such conflict: it is 256 for every K-/I-quant and 32 for Q8_0 and IQ4_NL.
//
// Specialised for the 11 types this checkpoint's quantized dispatch actually sees
// (plan section 1.1). Adding another type is one four-line block.
// -------------------------------------------------------------------------------------------------------

enum ftmma_type : int {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
    GGML_TYPE_COUNT   = 43,
};

// Upstream templates read `template <ggml_type type, ...>`; keep that spelling working.
using ggml_type = ftmma_type;

// Local block-size constants. Deliberately NOT the QK_K / QK4_NL / QK8_0 macros of the vendored
// ggml-common.h -- this subdirectory includes nothing from that header.
static constexpr int FTMMA_QK_K   = 256;
static constexpr int FTMMA_QK8_0  = 32;
static constexpr int FTMMA_QK8_1  = 32;
static constexpr int FTMMA_QK4_NL = 32;

template <ftmma_type type>
struct ftmma_type_traits;

template <> struct ftmma_type_traits<GGML_TYPE_Q8_0>    { static constexpr int qk = FTMMA_QK8_0;  };
template <> struct ftmma_type_traits<GGML_TYPE_Q3_K>    { static constexpr int qk = FTMMA_QK_K;   };
template <> struct ftmma_type_traits<GGML_TYPE_Q4_K>    { static constexpr int qk = FTMMA_QK_K;   };
template <> struct ftmma_type_traits<GGML_TYPE_Q5_K>    { static constexpr int qk = FTMMA_QK_K;   };
template <> struct ftmma_type_traits<GGML_TYPE_Q6_K>    { static constexpr int qk = FTMMA_QK_K;   };
template <> struct ftmma_type_traits<GGML_TYPE_IQ2_XS>  { static constexpr int qk = FTMMA_QK_K;   };
template <> struct ftmma_type_traits<GGML_TYPE_IQ2_S>   { static constexpr int qk = FTMMA_QK_K;   };
template <> struct ftmma_type_traits<GGML_TYPE_IQ3_XXS> { static constexpr int qk = FTMMA_QK_K;   };
template <> struct ftmma_type_traits<GGML_TYPE_IQ3_S>   { static constexpr int qk = FTMMA_QK_K;   };
template <> struct ftmma_type_traits<GGML_TYPE_IQ4_NL>  { static constexpr int qk = FTMMA_QK4_NL; };
template <> struct ftmma_type_traits<GGML_TYPE_IQ4_XS>  { static constexpr int qk = FTMMA_QK_K;   };

// Upstream spelling, so `ggml_cuda_type_traits<type>::qk` (mmq.cuh:877, :965, :1414) ports verbatim.
template <ftmma_type type>
using ggml_cuda_type_traits = ftmma_type_traits<type>;

// ===========================================================================================================
}  // namespace ftmma
// ===========================================================================================================
