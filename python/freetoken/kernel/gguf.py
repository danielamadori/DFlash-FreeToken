"""Borrowed llama.cpp GGUF dequant/GEMM CUDA kernels, JIT-compiled on first use.

The ``.cu``/``.cuh`` under ``csrc/gguf/`` are vendored verbatim from sgl-kernel
(``csrc/quantization/gguf/``), which are themselves ports of llama.cpp. We compile
them through ``torch.utils.cpp_extension.load`` (the same toolchain sglang/vllm use)
into a torch-op module and expose the handful of ops the GGUF path needs. This is a
separate, torch-native extension that sits alongside FreeToken's tvm-ffi kernels.

All ops keep the weight in its native GGUF block layout (packed ``uint8`` rows) and
dequantize *inside* the kernel -- no bf16 copy of the weight is ever materialized.
"""

from __future__ import annotations

import functools
import os
import pathlib
import re
import shutil

import torch

_CSRC = pathlib.Path(__file__).parent / "csrc" / "gguf"


def _compiles_cxx(cxx: str) -> bool:
    """True when ``cxx`` can actually compile a C++ translation unit.

    Being on PATH is not the same as working. Ubuntu's clang++ 14 is installed here and
    fails on ``#include <new>``: it looks for a libstdc++ toolchain it cannot find, so
    every C++ header is missing. shutil.which reports it happily, nvcc is then pointed at
    it, and the build dies deep inside CUDA's own headers with ``'new' file not found`` --
    an error that names neither the compiler nor the reason.
    """
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        src = pathlib.Path(tmp) / "probe.cpp"
        src.write_text("#include <new>\nint main() { return 0; }\n", encoding="utf-8")
        try:
            done = subprocess.run(
                [cxx, "-c", str(src), "-o", str(pathlib.Path(tmp) / "probe.o")],
                capture_output=True, timeout=120,
            )
        except (OSError, subprocess.SubprocessError):
            return False
    return done.returncode == 0


@functools.cache
def _host_compiler() -> str | None:
    """A host compiler nvcc + libtorch headers accept, verified by compiling with it.

    The system default gcc can be too new for the torch headers (gcc 16 hard-errors), and
    on some toolchains nvcc+gcc-13 trips a non-conformant ``typename decltype`` in
    ``List_inl.h`` once ``torch::Tensor`` is instantiated, where nvcc with clang++ compiles
    cleanly. So clang++ stays the first preference -- but only if it works: each candidate
    is probed with a real compile before being handed to nvcc, because a broken-but-present
    compiler produced a build failure pointing at CUDA's headers instead of at itself.

    Override with ``FREETOKEN_GGUF_HOST_CXX`` (taken as given, not probed).
    """
    override = os.environ.get("FREETOKEN_GGUF_HOST_CXX")
    if override:
        return override
    for cxx in ("clang++", "g++-13", "g++-14", "g++-15", "g++"):
        path = shutil.which(cxx)
        if path and _compiles_cxx(path):
            return cxx
    return None


def _c_compiler_for(cxx: str) -> str:
    base = os.path.basename(cxx)
    if "clang" in base:
        return shutil.which("clang") or "clang"
    cc = base.replace("g++", "gcc")
    return shutil.which(cc) or cc

def _assert_nvcc_supports(major: int, minor: int) -> None:
    """Fail with the actual problem when nvcc is too old for the card.

    torch resolves CUDA_HOME to /usr when nothing else is set, and a distro nvcc there can
    predate the GPU: CUDA 11.5 against an Ada card gives ``nvcc fatal: Unsupported gpu
    architecture 'compute_89'``, which says nothing about WHICH nvcc or where a newer one
    is. Ada (sm_89) needs CUDA >= 11.8, Blackwell (sm_100) >= 12.8.
    """
    import subprocess

    from torch.utils.cpp_extension import CUDA_HOME

    nvcc = os.path.join(CUDA_HOME, "bin", "nvcc") if CUDA_HOME else shutil.which("nvcc")
    if not nvcc or not os.path.exists(nvcc):
        return
    try:
        out = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return
    match = re.search(r"release (\d+)\.(\d+)", out)
    if not match:
        return
    version = (int(match.group(1)), int(match.group(2)))
    needed = (12, 8) if major >= 10 else (11, 8) if (major, minor) >= (8, 9) else (11, 0)
    if version < needed:
        raise RuntimeError(
            f"nvcc {version[0]}.{version[1]} at {nvcc} cannot target this GPU "
            f"(sm_{major}{minor} needs CUDA >= {needed[0]}.{needed[1]}). Point CUDA_HOME at a "
            f"newer toolkit, e.g. CUDA_HOME=/usr/local/cuda-12.9."
        )


@functools.cache
def _module():
    from torch.utils.cpp_extension import load

    # Built for THIS card, at -O3, on both passes.
    #
    # Without an explicit target nvcc emits a fat binary for its default architecture
    # list: longer builds, and no Ada-specific scheduling. Deriving the arch from the
    # device present means the kernel is compiled for what will run it -- sm_89 here --
    # and `code=sm_XX` emits real SASS rather than PTX the driver must JIT on first launch.
    #
    # Deliberately NOT --use_fast_math: it relaxes IEEE semantics, and this fork has just
    # spent its time establishing that a one-ULP difference in a logit is the whole
    # explanation for two engines disagreeing. Speed that changes numbers is not free
    # here. FREETOKEN_GGUF_FAST_MATH=1 opts in for anyone who wants it.
    extra_cuda_cflags = ["-O3", "--expt-relaxed-constexpr", "-DNDEBUG"]
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        _assert_nvcc_supports(major, minor)
        extra_cuda_cflags += [f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"]
    if os.environ.get("FREETOKEN_GGUF_FAST_MATH", "").lower() in {"1", "true", "yes", "on"}:
        extra_cuda_cflags += ["--use_fast_math"]
    extra_cflags = ["-O3", "-DNDEBUG"]
    host_cxx = _host_compiler()
    if host_cxx is not None:
        # Point both nvcc's host pass (-ccbin) and torch's C++ compile (CXX) at a
        # libtorch/nvcc-compatible compiler. Force (not setdefault): the system
        # default (CXX unset -> g++) can be a gcc too new for the torch headers.
        cxx_path = shutil.which(host_cxx) or host_cxx
        extra_cuda_cflags += ["-ccbin", cxx_path]
        os.environ["CXX"] = cxx_path
        os.environ["CC"] = _c_compiler_for(cxx_path)

    # gguf_kernel.cu carries its own PYBIND11_MODULE (appended at the end), so a
    # plain `load` of the single source compiles + binds the ggml_* ops.
    return load(
        name="freetoken_gguf_kernels",
        sources=[str(_CSRC / "gguf_kernel.cu")],
        extra_include_paths=[str(_CSRC)],
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=True,
    )


# ---- thin typed wrappers (signatures mirror sgl_kernel.quantization.gguf) ----


def ggml_dequantize(
    weight: torch.Tensor, quant_type: int, m: int, n: int, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Dequantize a packed GGUF weight ``[m, row_bytes]`` to a dense ``[m, n]`` tensor."""
    return _module().ggml_dequantize(weight, quant_type, m, n, dtype)


def ggml_mul_mat_vec_a8(
    weight: torch.Tensor, x: torch.Tensor, quant_type: int, row: int
) -> torch.Tensor:
    """MMVQ: small-batch GEMV with on-the-fly dequant. ``row`` = output features."""
    return _module().ggml_mul_mat_vec_a8(weight, x, quant_type, row)


def ggml_mul_mat_a8(
    weight: torch.Tensor, x: torch.Tensor, quant_type: int, row: int
) -> torch.Tensor:
    """MMQ: large-batch quantized matmul. ``row`` = output features."""
    return _module().ggml_mul_mat_a8(weight, x, quant_type, row)


def ggml_mul_mat_mma(
    weight: torch.Tensor, x: torch.Tensor, quant_type: int, row: int
) -> torch.Tensor:
    """Tensor-core MMQ (kernel/csrc/gguf/mma/): quantized matmul without dequantizing.

    Raises rather than falling back when the type or the shape is outside what the port
    covers, so a dispatch mistake is loud; ``layers.gguf`` checks the same conditions before
    calling. ``row`` = output features.
    """
    return _module().ggml_mul_mat_mma(weight, x, quant_type, row)


def ggml_mul_mat_mma_swiglu(
    weight: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, quant_type: int, row: int
) -> torch.Tensor:
    """``ggml_mul_mat_mma`` over ``silu(gate) * up``, with the activation folded into the
    quantization the MMQ has to do anyway. Same conditions as ``ggml_mul_mat_mma``; raises
    rather than falling back. ``row`` = output features."""
    return _module().ggml_mul_mat_mma_swiglu(weight, gate, up, quant_type, row)


def ggml_moe_a8(
    x: torch.Tensor,
    weight: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    quant_type: int,
    row: int,
    top_k: int,
    tokens: int,
) -> torch.Tensor:
    """MMQ grouped expert matmul over stacked experts ``weight[E, row, *]``."""
    return _module().ggml_moe_a8(
        x, weight, sorted_token_ids, expert_ids, num_tokens_post_padded,
        quant_type, row, top_k, tokens,
    )


def ggml_moe_a8_vec(
    x: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    quant_type: int,
    row: int,
    tokens: int,
) -> torch.Tensor:
    """MMVQ grouped expert GEMV over stacked experts ``weight[E, row, *]``."""
    return _module().ggml_moe_a8_vec(x, weight, topk_ids, top_k, quant_type, row, tokens)


def ggml_moe_get_block_size(quant_type: int) -> int:
    return _module().ggml_moe_get_block_size(quant_type)


__all__ = [
    "ggml_dequantize",
    "ggml_mul_mat_vec_a8",
    "ggml_mul_mat_a8",
    "ggml_moe_a8",
    "ggml_moe_a8_vec",
    "ggml_moe_get_block_size",
]
