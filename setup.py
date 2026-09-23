from __future__ import annotations

import importlib.util
from pathlib import Path

import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME, CppExtension


ROOT = Path(__file__).parent


def _check_toolchain() -> None:
    path = ROOT / "python" / "freetoken" / "kernel" / "_toolchain.py"
    spec = importlib.util.spec_from_file_location("_freetoken_toolchain", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_nvcc_matches_torch()


def _cuda_runtime_paths() -> tuple[list[str], list[str]]:
    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA_HOME is required to build freetoken.kernel._pinned_tensor "
            "because it links against the CUDA runtime API."
        )
    cuda_home = Path(CUDA_HOME)
    # lib64 is where Linux keeps them and lib/x64 is where Windows does, so both
    # are offered and only the ones that exist are handed to the linker.
    #
    # WITHOUT THE WINDOWS PATH THIS CANNOT LINK THERE, AND SAYS SO BADLY. The
    # toolkit has no lib64 at all on Windows, and lib/ holds only the
    # subdirectories cmake, Win32 and x64 -- so the two directories computed
    # here were both real-looking and both empty of import libraries, and the
    # build died with "LNK1181: cannot open input file 'cudart.lib'". That
    # message names the library and not the search path, which sends a reader
    # looking for a missing CUDA install that is in fact present and correct.
    # Measured on Windows with CUDA 12.6 on 2026-09-23.
    #
    # Filtered rather than appended blindly: passing a /LIBPATH that does not
    # exist is how the original produced two useless entries, and a linker does
    # not complain about those -- it complains about what it could not find
    # because of them.
    candidates = [
        cuda_home / "lib64",        # Linux
        cuda_home / "lib" / "x64",  # Windows
        cuda_home / "lib",          # some layouts keep the libraries directly here
    ]
    library_dirs = [str(p) for p in candidates if p.is_dir()]
    if not library_dirs:
        raise RuntimeError(
            f"no CUDA library directory under {cuda_home}: looked for "
            + ", ".join(str(p.relative_to(cuda_home)) for p in candidates)
        )
    return [str(cuda_home / "include")], library_dirs


cuda_include_dirs, cuda_library_dirs = _cuda_runtime_paths()
_check_toolchain()


setup(
    ext_modules=[
        CppExtension(
            name="freetoken.kernel._pinned_tensor",
            sources=[
                "python/freetoken/kernel/csrc/pinned_tensor.cpp",
            ],
            include_dirs=cuda_include_dirs,
            library_dirs=cuda_library_dirs,
            libraries=["cudart"],
            extra_compile_args=["-O3", "-std=c++17"],
        ),
        # CPU-compute MoE executor for --moe-backend cpu. Links cudart for the
        # cudaLaunchHostFunc submit/sync graph nodes; the bf16 GEMV microkernels
        # use per-function target attributes (avx512bf16/avx512f) + a runtime
        # __builtin_cpu_supports dispatch, so the single binary stays portable
        # (scalar fallback) -- no global -march is set.
        CppExtension(
            name="freetoken.kernel._cpu_moe",
            sources=[
                "python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp",
            ],
            include_dirs=cuda_include_dirs,
            library_dirs=cuda_library_dirs,
            libraries=["cudart"],
            extra_compile_args=["-O3", "-std=c++17", "-pthread"],
        ),
        # --ple-backend disk row store; Linux-only until the TableFile/BatchReader seams grow Windows bodies
        *([
            CppExtension(
                name="freetoken.kernel._ple_store",
                sources=[
                    "python/freetoken/kernel/csrc/ple_store/ple_store_ext.cpp",
                ],
                extra_compile_args=["-O3", "-std=c++17"],
            )
        ] if sys.platform == "linux" else []),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
