from __future__ import annotations

import contextlib
import importlib
import os
import pathlib
import re
import sys
import threading
from typing import (
    TYPE_CHECKING,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Protocol,
    Sequence,
    Tuple,
    TypeAlias,
    Union,
)

if TYPE_CHECKING:
    from tvm_ffi import Module

KERNEL_PATH = pathlib.Path(__file__).parent / "csrc"
KERNEL_CACHE_PACKAGE = "freetoken_kernel_cache"
KERNEL_CACHE_DIR_ENV = "FREETOKEN_KERNEL_CACHE_DIR"
DISABLE_KERNEL_CACHE_ENV = "FREETOKEN_DISABLE_KERNEL_CACHE"
DISABLE_KERNEL_CACHE_VERSION_CHECK_ENV = "FREETOKEN_DISABLE_KERNEL_CACHE_VERSION_CHECK"
DISABLE_JIT_ENV = "FREETOKEN_DISABLE_JIT"
_TRUE_VALUES = {"1", "true", "yes", "on"}
DEFAULT_INCLUDE = [str(KERNEL_PATH / "include")]
# MSVC DOES NOT SPEAK GCC'S FLAG SYNTAX. Handed -std=c++20 and -O3 it warns
# D9002 "ignoring unknown option" and compiles anyway, so on Windows every host
# C++ module was built at tvm-ffi's default /std:c++17 -- where <source_location>
# does not exist, and utils.h stopped the build with
#   error C2039: 'source_location': is not a member of 'std'
# while the flag that was meant to prevent it sat unread on the command line.
# /Zc:__cplusplus goes with the standard switch: without it MSVC keeps reporting
# __cplusplus as 199711L and every feature test in those headers reads false.
DEFAULT_CFLAGS = (
    ["/std:c++20", "/O2", "/Zc:__cplusplus"]
    if sys.platform == "win32"
    else ["-std=c++20", "-O3"]
)


def _default_cuda_cflags(platform: str) -> List[str]:
    """nvcc flags every kernel build starts from.

    On Windows the host pass of nvcc needs the same C++20 as the host modules
    above: utils.cuh includes <concepts> and <source_location> unconditionally and
    MSVC at /std:c++17 compiles neither. nvcc does hand its own -std=c++20 to the
    host compiler, but it puts that first and every -Xcompiler group after it
    (measured with CUDA 12.9), so tvm-ffi's default /std:c++17 passthrough lands
    later on the cl line and wins. MSVC takes the last /std:, so ours has to come
    after tvm-ffi's -- which is where tvm-ffi appends the flags we pass it."""
    flags = ["-std=c++20", "-O3", "--expt-relaxed-constexpr"]
    if platform == "win32":
        flags += ["-Xcompiler", "/std:c++20,/Zc:__cplusplus"]
    return flags


DEFAULT_CUDA_CFLAGS = _default_cuda_cflags(sys.platform)
DEFAULT_LDFLAGS: List[str] = []


ARCH_LIST_ENV = "TVM_FFI_CUDA_ARCH_LIST"


def _cuda_arch_list() -> List[str]:
    """Archs a CUDA build targets: the AOT build's TVM_FFI_CUDA_ARCH_LIST, else the GPU this process is bound to."""
    arch_list = os.getenv(ARCH_LIST_ENV, "").split()
    if arch_list:
        return arch_list
    import torch

    if not torch.cuda.is_available():
        return []
    major, minor = torch.cuda.get_device_capability()
    return [f"{major}.{minor}"]


@contextlib.contextmanager
def _pin_tvm_ffi_arch_ctx(arch_list: List[str]) -> Iterator[None]:
    """Hand tvm-ffi the arch list through its env var for one build. Left unset, tvm-ffi asks nvidia-smi and takes the first GPU listed, which under --gpu or CUDA_VISIBLE_DEVICES on a mixed box is not the bound one."""
    if os.getenv(ARCH_LIST_ENV) or not arch_list:
        yield
        return
    os.environ[ARCH_LIST_ENV] = " ".join(arch_list)
    try:
        yield
    finally:
        os.environ.pop(ARCH_LIST_ENV, None)


def _cuda_cflags(extra: List[str], arch_list: List[str]) -> List[str]:
    """CUDA nvcc flags for a kernel build. tvm-ffi emits one SASS cubin per arch in ``arch_list`` and no PTX, so add the PTX of the highest arch: a GPU newer than every listed arch still runs through the driver's PTX JIT. This flag also carries the arch into tvm-ffi's build hash, which skips tvm-ffi's own -gencode, so GPUs of different archs never share a cached .so."""
    flags = DEFAULT_CUDA_CFLAGS + extra
    if arch_list:
        def _rank(a: str) -> int:
            major, minor = a.rstrip("a").split(".")
            return int(major) * 100 + int(minor)

        cc = max(arch_list, key=_rank).rstrip("a").replace(".", "")
        flags = flags + [f"-gencode=arch=compute_{cc},code=compute_{cc}"]
    return flags


# -Xcompiler TAKES ONE ARGUMENT. tvm-ffi's Windows CUDA defaults are
# ["-Xcompiler", "/std:c++17", "/O2"] (tvm_ffi/cpp/extension.py:418 in
# 0.1.13.post3): /std:c++17 reaches the host compiler and /O2 stays a free token
# that nvcc reads as a second input file. Both halves measured here with CUDA 12.9:
#   nvcc -Xcompiler /std:c++17 /O2 -c x.cu -o x.o
#     -> nvcc fatal : A single input file is required for a non-link phase when an
#        outputfile is specified
#   nvcc -Xcompiler /std:c++17,/O2 -c x.cu -o x.o   -> argument analysis passes
# The Linux branch of the same function (extension.py:427) is ["-std=c++17",
# "-O2"], no -Xcompiler and no defect, so this is Windows-only -- which is why it
# has survived: almost nobody builds CUDA kernels through that package there.
# Two things that do NOT fix it. Passing our own extra_cuda_cflags: tvm-ffi
# appends them to the defaults (extension.py:450), so the free token stays where
# it is. Moving the pin: 0.1.14.post1, the newest release on PyPI, carries the
# same line. And editing the file inside the venv would make one machine differ
# from a clean install with nothing to show it, so the repair lives here.
_NINJA_CUDA_CFLAGS_PREFIX = "cuda_cflags = "
_TVM_FFI_WINDOWS_CUDA_CFLAGS_DEFECT = "-Xcompiler /std:c++17 /O2"
_TVM_FFI_WINDOWS_CUDA_CFLAGS_REPAIR = "-Xcompiler /std:c++17,/O2"


def _repair_windows_cuda_cflags(ninja: str) -> str:
    """Give tvm-ffi's -Xcompiler its single argument in a generated build.ninja.

    Raises when the defect is not there: the day upstream fixes it or moves it, this
    must be read and removed rather than left searching for a string that is gone."""
    lines = ninja.split("\n")
    targets = [
        index
        for index, line in enumerate(lines)
        if line.startswith(_NINJA_CUDA_CFLAGS_PREFIX)
    ]
    if not targets:
        # A host-only module: tvm-ffi writes no cuda_cflags line and never runs nvcc.
        return ninja
    for index in targets:
        if _TVM_FFI_WINDOWS_CUDA_CFLAGS_DEFECT not in lines[index]:
            import tvm_ffi

            raise RuntimeError(
                f"apache-tvm-ffi {tvm_ffi.__version__} generated a CUDA build line "
                f"without {_TVM_FFI_WINDOWS_CUDA_CFLAGS_DEFECT!r} in it:\n"
                f"  {lines[index]}\n"
                "That string is the Windows -Xcompiler defect freetoken repairs in "
                "freetoken.kernel.utils. If upstream has fixed it, delete the repair; "
                "if it only changed shape, update it. Do not skip it: without the "
                "repair nvcc stops with 'A single input file is required for a "
                "non-link phase when an outputfile is specified'."
            )
        lines[index] = lines[index].replace(
            _TVM_FFI_WINDOWS_CUDA_CFLAGS_DEFECT, _TVM_FFI_WINDOWS_CUDA_CFLAGS_REPAIR, 1
        )
    return "\n".join(lines)


class _NinjaGenerator(Protocol):
    """tvm-ffi's _generate_ninja_build, spelled out: a call that no longer matches
    this fails here rather than reaching a build.ninja nobody repaired."""

    def __call__(
        self,
        name: str,
        extra_cflags: Sequence[str],
        extra_cuda_cflags: Sequence[str],
        extra_ldflags: Sequence[str],
        extra_include_paths: Sequence[str],
        sources: Sequence[str],
        embed_cubin: Mapping[str, bytes] | None = None,
        backend: str | None = None,
        output: str | None = None,
    ) -> str: ...


_ninja_repair_lock = threading.Lock()
_tvm_ffi_ninja_generator: _NinjaGenerator | None = None


def _generate_repaired_ninja_build(
    name: str,
    extra_cflags: Sequence[str],
    extra_cuda_cflags: Sequence[str],
    extra_ldflags: Sequence[str],
    extra_include_paths: Sequence[str],
    sources: Sequence[str],
    embed_cubin: Mapping[str, bytes] | None = None,
    backend: str | None = None,
    output: str | None = None,
) -> str:
    generator = _tvm_ffi_ninja_generator
    if generator is None:
        raise RuntimeError(
            "freetoken's build.ninja repair is installed in tvm-ffi but holds no "
            "generator to call -- it was uninstalled while a build was running"
        )
    return _repair_windows_cuda_cflags(
        generator(
            name=name,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_ldflags=extra_ldflags,
            extra_include_paths=extra_include_paths,
            sources=sources,
            embed_cubin=embed_cubin,
            backend=backend,
            output=output,
        )
    )


def _install_windows_cuda_cflags_repair() -> None:
    """Put the repair in front of tvm-ffi's build.ninja generator, on Windows only.

    _generate_ninja_build is a module function that returns build.ninja as a string,
    so the repair lives outside the package -- the same idea as the arch pin above,
    which hands tvm-ffi a value through its env var.

    Installed once per process and left in place, not wrapped and unwrapped around
    each build: kernels are built from a thread pool (kernel/aot.py), and one
    install/restore pair per build would nest one build's wrapper inside another's,
    where the outer repair reads an already repaired line as upstream's fix -- and
    the last thread out would restore a wrapper as if it were tvm-ffi's own. The
    identity check below is what keeps the generator from being captured as itself,
    which recurses until the stack ends. tvm-ffi rewrites build.ninja whenever the
    content differs, so a cache directory holding a broken one from an earlier run
    is corrected rather than reused."""
    if sys.platform != "win32":
        return
    global _tvm_ffi_ninja_generator
    from tvm_ffi.cpp import extension

    with _ninja_repair_lock:
        if extension._generate_ninja_build is _generate_repaired_ninja_build:
            return
        _tvm_ffi_ninja_generator = extension._generate_ninja_build
        extension._generate_ninja_build = _generate_repaired_ninja_build


CPP_TEMPLATE_TYPE: TypeAlias = Union[int, float, bool]


class CppArgList(list[str]):
    def __str__(self) -> str:
        return ", ".join(self)


class KernelConfig(NamedTuple):
    num_threads: int
    max_occupancy: int
    use_pdl: bool

    @property
    def template_args(self) -> str:
        pdl = "true" if self.use_pdl else "false"
        return f"{self.num_threads},{self.max_occupancy},{pdl}"


def _make_name(*args: str) -> str:
    return "freetoken__" + "_".join(str(arg) for arg in args)


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUE_VALUES


def _freetoken_version() -> str:
    from freetoken.version import __version__

    return __version__


def _version_parts(version: str) -> Tuple[str, List[str]]:
    """Split a PEP 440 version string into (release, local segments):
    "0.1.1+cu130.g3f01615" -> ("0.1.1", ["cu130", "g3f01615"])."""
    base, _, local = version.partition("+")
    return base, local.split(".") if local else []


def _build_stamps(segments: List[str]) -> set[str]:
    """The `g<sha>` commit-stamp tokens of a local version segment list
    (stamped by scripts/build-release-wheels.sh)."""
    return {s for s in segments if re.fullmatch(r"g[0-9a-f]{7,40}", s)}


def _kernel_cache_version_ok(cache_version: str, runtime_version: str) -> bool:
    """Same release -- and, when both sides carry a `g<sha>` stamp, the same build.

    The cache wheel extends the runtime's version with local segments (`+cu130`,
    `.g<sha>`), so the old string-prefix test cannot pair a stamped runtime with its
    cache; and comparing the stamps rejects a runtime/cache pair from two different
    builds, which bare release numbers (both `0.1.1`) could never detect. Either side
    may lack a stamp (dev builds) -- then only the release part is compared."""
    cache_base, cache_local = _version_parts(cache_version)
    runtime_base, runtime_local = _version_parts(runtime_version)
    if cache_base != runtime_base:
        return False
    cache_stamps = _build_stamps(cache_local)
    runtime_stamps = _build_stamps(runtime_local)
    return not (cache_stamps and runtime_stamps and cache_stamps != runtime_stamps)


def _kernel_cache_dir() -> pathlib.Path | None:
    if _env_enabled(DISABLE_KERNEL_CACHE_ENV):
        return None

    override = os.getenv(KERNEL_CACHE_DIR_ENV)
    if override:
        return pathlib.Path(override).expanduser()

    try:
        package = importlib.import_module(KERNEL_CACHE_PACKAGE)
    except ModuleNotFoundError as exc:
        if exc.name == KERNEL_CACHE_PACKAGE:
            return None
        raise

    package_version = str(getattr(package, "__version__", "0.0.0+unknown"))
    runtime_version = _freetoken_version()
    if not _env_enabled(DISABLE_KERNEL_CACHE_VERSION_CHECK_ENV):
        if not _kernel_cache_version_ok(package_version, runtime_version):
            raise RuntimeError(
                "freetoken-kernel-cache version "
                f"{package_version!r} does not match freetoken version {runtime_version!r}"
            )
        cache_cuda = re.search(r"\+cu(\d{2,})", package_version)
        if cache_cuda is not None:
            from freetoken.kernel._toolchain import torch_cuda_major

            cache_major = int(cache_cuda.group(1)[:-1])
            torch_major = torch_cuda_major()
            if torch_major is not None and cache_major != torch_major:
                raise RuntimeError(
                    f"freetoken-kernel-cache {package_version!r} was built for CUDA "
                    f"{cache_major}.x but torch runs CUDA {torch_major}.x -- install "
                    "the kernel-cache wheel matching this torch build"
                )

    get_jit_cache_dir = getattr(package, "get_jit_cache_dir", None)
    if get_jit_cache_dir is None:
        raise RuntimeError(f"{KERNEL_CACHE_PACKAGE} does not expose get_jit_cache_dir()")
    return pathlib.Path(get_jit_cache_dir()).expanduser()


def _load_prebuilt(name: str) -> Module | None:
    cache_dir = _kernel_cache_dir()
    if cache_dir is None:
        if _env_enabled(DISABLE_JIT_ENV):
            raise RuntimeError(
                "JIT compilation is disabled by FREETOKEN_DISABLE_JIT, "
                f"but no prebuilt kernel cache is configured for {name!r}"
            )
        return None

    so_path = cache_dir / name / f"{name}.so"
    if so_path.exists():
        import tvm_ffi

        return tvm_ffi.load_module(str(so_path))

    if _env_enabled(DISABLE_JIT_ENV):
        raise RuntimeError(
            "JIT compilation is disabled by FREETOKEN_DISABLE_JIT, "
            f"but prebuilt kernel {name!r} was not found at {so_path}"
        )
    return None


def _make_wrapper(tup: Tuple[str, str]) -> str:
    export_name, kernel_name = tup
    return f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({export_name}, ({kernel_name}));"


def make_cpp_args(*args: CPP_TEMPLATE_TYPE) -> CppArgList:
    def _convert(arg: CPP_TEMPLATE_TYPE) -> str:
        if isinstance(arg, bool):
            return "true" if arg else "false"
        if isinstance(arg, (int, float)):
            return str(arg)
        raise TypeError(f"Unsupported argument type for cpp template: {type(arg)}")

    return CppArgList(_convert(arg) for arg in args)


def load_aot(
    *args: str,
    cpp_files: List[str] | None = None,
    cuda_files: List[str] | None = None,
    extra_cflags: List[str] | None = None,
    extra_cuda_cflags: List[str] | None = None,
    extra_ldflags: List[str] | None = None,
    extra_include_paths: List[str] | None = None,
    build_directory: str | None = None,
) -> Module:
    name = _make_name(*args)
    prebuilt = _load_prebuilt(name)
    if prebuilt is not None:
        return prebuilt

    arch_list: List[str] = []
    if cuda_files:
        from freetoken.kernel._toolchain import check_nvcc_matches_torch

        check_nvcc_matches_torch()
        arch_list = _cuda_arch_list()

    from tvm_ffi.cpp import load

    cpp_files = cpp_files or []
    cuda_files = cuda_files or []
    extra_cflags = extra_cflags or []
    extra_cuda_cflags = extra_cuda_cflags or []
    extra_ldflags = extra_ldflags or []
    extra_include_paths = extra_include_paths or []

    cpp_files = [str((KERNEL_PATH / "src" / f).resolve()) for f in cpp_files]
    cuda_files = [str((KERNEL_PATH / "src" / f).resolve()) for f in cuda_files]

    _install_windows_cuda_cflags_repair()
    with _pin_tvm_ffi_arch_ctx(arch_list):
        return load(
            name,
            cpp_files=cpp_files,
            cuda_files=cuda_files,
            extra_cflags=DEFAULT_CFLAGS + extra_cflags,
            extra_cuda_cflags=_cuda_cflags(extra_cuda_cflags, arch_list),
            extra_ldflags=DEFAULT_LDFLAGS + extra_ldflags,
            extra_include_paths=DEFAULT_INCLUDE + extra_include_paths,
            build_directory=build_directory,
        )


def load_jit(
    *args: str,
    cpp_files: List[str] | None = None,
    cuda_files: List[str] | None = None,
    cpp_wrappers: List[Tuple[str, str]] | None = None,
    cuda_wrappers: List[Tuple[str, str]] | None = None,
    extra_cflags: List[str] | None = None,
    extra_cuda_cflags: List[str] | None = None,
    extra_ldflags: List[str] | None = None,
    extra_include_paths: List[str] | None = None,
    build_directory: str | None = None,
) -> Module:
    name = _make_name(*args)
    prebuilt = _load_prebuilt(name)
    if prebuilt is not None:
        return prebuilt

    arch_list: List[str] = []
    if cuda_files or cuda_wrappers:
        from freetoken.kernel._toolchain import check_nvcc_matches_torch

        check_nvcc_matches_torch()
        arch_list = _cuda_arch_list()

    from tvm_ffi.cpp import load_inline

    cpp_files = cpp_files or []
    cuda_files = cuda_files or []
    cpp_wrappers = cpp_wrappers or []
    cuda_wrappers = cuda_wrappers or []
    extra_cflags = extra_cflags or []
    extra_cuda_cflags = extra_cuda_cflags or []
    extra_ldflags = extra_ldflags or []
    extra_include_paths = extra_include_paths or []

    # include cpp files
    cpp_paths = [(KERNEL_PATH / "jit" / f).resolve() for f in cpp_files]
    cpp_sources = [f'#include "{path}"' for path in cpp_paths]
    cpp_sources += [_make_wrapper(tup) for tup in cpp_wrappers]

    # include cuda files
    cuda_paths = [(KERNEL_PATH / "jit" / f).resolve() for f in cuda_files]
    cuda_sources = [f'#include "{path}"' for path in cuda_paths]
    cuda_sources += [_make_wrapper(tup) for tup in cuda_wrappers]

    _install_windows_cuda_cflags_repair()
    with _pin_tvm_ffi_arch_ctx(arch_list):
        return load_inline(
            name,
            cpp_sources=cpp_sources,
            cuda_sources=cuda_sources,
            extra_cflags=DEFAULT_CFLAGS + extra_cflags,
            extra_cuda_cflags=_cuda_cflags(extra_cuda_cflags, arch_list),
            extra_ldflags=DEFAULT_LDFLAGS + extra_ldflags,
            extra_include_paths=DEFAULT_INCLUDE + extra_include_paths,
            build_directory=build_directory,
        )
