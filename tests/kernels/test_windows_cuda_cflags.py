"""The Windows repair of tvm-ffi's generated build.ninja.

tvm-ffi's Windows CUDA defaults are ["-Xcompiler", "/std:c++17", "/O2"], and
-Xcompiler takes one argument, so /O2 is left over for nvcc to read as a second
input file:

    nvcc -Xcompiler /std:c++17 /O2 -c x.cu -o x.o
      -> nvcc fatal : A single input file is required for a non-link phase when an
         outputfile is specified

freetoken.kernel.utils repairs that line on the way out of _generate_ninja_build.
The platform is simulated here: these tests need neither Windows nor a GPU, and on
Linux the whole point is that the generated file comes back untouched."""

from __future__ import annotations

import sys
from typing import List, Mapping, Sequence

import pytest

from freetoken.kernel import utils as kernel_utils
from freetoken.kernel.utils import (
    _default_cuda_cflags,
    _install_windows_cuda_cflags_repair,
    _repair_windows_cuda_cflags,
)

# The line under test, as tvm-ffi 0.1.13.post3 writes it on Windows for a freetoken
# CUDA kernel: its own defaults first, the flags freetoken passes appended after.
WINDOWS_CUDA_CFLAGS = (
    "cuda_cflags = -Xcompiler /std:c++17 /O2 -std=c++20 -O3 --expt-relaxed-constexpr "
    "-Xcompiler /std:c++20,/Zc:__cplusplus -gencode=arch=compute_86,code=compute_86 "
    "-IC$:\\ft\\include"
)
WINDOWS_NINJA = f"""ninja_required_version = 1.3
cxx = cl
cxxflags = /std:c++17 /MD /EHsc /std:c++20 /O2 /Zc:__cplusplus
nvcc = C:\\CUDA\\v12.6\\bin\\nvcc.exe
{WINDOWS_CUDA_CFLAGS}
ldflags = /DLL

rule cuda_compile
  command = $nvcc $cuda_cflags -c $in -o $out

default probe.dll
"""

# The same module on Linux: -Xcompiler carries one argument, nothing to repair.
LINUX_NINJA = """ninja_required_version = 1.3
cxx = c++
cxxflags = -std=c++17 -fPIC -O2 -std=c++20 -O3
nvcc = /usr/local/cuda/bin/nvcc
cuda_cflags = -Xcompiler -fPIC -std=c++17 -O2 -std=c++20 -O3 --expt-relaxed-constexpr -I/ft/include
ldflags = -shared -ltvm_ffi

default probe.so
"""

# A host-only module (radix.cpp, tensor.cpp): tvm-ffi writes no cuda_cflags line.
HOST_ONLY_NINJA = """ninja_required_version = 1.3
cxx = cl
cxxflags = /std:c++17 /MD /EHsc /std:c++20 /O2 /Zc:__cplusplus
ldflags = /DLL

default probe.dll
"""


def _cuda_cflags_line(ninja: str) -> str:
    lines = [line for line in ninja.split("\n") if line.startswith("cuda_cflags = ")]
    assert len(lines) == 1
    return lines[0]


def test_repair_pairs_xcompiler_with_its_argument() -> None:
    repaired = _repair_windows_cuda_cflags(WINDOWS_NINJA)
    assert "-Xcompiler /std:c++17,/O2 -std=c++20" in _cuda_cflags_line(repaired)
    assert "-Xcompiler /std:c++17 /O2" not in repaired
    # Only that line moves: same number of lines, host flags left alone.
    assert repaired.split("\n")[2] == WINDOWS_NINJA.split("\n")[2]
    assert len(repaired.split("\n")) == len(WINDOWS_NINJA.split("\n"))


def test_repaired_line_has_no_free_msvc_token() -> None:
    """What nvcc needs: every MSVC-syntax token is an argument of an -Xcompiler."""
    tokens = _cuda_cflags_line(_repair_windows_cuda_cflags(WINDOWS_NINJA)).split()
    msvc = [index for index, token in enumerate(tokens) if token.startswith("/")]
    assert msvc, "the Windows line is supposed to carry MSVC flags"
    for index in msvc:
        assert tokens[index - 1] == "-Xcompiler", tokens[index]
    # Before the repair it is exactly this property that fails.
    broken = _cuda_cflags_line(WINDOWS_NINJA).split()
    assert any(
        token.startswith("/") and broken[index - 1] != "-Xcompiler"
        for index, token in enumerate(broken)
    )


def test_repair_raises_when_the_defect_is_gone() -> None:
    fixed_upstream = WINDOWS_NINJA.replace(
        "-Xcompiler /std:c++17 /O2", "-Xcompiler /std:c++17,/O2"
    )
    with pytest.raises(RuntimeError) as excinfo:
        _repair_windows_cuda_cflags(fixed_upstream)
    message = str(excinfo.value)
    assert "-Xcompiler /std:c++17 /O2" in message  # what it looked for
    assert "apache-tvm-ffi" in message  # and in which version
    assert "0.1" in message


def test_host_only_module_is_not_a_defect() -> None:
    assert _repair_windows_cuda_cflags(HOST_ONLY_NINJA) == HOST_ONLY_NINJA


def _fake_generate_factory(ninja: str, calls: List[str]) -> kernel_utils._NinjaGenerator:
    def _fake_generate(
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
        calls.append(name)
        return ninja

    return _fake_generate


def _generate(generator: kernel_utils._NinjaGenerator, name: str) -> str:
    return generator(
        name=name,
        extra_cflags=[],
        extra_cuda_cflags=[],
        extra_ldflags=[],
        extra_include_paths=[],
        sources=["k.cu"],
    )


def test_windows_install_repairs_the_generated_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tvm_ffi.cpp import extension

    calls: List[str] = []
    fake = _fake_generate_factory(WINDOWS_NINJA, calls)
    monkeypatch.setattr(kernel_utils, "_tvm_ffi_ninja_generator", None)
    monkeypatch.setattr(extension, "_generate_ninja_build", fake)
    monkeypatch.setattr(sys, "platform", "win32")

    _install_windows_cuda_cflags_repair()
    assert extension._generate_ninja_build is not fake
    generated = _generate(extension._generate_ninja_build, "probe")

    assert calls == ["probe"]
    assert "-Xcompiler /std:c++17,/O2" in generated
    assert "-Xcompiler /std:c++17 /O2" not in generated


def test_installing_twice_does_not_repair_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """kernel/aot.py builds kernels from a thread pool, so the install runs once per
    build and has to be idempotent. Without the identity check it captures its own
    repair as the generator to call and recurses until the stack ends (measured);
    wrapping and unwrapping per build instead would hand an already repaired line to
    the repair, which reads a missing defect as upstream's fix."""
    from tvm_ffi.cpp import extension

    calls: List[str] = []
    fake = _fake_generate_factory(WINDOWS_NINJA, calls)
    monkeypatch.setattr(kernel_utils, "_tvm_ffi_ninja_generator", None)
    monkeypatch.setattr(extension, "_generate_ninja_build", fake)
    monkeypatch.setattr(sys, "platform", "win32")

    _install_windows_cuda_cflags_repair()
    installed = extension._generate_ninja_build
    _install_windows_cuda_cflags_repair()
    assert extension._generate_ninja_build is installed

    generated = _generate(extension._generate_ninja_build, "probe")
    assert generated == _repair_windows_cuda_cflags(WINDOWS_NINJA)
    assert calls == ["probe"]


def test_linux_is_left_exactly_as_it_was(monkeypatch: pytest.MonkeyPatch) -> None:
    from tvm_ffi.cpp import extension

    fake = _fake_generate_factory(LINUX_NINJA, [])
    monkeypatch.setattr(kernel_utils, "_tvm_ffi_ninja_generator", None)
    monkeypatch.setattr(extension, "_generate_ninja_build", fake)
    monkeypatch.setattr(sys, "platform", "linux")

    _install_windows_cuda_cflags_repair()
    # Nothing is wrapped at all: on Linux the defect does not exist.
    assert extension._generate_ninja_build is fake
    assert _generate(extension._generate_ninja_build, "probe").encode() == LINUX_NINJA.encode()
    assert kernel_utils._tvm_ffi_ninja_generator is None


def test_windows_asks_the_host_pass_for_cpp20() -> None:
    """utils.cuh includes <concepts> and <source_location>; MSVC needs /std:c++20 for
    both, and nvcc places -Xcompiler groups after its own -std, so the last
    -Xcompiler on the line is the one that decides the host standard."""
    linux = _default_cuda_cflags("linux")
    windows = _default_cuda_cflags("win32")
    assert linux == ["-std=c++20", "-O3", "--expt-relaxed-constexpr"]
    assert windows[: len(linux)] == linux
    assert windows[len(linux) :] == ["-Xcompiler", "/std:c++20,/Zc:__cplusplus"]
