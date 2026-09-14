"""Patch the installed tvm-ffi so its Windows branch can build a CUDA kernel.

FreeToken's JIT kernels are compiled by tvm-ffi, whose Windows branch is missing
four things the POSIX branch has. None of this is FreeToken's code and every
change belongs upstream; until it lands, the engine cannot build a single kernel
on Windows without them, and `pip install -U apache-tvm-ffi` silently reverts
every one. Run this after any install or upgrade of that package.

The four:

1. `-Xcompiler` takes ONE argument. Written as a single `-Xcompiler` followed by
   two flags, nvcc hands `/std:c++17` to the host compiler and then reads `/O2`
   as a second INPUT FILE -- on Windows a leading slash is a path -- so with
   `-c source.cu -o out.o` it sees two inputs and refuses with "nvcc fatal: A
   single input file is required for a non-link phase when an outputfile is
   specified". The host standard must also match the device one: callers compile
   device code as `-std=c++20` while this told MSVC `/std:c++17`, and a header
   shared by both passes then failed on the host with "namespace std has no
   member source_location".

2. The target architecture. The POSIX branch adds `_get_cuda_target()`; the
   Windows branch left nvcc at its default compute_52, and a kernel using
   `__grid_constant__` was refused with "annotation is only allowed for
   architecture compute_70 or later" -- a message about the code, produced by a
   missing flag.

3. The CUDA runtime at link time. The POSIX branch adds `-L<cuda>/lib64
   -lcudart`; the Windows link step had no CUDA library at all, so every kernel
   that compiled then failed with LNK2019 on cudaLaunchKernel,
   cudaLaunchKernelExC and cudaGetErrorString. On Windows the import library
   sits in `lib/x64`.

4. Quoting of the nvcc path in build.ninja. The real toolkit lives under
   "C:\\Program Files\\...", and written unquoted the command line splits at the
   first space. (The toolkit's include paths are unquoted too, which a junction
   without spaces works around; this one is cheap to fix properly.)

Usage:
    python scripts/windows/patch_tvm_ffi.py            # apply
    python scripts/windows/patch_tvm_ffi.py --check    # report only, exit 1 if not applied
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

# (name, marker, original, replacement).
#
# The MARKER is text that exists only once the change is in, and it -- not the
# replacement -- is what decides "already applied". An exact-match test against
# the replacement would call the file unpatched the moment anyone added a line
# inside the patched block, and the script would then fail to find the original
# text and refuse to run at all.
CAMBI: list[tuple[str, str, str, str]] = [
    (
        "host flags for nvcc",
        '"-Xcompiler", "/Zc:__cplusplus"',
        '        default_cuda_cflags = ["-Xcompiler", "/std:c++17", "/O2"]\n',
        '        default_cuda_cflags = ["-Xcompiler", "/std:c++20", "-Xcompiler", "/O2",\n'
        '                               "-Xcompiler", "/Zc:__cplusplus"]\n',
    ),
    (
        "target architecture and the CUDA runtime",
        '"cudart.lib",',
        '        default_ldflags = [\n'
        '            "/DLL",\n'
        '            f"/LIBPATH:{tvm_ffi_lib_path}",\n'
        '            f"{tvm_ffi_lib_name}.lib",\n'
        '        ]\n',
        '        if with_cuda:\n'
        '            default_cuda_cflags += [_get_cuda_target()]\n'
        '        default_ldflags = [\n'
        '            "/DLL",\n'
        '            f"/LIBPATH:{tvm_ffi_lib_path}",\n'
        '            f"{tvm_ffi_lib_name}.lib",\n'
        '        ]\n'
        '        if with_cuda:\n'
        '            default_ldflags += [\n'
        '                "/LIBPATH:{}".format(Path(_find_cuda_home()) / "lib" / "x64"),\n'
        '                "cudart.lib",\n'
        '            ]\n',
    ),
    (
        "quoting of hipcc in build.ninja",
        "'nvcc = \"{}\"'.format(str(Path(_find_rocm_home())",
        '            ninja.append("nvcc = {}".format(str(Path(_find_rocm_home()) / "bin" / "hipcc")))\n',
        '            ninja.append(\'nvcc = "{}"\'.format(str(Path(_find_rocm_home()) / "bin" / "hipcc")))\n',
    ),
    (
        "quoting of nvcc in build.ninja",
        "'nvcc = \"{}\"'.format(str(Path(_find_cuda_home())",
        '            ninja.append("nvcc = {}".format(str(Path(_find_cuda_home()) / "bin" / "nvcc")))\n',
        '            ninja.append(\'nvcc = "{}"\'.format(str(Path(_find_cuda_home()) / "bin" / "nvcc")))\n',
    ),
]


def bersaglio() -> Path:
    import tvm_ffi  # noqa: PLC0415  -- imported here so --help works without it

    return Path(tvm_ffi.__file__).parent / "cpp" / "extension.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report whether the patch is applied; change nothing. Exit 1 if not.",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="act on this file instead of the installed tvm_ffi (for testing).",
    )
    args = parser.parse_args()

    percorso = args.file if args.file is not None else bersaglio()
    testo = io.open(percorso, encoding="utf-8").read()
    print(f"tvm-ffi: {percorso}")

    da_fare: list[tuple[str, str, str]] = []
    for nome, marcatore, vecchio, nuovo in CAMBI:
        if marcatore in testo:
            print(f"  [already in] {nome}")
            continue
        if testo.count(vecchio) == 1:
            da_fare.append((nome, vecchio, nuovo))
            continue
        # NOT a warning to step over. Neither the marker nor the original text is
        # there, so this is a tvm-ffi whose Windows branch has changed; applying
        # the rest would leave a half-patched file that fails later, elsewhere,
        # with a message about something unrelated.
        raise SystemExit(
            f"cannot patch '{nome}': the expected text is not in {percorso}\n"
            "tvm-ffi has changed. Re-read its Windows branch and update this script."
        )

    if args.check:
        if da_fare:
            print(f"NOT patched: {len(da_fare)} change(s) missing")
            return 1
        print("patched")
        return 0

    if not da_fare:
        print("nothing to do")
        return 0

    for nome, vecchio, nuovo in da_fare:
        testo = testo.replace(vecchio, nuovo)
        print(f"  [applied]    {nome}")
    io.open(percorso, "w", encoding="utf-8", newline="").write(testo)
    print(f"{len(da_fare)} change(s) written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
