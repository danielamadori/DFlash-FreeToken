"""Availability probes for the optional native kernel packages.

When flashinfer / sgl_kernel are USABLE the call-sites use their fused CUDA ops;
otherwise they fall back to the pure-Triton kernels in ``freetoken.kernel.triton``.

Usable, not merely installed: these packages ship prebuilt per-architecture binaries
and raise from their own __init__ when none matches the GPU or its CUDA runtime is
absent. Probing with find_spec alone reported them present in exactly that case and
sent the caller into an import that then failed, which is why each probe now imports
the package once. The result is cached, so the cost is paid at most once per name.
"""
from __future__ import annotations

import functools
import importlib
import importlib.util


def _importable(name: str) -> bool:
    """True when ``name`` can actually be IMPORTED, not merely located on disk.

    find_spec answers "is this installed", and callers need "can I use this". The two
    differ for exactly the packages here: sgl_kernel installs cleanly and then RAISES from
    its __init__ when no prebuilt variant matches the GPU -- an sm_89 card against wheels
    shipping sm90/sm100, or a binary linked against a CUDA runtime that is not present
    (libnvrtc.so.13 with a CUDA 12 install). find_spec still returns a spec for that, so
    every caller took the accelerated branch and died on the import instead of falling
    back to triton, which is the whole point of asking.

    The import is attempted once per name (the callers are cached) and any failure means
    "not available", as the callers' fallbacks already assume.
    """
    try:
        if importlib.util.find_spec(name) is None:
            return False
        importlib.import_module(name)
        return True
    except Exception:
        return False


@functools.cache
def is_flashinfer_installed() -> bool:
    return _importable("flashinfer")


@functools.cache
def is_sgl_kernel_installed() -> bool:
    return _importable("sgl_kernel")


@functools.cache
def is_triton_kernels_installed() -> bool:
    """OpenAI's ``triton_kernels`` (the fused MoE router used by ``moe.fused.fused_topk``).

    Distinct from the ``triton`` runtime we always depend on: it ships with the Triton
    source tree and has no Windows wheel. It is also not one of the six ops
    ``freetoken.kernel.triton`` reimplements, so its call-site carries its own fallback.
    """
    return _importable("triton_kernels")


@functools.cache
def driver_cuda_version() -> int | None:
    """Max CUDA version the installed NVIDIA driver supports (``13000`` == CUDA 13.0),
    or None if undetermined. Driver-JIT kernels (PTX compiled at runtime, e.g.
    flashinfer's CuTe-DSL paths) are gated by this, not by any package's build-time
    toolkit version. Resolved through the ``_pinned_tensor`` extension's link-time
    cudart, so it works wherever the extension builds (including Windows) -- no dlopen
    by soname."""
    try:
        from freetoken.kernel.pinned import _load_pinned_extension

        version = int(_load_pinned_extension().driver_cuda_version())
    except Exception:
        return None
    return version or None  # 0 == no driver installed
