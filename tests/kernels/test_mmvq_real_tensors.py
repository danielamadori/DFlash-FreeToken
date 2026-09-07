"""The batched MMVQ kernel against a dequantized reference, on real tensors of the 27B.

The kernel carries the shape of llama.cpp's mul_mat_vec_q (two warps splitting K, two weight
rows per thread, one exact instantiation per column count up to 8, a guarded group-of-8 path
above). What this checks, per quant type: every column count 1..9 and 16/17 (both the exact
instantiations and the guarded path), an odd number of rows (the row clamp at the tail of the
grid), and the 1-column decode path. The reference is ggml_dequantize + a float matmul on the
same Q8_1-rounded activation the kernel consumes, so the two differ only by summation order.

Skipped without CUDA or the GGUF file (this is a fork-local fixture: the file is 14 GiB).
"""
from __future__ import annotations

import os

import pytest
import torch

GGUF = os.environ.get(
    "FREETOKEN_TEST_GGUF",
    os.path.expanduser("~/.lmstudio/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_S.gguf"),
)
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.exists(GGUF), reason="needs CUDA and the 27B GGUF"
)

WANT = ["Q4_K", "Q5_K", "Q6_K", "Q3_K", "Q8_0", "IQ4_XS", "IQ3_S", "IQ2_S", "IQ4_NL"]


def _pick():
    from freetoken.layers.gguf import GGML_NAME
    from freetoken.models.gguf.reader import iter_gguf_tensors

    picked = {}
    for t in iter_gguf_tensors(GGUF):
        name = GGML_NAME.get(t.ggml_type)
        if len(t.shape) == 2 and name in WANT and name not in picked and t.shape[0] <= 20000:
            picked[name] = t
    return picked


@pytest.fixture(scope="module")
def tensors():
    return _pick()


def _q8_1_round(x: torch.Tensor) -> torch.Tensor:
    """What quantize_q8_1 does to the activation: per-32 absmax/127, round, back to float."""
    n, k = x.shape
    pad = (-k) % 512
    xp = torch.nn.functional.pad(x.float(), (0, pad)).view(n, -1, 32)
    amax = xp.abs().amax(dim=-1, keepdim=True)
    d = amax / 127.0
    q = torch.where(amax == 0, torch.zeros_like(xp), torch.round(xp / d))
    return (q * d).view(n, -1)[:, :k]


def _tol(ref: torch.Tensor) -> float:
    import math

    scale = ref.abs().max().item()
    ulp = 2.0 ** (math.floor(math.log2(scale)) - 7) if scale > 0 else 0.0
    return ulp + 1e-3 * scale + 1e-4


@pytest.mark.parametrize("name", WANT)
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17])
def test_matches_dequantized_reference(tensors, name, rows):
    from freetoken.kernel.gguf import ggml_dequantize, ggml_mul_mat_vec_a8

    t = tensors.get(name)
    if t is None:
        pytest.skip(f"no {name} tensor in the file")
    out_f, in_f = t.shape
    w = t.packed().to("cuda")
    torch.manual_seed(rows)
    x = (torch.randn(rows, in_f, device="cuda") * 0.1).to(torch.bfloat16)
    got = ggml_mul_mat_vec_a8(w, x, t.ggml_type, out_f).float()
    dense = ggml_dequantize(w, t.ggml_type, out_f, in_f).float()
    ref = _q8_1_round(x) @ dense.T
    assert got.shape == (rows, out_f)
    # The kernel returns bf16, so one bf16 ULP at the output magnitude is rounding, and the
    # rest is summation order (the same integer products, added in a different order).
    assert (got - ref).abs().max().item() <= _tol(ref), (name, rows)


@pytest.mark.parametrize("name", ["Q4_K", "IQ4_XS", "Q6_K"])
@pytest.mark.parametrize("rows", [1, 8])
def test_odd_row_count_is_clamped_not_read_past(tensors, name, rows):
    """An odd out_features leaves one row alone in the last block of two: it must be computed
    and the phantom row must neither be written nor read out of bounds."""
    from freetoken.kernel.gguf import ggml_dequantize, ggml_mul_mat_vec_a8

    t = tensors.get(name)
    if t is None:
        pytest.skip(f"no {name} tensor in the file")
    out_f, in_f = t.shape
    odd = out_f - 1
    w = t.packed()[:odd].contiguous().to("cuda")
    torch.manual_seed(0)
    x = (torch.randn(rows, in_f, device="cuda") * 0.1).to(torch.bfloat16)
    got = ggml_mul_mat_vec_a8(w, x, t.ggml_type, odd).float()
    dense = ggml_dequantize(w, t.ggml_type, odd, in_f).float()
    ref = _q8_1_round(x) @ dense.T
    assert got.shape == (rows, odd)
    assert (got - ref).abs().max().item() <= _tol(ref)
    torch.cuda.synchronize()  # an out-of-bounds read would surface here as an illegal access
