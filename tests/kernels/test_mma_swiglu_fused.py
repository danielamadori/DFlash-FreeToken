"""Folding the SwiGLU into the MMQ's activation quantizer must not change a single bit.

The tensor-core MMQ quantizes its activation to q8_1 before multiplying, so it can apply
silu(gate) * up while it is there instead of reading back a bf16 tensor another kernel just
wrote. The fused kernel therefore computes the activation with the same ex2.approx.f32
instruction the triton act_and_mul kernel uses, and rounds it to bf16 before quantizing --
because that is exactly what the unfused pair does. Anything less and the model's text moves.

Both layouts are covered: gate and up as separate contiguous tensors (the layers whose ffn_gate
and ffn_up carry different ggml types) and as the two halves of one (the layers where they
match, where the halves are strided views).
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


@pytest.fixture(scope="module")
def down_tensors():
    """One ffn_down-shaped tensor per ported quant type: out 5120, in 17408."""
    from freetoken.layers.gguf import GGML_NAME, MMA_TYPES
    from freetoken.models.gguf.reader import iter_gguf_tensors

    picked = {}
    for t in iter_gguf_tensors(GGUF):
        if len(t.shape) == 2 and t.ggml_type in MMA_TYPES and t.shape == (5120, 17408):
            picked.setdefault(GGML_NAME.get(t.ggml_type), t)
    return picked


@pytest.mark.parametrize("rows", [24, 64, 127, 128, 256])
@pytest.mark.parametrize("halves", [False, True])
def test_fused_matches_the_two_step_form(down_tensors, rows, halves):
    from freetoken.kernel.gguf import ggml_mul_mat_mma, ggml_mul_mat_mma_swiglu
    from freetoken.layers.activation import silu_and_mul, silu_and_mul_pair

    if not down_tensors:
        pytest.skip("no 5120x17408 tensor of a ported type in the file")

    for name, t in down_tensors.items():
        w = t.packed().to("cuda")
        d = t.shape[1]
        torch.manual_seed(31 + rows)
        if halves:
            fused_act = torch.randn(rows, 2 * d, dtype=torch.bfloat16, device="cuda") * 0.4
            gate, up = fused_act[:, :d], fused_act[:, d:]
            act = silu_and_mul(fused_act)
        else:
            gate = torch.randn(rows, d, dtype=torch.bfloat16, device="cuda") * 0.4
            up = torch.randn(rows, d, dtype=torch.bfloat16, device="cuda") * 0.4
            act = silu_and_mul_pair(gate, up)

        want = ggml_mul_mat_mma(w, act, t.ggml_type, 5120)
        got = ggml_mul_mat_mma_swiglu(w, gate, up, t.ggml_type, 5120)
        assert torch.equal(got, want), (
            f"{name} rows={rows} halves={halves}: "
            f"max |diff| = {(got.float() - want.float()).abs().max().item()}"
        )
        del w, gate, up, act, want, got
        torch.cuda.empty_cache()
