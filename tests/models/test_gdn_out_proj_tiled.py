"""out_proj keeps llama.cpp's tiled V-head columns; the GDN op permutes its activation.

The identity the converter relies on, with the dense un-tiling the loader used to apply to
the weight as the reference:

    x_tiled @ W_tiled.T == x_grouped @ _ungroup_v(W_tiled, 1, ...).T
"""
import torch

from freetoken.models.qwen3_5_moe.gdn import v_grouped_to_tiled
from freetoken.models.qwen3_5_moe.gguf import _ungroup_v

# Qwen3.8-27B geometry: 16 K heads, 32 V heads (R = 2), head_v_dim 128.
K, R, D = 16, 2, 128
V = K * R * D


def _tiled(x, rows):
    return v_grouped_to_tiled(x, rows, K, K * R, D).reshape(rows, -1)


def test_activation_permutation_matches_weight_untiling():
    # Small integers so both matmuls are exact and the comparison can be bit-for-bit;
    # a wrong permutation cannot hide behind summation-order rounding.
    torch.manual_seed(0)
    rows, out_f = 8, 96
    w_tiled = torch.randint(-3, 4, (out_f, V)).to(torch.float64)
    x = torch.randint(-3, 4, (rows, V)).to(torch.float64)
    w_grouped = _ungroup_v(w_tiled, 1, K, R, D)
    ref = x @ w_grouped.T
    got = _tiled(x, rows) @ w_tiled.T
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


def test_permutation_index_is_r_k_d():
    # Column r*K*D + k*D + d of the tiled activation reads grouped column k*R*D + r*D + d.
    x = torch.arange(V, dtype=torch.float32).unsqueeze(0)
    idx = _tiled(x, 1)[0].long()
    expected = torch.arange(V).view(K, R, D).permute(1, 0, 2).reshape(-1)
    assert torch.equal(idx, expected)
    # It is a permutation: every column appears exactly once.
    assert torch.equal(idx.sort().values, torch.arange(V))


def test_single_row_and_norm_layout():
    # The op receives the gated-norm output as [rows*num_v_heads, head_v_dim]; a decode
    # step (one row) must go through the same reshape without a batch axis to lean on.
    x = torch.randn(1 * K * R, D)
    y = _tiled(x, 1)
    assert y.shape == (1, V)
    assert torch.equal(y[0].view(R, K, D), x.view(K, R, D).transpose(0, 1))


def test_no_reorder_when_heads_match():
    # R == 1: tiled and grouped layouts coincide; the transform must be the identity.
    k, d = 4, 8
    x = torch.randn(3 * k, d)
    assert torch.equal(v_grouped_to_tiled(x, 3, k, k, d).reshape(3, -1), x.reshape(3, -1))
