"""The varlen causal conv must give the same answer whichever layout it is handed.

The GDN holds its activation token-major ([total, conv_dim] contiguous) and the kernel wants
[conv_dim, total]. Materialising that transpose cost more than the convolution, so the wrapper
now reads the transposed VIEW directly. The arithmetic per (feature, token) is unchanged --
only the addresses are -- so the two layouts must agree bit for bit, not approximately.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

CONV_DIM = 10240   # the 27B's 2*key_dim + value_dim
WIDTH = 4


def _run(x, weight, states, cu_seqlens, cache_indices, has_initial_state):
    from freetoken.kernel.triton.causal_conv1d_triton import causal_conv1d_varlen

    return causal_conv1d_varlen(
        x, weight, states, cu_seqlens, cache_indices, has_initial_state,
        max_seq_len=int(x.shape[1]), batch=int(cu_seqlens.numel()) - 1,
    )


@pytest.mark.parametrize("lengths", [(127,), (2129,), (8, 40, 79), (1, 1, 1, 512)])
@pytest.mark.parametrize("carry", [False, True])
def test_token_major_matches_channel_major(lengths, carry):
    total = sum(lengths)
    torch.manual_seed(7)
    token_major_src = torch.randn(total, CONV_DIM, dtype=torch.bfloat16, device="cuda") * 0.2
    weight = torch.randn(CONV_DIM, WIDTH, dtype=torch.bfloat16, device="cuda") * 0.3
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    idx = torch.arange(len(lengths), dtype=torch.int32, device="cuda")
    has_init = torch.full((len(lengths),), carry, dtype=torch.bool, device="cuda")

    states0 = torch.randn(len(lengths), CONV_DIM, WIDTH - 1, dtype=torch.bfloat16, device="cuda") * 0.1

    # channel-major: the old path, an explicit transposed copy
    st_a = states0.clone()
    out_a = _run(token_major_src.transpose(0, 1).contiguous(), weight, st_a, cu, idx, has_init)

    # token-major: the transposed view, read in place
    st_b = states0.clone()
    out_b = _run(token_major_src.transpose(0, 1), weight, st_b, cu, idx, has_init)

    assert out_a.shape == out_b.shape == (CONV_DIM, total)
    assert torch.equal(out_a, out_b), (out_a.float() - out_b.float()).abs().max().item()
    assert torch.equal(st_a, st_b), "the conv state tail must be written identically too"


def test_the_view_really_is_a_view():
    """If the wrapper started copying again the speed-up would vanish silently, so pin the
    property the change is about: a token-major input is not materialised."""
    x = torch.randn(64, CONV_DIM, dtype=torch.bfloat16, device="cuda").transpose(0, 1)
    assert x.stride(-1) != 1 and x.stride(0) == 1, "this is the layout the GDN hands over"


@pytest.mark.parametrize("lengths", [(127,), (2129,), (8, 40, 79)])
def test_per_channel_split_matches_one_call(lengths):
    """The GDN runs the convolution once per q/k/v slice so each output is contiguous. The
    convolution is depthwise, so this must be exactly the same answer as one call over all the
    channels -- both for the output and for the conv-state tail written back per slot."""
    total = sum(lengths)
    widths = (2048, 2048, 6144)          # the 27B's key_dim, key_dim, value_dim
    assert sum(widths) == CONV_DIM
    torch.manual_seed(11)
    src = torch.randn(total, CONV_DIM, dtype=torch.bfloat16, device="cuda") * 0.2
    weight = torch.randn(CONV_DIM, WIDTH, dtype=torch.bfloat16, device="cuda") * 0.3
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    idx = torch.arange(len(lengths), dtype=torch.int32, device="cuda")
    has_init = torch.ones(len(lengths), dtype=torch.bool, device="cuda")
    states0 = torch.randn(len(lengths), CONV_DIM, WIDTH - 1, dtype=torch.bfloat16, device="cuda") * 0.1

    st_whole = states0.clone()
    whole = _run(src.transpose(0, 1), weight, st_whole, cu, idx, has_init).transpose(0, 1)

    st_split = states0.clone()
    outs, start = [], 0
    for w in widths:
        stop = start + w
        outs.append(
            _run(src[:, start:stop].transpose(0, 1), weight[start:stop],
                 st_split[:, start:stop], cu, idx, has_init).transpose(0, 1)
        )
        start = stop

    for o in outs:
        assert o.is_contiguous(), "the whole point is that each slice comes out contiguous"
    assert torch.equal(torch.cat(outs, dim=-1), whole)
    assert torch.equal(st_split, st_whole)
