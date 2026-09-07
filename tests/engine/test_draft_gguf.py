"""Loading a DFlash draft whose weights stay in ggml blocks.

These cover the parts that need neither a GPU nor the 1 GB checkpoint: that an N-D activation
survives the fused matmul's 2-D-only contract, and that a leaf the name map forgets is refused
instead of running as noise. The arithmetic -- that a Q4_K draft proposes what the bf16 one
proposes -- needs the real weights and is checked by acceptance rate against a GPU run.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from freetoken.engine import draft_gguf
from freetoken.engine.draft_gguf import GGUFDraftLinear, _assert_materialised


def test_a_draft_block_survives_the_two_dimensional_matmul(monkeypatch):
    """fused_mul_mat_gguf takes [tokens, features]; a draft block arrives with a batch dim.

    Handing it a 3-D tensor does not raise -- it returns a wrong rank that fails somewhere far
    from here, which is how this cost an earlier debugging session on the output head.
    """
    seen: list[tuple[int, ...]] = []

    def fake_matmul(x, qweight, quant_type):
        seen.append(tuple(x.shape))
        assert x.dim() == 2, f"the fused matmul takes 2-D, got {tuple(x.shape)}"
        return torch.zeros(x.shape[0], 7, dtype=x.dtype)

    monkeypatch.setattr(draft_gguf, "fused_mul_mat_gguf", fake_matmul)

    layer = GGUFDraftLinear(in_features=256, out_features=7, quant_type=12)
    out = layer(torch.zeros(2, 3, 256))

    assert seen == [(6, 256)], "the leading dims must be folded, not passed through"
    assert out.shape == (2, 3, 7), "and restored on the way out"


def test_a_two_dimensional_activation_is_unchanged(monkeypatch):
    monkeypatch.setattr(
        draft_gguf,
        "fused_mul_mat_gguf",
        lambda x, qweight, quant_type: torch.zeros(x.shape[0], 7, dtype=x.dtype),
    )
    layer = GGUFDraftLinear(in_features=256, out_features=7, quant_type=12)
    assert layer(torch.zeros(5, 256)).shape == (5, 7)


def test_a_leaf_the_map_forgot_is_refused():
    """The skeleton is built on meta, so an unmapped leaf stays meta rather than random.

    This is the guard that caught rotary_emb.inv_freq, which no GGUF carries because it is
    derived from theta rather than stored. Without it the draft would have run on an
    uninitialised rotary table and simply proposed candidates the target rejects -- a poor
    acceptance rate, not a failure.
    """

    class Partly(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.loaded = nn.Parameter(torch.zeros(2))
            self.forgotten = nn.Parameter(torch.empty(2, device="meta"))

    with pytest.raises(ValueError, match="never loaded"):
        _assert_materialised(Partly(), "draft.gguf")


def test_a_fully_loaded_model_passes_the_guard():
    class Whole(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.zeros(2))
            self.register_buffer("packed", torch.zeros(2, dtype=torch.uint8))

    _assert_materialised(Whole(), "draft.gguf")  # must not raise


def test_the_packed_weight_is_a_buffer_so_it_moves_with_the_model():
    """It has to ride .to(device) and appear in state_dict, which a plain attribute would not."""
    layer = GGUFDraftLinear(in_features=256, out_features=8, quant_type=12)
    assert "qweight" in dict(layer.named_buffers())
    assert "qweight" in layer.state_dict()
    assert layer.qweight.dtype == torch.uint8


def test_the_maps_do_not_claim_the_same_tensor_twice():
    """A tensor in two maps would be installed twice, the second silently winning."""
    names: list[str] = []
    for mapping in (
        draft_gguf._LINEAR_MAP,
        draft_gguf._GLOBAL_LINEAR_MAP,
        draft_gguf._EMBEDDING_MAP,
        draft_gguf._PARAM_MAP,
        draft_gguf._GLOBAL_PARAM_MAP,
    ):
        names.extend(mapping)
    assert len(names) == len(set(names))


def test_the_maps_do_not_send_two_tensors_to_one_destination():
    """Two sources for one attribute means one of them is quietly discarded."""
    paths: list[str] = []
    for mapping in (
        draft_gguf._LINEAR_MAP,
        draft_gguf._GLOBAL_LINEAR_MAP,
        draft_gguf._EMBEDDING_MAP,
        draft_gguf._PARAM_MAP,
        draft_gguf._GLOBAL_PARAM_MAP,
    ):
        paths.extend(mapping.values())
    assert len(paths) == len(set(paths))
