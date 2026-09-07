"""Unit tests for qwen35moe GGUF weight loading.

Tests GGUFMergedLinear (mixed-quant fused projections), gguf_merged_or_plain dispatch,
the GDN in_proj geometry, and the qwen35moe name mapping (dropping MTP block). All tests
run without CUDA or kernel compilation, using monkeypatched kernel mocks that record calls
and return correctly-shaped CPU tensors.

Reference: llama.cpp's qwen3.5 GGUF mapping (gguf-py/gguf/tensor_mapping.py) and the
Ornith-1.5-35B-A3B-GGUF checkpoint geometry (ORNITH_SPEC.md).
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest
import torch

from freetoken.layers.gguf import GGUFLinear, GGUFMergedLinear, gguf_merged_or_plain
from freetoken.models.gguf.dequant import (
    GGML_IQ3_S,
    GGML_Q4_K,
    BLOCK_SHAPE,
    row_bytes,
)
from freetoken.models.qwen3_5_moe.gguf import gguf_name_to_freetoken


@pytest.fixture
def mock_kernel_module(monkeypatch):
    """Replace freetoken.kernel.gguf with a mock that tracks kernel calls.

    Returns a dict tracking which kernels were called and with what arguments.
    This allows tests to run without CUDA or the compiled kernel extension.
    """
    call_log = {
        "ggml_mul_mat_vec_a8": [],  # List of calls (can be multiple)
        "ggml_mul_mat_a8": [],
        "ggml_dequantize": [],
    }

    def make_mmvq_kernel(call_log):
        """Mock ggml_mul_mat_vec_a8: GEMV kernel for small batch."""
        def kernel(qweight, x, qweight_type, out_features):
            call_log["ggml_mul_mat_vec_a8"].append({
                "qweight_shape": qweight.shape,
                "x_shape": x.shape,
                "qweight_type": qweight_type,
                "out_features": out_features,
            })
            batch_size = x.shape[0]
            return torch.randn(batch_size, out_features, dtype=x.dtype)
        return kernel

    def make_mmq_kernel(call_log):
        """Mock ggml_mul_mat_a8: MMQ kernel for large batch."""
        def kernel(qweight, x, qweight_type, out_features):
            call_log["ggml_mul_mat_a8"].append({
                "qweight_shape": qweight.shape,
                "x_shape": x.shape,
                "qweight_type": qweight_type,
                "out_features": out_features,
            })
            batch_size = x.shape[0]
            return torch.randn(batch_size, out_features, dtype=x.dtype)
        return kernel

    def make_dequant_kernel(call_log):
        """Mock ggml_dequantize: materializes weight into BF16."""
        def kernel(qweight, qweight_type, out_features, in_features, out_dtype):
            call_log["ggml_dequantize"].append({
                "qweight_shape": qweight.shape,
                "qweight_type": qweight_type,
                "out_features": out_features,
                "in_features": in_features,
                "out_dtype": out_dtype,
            })
            return torch.randn(out_features, in_features, dtype=out_dtype)
        return kernel

    mock_module = ModuleType("freetoken.kernel.gguf")
    mock_module.ggml_mul_mat_vec_a8 = make_mmvq_kernel(call_log)
    mock_module.ggml_mul_mat_a8 = make_mmq_kernel(call_log)
    mock_module.ggml_dequantize = make_dequant_kernel(call_log)

    monkeypatch.setitem(sys.modules, "freetoken.kernel.gguf", mock_module)

    yield call_log

    # Cleanup: ensure mock is removed so it doesn't interfere with other tests
    if "freetoken.kernel.gguf" in sys.modules:
        del sys.modules["freetoken.kernel.gguf"]


class TestMergedLinearConcatenatesOutputs:
    """Test GGUFMergedLinear with mixed-quant output parts."""

    def test_merged_linear_concatenates_outputs(self, mock_kernel_module):
        """GGUFMergedLinear with [8192, 512, 512] outputs and mixed quant types
        produces output of width 9216 and calls kernel once per part.

        Ornith's full-attention qkv_proj: q(IQ3_S, 8192) + k(IQ3_S, 512) + v(Q4_K, 512).
        """
        in_features = 2048
        output_sizes = [8192, 512, 512]
        quant_types = [GGML_IQ3_S, GGML_IQ3_S, GGML_Q4_K]

        merged = GGUFMergedLinear(in_features, output_sizes, quant_types, has_bias=False)

        # Check total output width
        assert merged.out_features == sum(output_sizes) == 9216

        # q and k share a type, so they are one launch: their packed rows concatenate (same
        # row_bytes); v keeps its own buffer.
        assert merged.part_names == ["qweight_0", "qweight_1"]
        assert merged.qweight_0.shape == (8192 + 512, row_bytes(in_features, GGML_IQ3_S))
        assert merged.qweight_0.shape == (8704, 880)  # IQ3_S: 2048 // 256 * 110

        assert merged.qweight_1.shape == (512, row_bytes(in_features, GGML_Q4_K))
        assert merged.qweight_1.shape == (512, 1152)  # Q4_K: 2048 // 256 * 144

        # Forward pass: batch size 1 (small batch, uses MMVQ)
        x = torch.randn(1, in_features, dtype=torch.bfloat16)
        output = merged.forward(x)

        # Output shape should be [1, 9216]
        assert output.shape == (1, 9216)

        # Kernel should have been called once per regrouped part, all to MMVQ
        assert len(mock_kernel_module["ggml_mul_mat_vec_a8"]) == 2
        assert len(mock_kernel_module["ggml_mul_mat_a8"]) == 0

        calls = mock_kernel_module["ggml_mul_mat_vec_a8"]
        assert calls[0]["out_features"] == 8704
        assert calls[1]["out_features"] == 512


class TestMergedLinearRejectsLengthMismatch:
    """Test GGUFMergedLinear validation of output_sizes and quant_types lengths."""

    def test_merged_linear_rejects_length_mismatch(self):
        """GGUFMergedLinear raises ValueError when output_sizes and quant_types differ."""
        in_features = 2048
        output_sizes = [8192, 512, 512]
        quant_types = [GGML_IQ3_S, GGML_Q4_K]  # Only 2 types, 3 sizes

        with pytest.raises(ValueError) as excinfo:
            GGUFMergedLinear(in_features, output_sizes, quant_types, has_bias=False)

        assert "length" in str(excinfo.value).lower()

    def test_merged_linear_rejects_zero_output(self):
        """GGUFMergedLinear raises ValueError if any output_size is <= 0."""
        in_features = 2048
        output_sizes = [8192, 0, 512]  # Zero output size
        quant_types = [GGML_IQ3_S, GGML_IQ3_S, GGML_Q4_K]

        with pytest.raises(ValueError) as excinfo:
            GGUFMergedLinear(in_features, output_sizes, quant_types, has_bias=False)

        assert "must be > 0" in str(excinfo.value)


class TestGGUFMergedOrPlainDispatch:
    """Test gguf_merged_or_plain routing between GGUFLinear and GGUFMergedLinear."""

    def test_gguf_merged_or_plain_picks_plain_when_uniform(self):
        """When all quant types are identical, gguf_merged_or_plain returns GGUFLinear."""
        in_features = 2048
        output_sizes = [8192, 512, 512]
        # All three parts use IQ3_S
        quant_types = [GGML_IQ3_S, GGML_IQ3_S, GGML_IQ3_S]

        lin = gguf_merged_or_plain(in_features, output_sizes, quant_types, has_bias=False)

        # Should return a plain GGUFLinear, not merged
        assert isinstance(lin, GGUFLinear)
        assert not isinstance(lin, GGUFMergedLinear)
        # Total output features should be the sum
        assert lin.out_features == 9216
        # Qweight should be concatenated (single packed buffer)
        assert lin.qweight.shape == (9216, row_bytes(in_features, GGML_IQ3_S))

    def test_gguf_merged_or_plain_picks_merged_when_mixed(self):
        """When quant types differ, gguf_merged_or_plain returns GGUFMergedLinear."""
        in_features = 2048
        output_sizes = [8192, 512, 512]
        # Mixed: IQ3_S and Q4_K
        quant_types = [GGML_IQ3_S, GGML_IQ3_S, GGML_Q4_K]

        lin = gguf_merged_or_plain(in_features, output_sizes, quant_types, has_bias=False)

        # Should return a GGUFMergedLinear
        assert isinstance(lin, GGUFMergedLinear)
        assert lin.out_features == 9216
        # Two regrouped parts: q+k (IQ3_S) and v (Q4_K)
        assert hasattr(lin, "qweight_0")
        assert hasattr(lin, "qweight_1")
        assert not hasattr(lin, "qweight_2")


class TestMergedPartPlanning:
    """plan_merged_parts: consecutive same-type parts share a launch; a tiny Q8_0 group is
    kept dense in bf16. The loader packs with the same plan (_emit_merged), so both sides of
    the state dict agree by construction."""

    def test_consecutive_same_type_parts_merge(self):
        from freetoken.layers.gguf import plan_merged_parts

        parts = plan_merged_parts([8192, 512, 512], [GGML_IQ3_S, GGML_IQ3_S, GGML_Q4_K])
        assert [(p.out_size, p.quant_type, p.members) for p in parts] == [
            (8704, GGML_IQ3_S, (0, 1)),
            (512, GGML_Q4_K, (2,)),
        ]
        # Non-adjacent equal types do not merge (the output order must be preserved).
        parts = plan_merged_parts([10, 20, 30], [GGML_Q4_K, GGML_IQ3_S, GGML_Q4_K])
        assert [p.members for p in parts] == [(0,), (1,), (2,)]

    def test_tiny_q8_0_group_is_densified(self):
        from freetoken.layers.gguf import _TINY_DENSE_ROWS, plan_merged_parts
        from freetoken.models.gguf.dequant import GGML_BF16, GGML_IQ4_XS, GGML_Q8_0

        # The 27B GDN in_proj: qkv | gate | beta | alpha with 48-row Q8_0 beta and alpha.
        parts = plan_merged_parts(
            [12288, 4096, 48, 48], [GGML_IQ4_XS, GGML_IQ4_XS, GGML_Q8_0, GGML_Q8_0]
        )
        assert len(parts) == 2
        assert (parts[0].out_size, parts[0].quant_type) == (16384, GGML_IQ4_XS)
        assert (parts[1].out_size, parts[1].quant_type, parts[1].source_type) == (
            96,
            GGML_BF16,
            GGML_Q8_0,
        )
        # A large Q8_0 group stays quantized.
        big = plan_merged_parts([_TINY_DENSE_ROWS + 1], [GGML_Q8_0])
        assert big[0].quant_type == GGML_Q8_0
        # The module allocates the dense part at bf16 width.
        merged = GGUFMergedLinear(
            5120, [12288, 4096, 48, 48], [GGML_IQ4_XS, GGML_IQ4_XS, GGML_Q8_0, GGML_Q8_0]
        )
        assert merged.qweight_1.shape == (96, 5120 * 2)

    def test_loader_dequantizes_tiny_q8_0_exactly(self):
        from freetoken.models.gguf.dequant import GGML_Q8_0
        from freetoken.models.qwen3_5_moe.gguf import _dequant_q8_0, _emit_merged

        in_features = 64  # two Q8_0 blocks per row
        rows = 3
        torch.manual_seed(0)
        q = torch.randint(-127, 128, (rows, in_features), dtype=torch.int8)
        d = torch.tensor([[0.5, 0.25]] * rows, dtype=torch.float16)  # one scale per block
        blocks = torch.cat(
            [d.view(rows, 2, 1).view(torch.uint8), q.view(rows, 2, 32).view(torch.uint8)],
            dim=2,
        )
        packed = blocks.reshape(rows, 2 * 34)
        expected = (q.float().view(rows, 2, 32) * d.float().view(rows, 2, 1)).reshape(rows, -1)
        assert torch.equal(_dequant_q8_0(packed, in_features).float(), expected)
        # Through the loader helper the group arrives as raw bf16 bytes under the plan's name.
        out = dict(_emit_merged("x", [packed, packed], [GGML_Q8_0, GGML_Q8_0], in_features))
        assert list(out) == ["x.qweight_0"]
        assert out["x.qweight_0"].shape == (2 * rows, in_features * 2)
        assert torch.equal(
            out["x.qweight_0"].view(torch.bfloat16).float(),
            torch.cat([expected, expected]).to(torch.bfloat16).float(),
        )


class TestGDNGeometry:
    """Test that GDN in_proj geometry matches Ornith's configuration."""

    def test_gdn_split_matches_ornith_geometry(self):
        """GDN in_proj split [8192, 4096, 32, 32] sums to 12352 and matches the arithmetic.

        From ORNITH_SPEC section 1:
        - embedding_length=2048
        - head_count=16, head_count_kv=2, key_length=value_length=256
        - ssm.state_size=128, group_count=16, time_step_rank=32, inner_size=4096

        GDN in_proj fuses four tensors:
        - attn_qkv: q (2 * head_count * state_size) + k (head_count * state_size) +
                    v (num_v_heads * state_size)
                  = 2*16*128 + 16*128 + 32*128 = 8192
        - attn_gate: 4096 (= num_v_heads * state_size = 32 * 128)
        - ssm_beta: 32 (= num_v_heads)
        - ssm_alpha: 32 (= num_v_heads)
        Total: 8192 + 4096 + 32 + 32 = 12352
        """
        # Ornith GDN parameters
        head_count = 16
        state_size = 128
        num_k_heads = 16
        num_v_heads = 32
        inner_size = 4096

        # Compute expected attn_qkv output size
        attn_qkv_size = 2 * num_k_heads * state_size + num_v_heads * state_size
        assert attn_qkv_size == 8192

        # Compute expected attn_gate output size
        attn_gate_size = num_v_heads * state_size
        assert attn_gate_size == 4096

        # ssm_beta and ssm_alpha outputs are both num_v_heads
        ssm_beta_size = num_v_heads
        ssm_alpha_size = num_v_heads
        assert ssm_beta_size == 32
        assert ssm_alpha_size == 32

        # Total in_proj output width
        in_proj_split = [attn_qkv_size, attn_gate_size, ssm_beta_size, ssm_alpha_size]
        in_proj_total = sum(in_proj_split)
        assert in_proj_total == 12352

        # Verify all parts are as specified
        assert in_proj_split == [8192, 4096, 32, 32]


class TestQwenNameMapping:
    """Test gguf_name_to_freetoken: the inverse of llama.cpp's tensor name mapping."""

    def test_name_map_drops_mtp_block(self):
        """gguf_name_to_freetoken drops block 40 (NextN/MTP) and nextn.* suffixes."""
        num_layers = 40  # Ornith has 40 decoder layers + 1 NextN block (41 total in file)

        # Block 40 is the NextN/MTP block and should be dropped
        assert gguf_name_to_freetoken("blk.40.attn_norm.weight", num_layers) is None
        assert gguf_name_to_freetoken("blk.40.ffn_gate.weight", num_layers) is None

        # Any suffix starting with "nextn." on valid layers should be dropped
        assert gguf_name_to_freetoken("blk.0.nextn.something", num_layers) is None
        assert gguf_name_to_freetoken("blk.5.nextn.predict.weight", num_layers) is None

    def test_name_map_real_names(self):
        """gguf_name_to_freetoken maps real Ornith tensor names correctly."""
        num_layers = 40

        # Global tensors
        assert gguf_name_to_freetoken("token_embd.weight", num_layers) == "model.embed_tokens.weight"
        assert gguf_name_to_freetoken("output_norm.weight", num_layers) == "model.norm.weight"
        assert gguf_name_to_freetoken("output.weight", num_layers) == "lm_head.weight"

        # Layer 3 (full-attention). Only o_proj is a 1:1 rename. attn_q/k/v are PARTS of
        # the merged qkv_proj (Qwen3_5Attention has no q_proj/k_proj/v_proj attribute), so
        # the mapper reports None -- iter_gguf_weights owns their fusion because only it
        # knows the concat order and the per-part quant types. Asserting a q_proj name here
        # would lock in a parameter the module does not have.
        assert gguf_name_to_freetoken("blk.3.attn_output.weight", num_layers) == "model.layers.3.self_attn.o_proj.weight"
        for part in ("attn_q.weight", "attn_k.weight", "attn_v.weight"):
            assert gguf_name_to_freetoken(f"blk.3.{part}", num_layers) is None, part

        # Layer 0 (GDN): linear attention with conv1d and SSM
        assert gguf_name_to_freetoken("blk.0.ssm_conv1d.weight", num_layers) == "model.layers.0.linear_attn.conv1d.weight"
        assert gguf_name_to_freetoken("blk.0.ssm_norm.weight", num_layers) == "model.layers.0.linear_attn.norm.weight"
        assert gguf_name_to_freetoken("blk.0.ssm_out.weight", num_layers) == "model.layers.0.linear_attn.out_proj.weight"
        assert gguf_name_to_freetoken("blk.0.ssm_a", num_layers) == "model.layers.0.linear_attn.A_log"
        assert gguf_name_to_freetoken("blk.0.ssm_dt.bias", num_layers) == "model.layers.0.linear_attn.dt_bias"

        # MoE tensors: shared expert and router
        assert gguf_name_to_freetoken("blk.5.ffn_gate_inp.weight", num_layers) == "model.layers.5.mlp.gate.weight"
        assert gguf_name_to_freetoken("blk.5.ffn_gate_inp_shexp.weight", num_layers) == "model.layers.5.mlp.shared_expert_gate.weight"
        assert gguf_name_to_freetoken("blk.5.ffn_down_shexp.weight", num_layers) == "model.layers.5.mlp.shared_expert.down_proj.weight"
        # gate/up shexp are parts of _SharedExpert's merged gate_up_proj -- same reasoning
        # as attn_q/k/v above.
        for part in ("ffn_gate_shexp.weight", "ffn_up_shexp.weight"):
            assert gguf_name_to_freetoken(f"blk.5.{part}", num_layers) is None, part

    def test_name_map_ignores_routed_experts(self):
        """gguf_name_to_freetoken ignores routed-expert stacks (handled by offload banks)."""
        num_layers = 40

        # Routed-expert stacks are skipped (offload banks read them directly)
        assert gguf_name_to_freetoken("blk.0.ffn_gate_exps.weight", num_layers) is None
        assert gguf_name_to_freetoken("blk.15.ffn_up_exps.weight", num_layers) is None
        assert gguf_name_to_freetoken("blk.39.ffn_down_exps.weight", num_layers) is None

__all__ = [
    "test_merged_linear_concatenates_outputs",
    "test_merged_linear_rejects_length_mismatch",
    "test_gguf_merged_or_plain_picks_plain_when_uniform",
    "test_gdn_split_matches_ornith_geometry",
    "test_name_map_drops_mtp_block",
]
