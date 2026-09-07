from __future__ import annotations

import os

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.utils import init_logger
from freetoken.layers import (
    BaseOP,
    GemmaRMSNorm,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from freetoken.models.blocks import BaseLLMModel, HiddenStateCapture
from freetoken.utils import nvtx_annotate

from .attention import Qwen3_5Attention
from .gdn import Qwen3_5GatedDeltaNet
from .moe import Qwen3_5DenseMLP, Qwen3_5MoE

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


logger = init_logger(__name__)

_FORWARD_TIMING = bool(os.environ.get("FREETOKEN_FORWARD_TIMING"))
# rows -> [transformer_ms, head_ms, calls]; keyed by row count so a one-row decode and an
# eight-row verification are never averaged into one meaningless number.
_FORWARD_MS: dict[int, list[float]] = {}
# (kind, start_event, end_event) for every sublayer of the forward in flight; drained and
# bucketed by kind once the forward has been synchronised.
_PENDING_EVENTS: list = []
# rows -> kind -> accumulated ms across calls
_SUBLAYER_MS: dict[int, dict[str, float]] = {}


class Qwen3_5DecoderLayer(BaseOP):
    """Pre-norm hybrid block: ``x = x + mixer(input_norm(x)); x = x + moe(post_norm(x))``,
    where the mixer is a GatedDeltaNet (linear layers) or gated attention (full layers).
    All norms are Gemma-style (1+weight)."""

    def __init__(self, config: ModelConfig, layer_id: int):
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        if self._is_linear:
            g = config.linear_attention_group()
            assert g is not None
            self.linear_attn = Qwen3_5GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_k_heads=g.num_key_heads,
                num_v_heads=g.num_value_heads,
                head_k_dim=g.key_head_dim,
                head_v_dim=g.value_head_dim,
                conv_kernel_size=g.conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                layer_id=layer_id,
                expert_quant=config.expert_quant,
                attn_quant=config.attn_quant,
            )
        else:
            self.self_attn = Qwen3_5Attention(config, layer_id)
        # Dense variants (num_experts==0, e.g. Qwen3.6-27B) use a plain SwiGLU MLP instead of
        # the routed MoE block; both expose ``forward(hidden)->hidden`` and the same key prefix.
        self.mlp = Qwen3_5MoE(config, layer_id) if config.moe_enabled else Qwen3_5DenseMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor, residual: torch.Tensor | None):
        # Residual-stream form: fuse each residual-add into the next RMSNorm
        # (GemmaRMSNorm.forward_add_residual) so add + norm are one kernel per sublayer.
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm.forward(hidden)
        else:
            hidden, residual = self.input_layernorm.forward_add_residual(hidden, residual)
        if _FORWARD_TIMING and not torch.cuda.is_current_stream_capturing():
            # CUDA events, not host syncs: a sync per layer would serialise the stream and time
            # the stalls it created. Elapsed times are read once, after the whole forward.
            e0 = torch.cuda.Event(enable_timing=True); e0.record()
            hidden = self.linear_attn.forward(hidden) if self._is_linear else self.self_attn.forward(hidden)
            e1 = torch.cuda.Event(enable_timing=True); e1.record()
            hidden, residual = self.post_attention_layernorm.forward_add_residual(hidden, residual)
            e2 = torch.cuda.Event(enable_timing=True); e2.record()
            hidden = self.mlp.forward(hidden)
            e3 = torch.cuda.Event(enable_timing=True); e3.record()
            _PENDING_EVENTS.append(("gdn" if self._is_linear else "attn", e0, e1))
            _PENDING_EVENTS.append(("norm", e1, e2))
            _PENDING_EVENTS.append(("mlp", e2, e3))
            return hidden, residual
        hidden = self.linear_attn.forward(hidden) if self._is_linear else self.self_attn.forward(hidden)
        hidden, residual = self.post_attention_layernorm.forward_add_residual(hidden, residual)
        hidden = self.mlp.forward(hidden)
        return hidden, residual


class Qwen3_5Model(BaseOP, HiddenStateCapture):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen3_5DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        if not self._capture_layer_ids:
            for layer in self.layers.op_list:
                x, residual = layer.forward(x, residual)
            x, _ = self.norm.forward_add_residual(x, residual)
            return x

        # Same carried-residual shape as the dense Qwen3: a layer returns (mlp_out, residual)
        # and the add itself happens in the next layer's fused add+norm, so layer i's hidden
        # state in HF terms is residual + x and has to be materialised -- both buffers are
        # rewritten in place further down the stack, so a reference would go stale.
        captured = self._new_capture_store()
        if -1 in self._capture_layer_ids:
            captured[0] = x.clone()
        for layer_id, layer in enumerate(self.layers.op_list):
            x, residual = layer.forward(x, residual)
            if layer_id in self._capture_layer_ids:
                captured[layer_id + 1] = residual + x
        self._captured_hidden_states = captured
        x, _ = self.norm.forward_add_residual(x, residual)
        return x


class Qwen3_5MoEForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3_5Model(config)
        if getattr(config, "lm_head_quant", "none") == "nvfp4":
            # checkpoint stores the (untied) lm_head as NVFP4: keep it native (W4A16) -- the
            # bf16 dequant of this ~1 GB matrix was the single largest decode kernel.
            from freetoken.kernel.triton.nvfp4_linear import Nvfp4LMHead

            assert not config.tie_word_embeddings, "NVFP4 lm_head assumes untied embeddings"
            self.lm_head = Nvfp4LMHead(
                num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
            )
        else:
            self.lm_head = ParallelLMHead(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                tie_word_embeddings=config.tie_word_embeddings,
                tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            )
        super().__init__()

        # A GGUF checkpoint carries native block-quantized weights: swap the dense
        # projections + embedding for GGUF-quant ops so the packed buffers have somewhere
        # to land (routed experts stay on the offload cache). Mirrors gemma4/model.py.
        from .gguf import convert_qwen35_to_gguf, is_gguf_model

        if is_gguf_model(config):
            assert config.gguf_model_path is not None, (
                "expert_quant=='gguf' but ModelConfig.gguf_model_path is unset; the per-tensor "
                "ggml types can only be read from the file"
            )
            convert_qwen35_to_gguf(self, config, model_path=config.gguf_model_path)

    def forward(self) -> torch.Tensor:
        # Never time during graph capture: torch.cuda.synchronize() is not permitted on a
        # capturing stream, and the failure surfaces as "operation not permitted when stream
        # is capturing" from whichever kernel is capturing at the time.
        if not _FORWARD_TIMING or torch.cuda.is_current_stream_capturing():
            output = self.model.forward(get_global_ctx().batch.input_ids)
            return self.lm_head.forward(output)
        # Split the forward in two under FREETOKEN_FORWARD_TIMING, so a speculative
        # verification that costs four decode steps can be attributed to the transformer or to
        # the head instead of guessed at. Synchronised: CUDA's asynchrony would otherwise
        # charge the transformer's cost to whichever call waits for it.
        import time

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        output = self.model.forward(get_global_ctx().batch.input_ids)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        logits = self.lm_head.forward(output)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        rows = int(output.shape[0])
        _FORWARD_MS.setdefault(rows, [0.0, 0.0, 0])
        entry = _FORWARD_MS[rows]
        entry[0] += (t1 - t0) * 1000.0
        entry[1] += (t2 - t1) * 1000.0
        entry[2] += 1
        by_kind = _SUBLAYER_MS.setdefault(rows, {})
        for kind, e_start, e_end in _PENDING_EVENTS:
            by_kind[kind] = by_kind.get(kind, 0.0) + e_start.elapsed_time(e_end)
        _PENDING_EVENTS.clear()
        if entry[2] % 40 == 0:
            parts = ", ".join(f"{k}={v / entry[2]:.1f}" for k, v in sorted(by_kind.items()))
            logger.info_rank0(
                f"forward @{rows} rows x{entry[2]}: transformer={entry[0] / entry[2]:.1f}ms "
                f"head={entry[1] / entry[2]:.1f}ms | sublayers(ms): {parts}"
            )
        return logits


__all__ = ["Qwen3_5MoEForCausalLM"]
