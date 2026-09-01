from __future__ import annotations

import sys
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from freetoken.utils import init_logger

logger = init_logger(__name__)


def _ensure_dflash_importable() -> None:
    """Ensure dflash package can be imported from Agents submodule if not in sys.path."""
    try:
        import dflash  # noqa: F401
    except ImportError:
        import os
        from pathlib import Path

        # Search sibling or parent directories for dflash submodule
        current = Path(__file__).resolve()
        for parent in current.parents:
            candidate = parent / "dflash"
            if candidate.is_dir() and (candidate / "dflash").is_dir():
                sys.path.insert(0, str(candidate))
                break


def _output_head(target: nn.Module) -> nn.Module:
    """Output head of the target, transformers-style or FreeToken-style."""
    head = getattr(target, "lm_head", None)
    if head is not None:
        return head
    get_output_embeddings = getattr(target, "get_output_embeddings", None)
    if get_output_embeddings is None:
        raise TypeError(f"cannot locate the output head of {type(target).__name__}")
    return get_output_embeddings()


def _target_embedding_weight(target: nn.Module) -> torch.Tensor:
    """Raw, un-normalised input embedding matrix of the target model.

    DFlash's own helper reaches it through `get_input_embeddings()`, which exists only on
    transformers modules. FreeToken models are plain objects holding a
    VocabParallelEmbedding, and a vocabulary sharded across ranks would have to be
    gathered first, so refuse that instead of drafting against a slice of the vocabulary.
    """
    get_input_embeddings = getattr(target, "get_input_embeddings", None)
    if get_input_embeddings is not None:
        return get_input_embeddings().weight

    embedding = getattr(getattr(target, "model", None), "embed_tokens", None)
    if embedding is None:
        raise TypeError(
            f"cannot locate the input embeddings of {type(target).__name__}: DFlash needs "
            "either a transformers `get_input_embeddings()` or a `.model.embed_tokens`"
        )
    if getattr(embedding, "tp_size", 1) != 1:
        raise NotImplementedError(
            "DFlash drafting against a tensor-parallel sharded vocabulary is not supported"
        )
    return embedding.weight


def _target_output_logits(target: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Project draft hidden states with the target's output head.

    FreeToken's ParallelLMHead.forward reads the engine's current batch to slice the last
    prefill position out, which is wrong for a [1, K, hidden] draft block, so apply the
    linear it wraps directly.
    """
    head = _output_head(target)
    if callable(head):
        return head(hidden_states)
    if getattr(head, "tp_size", 1) != 1:
        raise NotImplementedError(
            "DFlash drafting against a tensor-parallel sharded output head is not supported"
        )
    weight_owner = getattr(head, "tied_embedding", None) or head
    return F.linear(hidden_states, weight_owner.weight, getattr(head, "bias", None))


def _sampling_probs(
    logits: torch.Tensor,
    temperature: float,
    top_p: float = 1.0,
    top_k: int = 0,
) -> torch.Tensor:
    if temperature <= 0:
        probs = torch.zeros_like(logits)
        argmax_idx = torch.argmax(logits, dim=-1, keepdim=True)
        probs.scatter_(-1, argmax_idx, 1.0)
        return probs

    scores = logits.float() / temperature
    vocab_size = scores.shape[-1]
    if 0 < top_k < vocab_size:
        scores, indices = torch.topk(scores, top_k, dim=-1)
    else:
        indices = None

    probs = torch.softmax(scores, dim=-1)
    if top_p < 1.0:
        sorted_probs, order = probs.sort(dim=-1, descending=True)
        keep = sorted_probs.cumsum(dim=-1) - sorted_probs < top_p
        sorted_probs = sorted_probs * keep
        probs = torch.zeros_like(probs).scatter(-1, order, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True)

    if indices is not None:
        probs = torch.zeros_like(logits, dtype=probs.dtype).scatter(-1, indices, probs)
    return probs


def _sample_probs(probs: torch.Tensor) -> torch.Tensor:
    shape = probs.shape[:-1]
    return torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(shape)


def rejection_sample(
    draft_tokens: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    temperature: float = 0.0,
) -> Tuple[int, torch.Tensor]:
    """
    Standard speculative decoding rejection sampling.

    Args:
        draft_tokens: [batch_size, gamma] candidate token IDs
        target_probs: [batch_size, gamma + 1, vocab_size] verification probabilities from target model
        draft_probs: [batch_size, gamma, vocab_size] drafting probabilities
        temperature: sampling temperature

    Returns:
        (accepted_count, next_token_tensor)
    """
    gamma = draft_tokens.shape[1]
    if temperature <= 0:
        # Greedy acceptance: accept as long as argmax matches
        target_tokens = torch.argmax(target_probs[:, :gamma], dim=-1)
        matches = (target_tokens == draft_tokens)[0]
        accepted = 0
        for m in matches:
            if bool(m.item()):
                accepted += 1
            else:
                break
        next_token = torch.argmax(target_probs[:, accepted], dim=-1)
        return accepted, next_token

    # Stochastic rejection sampling
    p = target_probs[:, :gamma].gather(-1, draft_tokens[..., None])[..., 0]
    q = draft_probs.gather(-1, draft_tokens[..., None])[..., 0]
    
    rand_vals = torch.rand_like(q)
    accepted_mask = (rand_vals * q < p).to(torch.int32).cumprod(-1)[0]
    accepted = int(accepted_mask.sum().item())

    if accepted == gamma:
        return accepted, _sample_probs(target_probs[:, -1])[0]

    residual = target_probs[0, accepted].clone()
    residual.sub_(draft_probs[0, accepted])
    residual.clamp_min_(0)
    total = residual.sum()
    residual = torch.where(
        total > 0,
        residual / total.clamp_min(torch.finfo(residual.dtype).tiny),
        target_probs[0, accepted],
    )
    return accepted, _sample_probs(residual[None])[0]


class DFlashRunner:
    """
    In-engine runner for DFlash and DFlash 2 block diffusion draft models.
    """

    def __init__(
        self,
        draft_model_path: str,
        target_model: nn.Module,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        block_size: int = 5,
    ):
        _ensure_dflash_importable()
        from transformers import AutoConfig
        from dflash.model import (
            DFlashDraftModel,
            DFlash2DraftModel,
            _make_cache,
            _crop_to,
            _draft_value,
            extract_context_feature,
        )

        self.device = device
        self.dtype = dtype
        self.block_size = block_size
        self._target = target_model

        logger.info_rank0(f"Loading DFlash draft model from '{draft_model_path}' on {device} ({dtype})...")
        config = AutoConfig.from_pretrained(draft_model_path, trust_remote_code=True)
        draft_class = (
            DFlash2DraftModel
            if "DFlash2DraftModel" in (getattr(config, "architectures", None) or [])
            else DFlashDraftModel
        )
        self.draft_model = draft_class.from_pretrained(
            draft_model_path,
            config=config,
            torch_dtype=dtype,
        ).to(device)
        self.draft_model.eval()

        self.target_layer_ids: List[int] = getattr(
            self.draft_model, "target_layer_ids", [len(getattr(config, "architectures", [])) // 2]
        )
        self.mask_token_id: int = getattr(self.draft_model, "mask_token_id", 0)
        self.input_embedding_scale: float = float(_draft_value(config, "input_embedding_scale", 1.0))
        
        self._make_cache = _make_cache
        self._crop_to = _crop_to
        self._extract_context_feature = extract_context_feature
        self._draft_cache = None

        logger.info_rank0(
            f"DFlash draft model initialized: class={draft_class.__name__}, "
            f"target_layers={self.target_layer_ids}, block_size={self.block_size}"
        )

    def reset_cache(self) -> None:
        self._draft_cache = self._make_cache(self.draft_model.config)

    @torch.inference_mode()
    def draft(
        self,
        target_hidden_states: List[torch.Tensor],
        current_token_id: torch.Tensor,
        position_ids: torch.Tensor,
        seq_len: int,
        block_size: Optional[int] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate candidate tokens block via diffusion forward.

        Returns:
            draft_tokens: [1, K] candidate token IDs
            draft_probs: [1, K, vocab_size] probability distribution for rejection sampling
        """
        k = block_size or self.block_size
        if self._draft_cache is None:
            self.reset_cache()

        target_hidden = self._extract_context_feature(target_hidden_states, self.target_layer_ids)
        
        block_output_ids = torch.full(
            (1, k), self.mask_token_id, dtype=torch.long, device=self.device
        )
        block_output_ids[:, 0] = current_token_id.view(1)

        noise_emb = (
            F.embedding(block_output_ids, _target_embedding_weight(self._target))
            * self.input_embedding_scale
        )
        
        pos = position_ids[:, seq_len - target_hidden.shape[1] : seq_len + k]
        draft_hidden = self.draft_model(
            target_hidden=target_hidden,
            noise_embedding=noise_emb,
            position_ids=pos,
            past_key_values=self._draft_cache,
            use_cache=True,
        )[:, 1 - k :, :]
        
        self._crop_to(self._draft_cache, seq_len)
        
        draft_logits = _target_output_logits(self._target, draft_hidden)
        draft_probs = _sampling_probs(draft_logits, temperature, top_p, top_k)
        
        if temperature <= 0:
            draft_tokens = torch.argmax(draft_logits, dim=-1)
        else:
            draft_tokens = _sample_probs(draft_probs)

        return draft_tokens, draft_probs
