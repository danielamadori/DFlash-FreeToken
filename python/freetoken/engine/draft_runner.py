from __future__ import annotations

import sys
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from freetoken.utils import init_logger

logger = init_logger(__name__)


def _ensure_dflash_importable() -> None:
    """Put the real dflash package on sys.path, whatever else answers to that name.

    WHY NOT JUST `import dflash`. The checkout carries dflash as a submodule at its own
    root, so a process started from the repository root has a bare `dflash/` directory
    on its path -- and Python 3 imports a directory with no __init__.py as a NAMESPACE
    package. `import dflash` therefore SUCCEEDS, binding the outer submodule directory
    rather than the package inside it, and `dflash.model` is then missing. Probing for
    the submodule that actually holds the model is what distinguishes the two.
    """
    from pathlib import Path

    def _has_model() -> bool:
        try:
            import dflash.model  # noqa: F401
        except Exception:
            return False
        return True

    if _has_model():
        return

    # `dflash/dflash/model.py` is the layout of the z-lab checkout, whether it sits in
    # this repository (as a submodule) or one level up in the workspace that vendors it.
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "dflash"
        if (candidate / "dflash" / "model.py").is_file():
            path = str(candidate)
            if path in sys.path:
                sys.path.remove(path)
            sys.path.insert(0, path)
            # Drop the namespace-package binding the bare directory may have created,
            # otherwise the stale entry keeps shadowing the real one.
            sys.modules.pop("dflash", None)
            if _has_model():
                return
    raise ImportError(
        "cannot locate the dflash package: expected a dflash/dflash/model.py under this "
        "repository (git submodule update --init dflash) or under the workspace above it"
    )


def _output_head(target: nn.Module) -> nn.Module:
    """Output head of the target, transformers-style or FreeToken-style."""
    head = getattr(target, "lm_head", None)
    if head is not None:
        return head
    get_output_embeddings = getattr(target, "get_output_embeddings", None)
    if get_output_embeddings is None:
        raise TypeError(f"cannot locate the output head of {type(target).__name__}")
    return get_output_embeddings()


def _embed_with_target(target: nn.Module, ids: torch.Tensor) -> torch.Tensor:
    """Embed ``ids`` with the target's own input embedding.

    Asking the module rather than taking a weight matrix off it: a GGUF checkpoint keeps the
    table block-quantized and dequantizes only the rows a lookup touches, so it has ``qweight``
    and no ``weight`` at all. Reaching for ``.weight`` worked on safetensors targets and broke
    the moment the target was the GGUF this fork exists to serve.

    DFlash's own helper reaches it through `get_input_embeddings()`, which exists only on
    transformers modules. FreeToken models are plain objects holding a
    VocabParallelEmbedding, and a vocabulary sharded across ranks would have to be
    gathered first, so refuse that instead of drafting against a slice of the vocabulary.
    """
    get_input_embeddings = getattr(target, "get_input_embeddings", None)
    if get_input_embeddings is not None:
        return F.embedding(ids, get_input_embeddings().weight)

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
    # FreeToken's layers are plain objects with a `forward` method, not nn.Modules, so they
    # are NOT callable: GGUFEmbedding raises "object is not callable" under `embedding(ids)`.
    # Transformers modules have both, and `forward` means the same thing on each.
    forward = getattr(embedding, "forward", None)
    if forward is not None:
        return forward(ids)
    return embedding(ids)


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
    if target_probs.shape[1] != gamma + 1:
        # The verification forward must score every candidate plus the bonus position.
        # Handed a single decode row instead, the greedy branch below broadcasts one target
        # token against all gamma candidates: it silently accepts nothing (speculation
        # becomes pure overhead) until the draft happens to repeat that token, and then it
        # indexes past the end. Refuse the shape instead of returning a plausible number.
        raise ValueError(
            f"target_probs must score all {gamma + 1} positions of the draft block "
            f"(gamma={gamma} candidates + 1 bonus), got {tuple(target_probs.shape)}: the "
            "verification forward has to run over the drafted tokens, not the last one"
        )
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

        if not target_hidden_states:
            raise RuntimeError(
                "DFlash drafting needs the target's per-layer hidden states, but the target "
                "published none: either enable_hidden_state_capture was never called or the "
                "model does not implement it."
            )
        missing = [
            layer_id
            for layer_id in self.target_layer_ids
            if target_hidden_states[layer_id + 1] is None
        ]
        if missing:
            raise RuntimeError(
                f"the target published no hidden states for layers {missing}, which this "
                "draft checkpoint was trained against; capture was enabled for a different "
                "set of layers"
            )

        target_hidden = self._extract_context_feature(target_hidden_states, self.target_layer_ids)
        
        block_output_ids = torch.full(
            (1, k), self.mask_token_id, dtype=torch.long, device=self.device
        )
        block_output_ids[:, 0] = current_token_id.view(1)

        noise_emb = (
            _embed_with_target(self._target, block_output_ids)
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
