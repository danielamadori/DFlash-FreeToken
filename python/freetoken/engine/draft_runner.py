from __future__ import annotations

import sys
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from freetoken.env import ENV
from freetoken.models.gguf.reader import resolve_gguf_path
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

    # A GGUF head keeps its weights block-quantized, so there is no matrix to hand F.linear.
    # Its own forward cannot be used either: GGUFLMHead slices the last position of each
    # sequence, for the same reason ParallelLMHead does, and a draft block needs every row.
    # The projection it inherits does exactly the wanted thing, so call that.
    from freetoken.layers.gguf import GGUFLinear

    if isinstance(head, GGUFLinear):
        # fused_mul_mat_gguf treats dim 0 as the batch and accepts no further leading dims,
        # unlike F.linear, so a [1, K, hidden] draft block is folded and restored around it.
        # Leaving it 3-D does not raise: it comes back with a rank the caller then indexes
        # past the end of, far from here.
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        out = GGUFLinear.forward(head, flat)
        return out.reshape(*hidden_states.shape[:-1], out.shape[-1])

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


def _resolve_draft_gguf(draft_model_path: str) -> str | None:
    """The GGUF holding this draft's weights, or None if the path is an HF checkpoint.

    ``resolve_gguf_path`` accepts a file, or a directory holding a multi-shard set. A draft is
    one small file, and pointing at a directory that holds it beside its config.json is the
    natural way to pass one, so that case is resolved here rather than by widening the shared
    resolver, which serves target models with their own conventions.
    """
    import glob
    import os

    resolved = resolve_gguf_path(draft_model_path)
    if resolved is not None:
        return resolved
    if not os.path.isdir(draft_model_path):
        return None
    candidates = sorted(glob.glob(os.path.join(draft_model_path, "*.gguf")))
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        raise ValueError(
            f"{draft_model_path} holds {len(candidates)} .gguf files and none is a shard set; "
            "point --draft-model at the one to use"
        )
    return None


def _draft_config_source(draft_model_path: str, gguf_weights: str) -> str:
    """Where to read the draft's architecture from when its weights are a GGUF file.

    A DFlash GGUF ships no config.json. If the path given is a directory that has one, that is
    the source; otherwise the caller has to point at a directory that does, because the draft's
    own fields (block size, target layer ids, mask token) are not in the file's metadata and a
    wrong guess produces candidates the target quietly rejects.
    """
    import os

    directory = draft_model_path if os.path.isdir(draft_model_path) else os.path.dirname(gguf_weights)
    if os.path.isfile(os.path.join(directory, "config.json")):
        return directory
    raise ValueError(
        f"GGUF draft {gguf_weights} has no config.json beside it: point --draft-model at a "
        "directory holding both, or at the HF snapshot whose config describes this draft"
    )


def _draft_is_causal(draft_model: nn.Module) -> bool:
    """The causality flag the model's own mask builder would use, read off the layers.

    The model resolves `is_causal` per layer from the layer type when the config leaves it
    unset (dflash/model.py:363-367), so the instantiated layers are the ground truth. Every
    layer must agree: one cache mask serves all of them.
    """
    flags = {bool(layer.self_attn.is_causal) for layer in draft_model.layers}
    if len(flags) != 1:
        raise ValueError(
            f"the draft's layers disagree on is_causal ({sorted(flags)}): one static mask "
            "cannot serve them all"
        )
    return flags.pop()


def _build_static_cache(
    config: Any, draft_model: nn.Module, *, block: int, device: torch.device, dtype: torch.dtype
) -> Any | None:
    """The fixed-address ring cache of a windowed draft (DFlash 2), or None.

    None keeps the DynamicCache path: a full-attention draft (DFlash 1) has no window, so
    a ring holding one window of keys would silently drop everything older. The support
    rule is the cache's own, so the runner and the cache can never disagree on it.
    """
    from freetoken.engine.draft_cache import StaticDraftCache, _static_cache_supported

    if not _static_cache_supported(config):
        return None
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    return StaticDraftCache(
        num_layers=int(config.num_hidden_layers),
        num_kv_heads=int(config.num_key_value_heads),
        head_dim=int(head_dim),
        window=int(config.sliding_window),
        block=block,
        causal=_draft_is_causal(draft_model),
        device=device,
        dtype=dtype,
    )


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
        gguf_weights = _resolve_draft_gguf(draft_model_path)
        # A GGUF draft has no config.json beside it, so the architecture is read from the
        # companion HF directory when there is one, and otherwise from the file's own metadata.
        config_source = draft_model_path if gguf_weights is None else _draft_config_source(
            draft_model_path, gguf_weights
        )
        config = AutoConfig.from_pretrained(config_source, trust_remote_code=True)
        draft_class = (
            DFlash2DraftModel
            if "DFlash2DraftModel" in (getattr(config, "architectures", None) or [])
            else DFlashDraftModel
        )
        if gguf_weights is not None:
            # Quantized weights, kept quantized: 1.05 GiB for the DFlash2 draft of Qwen3.8-27B
            # against 3.85 GiB in bf16, which is what lets the pair fit on a 24 GB card at all.
            from freetoken.engine.draft_gguf import load_gguf_draft

            self.draft_model = load_gguf_draft(
                gguf_weights, draft_class, config, device=device, dtype=dtype
            )
        else:
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
        # Which request the draft cache belongs to. The cache holds that request's context
        # keys, built up block by block; it is meaningless for any other request.
        self._cache_owner: int | None = None
        self._warned_sampling_selector = False

        # A windowed draft (DFlash 2) keeps its keys in one ring allocated here and never
        # replaced: a CUDA graph bakes the K/V addresses, and a second cache object for the
        # eager blocks would leave the graph blocks drafting with no prompt context. The same
        # object serves the eager path, so the two paths run the same math on the same rows.
        self._static_cache = _build_static_cache(
            config, self.draft_model, block=block_size, device=device, dtype=dtype
        )
        self._window: int | None = (
            None if self._static_cache is None else int(self._static_cache.window)
        )
        self._draft_cache = self._static_cache
        # Block geometry as device tensors, so a captured forward reads the block's positions
        # from a buffer instead of baking them: row i of a block is seq_len - c + i.
        self._seq_len_t = torch.zeros(1, dtype=torch.int64, device=device)
        self._ar = torch.arange(
            (self._window or 0) + 2 * block_size, dtype=torch.int64, device=device
        )
        # Set by the engine once the draft forward is captured (engine/draft_graph.py).
        self.graph: Any | None = None
        self._shadow_blocks = 0
        self._shadow_mismatches = 0

        logger.info_rank0(
            f"DFlash draft model initialized: class={draft_class.__name__}, "
            f"target_layers={self.target_layer_ids}, block_size={self.block_size}, "
            f"cache={'static ring' if self._static_cache is not None else 'dynamic'}"
        )

    def reset_cache(self) -> None:
        if self._static_cache is not None:
            # Same object, emptied in place: a graph replay reads the ring at the address it
            # was captured with, and a fresh cache here would leave it reading a dead one.
            self._static_cache.reset()
            self._draft_cache = self._static_cache
        else:
            self._draft_cache = self._make_cache(self.draft_model.config)

    def _select_tokens(
        self,
        draft_hidden: torch.Tensor,
        draft_logits: torch.Tensor,
        draft_probs: torch.Tensor,
        anchor_ids: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        selector = getattr(self.draft_model, "candidate_selector", None)
        if selector is not None and temperature <= 0:
            # DFlash 2 does not pick each drafted token on its own. Its selector takes the
            # top-k candidates per position and then walks the block in order, scoring each
            # candidate against the token just chosen through the predecessor/successor
            # codebooks. Choosing every position independently -- which is DFlash 1's rule --
            # yields a block whose tokens do not follow one another, and the target rejects it:
            # the cost shows up as a low acceptance rate, never as an error.
            draft_tokens, _candidates, _q = selector.select(
                draft_hidden, draft_logits, anchor_ids, temperature
            )
            return draft_tokens
        if temperature <= 0:
            return torch.argmax(draft_logits, dim=-1)
        if selector is not None and not self._warned_sampling_selector:
            # Not a silent fallback: the selector returns scores over its candidate set,
            # and rejection sampling here wants a full-vocabulary distribution, so wiring
            # the two together is a change to the sampler, not to this call.
            logger.info_rank0(
                "DFlash 2 selector is bypassed at temperature > 0: candidates are drawn "
                "per position, which lowers acceptance. Greedy drafting uses the selector."
            )
            self._warned_sampling_selector = True
        return _sample_probs(draft_probs)

    def _run_block(
        self,
        th: torch.Tensor,
        block_ids: torch.Tensor,
        c: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One draft forward over the static ring: the body eager, warm-up, capture and
        shadow all run, so a replay is op-for-op the eager block.

        ``th`` is [1, c, features] context (a view of the graph's static buffer when
        captured), ``block_ids`` [1, k] the anchor id followed by mask tokens, ``c`` a Python
        int (each captured graph runs exactly its own row count, since MMVQ's summation order
        and the hidden norm's reduction depend on it). Reads the block's start from
        ``_seq_len_t`` so nothing host-side is baked. Returns (tokens, probs, logits).
        """
        cache = self._static_cache
        assert cache is not None, "_run_block needs the static ring cache"
        k = block_ids.shape[1]
        # Positions seq_len - c .. seq_len + k - 1: the context rows, then the noise rows,
        # which is the order the model concatenates K/V in (ctx first, model.py:388-391).
        row_pos = self._seq_len_t - c + self._ar[: c + k]
        cache.stage(row_pos)
        attention_mask = cache.mask(row_pos[c:])
        noise_emb = _embed_with_target(self._target, block_ids) * self.input_embedding_scale
        draft_hidden = self.draft_model(
            target_hidden=th,
            noise_embedding=noise_emb,
            position_ids=row_pos[None],
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
        )[:, 1 - k :, :]
        # Drop exactly the k noise rows this block staged (the crop(-k) of the dynamic path):
        # their keys stay in the ring but no later block may see them.
        cache.retire()
        draft_logits = _target_output_logits(self._target, draft_hidden)
        draft_probs = _sampling_probs(draft_logits, temperature, top_p, top_k)
        draft_tokens = self._select_tokens(
            draft_hidden, draft_logits, draft_probs, block_ids[:, 0], temperature
        )
        return draft_tokens, draft_probs, draft_logits

    def _block_ids(self, k: int, current_token_id: torch.Tensor) -> torch.Tensor:
        block_output_ids = torch.full(
            (1, k), self.mask_token_id, dtype=torch.long, device=self.device
        )
        block_output_ids[:, 0] = current_token_id.view(1)
        return block_output_ids

    def _draft_static(
        self,
        target_hidden: torch.Tensor,
        current_token_id: torch.Tensor,
        seq_len: int,
        k: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if k != self.block_size:
            # The ring retires exactly one block of noise rows per forward, and its size is
            # fixed at construction; a different k would leave stale noise keys visible.
            raise ValueError(
                f"the static draft cache retires {self.block_size} noise rows per block, "
                f"cannot draft a block of {k}"
            )
        assert self._window is not None
        if target_hidden.shape[1] > self._window:
            # First block after a long prefill. The dynamic cache kept only the last
            # window - 1 keys and each row saw at most that many (model.py:157-171), so the
            # rows beyond the window would be staged only to be masked, and the ring has no
            # room for them (window + 2 blocks slots).
            target_hidden = target_hidden[:, -self._window :]
        c = target_hidden.shape[1]
        self._seq_len_t.fill_(seq_len)
        return self._run_block(
            target_hidden, self._block_ids(k, current_token_id), c, temperature, top_p, top_k
        )

    def _draft_dynamic(
        self,
        cache: Any,
        target_hidden: torch.Tensor,
        current_token_id: torch.Tensor,
        position_ids: torch.Tensor,
        seq_len: int,
        k: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        block_output_ids = self._block_ids(k, current_token_id)
        noise_emb = (
            _embed_with_target(self._target, block_output_ids)
            * self.input_embedding_scale
        )
        
        pos = position_ids[:, seq_len - target_hidden.shape[1] : seq_len + k]
        draft_hidden = self.draft_model(
            target_hidden=target_hidden,
            noise_embedding=noise_emb,
            position_ids=pos,
            past_key_values=cache,
            use_cache=True,
        )[:, 1 - k :, :]
        
        # Drop exactly the k noise rows this block appended, not "crop to seq_len". The two
        # agree only when the cache already held every position below seq_len. After a prefix
        # cache hit or a chunked prefill it does not -- the draft was fed only the rows this
        # prefill computed -- and cropping to seq_len then asks for a positive crop, which
        # transformers treats as an absolute length: the mask-token keys stay in the cache and
        # every later block attends to them. Dropping k is right in both cases; in the short one
        # the draft simply has less context, all of it real, which is what llama.cpp does too.
        self._crop_to(cache, cache.get_seq_length() - k)

        draft_logits = _target_output_logits(self._target, draft_hidden)
        draft_probs = _sampling_probs(draft_logits, temperature, top_p, top_k)
        draft_tokens = self._select_tokens(
            draft_hidden, draft_logits, draft_probs, block_output_ids[:, 0], temperature
        )
        return draft_tokens, draft_probs

    def _shadow_draft(
        self,
        target_hidden_states: List[torch.Tensor],
        current_token_id: torch.Tensor,
        seq_len: int,
        c: int,
        k: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Replay, then draft the same block eagerly and compare the tokens.

        A replay that reads a stale ring or hidden buffer produces plausible tokens the target
        merely rejects more often; nothing raises. Both runs stage the same positions to the
        same slots and retire the same noise rows, so running them back to back leaves the
        ring as one run would. The eager pair is what gets returned. One host sync per block.
        """
        assert self.graph is not None
        tokens_g, _probs_g = self.graph.replay(target_hidden_states, current_token_id, seq_len, c)
        # The replay's outputs live in static buffers the next replay rewrites: keep a copy so
        # a later warm-up or replay cannot alter what is compared here.
        tokens_g = tokens_g.clone()
        target_hidden = self._extract_context_feature(target_hidden_states, self.target_layer_ids)
        tokens_e, probs_e, logits_e = self._draft_static(
            target_hidden, current_token_id, seq_len, k, temperature, top_p, top_k
        )
        self._shadow_blocks += 1
        mismatch = tokens_g[0] != tokens_e[0]
        if bool(mismatch.any().item()):
            self._shadow_mismatches += 1
            first = int(mismatch.nonzero()[0].item())
            top2 = logits_e[0, first].float().topk(2).values
            logger.warning_rank0(
                f"Draft graph shadow mismatch at block {self._shadow_blocks} "
                f"(seq_len={seq_len}, c={c}): {int(mismatch.sum().item())} of {k - 1} tokens "
                f"differ, first at position {first} (graph {int(tokens_g[0, first])} vs eager "
                f"{int(tokens_e[0, first])}, eager top-2 logit margin "
                f"{float(top2[0] - top2[1]):.4f}); {self._shadow_mismatches} mismatching "
                f"blocks so far"
            )
        elif self._shadow_blocks == 1:
            logger.info_rank0("Draft graph shadow mode: first replayed block matches eager")
        return tokens_e, probs_e

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
        request_uid: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate candidate tokens block via diffusion forward.

        ``position_ids`` is read only by the dynamic-cache path; the static ring derives the
        block's positions from ``seq_len`` on the device.

        Returns:
            draft_tokens: [1, K] candidate token IDs
            draft_probs: [1, K, vocab_size] probability distribution for rejection sampling
        """
        k = block_size or self.block_size
        if self._draft_cache is None or request_uid != self._cache_owner:
            # The draft's KV cache is this request's context, built up block by block. Kept
            # across requests it is not empty for the next one -- it is the previous request's
            # keys, cropped to the new request's position and used as its context. Every block
            # after the first request then drafts against the wrong text. The target still
            # verifies, so output stays correct; what drops is acceptance, silently: on the
            # 27B this fork accepted 27.9% from the identical draft file llama.cpp accepts
            # 35.4% from.
            self.reset_cache()
            self._cache_owner = request_uid
        cache = self._draft_cache
        assert cache is not None  # reset_cache() just guaranteed it; this narrows the type

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

        # After the owner check: a new request's device reset is then stream-ordered before
        # the replay that reads the ring, and a same-request block replays over its context.
        graph = self.graph
        if graph is not None:
            c = graph.can_replay(target_hidden_states, k, temperature)
            if c is not None:
                if bool(getattr(ENV, "SPEC_DRAFT_GRAPH_SHADOW", False)):
                    return self._shadow_draft(
                        target_hidden_states, current_token_id, seq_len, c, k,
                        temperature, top_p, top_k,
                    )
                return graph.replay(target_hidden_states, current_token_id, seq_len, c)

        target_hidden = self._extract_context_feature(target_hidden_states, self.target_layer_ids)
        if self._static_cache is not None:
            draft_tokens, draft_probs, _logits = self._draft_static(
                target_hidden, current_token_id, seq_len, k, temperature, top_p, top_k
            )
            return draft_tokens, draft_probs
        return self._draft_dynamic(
            cache, target_hidden, current_token_id, position_ids, seq_len, k,
            temperature, top_p, top_k,
        )
