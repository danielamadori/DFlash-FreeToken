from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from freetoken.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from freetoken.core import Batch


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None
    greedy_mask: torch.Tensor | None = None


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def greedy_argmax(logits: torch.Tensor) -> torch.Tensor:
    """Greedy pick over the last dim, ties going to the LOWEST id, everywhere the same.

    ``torch.argmax`` documents that it returns the first maximal index, but on CUDA the answer
    at an exact tie follows the reduction order, which follows the tensor's shape. The plain
    decode reduces over ``[batch, vocab]`` and the speculative verify over a slice of
    ``[1, block, vocab]``, so the two disagreed -- and they disagreed on a real generation:
    position 255 of a 300-token greedy run had token 25 and token 11 both at logit 21.25,
    exactly tied, and the plain path committed 25 while the speculative one committed 11. From
    there the two streams say different things for the rest of the answer.

    That is not the draft being wrong. Greedy speculative decoding is supposed to reproduce
    greedy decoding token for token -- the draft only proposes, the target decides -- and up to
    that tie it did, for 255 positions. But because the tie went two ways, the block size
    changed the generated text, which is also why no comparison between configurations could
    be read off the output.

    Exact ties are not rare enough to wave away: four positions in 298 on that run, roughly one
    in seventy, each one a fork in the rest of the answer.

    The cost is a comparison and a min over the vocabulary, next to a forward that reads
    seventeen gigabytes of weights.
    """
    top = logits.max(dim=-1, keepdim=True).values
    ids = torch.arange(logits.shape[-1], device=logits.device, dtype=torch.int64)
    return torch.where(logits == top, ids, logits.shape[-1]).min(dim=-1).values


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    from freetoken.kernel.backend import is_flashinfer_installed

    if is_flashinfer_installed():
        import flashinfer.sampling as sampling
    else:
        import freetoken.kernel.triton.sampling as sampling

    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)

    if top_p is None:
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k)

    if top_k is None:
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p)

    assert top_k is not None and top_p is not None
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        is_greedy = [p.is_greedy for p in params]
        if all(is_greedy):
            return BatchSamplingArgs(temperatures=None)

        MIN_P = MIN_T = 1e-6
        # Greedy outputs are selected explicitly in sample(); use neutral sampling
        # parameters for those rows instead of approximating argmax at low temperature.
        ts = [1.0 if g else max(p.temperature, MIN_T) for p, g in zip(params, is_greedy)]
        top_ks = [
            p.top_k if not g and p.top_k >= 1 else self.vocab_size
            for p, g in zip(params, is_greedy)
        ]
        top_ps = [
            1.0 if g else min(max(p.top_p, MIN_P), 1.0)
            for p, g in zip(params, is_greedy)
        ]
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        greedy_mask = (
            make_device_tensor(is_greedy, torch.bool, self.device) if any(is_greedy) else None
        )
        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p, greedy_mask=greedy_mask)

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            if args.temperatures is None:  # greedy sampling
                return greedy_argmax(logits)
            tokens = sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
            if args.greedy_mask is not None:
                # Mixed batches still run probability sampling for all rows, but
                # greedy rows must follow argmax's deterministic tie-breaking.
                # ``greedy_argmax``, not ``torch.argmax``: on CUDA the latter does
                # not reliably return the first maximal index, which is the whole
                # reason greedy_argmax exists (see its docstring above).
                greedy_tokens = greedy_argmax(logits).to(tokens.dtype)
                tokens = torch.where(args.greedy_mask, greedy_tokens, tokens)
            return tokens
