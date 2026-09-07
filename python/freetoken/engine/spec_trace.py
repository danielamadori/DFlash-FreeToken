"""Per-position logit trace, used to decide whether a divergence is a bug or arithmetic.

Greedy speculative decoding should reproduce greedy decoding token for token: the draft
only proposes, the target verifies, so an accepted candidate is the token the target
would have chosen alone. When it does not, there are two possible explanations and they
call for opposite responses:

  * the verification forward computes its logits over several positions at once, and
    bf16 GEMM reductions are not batch-invariant, so a near-tie between the top two
    candidates can flip. Nothing to fix in the engine.
  * the verification or the KV rollback is wrong, and the target is being asked about
    the wrong context. Everything to fix.

The two are told apart by the MARGIN between the top two logits at the position where
the outputs first differ: a margin at the scale of bf16 rounding is the first, a clear
margin is the second. This writes that margin out for every generated position so the
two runs can be compared afterwards, and is off unless FREETOKEN_SPEC_TRACE names a file.
"""

from __future__ import annotations

import json
import os
from typing import TextIO

import torch

_TRACE_PATH = os.environ.get("FREETOKEN_SPEC_TRACE", "")
_handle: TextIO | None = None


def enabled() -> bool:
    return bool(_TRACE_PATH)


def _out() -> TextIO:
    global _handle
    if _handle is None:
        _handle = open(_TRACE_PATH, "a", encoding="utf-8")
    return _handle


def record(
    source: str,
    uid: int,
    position: int,
    logits_row: torch.Tensor,
    *,
    valid: bool = True,
    token: int | None = None,
) -> None:
    """Write the top two candidates and their gap for one predicted position.

    ``logits_row`` is the [vocab] row that decides the token at ``position``; ``source``
    says which path produced it ("plain" or "verify") so the two runs stay comparable.
    ``valid`` is False for a verification row whose context contains a rejected candidate:
    it is recorded for completeness but describes a stream that was never committed.
    """
    if not _TRACE_PATH:
        return
    top = torch.topk(logits_row.float(), 2)
    values = top.values.tolist()
    indices = top.indices.tolist()
    _out().write(
        json.dumps(
            {
                "source": source,
                "uid": uid,
                "position": position,
                "top1": indices[0],
                "top2": indices[1],
                "logit1": values[0],
                "logit2": values[1],
                "margin": values[0] - values[1],
                "valid": bool(valid),
                "token": token,
            }
        )
        + "\n"
    )
    _out().flush()
