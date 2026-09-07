# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/index.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Sequence

import torch
import triton

from freetoken.kernel.fla.utils import tensor_cache


def _refuse_under_capture(cu_seqlens: torch.Tensor) -> None:
    # A D2H readback or a pageable H2D copy inside a CUDA-graph capture is an invalid stream
    # operation; the identity cache hides it only when the same cu_seqlens object was seen
    # just before, which a capture cannot rely on. Callers on a captured path pass host-built
    # indices instead (chunk_indices_from_lens / chunk_offsets_from_lens). A host cu_seqlens
    # (the scheduler's track metadata) never touches a stream, and asking CUDA about capture
    # state from a process without a device raises, so only device tensors are checked.
    if cu_seqlens.device.type != "cpu" and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "fla chunk index computed under CUDA-graph capture; pass precomputed indices"
        )


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    _refuse_under_capture(cu_seqlens)
    indices = torch.cat(
        [
            torch.arange(n)
            for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
        ]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    _refuse_under_capture(cu_seqlens)
    return torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)


def chunk_indices_from_lens(lens: Sequence[int], chunk_size: int) -> list[list[int]]:
    """``prepare_chunk_indices`` over per-sequence lengths, computed on the host from Python
    ints -- no tensor, no sync -- so a caller can move the result to the device up front."""
    indices: list[list[int]] = []
    # The device formula numbers sequences by `indices.eq(0).cumsum(0) - 1`, so a sequence
    # without a chunk (length 0) does not take a number; mirror that exactly.
    seq = -1
    for n in lens:
        for c in range(triton.cdiv(n, chunk_size)):
            if c == 0:
                seq += 1
            indices.append([seq, c])
    return indices


def chunk_offsets_from_lens(lens: Sequence[int], chunk_size: int) -> list[int]:
    """``prepare_chunk_offsets`` over per-sequence lengths, on the host (see above)."""
    offsets = [0]
    for n in lens:
        offsets.append(offsets[-1] + triton.cdiv(n, chunk_size))
    return offsets
