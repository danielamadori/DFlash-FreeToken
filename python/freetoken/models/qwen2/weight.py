from __future__ import annotations

from typing import Iterator

import torch
from freetoken.distributed import get_tp_info
from freetoken.models.loader import MergeRule, iter_merged_tensors, iter_weight_files, shard_tensor, safe_open_device
from freetoken.utils import cached_load_hf_config
from tqdm import tqdm

from .config import parse_config

_MERGE_RULES = {
    ".q_proj": MergeRule(".qkv_proj", "q", ("q", "k", "v")),
    ".k_proj": MergeRule(".qkv_proj", "k", ("q", "k", "v")),
    ".v_proj": MergeRule(".qkv_proj", "v", ("q", "k", "v")),
    ".gate_proj": MergeRule(".gate_up_proj", "gate", ("gate", "up")),
    ".up_proj": MergeRule(".gate_up_proj", "up", ("gate", "up")),
}


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    if not include_non_moe:
        return

    config = parse_config(cached_load_hf_config(model_path))
    tp_info = get_tp_info()

    def sharded_tensors() -> Iterator[tuple[str, torch.Tensor]]:
        for file in tqdm(
            iter_weight_files(model_path),
            desc="Loading weights",
            disable=not tp_info.is_primary(),
        ):
            # Not safe_open(device=...) directly: on some builds that path raises
            # on the first read and the engine dies during load. The helper probes
            # it once and says which path it got, so the move below happens only
            # when the tensors really arrived on the host.
            handle, gia_su_device = safe_open_device(file, device)
            with handle as f:
                for raw_name in f.keys():
                    name = raw_name.removeprefix("language_model.")
                    raw = f.get_tensor(raw_name)
                    if not gia_su_device:
                        raw = raw.to(device)
                    tensor = shard_tensor(
                        name,
                        raw,
                        rank=tp_info.rank,
                        world_size=tp_info.size,
                        num_kv_heads=config.num_kv_heads,
                    )
                    del raw
                    yield name, tensor

    yield from iter_merged_tensors(
        sharded_tensors(),
        _MERGE_RULES,
        model_name="qwen2",
    )

__all__ = ["iter_weights"]
