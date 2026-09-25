from __future__ import annotations

import glob
import json
import logging
import os
import re
import struct
import sys
from dataclasses import dataclass
from typing import Iterable, Iterator

import torch
from freetoken.env import ENV
from freetoken.utils import div_ceil, download_hf_weight

logger = logging.getLogger(__name__)

SPLIT_DIM_0 = (".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj")
SPLIT_DIM_1 = (".o_proj", ".down_proj")

SAFETENSORS_BACKENDS = ("mmap", "pread")


def safetensors_backend() -> str:
    """``safe_open`` storage backend, from FREETOKEN_SAFETENSORS_BACKEND ("auto" = per platform).

    auto is pread on Windows and mmap elsewhere, because what it avoids is a Windows
    mechanism and not a Windows suspicion: the mmap backend takes one non-shared
    (FILE_MAP_COPY) view of the whole shard, and Windows charges system commit for the full
    file size against it, so a shard that fits in RAM several times over still fails when
    the commit limit says no -- and that limit moves, because the page file grows under
    pressure, which is what made the failure intermittent. env.py carries the numbers, and
    the one case this cluster cannot measure: a file larger than physical RAM, where mmap
    can drop clean pages and pread cannot.

    Requires safetensors >= 0.8: 0.7.0 and earlier have no ``backend`` keyword and raise
    TypeError. That is the floor pyproject.toml declares.
    """
    name = str(ENV.SAFETENSORS_BACKEND).lower()
    if name in ("auto", ""):
        return "pread" if sys.platform.startswith("win") else "mmap"
    if name not in SAFETENSORS_BACKENDS:
        raise ValueError(
            f"FREETOKEN_SAFETENSORS_BACKEND={name!r} is not one of "
            f"{'auto'!r}, {', '.join(map(repr, SAFETENSORS_BACKENDS))}"
        )
    return name


@dataclass(frozen=True)
class MergeRule:
    fused_suffix: str
    slot: str
    slots: tuple[str, ...]


def iter_shard_tensors(file: str, device: torch.device | str) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield every tensor of one safetensors shard, already on ``device``.

    ``safe_open(..., device="cuda:0")`` reads the shard straight into VRAM with
    no copy through host memory, which is what every model family here does.

    OPEN THE FILE ONCE. An earlier version of this module probed the path first
    -- open, read one tensor, close, then open again for real -- to find out
    whether the direct read worked on this build. The probe was the defect it
    was meant to detect: measured on Windows 11 with safetensors 0.8.0 and
    torch 2.11.0+cu126, opening the same shard twice in one process reads
    correctly 2 times in 4, while a single open read all 338 tensors 6 times in
    6. When the second open loses, the very first ``get_tensor`` raises

        RuntimeError: Attempted to access the data pointer on an invalid
        python storage

    and the backend worker dies during load -- so the engine came up only some
    of the time, and the message named neither the file nor the device.

    There is deliberately no fall back to ``device="cpu"`` on failure. On this
    machine that path does not raise, it takes the process down with a Windows
    access violation inside ``torch.storage.__getitem__``, which turns a
    readable error into a worker that vanishes with no traceback at all.

    NOT FOR EVERY FAMILY. This reads every tensor in the shard, so a loader that
    skips keys -- the MoE families dropping experts, the ones whose rename returns
    None -- must keep its own loop, or it pulls tensors into VRAM only to discard
    them. The same goes for loaders that need the handle itself (``get_slice``, a
    ``set(f.keys())`` to look ahead, a handle held open across shards). Those are
    already opening once, which is the part that matters.

    The single open is not enough on Windows, because one mapping is already too many
    there: see ``safetensors_backend``, which picks how the bytes are served and which the
    other families still do not ask -- they open with the library default.
    """
    import safetensors

    backend = safetensors_backend()
    with safetensors.safe_open(file, framework="pt", device=str(device), backend=backend) as f:
        names = list(f.keys())
        for index, raw_name in enumerate(names):
            try:
                tensor = f.get_tensor(raw_name)
            except RuntimeError as exc:
                raise RuntimeError(
                    _shard_read_failure(file, raw_name, index, len(names), device, backend, exc)
                ) from exc
            yield raw_name, tensor


def _shard_read_failure(
    file: str,
    name: str,
    index: int,
    total: int,
    device: torch.device | str,
    backend: str,
    exc: BaseException,
) -> str:
    """The facts a reader of a failed shard read needs, none of which torch's message carries.

    "Attempted to access the data pointer on an invalid python storage" names neither the
    shard, nor its size, nor the device it was being read into, so it reads like a corrupt
    checkpoint. safetensors maps the WHOLE shard as one torch storage and slices it per
    tensor, so the thing that failed is the shard mapping, not the tensor: the size and the
    position in the file are what tell a shared mapping apart from a bad tensor.

    The backend is in there because it is the first thing to change when this fires: under
    mmap the shard was mapped and the size is what Windows charged to the commit limit,
    under pread it was read and the size is what was copied.
    """
    try:
        size = f"{os.path.getsize(file)} bytes"
    except OSError as size_exc:  # reported, not swallowed: the size is part of the diagnosis
        size = f"size unreadable ({size_exc})"
    versions = [f"torch {torch.__version__}", f"platform {sys.platform}", f"backend {backend}"]
    safetensors_version = getattr(sys.modules.get("safetensors"), "__version__", None)
    if safetensors_version is not None:
        versions.insert(0, f"safetensors {safetensors_version}")
    if torch.cuda.is_initialized():
        try:
            free, total_vram = torch.cuda.mem_get_info()
            versions.append(f"free VRAM {free} of {total_vram} bytes")
        except RuntimeError as vram_exc:  # reported: a dead context is itself the diagnosis
            versions.append(f"free VRAM unreadable ({vram_exc})")
    return (
        f"{file} ({size}): reading tensor {index + 1} of {total}, {name!r}, into "
        f"{device} failed with {type(exc).__name__}: {exc}. safetensors maps the whole "
        f"shard as one torch storage and slices it per tensor, so a storage error here is "
        f"about the shard mapping, not about this tensor. [{', '.join(versions)}]"
    )


def shard_tensor(
    key: str,
    value: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    num_kv_heads: int,
) -> torch.Tensor:
    if any(key.count(sub) for sub in SPLIT_DIM_0):
        is_kv_proj = any(key.count(sub) for sub in (".k_proj", ".v_proj"))
        if is_kv_proj and num_kv_heads is not None and num_kv_heads < world_size:
            head_dim = value.shape[0] // num_kv_heads
            head_idx = rank * num_kv_heads // world_size
            return value[head_idx * head_dim : (head_idx + 1) * head_dim].clone()
        return value.chunk(world_size, dim=0)[rank].clone()
    if any(key.count(sub) for sub in SPLIT_DIM_1):
        return value.chunk(world_size, dim=1)[rank].clone()
    if key.count("lm_head") or key.count("embed_tokens"):
        num_embeddings = value.shape[0]
        num_embeddings_per_partition = div_ceil(num_embeddings, world_size)
        vocab_start_idx = rank * num_embeddings_per_partition
        vocab_end_idx = min((rank + 1) * num_embeddings_per_partition, num_embeddings)
        return value[vocab_start_idx:vocab_end_idx, :].clone()
    return value


def iter_weight_files(model_path: str) -> list[str]:
    model_folder = download_hf_weight(model_path)
    files = glob.glob(f"{model_folder}/*.safetensors")
    return [f for f in files if not f.endswith("consolidated.safetensors")] or files


def safetensors_weight_map(folder: str) -> dict[str, str]:
    """Tensor name -> shard basename, from the index or from each shard's header when the checkpoint ships none."""
    index = os.path.join(folder, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index, encoding="utf-8") as f:
            return json.load(f)["weight_map"]
    weight_map: dict[str, str] = {}
    for path in sorted(iter_weight_files(folder)):
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        for name in header:
            if name != "__metadata__":
                weight_map[name] = os.path.basename(path)
    return weight_map


def drop_page_cache(path: str) -> None:
    """drop a file's page cache: banks + full checkpoint cache don't both fit in host RAM (OOM)."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        pass


def iter_root_safetensor_files_from_index(
    model_path: str,
    *,
    index_file: str = "model.safetensors.index.json",
) -> list[str]:
    model_folder = download_hf_weight(model_path)
    if os.path.basename(os.path.normpath(model_folder)) in {"metal", "original"}:
        raise ValueError("GPT-OSS loading requires the root GPT-OSS model directory")

    root_files = sorted(glob.glob(os.path.join(model_folder, "*.safetensors")))
    index_path = os.path.join(model_folder, index_file)
    if not os.path.isfile(index_path):
        files = [f for f in root_files if not f.endswith("consolidated.safetensors")]
        if not files:
            raise ValueError("No root GPT-OSS safetensors shards found")
        return files

    with open(index_path, encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    indexed_files = []
    for filename in dict.fromkeys(weight_map.values()):
        if os.path.dirname(filename):
            continue
        path = os.path.join(model_folder, filename)
        if path in root_files:
            indexed_files.append(path)

    if not indexed_files:
        raise ValueError(
            "No root GPT-OSS safetensors shards found from model.safetensors.index.json"
        )
    return sorted(indexed_files)


def _merge_info(key: str, rules: dict[str, MergeRule]) -> tuple[str, MergeRule] | None:
    for suffix, rule in rules.items():
        if key.endswith(suffix + ".weight") or key.endswith(suffix) or key.count(suffix):
            return key.replace(suffix, rule.fused_suffix), rule
    return None


def iter_merged_tensors(
    tensors: Iterable[tuple[str, torch.Tensor]],
    rules: dict[str, MergeRule],
    *,
    model_name: str,
) -> Iterator[tuple[str, torch.Tensor]]:
    merge_buf: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in tensors:
        info = _merge_info(name, rules)
        if info is None:
            yield name, tensor
            continue
        merged_key, rule = info
        slots = merge_buf.setdefault(merged_key, {})
        slots[rule.slot] = tensor
        if not all(slot in slots for slot in rule.slots):
            continue
        parts = [slots[slot] for slot in rule.slots]
        del merge_buf[merged_key]
        yield merged_key, torch.cat(parts, dim=0)

    assert not merge_buf, (
        f"{model_name}: Incomplete merge groups in checkpoint: {list(merge_buf.keys())}"
    )


# ---------------------------------------------------------------------------------
# compressed-tensors NVFP4 (llm-compressor) dense-weight helpers, shared by the
# models that serve such checkpoints natively (muse_glimmer). Storage:
# ``weight_packed`` (uint8 [O, IN//2]) + ``weight_scale`` (fp8-e4m3 block [O, IN//16])
# + a scalar ``weight_global_scale``. The stored global is the *quant-side* scale, so
# the dequant/native global is its reciprocal (vLLM inverts it identically).
# ---------------------------------------------------------------------------------

# Quant scales consumed with their ``weight_packed``.
CT_SCALE_SUFFIXES = (
    ".weight_scale", ".weight_global_scale", ".input_global_scale", ".input_scale",
)


class ShardReader:
    """Serves tensors by name across safetensors shards (handles opened lazily).

    The quant scales of a ``weight_packed`` can land in a DIFFERENT shard than the
    packed weight itself (Muse-Glimmer-30B-NVFP4 splits layer 49's down_proj across
    the shard boundary), so sibling lookups must go through the index's weight_map
    rather than the shard the packed weight came from."""

    def __init__(self, model_path: str, device: torch.device):
        folder = download_hf_weight(model_path)
        self._map = {name: os.path.join(folder, shard) for name, shard in safetensors_weight_map(folder).items()}
        self._device = str(device)
        self._handles: dict[str, object] = {}

    def files(self) -> list[str]:
        return sorted(set(self._map.values()))

    def has(self, name: str) -> bool:
        return name in self._map

    def names_in(self, file: str) -> list[str]:
        return [name for name, shard in self._map.items() if shard == file]

    def get_tensor(self, name: str) -> torch.Tensor:
        import safetensors

        file = self._map[name]
        h = self._handles.get(file)
        if h is None:
            h = safetensors.safe_open(file, framework="pt", device=self._device).__enter__()
            self._handles[file] = h
        return h.get_tensor(name)

    def close(self) -> None:
        for h in self._handles.values():
            try:
                h.__exit__(None, None, None)
            except Exception:  # noqa: BLE001 -- best-effort handle cleanup
                pass
        self._handles.clear()


def nvfp4_parts_ct(f, raw_base: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """compressed-tensors NVFP4 -> ``(packed uint8 [O, IN//2], block scale fp8 [O, IN//16], per-output-row global fp16 [O], dequant-side input_scale fp32 scalar or None)`` for the NVFP4 linear method."""
    w = f.get_tensor(raw_base + ".weight_packed")
    s = f.get_tensor(raw_base + ".weight_scale")
    wg = f.get_tensor(raw_base + ".weight_global_scale").reshape(1).to(torch.float32)
    g = (1.0 / wg).to(torch.float16).expand(w.shape[0]).contiguous()
    a = None
    if f.has(raw_base + ".input_global_scale"):
        a = (1.0 / f.get_tensor(raw_base + ".input_global_scale").reshape(()).to(torch.float32))
    return w, s, g, a


def ct_nvfp4_fuse(base: str, parts_tuple: tuple, buf: dict, groups: dict[str, tuple[str, ...]]):
    """Buffer a native NVFP4 fusion part ``(w, s, g, a)``; once complete, emit the concatenated ``.weight`` / ``.weight_scale`` / ``.weight_global`` (output-dim concat, each part keeps its own scales, so the fused FP4 weight is exact) plus the largest ``.input_scale`` when every part has one.
    ``[]`` while incomplete, ``None`` if ``base`` is not a fusion part of any group in ``groups``."""
    for fused_suffix, parts in groups.items():
        for idx, part in enumerate(parts):
            if base.endswith(part):
                key = base[: -len(part)] + fused_suffix
                slots = buf.setdefault(key, {})
                slots[idx] = parts_tuple
                if len(slots) < len(parts):
                    return []
                del buf[key]
                ws = [slots[i][0] for i in range(len(parts))]
                ss = [slots[i][1] for i in range(len(parts))]
                gs = [slots[i][2] for i in range(len(parts))]
                acts = [slots[i][3] for i in range(len(parts))]
                out = [
                    (key + ".weight", torch.cat(ws, dim=0)),
                    (key + ".weight_scale", torch.cat(ss, dim=0)),
                    (key + ".weight_global", torch.cat(gs, dim=0)),
                ]
                if all(a is not None for a in acts):
                    out.append((key + ".input_scale", torch.stack(acts).max()))
                return out
    return None


def ct_bf16_fuse(base: str, tensor: torch.Tensor, buf: dict, groups: dict[str, tuple[str, ...]]):
    """Buffer a bf16 fusion part; emit the concatenated ``.weight`` once complete, ``[]``
    while incomplete, ``None`` if ``base`` is not a part of any group in ``groups``."""
    for fused_suffix, parts in groups.items():
        for idx, part in enumerate(parts):
            if base.endswith(part):
                key = base[: -len(part)] + fused_suffix
                slots = buf.setdefault(key, {})
                slots[idx] = tensor
                if len(slots) < len(parts):
                    return []
                del buf[key]
                return [(key + ".weight", torch.cat([slots[i] for i in range(len(parts))], dim=0))]
    return None


def _expert_stack_info(key: str, expert_pattern: re.Pattern[str]) -> tuple[str, int] | None:
    match = expert_pattern.match(key)
    if match is None:
        return None
    packed_name = match.group("name")
    if packed_name.endswith(".weight"):
        packed_name = packed_name.removesuffix(".weight")
    return f"{match.group('prefix')}.{packed_name}", int(match.group("idx"))


def iter_stacked_experts(
    tensors: Iterable[tuple[str, torch.Tensor]],
    *,
    num_experts: int,
    model_name: str,
    expert_pattern: re.Pattern[str],
) -> Iterator[tuple[str, torch.Tensor]]:
    expert_buf: dict[str, dict[int, torch.Tensor]] = {}
    for name, tensor in tensors:
        expert_info = _expert_stack_info(name, expert_pattern)
        if expert_info is None:
            yield name, tensor
            continue
        packed_key, expert_idx = expert_info
        slots = expert_buf.setdefault(packed_key, {})
        slots[expert_idx] = tensor
        if len(slots) != num_experts:
            continue
        experts = [slots[idx] for idx in range(num_experts)]
        del expert_buf[packed_key]
        yield packed_key, torch.stack(experts, dim=0)

    assert not expert_buf, (
        f"{model_name}: Incomplete expert tensors in checkpoint: {list(expert_buf.keys())}"
    )


__all__ = [
    "MergeRule",
    "iter_root_safetensor_files_from_index",
    "iter_weight_files",
    "safetensors_weight_map",
    "iter_merged_tensors",
    "iter_stacked_experts",
    "shard_tensor",
]
