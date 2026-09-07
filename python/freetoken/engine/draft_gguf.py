"""Loading a DFlash draft from a GGUF checkpoint, with its weights left quantized.

The draft was only ever loadable through transformers, which means bf16: 3.85 GiB for the
DFlash2 draft of Qwen3.8-27B. llama.cpp serves the same draft from GGUF at Q4_K, 1.05 GiB. On a
24 GB card that difference decides whether the pair fits at all -- the 27B target alone occupies
19.6 GiB, so the bf16 draft leaves a few hundred megabytes for everything else.

The draft's architecture stays where it belongs, in the vendored ``dflash`` package: this only
substitutes the weight-carrying leaves. Every ``nn.Linear`` becomes a module that keeps its
weight in native ggml blocks and multiplies through FreeToken's fused GGUF kernels, and the two
selector codebooks become embeddings that dequantize only the rows a lookup touches. The
convolution kernels, the norms and everything else load as they are -- they are F32 in the file
and about a megabyte in total.

Nothing here dequantizes a whole matrix. That is the entire point: a dequantize-on-load path
would give identical arithmetic to llama.cpp and still cost 3.85 GiB of VRAM, which is the
problem being solved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from freetoken.layers.gguf import fused_mul_mat_gguf
from freetoken.models.gguf.dequant import row_bytes
from freetoken.models.gguf.reader import iter_gguf_tensors
from freetoken.utils import init_logger

if TYPE_CHECKING:
    from freetoken.models.gguf.reader import GgufTensor

logger = init_logger(__name__)

# GGUF tensor name -> attribute path on DFlash2DraftModel. ``{i}`` is the layer index.
#
# The GGUF side follows llama.cpp's naming and stores shapes reversed (its ne[0] is the input
# dimension), so a torch weight of [out, in] appears as (in, out) in the file. Both agree on the
# packed row being one output feature, which is what the fused kernels index.
_LINEAR_MAP = {
    "blk.{i}.attn_q.weight": "layers.{i}.self_attn.q_proj",
    "blk.{i}.attn_k.weight": "layers.{i}.self_attn.k_proj",
    "blk.{i}.attn_v.weight": "layers.{i}.self_attn.v_proj",
    "blk.{i}.attn_output.weight": "layers.{i}.self_attn.o_proj",
    "blk.{i}.ffn_gate.weight": "layers.{i}.mlp.gate_proj",
    "blk.{i}.ffn_up.weight": "layers.{i}.mlp.up_proj",
    "blk.{i}.ffn_down.weight": "layers.{i}.mlp.down_proj",
    "blk.{i}.attn_conv_proj.weight": "layers.{i}.attention_conv.kernel_projection",
    "blk.{i}.ffn_conv_proj.weight": "layers.{i}.mlp_conv.kernel_projection",
}
_GLOBAL_LINEAR_MAP = {
    "fc.weight": "fc",
    "selector_hidden.weight": "candidate_selector.hidden_projection",
}
_EMBEDDING_MAP = {
    "selector_predecessor.weight": "candidate_selector.predecessor_codebook",
    "selector_successor.weight": "candidate_selector.successor_codebook",
}
# Plain parameters, F32 in the file. The two output norms are told apart by what they normalise
# in dflash/model.py: ``hidden_norm`` follows ``fc(target_hidden)``, the encoder side, so it is
# the one the file calls ``enc.output_norm``; ``norm`` is the draft's own final norm.
_PARAM_MAP = {
    "blk.{i}.attn_norm.weight": "layers.{i}.input_layernorm.weight",
    "blk.{i}.ffn_norm.weight": "layers.{i}.post_attention_layernorm.weight",
    "blk.{i}.attn_q_norm.weight": "layers.{i}.self_attn.q_norm.weight",
    "blk.{i}.attn_k_norm.weight": "layers.{i}.self_attn.k_norm.weight",
    "blk.{i}.attn_conv_base": "layers.{i}.attention_conv.base_kernel",
    "blk.{i}.ffn_conv_base": "layers.{i}.mlp_conv.base_kernel",
}
_GLOBAL_PARAM_MAP = {
    "enc.output_norm.weight": "hidden_norm.weight",
    "output_norm.weight": "norm.weight",
}


class GGUFDraftLinear(nn.Module):
    """``nn.Linear`` shaped, but the weight stays in native ggml blocks.

    An ``nn.Module`` rather than FreeToken's ``GGUFLinear`` because this lives inside a
    transformers module tree: it has to be callable, be moved by ``.to(device)``, and appear in
    ``state_dict``. The packed weight is a buffer for exactly those reasons.
    """

    def __init__(self, in_features: int, out_features: int, quant_type: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_type = quant_type
        self.register_buffer(
            "qweight",
            torch.empty(out_features, row_bytes(in_features, quant_type), dtype=torch.uint8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The fused GGUF matmul takes [tokens, in_features] and treats dim 0 as the batch; a
        # draft block arrives as [batch, seq, hidden]. Passing it whole does not raise, it
        # returns a wrong rank that fails somewhere else entirely.
        shape = x.shape
        out = fused_mul_mat_gguf(x.reshape(-1, shape[-1]), self.qweight, self.quant_type)
        return out.reshape(*shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, ggml_type={self.quant_type}"


class GGUFDraftEmbedding(nn.Module):
    """A codebook kept quantized, dequantizing only the rows a lookup gathers.

    The two selector codebooks are 248,320 x 256 each. Dequantizing them whole would undo the
    saving this module exists for, and a lookup touches a handful of rows.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, quant_type: int) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.quant_type = quant_type
        self.register_buffer(
            "qweight",
            torch.empty(num_embeddings, row_bytes(embedding_dim, quant_type), dtype=torch.uint8),
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.gguf import ggml_dequantize

        flat = ids.flatten()
        rows = self.qweight.index_select(0, flat)
        out = ggml_dequantize(
            rows, self.quant_type, flat.shape[0], self.embedding_dim, torch.bfloat16
        )
        return out.view(*ids.shape, self.embedding_dim)

    def extra_repr(self) -> str:
        return f"{self.num_embeddings}x{self.embedding_dim}, ggml_type={self.quant_type}"


def _resolve(root: nn.Module, path: str) -> tuple[nn.Module, str]:
    """Split ``a.b.c`` into the module owning ``c`` and the attribute name."""
    parts = path.split(".")
    owner = root
    for part in parts[:-1]:
        owner = getattr(owner, part)
    return owner, parts[-1]


def _expand(template: str, layers: int) -> dict[str, str]:
    return {template.format(i=i): i for i in range(layers)}


def load_gguf_draft(
    gguf_path: str,
    draft_class: Any,
    config: Any,
    device: torch.device | str,
    dtype: torch.dtype,
) -> nn.Module:
    """Build a DFlash draft from ``gguf_path`` with its large weights left quantized.

    The skeleton is built on the meta device so the bf16 weights this replaces are never
    allocated, then every leaf named in the maps above is either swapped for a quantized module
    or materialised from the file. A tensor the file does not carry is an error rather than a
    silently random weight: the draft would still run and simply produce candidates the target
    rejects, which reads as a poor acceptance rate rather than a broken load.
    """
    num_layers = int(config.num_hidden_layers)
    with torch.device("meta"):
        model = draft_class(config)

    wanted: dict[str, tuple[str, str]] = {}  # gguf name -> (kind, attribute path)
    for tmpl, path_tmpl in _LINEAR_MAP.items():
        for i in range(num_layers):
            wanted[tmpl.format(i=i)] = ("linear", path_tmpl.format(i=i))
    for name, path in _GLOBAL_LINEAR_MAP.items():
        wanted[name] = ("linear", path)
    for name, path in _EMBEDDING_MAP.items():
        wanted[name] = ("embedding", path)
    for tmpl, path_tmpl in _PARAM_MAP.items():
        for i in range(num_layers):
            wanted[tmpl.format(i=i)] = ("param", path_tmpl.format(i=i))
    for name, path in _GLOBAL_PARAM_MAP.items():
        wanted[name] = ("param", path)

    seen: set[str] = set()
    for tensor in iter_gguf_tensors(gguf_path):
        entry = wanted.get(tensor.name)
        if entry is None:
            continue
        kind, path = entry
        seen.add(tensor.name)
        if kind == "linear":
            _install_linear(model, path, tensor, device)
        elif kind == "embedding":
            _install_embedding(model, path, tensor, device)
        else:
            _install_param(model, path, tensor, device, dtype)

    missing = sorted(set(wanted) - seen)
    if missing:
        raise ValueError(
            f"GGUF draft {gguf_path}: {len(missing)} tensor(s) this adapter needs are absent, "
            f"starting with {missing[:3]}; the checkpoint does not match the DFlash2 layout"
        )

    _rebuild_rotary(model, config, device)
    _assert_materialised(model, gguf_path)
    model.eval()
    logger.info_rank0(
        f"DFlash draft loaded from GGUF, weights left quantized: {len(seen)} tensors"
    )
    return model


def _rebuild_rotary(model: nn.Module, config: Any, device) -> None:
    """Recompute the rotary tables, which no GGUF carries because they are derived.

    ``inv_freq`` follows from theta and the head dimension, so llama.cpp computes it at load
    too and the file has nothing to read. Building the skeleton on meta left these buffers
    unmaterialised; the module's constructor is what fills them, so it is run again for real.
    """
    rotary = getattr(model, "rotary_emb", None)
    if rotary is None:
        return
    setattr(model, "rotary_emb", type(rotary)(config, device=device))


def _install_linear(model: nn.Module, path: str, tensor: GgufTensor, device) -> None:
    owner, attr = _resolve(model, path)
    old = getattr(owner, attr)
    out_features, in_features = old.weight.shape
    if old.bias is not None:
        raise NotImplementedError(f"{path}: a biased draft projection is not handled")
    module = GGUFDraftLinear(in_features, out_features, tensor.ggml_type)
    packed = tensor.packed()
    if tuple(packed.shape) != tuple(module.qweight.shape):
        raise ValueError(
            f"{tensor.name} -> {path}: packed weight is {tuple(packed.shape)} but the layer "
            f"expects {tuple(module.qweight.shape)} for [out={out_features}, in={in_features}]"
        )
    module.qweight = packed.to(device)
    setattr(owner, attr, module)


def _install_embedding(model: nn.Module, path: str, tensor: GgufTensor, device) -> None:
    owner, attr = _resolve(model, path)
    old = getattr(owner, attr)
    num_embeddings, embedding_dim = old.weight.shape
    module = GGUFDraftEmbedding(num_embeddings, embedding_dim, tensor.ggml_type)
    packed = tensor.packed()
    if tuple(packed.shape) != tuple(module.qweight.shape):
        raise ValueError(
            f"{tensor.name} -> {path}: packed codebook is {tuple(packed.shape)} but the layer "
            f"expects {tuple(module.qweight.shape)}"
        )
    module.qweight = packed.to(device)
    setattr(owner, attr, module)


def _install_param(model: nn.Module, path: str, tensor: GgufTensor, device, dtype) -> None:
    owner, attr = _resolve(model, path)
    target = getattr(owner, attr)
    values = _unquantized_values(tensor)
    if values.shape != target.shape:
        values = values.reshape(target.shape)
    setattr(owner, attr, nn.Parameter(values.to(device=device, dtype=dtype), requires_grad=False))


def _unquantized_values(tensor: GgufTensor) -> torch.Tensor:
    """Read an F32/F16 tensor's values. Quantized here would mean the map is wrong."""
    from freetoken.layers.gguf import GGML_UNQUANTIZED, _UNQUANTIZED_DTYPE

    if tensor.ggml_type not in GGML_UNQUANTIZED:
        raise ValueError(
            f"{tensor.name} is {tensor.ggml_type}, a quantized type, but this adapter maps it "
            "to a plain parameter; norms and convolution kernels are stored unquantized"
        )
    return tensor.packed().view(_UNQUANTIZED_DTYPE[tensor.ggml_type]).reshape(tensor.shape)


def _assert_materialised(model: nn.Module, gguf_path: str) -> None:
    """Refuse a model with a parameter still on meta: it would run and produce noise."""
    stranded = [n for n, p in model.named_parameters() if p.is_meta]
    stranded += [n for n, b in model.named_buffers() if b.is_meta]
    if stranded:
        raise ValueError(
            f"GGUF draft {gguf_path}: {len(stranded)} parameter(s) were never loaded, "
            f"starting with {stranded[:3]}; this adapter's name map is incomplete"
        )


__all__ = ["load_gguf_draft", "GGUFDraftLinear", "GGUFDraftEmbedding"]
