from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

from freetoken.layers import (
    BaseOP,
    OPList,
    LinearColParallelMerged,
    LinearRowParallel,
    gelu_and_mul,
    gelu_tanh_and_mul,
    silu_and_mul,
)
from freetoken.utils import nvtx_annotate

if TYPE_CHECKING:
    import torch

    from .config import ModelConfig


class BaseLLMModel(ABC, BaseOP):
    @abstractmethod
    def forward(self) -> torch.Tensor: ...

    @property
    def last_hidden_states(self) -> list[torch.Tensor | None] | None:
        """Per-layer hidden states of the last forward, or None when capture is off.

        Laid out the way DFlash's ``extract_context_feature`` indexes them: entry 0 is
        the embedding output and entry ``i + 1`` the output of layer ``i``. Layers that
        were not requested stay None, so a draft needing five layers out of fifty-two
        does not pin the other forty-seven for the lifetime of the step.
        """
        return getattr(getattr(self, "model", None), "_captured_hidden_states", None)

    def enable_hidden_state_capture(self, layer_ids: Sequence[int]) -> None:
        """Ask the model to publish `last_hidden_states` for `layer_ids` on each forward.

        Families opt in by mixing HiddenStateCapture into their transformer stack. The
        default refuses instead of quietly capturing nothing: a draft model fed empty or
        stale context features produces garbage candidates that the target then rejects,
        which reads as a bad acceptance rate rather than the wiring bug it is.
        """
        set_capture_layer_ids = getattr(
            getattr(self, "model", None), "set_capture_layer_ids", None
        )
        if set_capture_layer_ids is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not implement hidden-state capture, so it "
                "cannot drive DFlash speculative decoding."
            )
        set_capture_layer_ids(layer_ids)


class HiddenStateCapture:
    """Records the output of selected layers on a transformer stack.

    Mixed into the inner ``*Model`` (the object that owns ``layers``); the enclosing
    ``*ForCausalLM`` exposes it through BaseLLMModel. Both attributes are
    underscore-prefixed so BaseOP.state_dict skips them: they are runtime scratch, not
    weights.
    """

    layers: OPList
    _capture_layer_ids: tuple[int, ...] = ()
    _captured_hidden_states: list[torch.Tensor | None] | None = None

    def set_capture_layer_ids(self, layer_ids: Sequence[int]) -> None:
        """Select which layers publish their output; an empty sequence disables it."""
        num_layers = len(self.layers.op_list)
        for layer_id in layer_ids:
            if layer_id < -1 or layer_id >= num_layers:
                raise ValueError(
                    f"hidden-state capture asked for layer {layer_id}, but this model has "
                    f"{num_layers} layers (-1 selects the embedding output)"
                )
        self._capture_layer_ids = tuple(layer_ids)
        self._captured_hidden_states = None

    def _new_capture_store(self) -> list[torch.Tensor | None]:
        return [None] * (len(self.layers.op_list) + 1)


class GatedMLP(BaseOP):
    def __init__(self, config: ModelConfig):
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            has_bias=False,
        )

        fn_map = {"silu": silu_and_mul, "gelu": gelu_and_mul, "gelu_tanh": gelu_tanh_and_mul}
        act_fn = fn_map.get(config.hidden_act, None)
        if act_fn is None:
            raise ValueError(f"Unsupported activation function: {config.hidden_act}")
        self.act_fn = act_fn
        self.down_proj = LinearRowParallel(
            config.intermediate_size,
            config.hidden_size,
            has_bias=False,
        )

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj.forward(x)
        del x
        y = self.act_fn(gate_up)
        del gate_up
        return self.down_proj.forward(y)


__all__ = ["BaseLLMModel", "GatedMLP", "HiddenStateCapture"]
