import pytest
import torch
from freetoken.engine.draft_runner import rejection_sample, _sampling_probs
from freetoken.server.args import parse_args


def test_rejection_sample_greedy_full_accept():
    gamma = 4
    vocab_size = 100
    draft_tokens = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    
    # Target logits strongly match draft tokens for all gamma positions
    target_probs = torch.zeros((1, gamma + 1, vocab_size))
    target_probs[0, 0, 10] = 1.0
    target_probs[0, 1, 20] = 1.0
    target_probs[0, 2, 30] = 1.0
    target_probs[0, 3, 40] = 1.0
    target_probs[0, 4, 50] = 1.0  # Next token after full acceptance
    
    draft_probs = torch.zeros((1, gamma, vocab_size))
    for i, tok in enumerate([10, 20, 30, 40]):
        draft_probs[0, i, tok] = 1.0
        
    accepted_count, next_tok = rejection_sample(
        draft_tokens=draft_tokens,
        target_probs=target_probs,
        draft_probs=draft_probs,
        temperature=0.0,
    )
    
    assert accepted_count == 4
    assert next_tok.item() == 50


def test_rejection_sample_greedy_partial_accept():
    gamma = 4
    vocab_size = 100
    draft_tokens = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    
    # Target matches pos 0 and 1, but diverges at pos 2
    target_probs = torch.zeros((1, gamma + 1, vocab_size))
    target_probs[0, 0, 10] = 1.0
    target_probs[0, 1, 20] = 1.0
    target_probs[0, 2, 99] = 1.0  # Divergence: target wants 99 instead of 30
    target_probs[0, 3, 40] = 1.0
    target_probs[0, 4, 50] = 1.0
    
    draft_probs = torch.zeros((1, gamma, vocab_size))
    for i, tok in enumerate([10, 20, 30, 40]):
        draft_probs[0, i, tok] = 1.0
        
    accepted_count, next_tok = rejection_sample(
        draft_tokens=draft_tokens,
        target_probs=target_probs,
        draft_probs=draft_probs,
        temperature=0.0,
    )
    
    assert accepted_count == 2
    assert next_tok.item() == 99


def test_server_args_spec_draft():
    args = [
        "--model", "Qwen/Qwen3.6-35B-A3B",
        "--spec-draft", "z-lab/Qwen3.6-35B-A3B-DFlash",
        "--spec-block-size", "7",
        "--spec-draft-dtype", "bfloat16",
    ]
    server_args, _ = parse_args(args)
    assert server_args.spec_draft_model == "z-lab/Qwen3.6-35B-A3B-DFlash"
    assert server_args.spec_block_size == 7
    assert server_args.spec_draft_dtype == "bfloat16"


def test_sampling_probs_temperature():
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    probs_greedy = _sampling_probs(logits, temperature=0.0)
    assert torch.allclose(probs_greedy, torch.tensor([[1.0, 0.0, 0.0]]))

    probs_temp = _sampling_probs(logits, temperature=1.0)
    assert probs_temp[0, 0] > probs_temp[0, 1] > probs_temp[0, 2]
    assert torch.allclose(probs_temp.sum(dim=-1), torch.tensor([1.0]))


def test_rejection_sample_stochastic():
    gamma = 2
    vocab_size = 10
    draft_tokens = torch.tensor([[1, 2]], dtype=torch.long)
    target_probs = torch.zeros((1, gamma + 1, vocab_size))
    target_probs[0, 0, 1] = 1.0
    target_probs[0, 1, 2] = 1.0
    target_probs[0, 2, 3] = 1.0

    draft_probs = torch.zeros((1, gamma, vocab_size))
    draft_probs[0, 0, 1] = 1.0
    draft_probs[0, 1, 2] = 1.0

    accepted, next_tok = rejection_sample(
        draft_tokens=draft_tokens,
        target_probs=target_probs,
        draft_probs=draft_probs,
        temperature=0.7,
    )
    assert accepted == 2
    assert next_tok.item() == 3


def test_dflash_runner_ensure_importable():
    from freetoken.engine.draft_runner import _ensure_dflash_importable
    # Should not raise exception even when called repeatedly
    _ensure_dflash_importable()


def test_engine_speculative_wiring_fallback():
    from unittest.mock import MagicMock
    from freetoken.engine.engine import Engine, ForwardOutput
    from freetoken.core import Batch, Req

    mock_engine = MagicMock()
    mock_engine.draft_runner = None
    mock_engine.forward_speculative_batch = Engine.forward_speculative_batch.__get__(mock_engine, Engine)
    mock_engine._forward_batch_standard.return_value = ForwardOutput(
        next_tokens_gpu=torch.tensor([42]),
        next_tokens_cpu=torch.tensor([42]),
        copy_done_event=None,
    )

    req = Req(
        input_ids=torch.tensor([1, 2, 3]),
        table_idx=0,
        cached_len=2,
        output_len=1,
        uid=1,
        sampling_params=None,
        cache_handle=None,
    )
    batch = Batch(reqs=[req], phase="decode")
    batch.padded_reqs = [req]

    out = mock_engine.forward_speculative_batch(batch, None)
    assert out.next_tokens_gpu.item() == 42
    mock_engine._forward_batch_standard.assert_called_once_with(batch, None)


def test_engine_speculative_execution_full_accept():
    from unittest.mock import MagicMock
    from freetoken.engine.engine import Engine, ForwardOutput
    from freetoken.core import Batch, Req

    mock_engine = MagicMock()
    mock_engine.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
    mock_engine.device = torch.device("cpu")
    mock_engine.forward_speculative_batch = Engine.forward_speculative_batch.__get__(mock_engine, Engine)

    # Setup draft runner mock
    mock_draft = MagicMock()
    # 2 draft tokens: [101, 102]
    draft_tokens = torch.tensor([[101, 102]], dtype=torch.long)
    draft_probs = torch.zeros((1, 2, 200))
    draft_probs[0, 0, 101] = 1.0
    draft_probs[0, 1, 102] = 1.0
    mock_draft.draft.return_value = (draft_tokens, draft_probs)
    mock_engine.draft_runner = mock_draft

    # Setup target model with hidden states
    mock_model = MagicMock()
    mock_model.last_hidden_states = [torch.zeros((1, 1, 128))]
    mock_engine.model = mock_model

    # Setup ctx and graph runner
    mock_engine.ctx.forward_batch.return_value.__enter__.return_value = None
    mock_engine.ctx.forward_batch.return_value.__exit__.return_value = None
    mock_engine.graph_runner.can_use_cuda_graph.return_value = False
    mock_engine.cpu_moe_executor = None

    # Target model returns logits matching draft tokens [101, 102] + bonus token [103]
    target_logits = torch.zeros((1, 3, 200))
    target_logits[0, 0, 101] = 10.0
    target_logits[0, 1, 102] = 10.0
    target_logits[0, 2, 103] = 10.0
    mock_model.forward.return_value = target_logits

    req = Req(
        input_ids=torch.tensor([1, 2, 3]),
        table_idx=0,
        cached_len=2,
        output_len=10,
        uid=1,
        sampling_params=None,
        cache_handle=None,
    )
    initial_device_len = req.device_len
    batch = Batch(reqs=[req], phase="decode")
    batch.padded_reqs = [req]
    batch.input_ids = torch.tensor([3])
    batch.positions = torch.tensor([3])

    if torch.cuda.is_available():
        with torch.cuda.stream(mock_engine.stream):
            out = mock_engine.forward_speculative_batch(batch, None)
    else:
        with unittest_mock_cuda_stream(mock_engine):
            out = mock_engine.forward_speculative_batch(batch, None)

    assert out.next_tokens_gpu.item() == 103
    # 2 accepted draft tokens + 1 bonus token = 3 complete_one() calls
    assert req.device_len == initial_device_len + 3


def test_engine_forward_batch_routing():
    from unittest.mock import MagicMock
    from freetoken.engine.engine import Engine, ForwardOutput
    from freetoken.core import Batch, Req

    mock_engine = MagicMock()
    mock_engine.forward_batch = Engine.forward_batch.__get__(mock_engine, Engine)

    req = Req(
        input_ids=torch.tensor([1, 2]),
        table_idx=0,
        cached_len=1,
        output_len=1,
        uid=1,
        sampling_params=None,
        cache_handle=None,
    )

    # 1. Prefill phase should always route to _forward_batch_standard
    mock_engine.draft_runner = MagicMock()
    batch_prefill = Batch(reqs=[req], phase="prefill")
    mock_engine.forward_batch(batch_prefill, None)
    mock_engine._forward_batch_standard.assert_called_once_with(batch_prefill, None)

    # 2. Decode phase with draft runner should route to forward_speculative_batch
    mock_engine.reset_mock()
    batch_decode = Batch(reqs=[req], phase="decode")
    mock_engine.forward_batch(batch_decode, None)
    mock_engine.forward_speculative_batch.assert_called_once_with(batch_decode, None)


def unittest_mock_cuda_stream(engine):
    from unittest.mock import patch
    import contextlib

    @contextlib.contextmanager
    def _mock_ctx():
        with patch("torch.cuda.current_stream", return_value=None), \
             patch("torch.cuda.Event"):
            yield

    return _mock_ctx()




def test_engine_speculative_missing_hidden_states_raises():
    """A target that publishes no hidden states must fail loudly, not decode at 1x."""
    from unittest.mock import MagicMock
    from freetoken.engine.engine import Engine
    from freetoken.core import Batch, Req

    mock_engine = MagicMock()
    mock_engine.stream = None
    mock_engine.draft_runner = MagicMock()
    # spec=[] -> the model exposes no attributes, so `last_hidden_states` is missing
    mock_engine.model = MagicMock(spec=[])
    mock_engine.forward_speculative_batch = Engine.forward_speculative_batch.__get__(
        mock_engine, Engine
    )

    req = Req(
        input_ids=torch.tensor([1, 2, 3]),
        table_idx=0,
        cached_len=2,
        output_len=1,
        uid=1,
        sampling_params=None,
        cache_handle=None,
    )
    batch = Batch(reqs=[req], phase="decode")
    batch.padded_reqs = [req]

    with unittest_mock_cuda_stream(mock_engine):
        with pytest.raises(RuntimeError, match="last_hidden_states"):
            mock_engine.forward_speculative_batch(batch, None)
    mock_engine._forward_batch_standard.assert_not_called()


def test_base_model_refuses_hidden_state_capture_by_default():
    """A model that cannot publish hidden states must say so, not capture nothing."""
    from freetoken.models.blocks import BaseLLMModel

    class _Plain(BaseLLMModel):
        def forward(self):  # noqa: ANN201 - test stub
            return torch.zeros(1)

    model = _Plain()
    assert model.last_hidden_states is None
    with pytest.raises(NotImplementedError, match="hidden-state capture"):
        model.enable_hidden_state_capture([0])


def test_muse_glimmer_capture_uses_the_hf_layer_offset():
    """Entry 0 is the embedding output and entry i+1 layer i, as DFlash indexes them."""
    from types import SimpleNamespace
    from freetoken.models.muse_glimmer.model import MuseGlimmerModel

    class _StubLayer:
        def __init__(self, delta: float) -> None:
            self._delta = delta

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self._delta

    # Bypass __init__: allocating real weights needs a checkpoint and a GPU, and the
    # layout logic under test does not depend on either.
    model = object.__new__(MuseGlimmerModel)
    model.embed_tokens = SimpleNamespace(forward=lambda ids: torch.zeros(1, len(ids), 4))
    model.embed_norm = SimpleNamespace(forward=lambda x: x)
    model.norm = SimpleNamespace(forward=lambda x: x)
    model.layers = SimpleNamespace(op_list=[_StubLayer(1.0), _StubLayer(2.0), _StubLayer(4.0)])
    model._capture_layer_ids = ()
    model._captured_hidden_states = None

    # capture off -> nothing published
    model.forward(torch.tensor([7]))
    assert model._captured_hidden_states is None

    model.set_capture_layer_ids([-1, 1])
    out = model.forward(torch.tensor([7]))
    captured = model._captured_hidden_states

    assert len(captured) == 4  # embeddings + 3 layers
    assert captured[1] is None and captured[3] is None  # layers 0 and 2 not requested
    assert torch.equal(captured[0], torch.zeros(1, 1, 4))  # embedding output
    assert torch.equal(captured[2], torch.full((1, 1, 4), 3.0))  # after layers 0 and 1
    assert torch.equal(out, torch.full((1, 1, 4), 7.0))  # all three layers applied

    with pytest.raises(ValueError, match="layer 9"):
        model.set_capture_layer_ids([9])


def test_draft_runner_accessors_speak_both_module_protocols():
    """The draft reads the target through FreeToken's plain objects and HF modules alike."""
    from types import SimpleNamespace
    from freetoken.engine.draft_runner import _target_embedding_weight, _target_output_logits

    embed_weight = torch.randn(10, 4)
    head_weight = torch.randn(10, 4)
    hidden = torch.randn(1, 3, 4)

    # FreeToken: plain objects, a non-callable head, no get_input_embeddings()
    ft_head = SimpleNamespace(weight=head_weight, bias=None, tied_embedding=None, tp_size=1)
    ft_target = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=embed_weight, tp_size=1)),
        lm_head=ft_head,
    )
    assert torch.equal(_target_embedding_weight(ft_target), embed_weight)
    assert torch.allclose(
        _target_output_logits(ft_target, hidden),
        torch.nn.functional.linear(hidden, head_weight),
    )

    # transformers: get_input_embeddings() and a callable head
    hf_head = torch.nn.Linear(4, 10, bias=False)
    hf_target = SimpleNamespace(
        get_input_embeddings=lambda: SimpleNamespace(weight=embed_weight),
        lm_head=hf_head,
    )
    assert torch.equal(_target_embedding_weight(hf_target), embed_weight)
    assert torch.allclose(_target_output_logits(hf_target, hidden), hf_head(hidden))

    # a sharded vocabulary is refused, not silently drafted against a slice
    sharded = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=embed_weight, tp_size=2))
    )
    with pytest.raises(NotImplementedError, match="tensor-parallel"):
        _target_embedding_weight(sharded)


def test_qwen3_capture_materialises_the_carried_residual():
    """Qwen3 carries the residual to the next layer, so the capture must add it in."""
    from types import SimpleNamespace
    from freetoken.models.qwen3.model import Qwen3Model

    class _StubLayer:
        """Mimics the fused-residual contract: returns (mlp_out, residual)."""

        def __init__(self, delta: float) -> None:
            self._delta = delta

        def forward(self, x, residual):  # noqa: ANN001 - test stub
            new_residual = x if residual is None else residual + x
            return torch.full_like(x, self._delta), new_residual

    model = object.__new__(Qwen3Model)
    model.embed_tokens = SimpleNamespace(forward=lambda ids: torch.ones(1, len(ids), 4))
    model.norm = SimpleNamespace(forward=lambda x, residual: (residual + x, None))
    model.layers = SimpleNamespace(op_list=[_StubLayer(2.0), _StubLayer(3.0)])

    model.set_capture_layer_ids([-1, 0, 1])
    model.forward(torch.tensor([5]))
    captured = model._captured_hidden_states

    assert len(captured) == 3
    assert torch.equal(captured[0], torch.ones(1, 1, 4))          # embeddings
    assert torch.equal(captured[1], torch.full((1, 1, 4), 3.0))   # 1 (residual) + 2 (mlp out)
    assert torch.equal(captured[2], torch.full((1, 1, 4), 6.0))   # 3 (residual) + 3 (mlp out)
