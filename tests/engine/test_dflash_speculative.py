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


def unittest_mock_cuda_stream(engine):
    from unittest.mock import patch
    import contextlib

    @contextlib.contextmanager
    def _mock_ctx():
        with patch("torch.cuda.current_stream", return_value=None), \
             patch("torch.cuda.Event"):
            yield

    return _mock_ctx()




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
    model.embed_tokens = SimpleNamespace(forward=lambda ids: torch.zeros(len(ids), 4))
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
    assert torch.equal(captured[0], torch.zeros(1, 4))  # embedding output
    assert torch.equal(captured[2], torch.full((1, 4), 3.0))  # after layers 0 and 1
    assert torch.equal(out, torch.full((1, 4), 7.0))  # all three layers applied

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
    model.embed_tokens = SimpleNamespace(forward=lambda ids: torch.ones(len(ids), 4))
    model.norm = SimpleNamespace(forward=lambda x, residual: (residual + x, None))
    model.layers = SimpleNamespace(op_list=[_StubLayer(2.0), _StubLayer(3.0)])

    model.set_capture_layer_ids([-1, 0, 1])
    model.forward(torch.tensor([5]))
    captured = model._captured_hidden_states

    assert len(captured) == 3
    assert torch.equal(captured[0], torch.ones(1, 4))          # embeddings
    assert torch.equal(captured[1], torch.full((1, 4), 3.0))   # 1 (residual) + 2 (mlp out)
    assert torch.equal(captured[2], torch.full((1, 4), 6.0))   # 3 (residual) + 3 (mlp out)


def test_rejection_sample_refuses_a_single_verification_row():
    """A decode-shaped [1, 1, V] target must raise, not silently accept nothing."""
    from freetoken.engine.draft_runner import rejection_sample

    vocab, gamma = 32, 5
    draft_tokens = torch.full((1, gamma), 7)
    target_probs = torch.zeros(1, 1, vocab)
    target_probs[0, 0, 7] = 1.0
    draft_probs = torch.zeros(1, gamma, vocab)

    with pytest.raises(ValueError, match=r"must score all 6 positions"):
        rejection_sample(draft_tokens, target_probs, draft_probs, temperature=0.0)


def test_rejection_sample_accepts_the_matching_prefix():
    """With all K+1 rows the greedy path accepts the leading run and takes the bonus."""
    from freetoken.engine.draft_runner import rejection_sample

    vocab, gamma = 32, 4
    draft_tokens = torch.tensor([[3, 4, 9, 9]])
    # target predicts 3, 4, 5, ... -> the first two candidates match, the third does not
    target_probs = torch.zeros(1, gamma + 1, vocab)
    for row, token in enumerate([3, 4, 5, 6, 7]):
        target_probs[0, row, token] = 1.0
    draft_probs = torch.zeros(1, gamma, vocab)

    accepted, next_token = rejection_sample(draft_tokens, target_probs, draft_probs, 0.0)
    assert accepted == 2
    assert int(next_token) == 5  # the target's own token at the first rejected position
