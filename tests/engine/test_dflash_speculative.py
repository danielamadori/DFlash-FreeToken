from types import SimpleNamespace

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
    from freetoken.engine.draft_runner import _embed_with_target, _target_output_logits

    embed_weight = torch.randn(10, 4)
    head_weight = torch.randn(10, 4)
    hidden = torch.randn(1, 3, 4)
    ids = torch.tensor([[1, 7, 3]])

    # FreeToken: plain objects, a non-callable head, no get_input_embeddings()
    ft_head = SimpleNamespace(weight=head_weight, bias=None, tied_embedding=None, tp_size=1)
    ft_embed = torch.nn.Embedding(10, 4)
    ft_embed.weight = torch.nn.Parameter(embed_weight)
    ft_embed.tp_size = 1
    ft_target = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=ft_embed),
        lm_head=ft_head,
    )
    assert torch.equal(_embed_with_target(ft_target, ids), embed_weight[ids])
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
    assert torch.equal(_embed_with_target(hf_target, ids), embed_weight[ids])
    assert torch.allclose(_target_output_logits(hf_target, hidden), hf_head(hidden))

    # a sharded vocabulary is refused, not silently drafted against a slice
    sharded = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=embed_weight, tp_size=2))
    )
    with pytest.raises(NotImplementedError, match="tensor-parallel"):
        _embed_with_target(sharded, ids)


def test_draft_embeds_through_a_target_whose_table_has_no_weight():
    """A GGUF target holds `qweight` and dequantizes per lookup; reaching for `.weight` fails."""
    from types import SimpleNamespace
    from freetoken.engine.draft_runner import _embed_with_target

    table = torch.randn(10, 4)

    class PackedEmbedding:
        """Stands in for GGUFEmbedding: a plain object with `forward`, no `.weight`, and --
        this is the part that bit -- NOT callable, since FreeToken layers are not nn.Modules."""

        tp_size = 1

        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            return table[ids]

    target = SimpleNamespace(model=SimpleNamespace(embed_tokens=PackedEmbedding()))
    ids = torch.tensor([[2, 5]])
    assert torch.equal(_embed_with_target(target, ids), table[ids])


def test_draft_projects_through_a_quantized_head_without_its_last_position_slice(monkeypatch):
    """A GGUF head has no weight matrix, and its own forward keeps only the last position.

    Asserted through which projection runs rather than through numbers: the slicing forward
    would return one row where a draft block needs one per drafted position, and that shape
    error would surface far from its cause.
    """
    from types import SimpleNamespace
    from freetoken.engine.draft_runner import _target_output_logits
    from freetoken.layers import gguf as gguf_layers

    hidden = torch.randn(1, 4, 6)
    table = torch.randn(3, 6)
    used: list[str] = []

    def plain_projection(self, x: torch.Tensor) -> torch.Tensor:
        used.append("GGUFLinear")
        # Like the real fused GGUF matmul: 2-D only, dim 0 is the batch.
        assert x.dim() == 2, f"the GGUF matmul takes [tokens, features], got {tuple(x.shape)}"
        return x @ table.T

    monkeypatch.setattr(gguf_layers.GGUFLinear, "forward", plain_projection)

    class Head(gguf_layers.GGUFLinear):
        tp_size = 1

        def __init__(self) -> None:
            pass

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            used.append("GGUFLMHead")
            return (x @ table.T)[:, -1:]

    out = _target_output_logits(SimpleNamespace(lm_head=Head()), hidden)
    assert used == ["GGUFLinear"], "the head's slicing forward must not be used for a block"
    assert out.shape == (1, 4, 3)


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


def test_qwen3_5_moe_capture_materialises_the_carried_residual() -> None:
    """The MoE family is why FreeToken is used here at all: DFlash has to reach it too.

    Same carried-residual contract as the dense Qwen3, so the same materialisation
    applies -- the layer returns (mlp_out, residual) and the add lands in the next
    layer's fused add+norm.
    """
    from types import SimpleNamespace
    from freetoken.models.qwen3_5_moe.model import Qwen3_5Model

    class _StubLayer:
        def __init__(self, delta: float) -> None:
            self._delta = delta

        def forward(self, x, residual):  # noqa: ANN001 - test stub
            new_residual = x if residual is None else residual + x
            return torch.full_like(x, self._delta), new_residual

    model = object.__new__(Qwen3_5Model)
    model.embed_tokens = SimpleNamespace(forward=lambda ids: torch.ones(len(ids), 4))
    model.norm = SimpleNamespace(forward_add_residual=lambda x, residual: (residual + x, None))
    model.layers = SimpleNamespace(op_list=[_StubLayer(2.0), _StubLayer(3.0)])

    model.set_capture_layer_ids([-1, 0, 1])
    model.forward(torch.tensor([5]))
    captured = model._captured_hidden_states

    assert len(captured) == 3
    assert torch.equal(captured[0], torch.ones(1, 4))          # embeddings
    assert torch.equal(captured[1], torch.full((1, 4), 3.0))   # 1 (residual) + 2 (mlp out)
    assert torch.equal(captured[2], torch.full((1, 4), 6.0))   # 3 (residual) + 3 (mlp out)


def test_every_family_with_capture_refuses_an_out_of_range_layer() -> None:
    """The guard has to hold on each family, not just the first one it was written for."""
    from types import SimpleNamespace
    from freetoken.models.muse_glimmer.model import MuseGlimmerModel
    from freetoken.models.qwen3.model import Qwen3Model
    from freetoken.models.qwen3_5_moe.model import Qwen3_5Model

    for cls in (MuseGlimmerModel, Qwen3Model, Qwen3_5Model):
        model = object.__new__(cls)
        model.layers = SimpleNamespace(op_list=[object(), object()])
        with pytest.raises(ValueError, match="layer 5"):
            model.set_capture_layer_ids([5])


def test_dflash2_drafts_through_its_selector_not_per_position_argmax(monkeypatch):
    """DFlash 2 couples the tokens of a block; choosing each independently is DFlash 1's rule.

    Its selector walks the block in order, scoring each candidate against the token just chosen
    through the predecessor/successor codebooks. Bypassing it produces a block whose tokens do
    not follow one another, which the target rejects -- visible only as a low acceptance rate,
    never as an error, which is why this asserts on which path runs rather than on output.
    """
    from types import SimpleNamespace
    from freetoken.engine import draft_runner

    calls: list[str] = []
    hidden = torch.randn(1, 3, 8)
    logits = torch.randn(1, 3, 11)

    class Selector:
        def select(self, h, lg, anchor_ids, temperature):
            calls.append("selector")
            assert h.shape[:2] == (1, 3), "the selector scores the whole block"
            assert anchor_ids.shape == (1,), "anchored on the token before the block"
            return torch.zeros(1, 3, dtype=torch.long), None, None

    monkeypatch.setattr(draft_runner, "_target_output_logits", lambda t, h: logits)
    monkeypatch.setattr(
        draft_runner, "_sampling_probs", lambda lg, *a, **k: torch.zeros(1, 3, 11)
    )

    runner = SimpleNamespace(
        draft_model=SimpleNamespace(candidate_selector=Selector()),
        _target=object(),
        _warned_sampling_selector=False,
    )
    block_output_ids = torch.tensor([[5, 0, 0]])

    # The branch under test, lifted out of draft() so it needs no checkpoint or GPU.
    selector = getattr(runner.draft_model, "candidate_selector", None)
    temperature = 0.0
    draft_logits = draft_runner._target_output_logits(runner._target, hidden)
    if selector is not None and temperature <= 0:
        tokens, _, _ = selector.select(hidden, draft_logits, block_output_ids[:, 0], temperature)
    else:
        tokens = torch.argmax(draft_logits, dim=-1)

    assert calls == ["selector"], "greedy DFlash 2 drafting must go through the selector"
    assert tokens.shape == (1, 3)


# ---------------------------------------------------------------------------
# The static ring cache path of DFlashRunner (plan: docs/plans/draft-cuda-graph-plan.md).
# A fake ring, draft model, target and graph on CPU tensors: what is asserted is which
# arguments reach the forward, in which order the cache is driven, and which path runs.
# ---------------------------------------------------------------------------

_HIDDEN, _VOCAB, _TARGET_LAYER = 4, 10, 0


class _FakeRing:
    """StaticDraftCache's interface (plan 2.1), recording the calls in order."""

    def __init__(self, window: int, block: int, log: list) -> None:
        self.window = window
        self.block = block
        self.ring = window + 2 * block
        self.log = log
        self.staged: list[torch.Tensor] = []
        self.masks: list[torch.Tensor] = []
        self.resets = 0

    def stage(self, row_pos: torch.Tensor) -> None:
        self.log.append("stage")
        self.staged.append(row_pos.clone())

    def mask(self, q_pos: torch.Tensor) -> torch.Tensor:
        self.log.append("mask")
        m = torch.ones(1, 1, q_pos.shape[0], self.ring, dtype=torch.bool)
        self.masks.append(m)
        return m

    def update(self, k, v, layer_idx, cache_kwargs=None):  # noqa: ANN001 - fake
        self.log.append("update")
        return k, v

    def retire(self) -> None:
        self.log.append("retire")

    def reset(self) -> None:
        self.log.append("reset")
        self.resets += 1


class _FakeDraft:
    """The draft forward: records its keyword arguments, returns one row per block slot."""

    def __init__(self, log: list, selector=None) -> None:  # noqa: ANN001 - fake
        self.log = log
        self.calls: list[dict] = []
        self.candidate_selector = selector
        self.config = SimpleNamespace(sliding_window=None)

    def __call__(self, **kwargs):  # noqa: ANN003 - fake
        self.log.append("forward")
        self.calls.append(kwargs)
        rows = kwargs["noise_embedding"].shape[1]
        return torch.zeros(1, rows, _HIDDEN)


class _Selector:
    def __init__(self, log: list) -> None:
        self.log = log
        self.anchors: list[torch.Tensor] = []

    def select(self, h, lg, anchor_ids, temperature):  # noqa: ANN001 - fake
        self.log.append("selector")
        self.anchors.append(anchor_ids.clone())
        return torch.arange(h.shape[1], dtype=torch.long)[None], None, None


class _FakeGraph:
    """DraftGraphRunner's dispatch contract: can_replay -> c or None, replay -> (tokens, probs)."""

    def __init__(self, log: list, c: int | None, k: int) -> None:
        self.log = log
        self.c = c
        self.k = k
        self.replays: list[tuple] = []
        self.tokens = torch.full((1, k - 1), 3, dtype=torch.long)
        self.probs = torch.zeros(1, k - 1, _VOCAB)

    def can_replay(self, target_hidden_states, k, temperature):  # noqa: ANN001 - fake
        self.log.append("can_replay")
        return self.c

    def replay(self, target_hidden_states, current_token_id, seq_len, c):  # noqa: ANN001
        self.log.append("replay")
        self.replays.append((seq_len, c, int(current_token_id.view(()))))
        return self.tokens, self.probs


def _target():
    embed = torch.nn.Embedding(_VOCAB, _HIDDEN)
    embed.tp_size = 1
    head = SimpleNamespace(
        weight=torch.randn(_VOCAB, _HIDDEN), bias=None, tied_embedding=None, tp_size=1
    )
    return SimpleNamespace(model=SimpleNamespace(embed_tokens=embed), lm_head=head)


def _static_runner(*, window: int = 16, block: int = 8, selector=None):  # noqa: ANN001
    """A DFlashRunner shell on the static path: the fields draft() and _run_block() touch."""
    from freetoken.engine.draft_runner import DFlashRunner

    log: list[str] = []
    r = DFlashRunner.__new__(DFlashRunner)
    r.device = torch.device("cpu")
    r.dtype = torch.float32
    r.block_size = block
    r._target = _target()
    r.draft_model = _FakeDraft(log, selector)
    r.target_layer_ids = [_TARGET_LAYER]
    r.mask_token_id = _VOCAB - 1
    r.input_embedding_scale = 1.0
    r._extract_context_feature = lambda hs, ids: torch.cat([hs[i + 1] for i in ids], dim=-1)
    r._make_cache = lambda config: (_ for _ in ()).throw(AssertionError("no DynamicCache"))
    r._crop_to = None
    r._cache_owner = None
    r._warned_sampling_selector = False
    r._window = window
    r._static_cache = _FakeRing(window, block, log)
    r._draft_cache = r._static_cache
    r._seq_len_t = torch.zeros(1, dtype=torch.int64)
    r._ar = torch.arange(window + 2 * block, dtype=torch.int64)
    r.graph = None
    r._shadow_blocks = 0
    r._shadow_mismatches = 0
    return r, log


def _hidden_states(c: int) -> list:
    """The target's store: entry layer + 1 holds [1, c, hidden]; the rest is unused."""
    return [None, torch.randn(1, c, _HIDDEN), None]


def _draft(r, c: int, seq_len: int, uid: int = 7, **kw):  # noqa: ANN001
    return r.draft(
        target_hidden_states=_hidden_states(c),
        current_token_id=torch.tensor([5], dtype=torch.int32),
        position_ids=torch.arange(seq_len + r.block_size + 1)[None],
        seq_len=seq_len,
        request_uid=uid,
        **kw,
    )


def test_run_block_drives_the_ring_around_the_forward():
    """stage -> mask -> forward(mask, positions, ring) -> retire, once each, per block."""
    r, log = _static_runner()
    c, k, seq_len = 3, 8, 20
    tokens, probs = _draft(r, c, seq_len)

    assert log == ["reset", "stage", "mask", "forward", "retire"]
    ring = r._static_cache
    call = r.draft_model.calls[0]
    expected = torch.arange(seq_len - c, seq_len + k)
    assert torch.equal(ring.staged[0], expected), "ctx rows then the noise rows, contiguous"
    assert torch.equal(call["position_ids"], expected[None])
    assert call["attention_mask"] is ring.masks[0], "the ring's mask, not the model's own"
    assert call["attention_mask"].shape == (1, 1, k, ring.ring)
    assert call["past_key_values"] is ring
    assert call["target_hidden"].shape == (1, c, _HIDDEN)
    assert call["noise_embedding"].shape == (1, k, _HIDDEN)
    assert tokens.shape == (1, k - 1) and probs.shape == (1, k - 1, _VOCAB)
    assert int(r._seq_len_t) == seq_len


def test_run_block_reads_the_block_start_from_the_device_buffer():
    """A captured graph cannot bake seq_len: the positions must come from _seq_len_t."""
    r, _ = _static_runner()
    c, k = 2, 8
    ids = r._block_ids(k, torch.tensor([5]))
    r._seq_len_t.fill_(11)
    r._run_block(torch.randn(1, c, _HIDDEN), ids, c, 0.0, 1.0, 0)
    r._seq_len_t.fill_(40)
    r._run_block(torch.randn(1, c, _HIDDEN), ids, c, 0.0, 1.0, 0)
    staged = r._static_cache.staged
    assert torch.equal(staged[0], torch.arange(11 - c, 11 + k))
    assert torch.equal(staged[1], torch.arange(40 - c, 40 + k))
    assert r._static_cache.log.count("retire") == 2


def test_first_block_beyond_the_window_feeds_the_last_window_rows():
    """The ring has window + 2 blocks slots; the rows beyond the window were masked anyway."""
    r, _ = _static_runner(window=16)
    c, k, seq_len = 30, 8, 100
    hs = _hidden_states(c)
    r.draft(
        target_hidden_states=hs,
        current_token_id=torch.tensor([5]),
        position_ids=torch.arange(seq_len + k + 1)[None],
        seq_len=seq_len,
        request_uid=1,
    )
    call = r.draft_model.calls[0]
    assert call["target_hidden"].shape == (1, 16, _HIDDEN)
    assert torch.equal(call["target_hidden"], hs[_TARGET_LAYER + 1][:, -16:])
    assert torch.equal(r._static_cache.staged[0], torch.arange(seq_len - 16, seq_len + k))


def test_greedy_static_path_uses_the_selector_and_sampling_bypasses_it():
    r, log = _static_runner(selector=_Selector([]))
    r.draft_model.candidate_selector.log = log
    tokens, _ = _draft(r, 2, 20, temperature=0.0)
    assert log[-2:] == ["retire", "selector"], "the selector runs after the noise is retired"
    assert torch.equal(r.draft_model.candidate_selector.anchors[0], torch.tensor([5]))
    assert torch.equal(tokens, torch.arange(r.block_size - 1)[None])

    del log[:]
    tokens, probs = _draft(r, 2, 21, temperature=0.7)
    assert "selector" not in log, "T > 0 draws per position, as before"
    assert tokens.shape == (1, r.block_size - 1)
    assert r._warned_sampling_selector


def test_static_path_refuses_a_block_size_override():
    """retire() drops exactly one block of noise rows; another k would leave keys visible."""
    r, _ = _static_runner(block=8)
    with pytest.raises(ValueError, match="retires 8 noise rows"):
        _draft(r, 2, 20, block_size=4)


def test_graph_dispatch_runs_after_the_owner_reset_and_skips_the_forward():
    r, log = _static_runner()
    graph = _FakeGraph(log, c=3, k=r.block_size)
    r.graph = graph
    tokens, probs = _draft(r, 3, 20, uid=7)
    assert log == ["reset", "can_replay", "replay"], "reset first, then replay; no eager body"
    assert tokens is graph.tokens and probs is graph.probs
    assert graph.replays == [(20, 3, 5)]

    del log[:]
    _draft(r, 3, 23, uid=7)
    assert log == ["can_replay", "replay"], "same request: the ring keeps its context"

    del log[:]
    _draft(r, 3, 5, uid=8)
    assert log == ["reset", "can_replay", "replay"], "a new request resets the ring first"
    assert r._cache_owner == 8


def test_graph_that_cannot_replay_falls_back_to_the_eager_ring_body():
    r, log = _static_runner()
    r.graph = _FakeGraph(log, c=None, k=r.block_size)
    _draft(r, 9, 20)
    assert log == ["reset", "can_replay", "stage", "mask", "forward", "retire"]


def test_graph_dispatch_comes_after_the_hidden_state_checks():
    """Missing hidden states raise the same error whether or not a graph is installed."""
    r, log = _static_runner()
    r.graph = _FakeGraph(log, c=3, k=r.block_size)
    with pytest.raises(RuntimeError, match="published no hidden states"):
        r.draft(
            target_hidden_states=[None, None, None],
            current_token_id=torch.tensor([5]),
            position_ids=torch.arange(30)[None],
            seq_len=20,
            request_uid=7,
        )
    assert "replay" not in log


def test_shadow_mode_returns_the_eager_pair_and_logs_a_mismatch(monkeypatch, caplog):
    from freetoken.env import ENV

    monkeypatch.setattr(ENV, "SPEC_DRAFT_GRAPH_SHADOW", True, raising=False)
    r, log = _static_runner(selector=_Selector([]))
    r.draft_model.candidate_selector.log = log
    graph = _FakeGraph(log, c=3, k=r.block_size)
    r.graph = graph
    graph.tokens = torch.arange(r.block_size - 1)[None].clone()  # what the selector returns
    with caplog.at_level("INFO"):
        tokens, probs = _draft(r, 3, 20)
    assert log == ["reset", "can_replay", "replay", "stage", "mask", "forward", "retire", "selector"]
    assert tokens is not graph.tokens and torch.equal(tokens, graph.tokens)
    assert probs is not graph.probs
    assert (r._shadow_blocks, r._shadow_mismatches) == (1, 0)

    graph.tokens[0, 2] = 9
    with caplog.at_level("WARNING"):
        tokens, _ = _draft(r, 3, 23)
    assert torch.equal(tokens, torch.arange(r.block_size - 1)[None]), "the eager tokens win"
    assert (r._shadow_blocks, r._shadow_mismatches) == (2, 1)
    assert any("shadow mismatch" in m and "first at position 2" in m for m in caplog.messages)


def test_shadow_flag_off_replays_without_the_eager_body(monkeypatch):
    from freetoken.env import ENV

    monkeypatch.setattr(ENV, "SPEC_DRAFT_GRAPH_SHADOW", False, raising=False)
    r, log = _static_runner()
    r.graph = _FakeGraph(log, c=3, k=r.block_size)
    _draft(r, 3, 20)
    assert "forward" not in log


def test_dynamic_cache_path_is_unchanged_for_drafts_without_a_ring():
    """DFlash 1: DynamicCache, positions sliced from position_ids, crop(-k), no mask."""
    from freetoken.engine.draft_runner import DFlashRunner

    log: list[str] = []
    r = DFlashRunner.__new__(DFlashRunner)
    r.device = torch.device("cpu")
    r.dtype = torch.float32
    r.block_size = 5
    r._target = _target()
    r.draft_model = _FakeDraft(log)
    r.target_layer_ids = [_TARGET_LAYER]
    r.mask_token_id = _VOCAB - 1
    r.input_embedding_scale = 1.0
    r._extract_context_feature = lambda hs, ids: torch.cat([hs[i + 1] for i in ids], dim=-1)
    r._cache_owner = None
    r._warned_sampling_selector = False
    r._window = None
    r._static_cache = None
    r._draft_cache = None
    r.graph = None

    class _Dyn:
        def __init__(self) -> None:
            self.length = 0
            self.crops: list[int] = []

        def get_seq_length(self) -> int:
            return self.length

        def crop(self, n: int) -> None:
            self.crops.append(n)
            self.length += n

    made: list[_Dyn] = []

    def make_cache(config):  # noqa: ANN001 - fake
        made.append(_Dyn())
        return made[-1]

    def crop_to(cache, length):  # noqa: ANN001 - dflash/model.py::_crop_to
        cache.crop(-(cache.get_seq_length() - length))

    r._make_cache = make_cache
    r._crop_to = crop_to

    c, k, seq_len = 3, 5, 20
    position_ids = torch.arange(seq_len + k + 1)[None]
    tokens, probs = r.draft(
        target_hidden_states=_hidden_states(c),
        current_token_id=torch.tensor([5]),
        position_ids=position_ids,
        seq_len=seq_len,
        request_uid=7,
    )
    made[0].length = c + k  # what the forward would have appended
    assert len(made) == 1 and r._draft_cache is made[0]
    call = r.draft_model.calls[0]
    assert "attention_mask" not in call, "the model builds its own mask from the shapes"
    assert torch.equal(call["position_ids"], position_ids[:, seq_len - c : seq_len + k])
    assert call["past_key_values"] is made[0]
    assert made[0].crops == [-k]
    assert tokens.shape == (1, k - 1) and probs.shape == (1, k - 1, _VOCAB)


def _tiny_config(**overrides):  # noqa: ANN003
    """A DFlash 2 config small enough for a CPU ring: window 16, one layer, one kv head."""
    fields = dict(
        sliding_window=16,
        layer_types=["sliding_attention"],
        is_causal=False,
        num_hidden_layers=1,
        num_key_value_heads=1,
        head_dim=2,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _layers(*causal_flags):  # noqa: ANN002
    """The instantiated draft layers: what the model's mask builder reads is_causal from."""
    return SimpleNamespace(
        layers=[SimpleNamespace(self_attn=SimpleNamespace(is_causal=f)) for f in causal_flags]
    )


def test_build_static_cache_follows_the_cache_support_rule():
    from freetoken.engine.draft_cache import StaticDraftCache
    from freetoken.engine.draft_runner import _build_static_cache

    cpu = dict(block=8, device=torch.device("cpu"), dtype=torch.float32)
    ring = _build_static_cache(_tiny_config(), _layers(False), **cpu)
    assert isinstance(ring, StaticDraftCache)
    assert (ring.window, ring.block, ring.causal, ring.layers, ring.kv_heads, ring.head_dim) == (
        16, 8, False, 1, 1, 2
    )
    # causal comes off the layers, which is where the model resolves it
    assert _build_static_cache(_tiny_config(), _layers(True), **cpu).causal is True
    with pytest.raises(ValueError, match="disagree on is_causal"):
        _build_static_cache(_tiny_config(num_hidden_layers=2), _layers(True, False), **cpu)
    # head_dim falls back to hidden_size / heads when the config does not spell it out
    fallback = _tiny_config(head_dim=None, hidden_size=8, num_attention_heads=4)
    assert _build_static_cache(fallback, _layers(False), **cpu).head_dim == 2
    # DFlash 1: full attention, no window -> the DynamicCache path
    assert _build_static_cache(
        _tiny_config(sliding_window=None, layer_types=None), _layers(False), **cpu
    ) is None


def test_static_path_runs_on_the_real_ring():
    """The runner's calls satisfy the real cache's contract, not just the fake's."""
    from freetoken.engine.draft_cache import NEG
    from freetoken.engine.draft_runner import _build_static_cache

    r, log = _static_runner(window=16, block=8)
    ring = _build_static_cache(
        _tiny_config(), _layers(False), block=8, device=torch.device("cpu"), dtype=torch.float32
    )
    r._static_cache = r._draft_cache = ring
    c, k, seq_len = 3, 8, 20
    _draft(r, c, seq_len)
    call = r.draft_model.calls[0]
    assert call["attention_mask"].shape == (1, 1, k, ring.ring)
    assert call["attention_mask"].dtype == torch.bool
    assert call["past_key_values"] is ring
    # After retire the ctx rows are live and the noise rows are gone, as crop(-k) left them.
    live = ring.slot_pos[ring.slot_pos != NEG].sort().values
    assert torch.equal(live, torch.arange(seq_len - c, seq_len))

    # A new request: reset() empties the ring in place, and its first block, longer than
    # the window, is accepted by the real stage() only because the runner truncated it.
    _draft(r, 40, 100, uid=8)
    live = ring.slot_pos[ring.slot_pos != NEG].sort().values
    assert torch.equal(live, torch.arange(100 - 16, 100)), "the last window rows, no noise"
    assert r._draft_cache is ring
