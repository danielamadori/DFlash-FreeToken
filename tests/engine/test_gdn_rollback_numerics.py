"""The rewound GDN state against the state a non-speculative run would have produced.

``test_gdn_rollback.py`` drives a ``FakePool``: it proves the bookkeeping -- when a rewind
happens, how many rows it walks, that the scratch slot is held across blocks -- and proves
nothing at all about the numbers. That is why ``FREETOKEN_SPEC_GDN_ROLLBACK`` is off by
default: the flag's own comment says it stays off until the rewound state is shown to match a
non-speculative run. This file is that comparison.

The shape of the claim: a request at ``P`` cached tokens verifies a window of ``W = K + 1``
rows and commits ``committed = accepted + 1`` of them. Afterwards its recurrent and convolution
state must be indistinguishable from the state of a request that was fed exactly those
``P + committed`` tokens and never speculated. Two slots of the same pool are run side by side
-- one straight, one speculate-then-rewind -- and compared.

A rewind that is merely close is not good enough. The error does not stay where it is made: the
state feeds the next block's draft, so a small drift compounds block over block, which is what
a decaying acceptance rate looks like from outside. So the tolerance here is one bf16 ulp, not
a comfortable 1e-2.

Measured, the rewind is better than that: the recurrent state comes back bit-identical, because
the rescan runs the same deterministic kernel over the same stashed rows from the same restored
state. The one place a difference survives is the newest column of the convolution window, and
only at model width: the verification forward computes the input projection over W rows while
the control computes it over ``committed`` rows, and a GEMM that tiles differently rounds a few
channels differently. At hidden 2048 that was 3 channels of 2048, off by 2**-7 -- exactly one
ulp. That floor is a property of running the projection over a different number of rows, not of
the rewind, and it is why the assertions below are one-ulp rather than exact.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.core import Batch, Context, Req, SamplingParams
from freetoken.engine.gdn_rollback import GDNRollback
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.models.qwen3_5_moe.gdn import Qwen3_5GatedDeltaNet, rescan_prefix_fused
from freetoken.utils import torch_dtype

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEV = torch.device("cuda")
HEAD_DIM, CONV_K, EPS = 128, 4, 1e-6
BLOCK = 8                    # DFlash2's draft block; the window is BLOCK + 1
# One bf16 ulp. The rewind reuses the verification forward's projected rows while the control
# reprojects them, so a few channels can round apart; anything past this is arithmetic drift.
ULP = 2.0 ** -7
# (hidden, num_k_heads, num_v_heads). The narrow one is bit-exact everywhere and runs in
# milliseconds; the wide one is where the projection actually tiles, which is the only place
# the one-ulp floor shows up at all.
SHAPES = ((256, 2, 4), (2048, 4, 8))


@pytest.fixture(autouse=True)
def _single_rank():
    """The op's fused in_proj is a tensor-parallel layer and asks for the world size at
    construction; nothing in a unit test sets one up."""
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _make_op(shape, seed: int = 0) -> Qwen3_5GatedDeltaNet:
    """One layer with random weights. The oracle here is another run of this same op, not HF
    reference math, so the weights only have to be valid -- but they must not be degenerate:
    a zero A_log or dt_bias makes the gate constant and hides a state that failed to advance."""
    hidden, num_k, num_v = shape
    torch.manual_seed(seed)
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        op = Qwen3_5GatedDeltaNet(
            hidden_size=hidden, num_k_heads=num_k, num_v_heads=num_v,
            head_k_dim=HEAD_DIM, head_v_dim=HEAD_DIM,
            conv_kernel_size=CONV_K, rms_norm_eps=EPS, layer_id=0,
        )
    conv_dim = 2 * num_k * HEAD_DIM + num_v * HEAD_DIM
    in_rows = conv_dim + num_v * HEAD_DIM + num_v + num_v
    op.load_state_dict({
        "in_proj.weight": torch.randn(in_rows, hidden, device=DEV, dtype=torch.bfloat16) * 0.05,
        "conv1d.weight": torch.randn(conv_dim, 1, CONV_K, device=DEV, dtype=torch.bfloat16) * 0.3,
        "dt_bias": torch.empty(num_v, device=DEV, dtype=torch.float32).uniform_(-1.0, 1.0),
        "A_log": torch.empty(num_v, device=DEV, dtype=torch.float32).uniform_(0.01, 16.0).log(),
        "norm.weight": torch.empty(HEAD_DIM, device=DEV, dtype=torch.float32).normal_(1.0, 0.1)
                            .to(torch.bfloat16),
        "out_proj.weight": torch.randn(hidden, num_v * HEAD_DIM, device=DEV,
                                       dtype=torch.bfloat16) * 0.05,
    })
    return op


def _ctx(shape, num_slots: int = 8) -> Context:
    import freetoken.core as core
    from freetoken.kvcache.linear_state_pool import LinearStatePool

    _, num_k, num_v = shape
    group = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=num_k, num_value_heads=num_v,
        key_head_dim=HEAD_DIM, value_head_dim=HEAD_DIM, conv_kernel_dim=CONV_K,
        output_gate="sigmoid",
    )
    core._GLOBAL_CTX = None
    ctx = Context(page_size=1)
    ctx.linear_state_pool = LinearStatePool(group, num_slots, torch.bfloat16, DEV, tp_size=1)
    core.set_global_ctx(ctx)
    return ctx


def _run(op, ctx, hidden: torch.Tensor, *, slot: int, cached_len: int) -> torch.Tensor:
    """One forward of ``hidden`` rows for a single request holding ``slot``.

    ``cached_len > 0`` is the continuation case -- the state in the slot is carried in rather
    than zeroed -- which is exactly what a speculative verification forward is.
    """
    n = hidden.shape[0]
    # input_ids is the whole sequence, cached rows included; the forward's new rows are the
    # tail past cached_len.
    req = Req(input_ids=torch.zeros(cached_len + n, dtype=torch.int32), table_idx=slot,
              cached_len=cached_len, output_len=1, uid=slot,
              sampling_params=SamplingParams(), cache_handle=None)
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = [req]
    with ctx.forward_batch(batch):
        return op.forward(hidden)


def _state(ctx, slot: int) -> tuple[torch.Tensor, torch.Tensor]:
    pool = ctx.linear_state_pool
    return (pool.recurrent_states[0, slot].clone(), pool.conv_states[0, slot].clone())


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"h{s[0]}")
@pytest.mark.parametrize("accepted", (0, 1, 3, 7))
@pytest.mark.parametrize("prefix", (5, 64, 130))
def test_the_rewound_state_matches_a_run_that_never_speculated(accepted, prefix, shape):
    """Rewinding after a partly-rejected block lands on the non-speculative state.

    ``prefix`` crosses the fla chunk size (64) in both directions, because the kernel emits one
    state per chunk and a rewind that happened to work only inside a single chunk would be a
    coincidence, not a property. ``accepted`` spans total rejection (0 candidates kept, but the
    pending token still commits) up to one short of the whole block -- full acceptance takes the
    no-rewind fast path and is covered separately.
    """
    hidden = shape[0]
    op, ctx = _make_op(shape, seed=accepted + prefix), _ctx(shape)
    committed = accepted + 1
    torch.manual_seed(101)
    ctx_rows = torch.randn(prefix, hidden, device=DEV, dtype=torch.bfloat16)
    window = torch.randn(BLOCK + 1, hidden, device=DEV, dtype=torch.bfloat16)

    # Slot 1: the control. Context, then only the rows that will turn out to be committed.
    _run(op, ctx, ctx_rows, slot=1, cached_len=0)
    _run(op, ctx, window[:committed], slot=1, cached_len=prefix)
    rec_ref, conv_ref = _state(ctx, 1)

    # Slot 2: the same context, then the whole verification window, then the rewind.
    _run(op, ctx, ctx_rows, slot=2, cached_len=0)
    rollback = GDNRollback(ctx.linear_state_pool)
    rollback.open(2)
    ctx.gdn_rollback = rollback
    try:
        _run(op, ctx, window, slot=2, cached_len=prefix)
        rollback.rewind(accepted)
    finally:
        ctx.gdn_rollback = None
    rec_rw, conv_rw = _state(ctx, 2)

    torch.testing.assert_close(conv_rw.float(), conv_ref.float(), rtol=ULP, atol=ULP)
    torch.testing.assert_close(rec_rw.float(), rec_ref.float(), rtol=ULP, atol=ULP)


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"h{s[0]}")
@pytest.mark.parametrize("prefix", (5, 64, 130))
def test_a_fully_accepted_block_needs_no_rewind_to_be_right(prefix, shape):
    """The fast path is a claim about the numbers too.

    ``rewind()`` returns without touching the state when every row committed, on the grounds
    that the live state already reflects the whole window. If that were wrong the common case --
    the one that makes speculation pay -- would be the one corrupting the state, silently,
    because no rewind ever runs to be blamed.
    """
    hidden = shape[0]
    op, ctx = _make_op(shape, seed=prefix), _ctx(shape)
    torch.manual_seed(202)
    ctx_rows = torch.randn(prefix, hidden, device=DEV, dtype=torch.bfloat16)
    window = torch.randn(BLOCK + 1, hidden, device=DEV, dtype=torch.bfloat16)

    _run(op, ctx, ctx_rows, slot=1, cached_len=0)
    _run(op, ctx, window, slot=1, cached_len=prefix)
    rec_ref, conv_ref = _state(ctx, 1)

    _run(op, ctx, ctx_rows, slot=2, cached_len=0)
    rollback = GDNRollback(ctx.linear_state_pool)
    rollback.open(2)
    ctx.gdn_rollback = rollback
    try:
        _run(op, ctx, window, slot=2, cached_len=prefix)
        rollback.rewind(BLOCK)  # every candidate accepted: window fully committed
    finally:
        ctx.gdn_rollback = None
    rec_rw, conv_rw = _state(ctx, 2)

    torch.testing.assert_close(conv_rw.float(), conv_ref.float(), rtol=ULP, atol=ULP)
    torch.testing.assert_close(rec_rw.float(), rec_ref.float(), rtol=ULP, atol=ULP)


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"h{s[0]}")
def test_consecutive_blocks_do_not_accumulate_drift(shape):
    """Eight rewound blocks in a row, against eight straight continuations.

    A per-block error small enough to pass a single-block tolerance still ruins a long
    generation, because every block's state is the next block's starting point: the drift
    compounds instead of averaging out. Comparing only at the end, after eight blocks, is what
    makes that visible -- and a decaying acceptance rate is exactly what compounding drift looks
    like from outside the engine.
    """
    hidden = shape[0]
    op, ctx = _make_op(shape, seed=7), _ctx(shape)
    torch.manual_seed(303)
    ctx_rows = torch.randn(48, hidden, device=DEV, dtype=torch.bfloat16)
    _run(op, ctx, ctx_rows, slot=1, cached_len=0)
    _run(op, ctx, ctx_rows, slot=2, cached_len=0)

    rollback = GDNRollback(ctx.linear_state_pool)
    ctx.gdn_rollback = rollback
    ref_len = spec_len = 48
    try:
        for block, accepted in enumerate((2, 0, 5, 1, 7, 3, 0, 4)):
            committed = accepted + 1
            window = torch.randn(BLOCK + 1, hidden, device=DEV, dtype=torch.bfloat16)

            ctx.gdn_rollback = None
            _run(op, ctx, window[:committed], slot=1, cached_len=ref_len)
            ref_len += committed

            ctx.gdn_rollback = rollback
            rollback.open(2)
            _run(op, ctx, window, slot=2, cached_len=spec_len)
            rollback.rewind(accepted)
            spec_len += committed
    finally:
        ctx.gdn_rollback = None

    assert ref_len == spec_len
    rec_ref, conv_ref = _state(ctx, 1)
    rec_rw, conv_rw = _state(ctx, 2)
    torch.testing.assert_close(conv_rw.float(), conv_ref.float(), rtol=ULP, atol=ULP)
    torch.testing.assert_close(rec_rw.float(), rec_ref.float(), rtol=ULP, atol=ULP)


def _batch_for(ctx, hidden, *, slot: int, cached_len: int):
    """A Req/Batch pair kept alive across capture and replay.

    The graph bakes the addresses of everything the forward touches, the fla metadata
    included, so capture and replay have to see the SAME batch object: a fresh one would
    build fresh index tensors at fresh addresses and the replay would read the captured
    ones, which still hold the capture's values.
    """
    n = hidden.shape[0]
    req = Req(input_ids=torch.zeros(cached_len + n, dtype=torch.int32), table_idx=slot,
              cached_len=cached_len, output_len=1, uid=slot,
              sampling_params=SamplingParams(), cache_handle=None)
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = [req]
    return batch


@pytest.mark.parametrize("accepted", (0, 3, 6))
def test_a_rewind_that_adopts_a_graphs_stash_lands_where_the_eager_one_does(accepted):
    """The rewind after a REPLAYED verify, against the rewind after an eager one.

    Under SPEC_VERIFY_GRAPH the verification forward runs no Python, so the layers never call
    stash() and the rollback would find nothing to walk forward over. The graph runner instead
    adopts the stash recorded at capture time, whose entries point at the graph's static
    activations -- the ones the replay has just rewritten for this block's rows.

    That is an argument, not a measurement, and it is the reason both SPEC_VERIFY_GRAPH and
    SPEC_GDN_ROLLBACK are off by default: a stale captured buffer would feed the rewind the
    PREVIOUS block's rows, and a recurrent state built from the wrong rows still produces
    fluent text. Nothing about it looks like an error from outside.

    So: the same tokens twice, one slot eager and one through a captured graph that is
    replayed on them, and the two states compared. If the adopted entries were stale the
    replayed slot would hold the warm-up block's state and this would fail.
    """
    shape = SHAPES[0]
    hidden_size = shape[0]
    op, ctx = _make_op(shape, seed=accepted), _ctx(shape)
    committed = accepted + 1
    prefix = 48
    torch.manual_seed(404)
    ctx_rows = torch.randn(prefix, hidden_size, device=DEV, dtype=torch.bfloat16)
    warmup = torch.randn(BLOCK + 1, hidden_size, device=DEV, dtype=torch.bfloat16)
    window = torch.randn(BLOCK + 1, hidden_size, device=DEV, dtype=torch.bfloat16)

    # Slot 1: the control, entirely eager.
    _run(op, ctx, ctx_rows, slot=1, cached_len=0)
    _run(op, ctx, window[:committed], slot=1, cached_len=prefix)
    rec_ref, conv_ref = _state(ctx, 1)

    # Slot 2: capture a graph over the forward, then replay it on this block's rows.
    _run(op, ctx, ctx_rows, slot=2, cached_len=0)
    pool = ctx.linear_state_pool
    before = (pool.recurrent_states[0, 2].clone(), pool.conv_states[0, 2].clone())

    statico = warmup.clone()                      # the graph's static input row buffer
    batch = _batch_for(ctx, statico, slot=2, cached_len=prefix)
    rollback = GDNRollback(pool)
    rollback.open(2)
    ctx.gdn_rollback = rollback
    try:
        # Warm-up on a side stream, as torch requires before capture, then the capture itself.
        flusso = torch.cuda.Stream()
        flusso.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(flusso), ctx.forward_batch(batch):
            op.forward(statico)
        torch.cuda.current_stream().wait_stream(flusso)

        grafo = torch.cuda.CUDAGraph()
        with torch.cuda.graph(grafo), ctx.forward_batch(batch):
            op.forward(statico)
        catturato = dict(rollback._stash)          # what the layers stashed while capturing
        assert catturato, "the capture recorded no layer: there would be nothing to adopt"
        rollback.close()

        # The warm-up and the capture both advanced the state; put it back where the eager
        # control started, so the replay below begins from the same place.
        pool.recurrent_states[0, 2] = before[0]
        pool.conv_states[0, 2] = before[1]

        rollback.open(2)
        statico.copy_(window)                      # this block's rows, into the baked address
        grafo.replay()
        rollback.adopt(catturato, rescan_prefix_fused)
        rollback.rewind(accepted)
    finally:
        ctx.gdn_rollback = None

    rec_rw, conv_rw = _state(ctx, 2)
    torch.testing.assert_close(conv_rw.float(), conv_ref.float(), rtol=ULP, atol=ULP)
    torch.testing.assert_close(rec_rw.float(), rec_ref.float(), rtol=ULP, atol=ULP)
