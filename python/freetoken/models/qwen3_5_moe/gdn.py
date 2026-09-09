from __future__ import annotations

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
import os

from freetoken.kernel.causal_conv1d import causal_conv1d_decode, causal_conv1d_varlen
from freetoken.layers import BaseOP, LinearColParallelMerged

from freetoken.kernel.triton.fp8_block_linear import Fp8BlockColMerged
from freetoken.kernel.triton.fp8_pertensor_linear import Fp8PerTensorColMerged

from .gdn_kernels import gdn_decode_fla, gdn_prefill_chunk_fla
from .quant_linear import make_replicated_quant


def v_grouped_to_tiled(
    x: torch.Tensor, rows: int, num_k_heads: int, num_v_heads: int, head_v_dim: int
) -> torch.Tensor:
    """Reorder a V-axis activation from grouped [K, R, D] to llama.cpp's tiled [R, K, D].

    The GGUF out_proj keeps the column order llama.cpp wrote (a column permutation cannot be
    applied to packed blocks), so the activation moves instead. This is the activation-side
    twin of gguf._ungroup_v on the weight: x_tiled @ W_tiled.T == x_grouped @ ungroup(W_tiled).T.
    Returns a [rows, num_v_heads, head_v_dim]-shaped view; reshape(rows, -1) materialises it.
    """
    R = num_v_heads // num_k_heads
    return x.reshape(rows, num_k_heads, R, head_v_dim).transpose(1, 2)


class _DepthwiseConv1d(BaseOP):
    """Holds the depthwise conv weight ``[conv_dim, 1, K]`` (key ``conv1d.weight``)."""

    def __init__(self, conv_dim: int, kernel: int):
        self.weight = torch.empty(conv_dim, 1, kernel)


class _GatedRMSNorm(BaseOP):
    """RMSNorm of x followed by a silu(z) gate (HF Qwen3_5MoeRMSNormGated).

    Uses the fused fla ``rms_norm_gated`` triton kernel (norm(x) * silu(z) in one
    kernel) instead of the unfused pow/mean/rsqrt/mul/silu chain, matching sglang's
    ``RMSNormGated`` -- collapses ~8 elementwise kernels per GDN layer into one."""

    def __init__(self, dim: int, eps: float):
        self.weight = torch.empty(dim)
        self.eps = eps

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.fla import rms_norm_gated

        return rms_norm_gated(
            x=x, weight=self.weight, bias=None, z=z, eps=self.eps,
            is_rms_norm=True, norm_before_gate=True, activation="silu",
        )


_FORWARD_TIMING = bool(os.environ.get("FREETOKEN_FORWARD_TIMING"))


def rescan_prefix_fused(entries, *, live_slot: int, scratch_slot: int, committed: int) -> None:
    """Walk every linear layer's state forward over a block's committed prefix, in one launch.

    The layers are independent sequences over the same kernel, so they do not need 48 separate
    calls: the state pool holds them in one ``[layers, slots, ...]`` tensor, and the chunk
    kernel already accepts a batch of variable-length sequences with a state slot each. Viewing
    the pool as ``[layers * slots, ...]`` turns the layer index into part of the slot index, and
    the whole rewind becomes one call over ``layers x committed`` tokens.

    Per-layer launches cost about 21 ms of a 88 ms speculative block on Qwen3.8-27B -- almost
    all of it launch overhead, since the scan itself covers three tokens.
    """
    ctx = get_global_ctx()
    pool = ctx.linear_state_pool
    stashes = list(entries)
    if not stashes:
        return
    n = len(stashes)
    device = stashes[0].q.device

    rec = pool.recurrent_states
    assert rec.is_contiguous(), (
        "the state pool must be contiguous to be viewed as [layers * slots, ...]; a reshape "
        "here would copy, and the kernel's in-place write-back would land in the copy"
    )
    num_slots = rec.shape[1]
    flat_state = rec.view(-1, *rec.shape[2:])
    # Host-staged like FLAMetadata: a pageable copy here would block until the verification
    # forward drains, and a fresh cu_seqlens object would make the kernels read their chunk
    # bookkeeping back from the device on every rewind (the index cache is keyed by identity).
    from freetoken.attention.linear import build_fla_chunk_indices

    pin = {"device": "cpu", "pin_memory": torch.cuda.is_available()}
    indices = torch.tensor(
        [st.local_index * num_slots + live_slot for st in stashes],
        dtype=torch.int32, **pin,
    ).to(device, non_blocking=True)
    cu_seqlens = torch.arange(
        0, (n + 1) * committed, committed, dtype=torch.int64, **pin
    ).to(device, non_blocking=True)
    chunks = build_fla_chunk_indices([committed] * n, device, pin_memory=pin["pin_memory"])

    def joined(name: str) -> torch.Tensor:
        return torch.cat([getattr(st, name)[:, :committed] for st in stashes], dim=1)

    gdn_prefill_chunk_fla(
        joined("q"), joined("k"), joined("v"), joined("g"), joined("beta"),
        state_source=flat_state, indices=indices,
        cu_seqlens=cu_seqlens, scale=stashes[0].head_k_dim ** -0.5,
        **chunks,
    )

    # The convolution state is a window over raw inputs, so it is rebuilt rather than rescanned:
    # the restored pre-block window followed by the committed rows, keeping the tail.
    cv = pool.conv_states
    width = cv.shape[-1]
    rows = torch.tensor([st.local_index for st in stashes], dtype=torch.long, device=device)
    prefix = torch.stack(
        [st.conv_in[:committed].transpose(0, 1).to(cv.dtype) for st in stashes], dim=0
    )
    window = torch.cat([cv[rows, scratch_slot], prefix], dim=-1)[..., -width:]
    cv[rows, live_slot] = window


class Qwen3_5GatedDeltaNet(BaseOP):
    """GatedDeltaNet op using the vendored flash-linear-attention triton kernels
    (``freetoken.kernel.fla``) for the recurrence and a per-request
    recurrent + conv state held in ``ctx.linear_state_pool`` (keyed by ``Req.table_idx``).

    Parameter names match HF (``in_proj_qkv``/``in_proj_z``/``in_proj_b``/``in_proj_a``/
    ``conv1d``/``A_log``/``dt_bias``/``norm``/``out_proj``). Handles prefill (incl. chunked
    continuation) and single-token decode; state is fresh when ``req.cached_len == 0``.
    """

    def __init__(
        self, hidden_size, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
        conv_kernel_size, rms_norm_eps, layer_id, expert_quant: str = "none",
        attn_quant: str = "none",
    ):
        self.layer_id = layer_id
        # The fla chunk/decode kernels read+write the recurrent state and the per-chunk h as
        # [V, K] while the LinearStatePool declares it [K, V]; these coincide (and the
        # hybrid-radix snapshot scatter h[h_row]->slot is a plain copy) only when the two head
        # dims are equal. Qwen3.5/3.6 satisfy this (128/128); guard any future config.
        assert head_k_dim == head_v_dim, (
            f"GatedDeltaNet requires head_k_dim == head_v_dim, got {head_k_dim} != {head_v_dim}"
        )
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        # Set by the GGUF converter when out_proj keeps llama.cpp's tiled V-head columns.
        self.out_proj_v_tiled = False
        self.key_dim = num_k_heads * head_k_dim
        self.value_dim = num_v_heads * head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.conv_kernel_size = conv_kernel_size
        # qkv|z carry a weight scale (block-fp8 weight_scale_inv, or per-tensor FP8
        # weight_scale); b|a stay bf16. Both quant modes therefore split the four-way
        # fusion into an fp8 qkvz GEMM + a bf16 ba GEMM (matches sglang/vLLM).
        self._block_fp8 = expert_quant == "fp8_block"
        self._pertensor_fp8 = attn_quant == "fp8_pertensor"
        self._fp8 = self._block_fp8 or self._pertensor_fp8

        self._in_proj_split = [self.conv_dim, self.value_dim, num_v_heads, num_v_heads]
        if self._fp8:
            ColMerged = Fp8BlockColMerged if self._block_fp8 else Fp8PerTensorColMerged
            self.in_proj_qkvz = ColMerged(
                hidden_size, [self.conv_dim, self.value_dim], has_bias=False
            )
            self.in_proj_ba = LinearColParallelMerged(
                hidden_size, [num_v_heads, num_v_heads], has_bias=False
            )
        else:
            # Fused input projection (one GEMM instead of four): qkv | z | b | a.
            self.in_proj = LinearColParallelMerged(hidden_size, self._in_proj_split, has_bias=False)
        self.conv1d = _DepthwiseConv1d(self.conv_dim, conv_kernel_size)
        # Recurrence-gating params kept in fp32 (exp/softplus is precision-sensitive,
        # and the fla kernel reads them as fp32) -- matches HF/sglang, and avoids a
        # per-call .float() upcast in the decode wrapper. The weight loader exempts
        # *.A_log / *.dt_bias from the model-dtype downcast.
        self.dt_bias = torch.empty(num_v_heads, dtype=torch.float32)
        self.A_log = torch.empty(num_v_heads, dtype=torch.float32)
        self.norm = _GatedRMSNorm(head_v_dim, eps=rms_norm_eps)
        # out_proj follows the checkpoint quant: block-fp8 / per-tensor-fp8 / compressed-tensors
        # NVFP4 (W4A16) / bf16. in_proj_* stay bf16 in every mode (above), so a compressed-tensors
        # NVFP4 checkpoint (attn_quant=="nvfp4") only makes out_proj native FP4.
        self.out_proj = make_replicated_quant(
            expert_quant, attn_quant, self.value_dim, hidden_size, has_bias=False
        )

    def _gate_params(self, a: torch.Tensor, b: torch.Tensor):
        beta = b.sigmoid()
        g = -self.A_log.exp() * F.softplus(a.float() + self.dt_bias)
        return g, beta

    def _conv_weight(self) -> torch.Tensor:
        return self.conv1d.weight.squeeze(1)  # [conv_dim, kernel] for the fused kernel

    def _conv_prefill(self, conv_in, pool, cu_seqlens, cache_indices, has_initial_state) -> torch.Tensor:
        """Varlen causal conv (fused sgl_kernel) with silu; reads/updates each request's
        conv state in place by ``cache_indices`` slot. ``conv_in`` [total, conv_dim].
        ``cu_seqlens`` / ``cache_indices`` / ``has_initial_state`` come from FLAMetadata."""
        li = pool.local_index(self.layer_id)
        # A VIEW, not a copy: the conv kernel reads either layout, and materialising this one
        # cost 14.4 ms per 2129-token prefill -- three times the convolution itself.
        x = conv_in.transpose(0, 1)  # [conv_dim, total]
        out = causal_conv1d_varlen(x, self._conv_weight(), pool.conv_states[li],
                                   cu_seqlens, cache_indices, has_initial_state)
        return out.transpose(0, 1)  # [total, conv_dim]

    def _conv_decode(self, conv_in: torch.Tensor, table_idx: torch.Tensor, pool) -> torch.Tensor:
        """Single-token causal conv update (fused sgl_kernel) by ``table_idx`` slot;
        updates conv state in place, no host loop -> CUDA-graph capturable.
        ``conv_in`` [B, conv_dim] -> silu(conv) [B, conv_dim]."""
        li = pool.local_index(self.layer_id)
        return causal_conv1d_decode(conv_in, pool.conv_states[li], self._conv_weight(), table_idx)

    def _write_track_snapshot(self, pool, li: int, conv_in: torch.Tensor,
                              h: torch.Tensor, fla) -> None:
        """Snapshot this layer's recurrent + conv state at the chunk-aligned track boundary
        into a donatable pool slot, on the forward stream (hybrid-radix extra_buffer path).
        SSM: ``recurrent_states[li, dst] = h[0, h_row]`` -- a DIRECT copy (h is [V,K], the
        state pool is [K,V]; they coincide because GDN requires head_k_dim == head_v_dim).
        Conv: the last (kernel-1) raw conv-input timesteps ending at the boundary."""
        rec = pool.recurrent_states[li]
        rec.index_copy_(0, fla.track_dst, h[0, fla.track_h_row].to(rec.dtype))
        cv = pool.conv_states[li]
        # conv_in [total, conv_dim]; gather the (kernel-1) window per tracked req.
        conv_win = conv_in[fla.track_conv_src].transpose(-1, -2).contiguous()  # [nt, conv_dim, K-1]
        cv.index_copy_(0, fla.track_dst, conv_win.to(cv.dtype))

    def rescan_prefix(self, *, live_slot: int, scratch_slot: int, committed: int, stash) -> None:
        """Re-run this layer's recurrence over the ``committed`` positions of a draft block.

        ``committed`` counts rows of the verification window, which is the token pending from
        the step before followed by the candidates -- so it is ``accepted + 1``, not the number
        of accepted candidates. GDNRollback.rewind() owns that conversion.

        The caller has already restored ``live_slot`` from ``scratch_slot``, so the state here is
        the one from before the block; this walks it forward over the prefix the target kept.
        Only the scan runs -- q/k/v/g/beta come from the verification forward, so no projection
        and no weight read is repeated.
        """
        from freetoken.engine.gdn_rollback import LayerStash

        assert isinstance(stash, LayerStash)
        ctx = get_global_ctx()
        pool = ctx.linear_state_pool
        li = pool.local_index(self.layer_id)
        device = stash.q.device
        indices = torch.tensor([live_slot], dtype=torch.int32, device=device)
        cu_seqlens = torch.tensor([0, committed], dtype=torch.int64, device=device)

        gdn_prefill_chunk_fla(
            stash.q[:, :committed], stash.k[:, :committed], stash.v[:, :committed],
            stash.g[:, :committed], stash.beta[:, :committed],
            state_source=pool.recurrent_states[li], indices=indices,
            cu_seqlens=cu_seqlens, scale=self.head_k_dim ** -0.5,
        )

        # The convolution state is the last (kernel-1) RAW inputs, so it is rebuilt rather than
        # rescanned: the window ending at the accepted position is the restored pre-block window
        # followed by the accepted rows, keeping the tail.
        width = self.conv_kernel_size - 1
        cv = pool.conv_states[li]
        prefix = stash.conv_in[:committed].transpose(0, 1).to(cv.dtype)  # [conv_dim, committed]
        window = torch.cat([cv[scratch_slot], prefix], dim=-1)[:, -width:]
        cv[live_slot] = window

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        pool = ctx.linear_state_pool
        total = hidden_states.shape[0]
        dtype = hidden_states.dtype

        # Per-forward GDN metadata (cu_seqlens / cache_indices / continuation flags),
        # built once and shared by all GDN layers. The scheduler/graph set it; build it
        # lazily here (cached on the batch) for direct-op callers (tests).
        fla = batch.fla_metadata
        if fla is None:
            from freetoken.attention.linear import build_fla_metadata

            fla = build_fla_metadata(batch, hidden_states.device)
            batch.fla_metadata = fla

        _t_proj0 = None
        # The verification forward is in the decode PHASE but takes the extend path (several
        # rows for one sequence), so the phase flag is the wrong test here; row count is right.
        if _FORWARD_TIMING and total > fla.cu_seqlens.numel() - 1 and not torch.cuda.is_current_stream_capturing():
            _t_proj0 = torch.cuda.Event(enable_timing=True); _t_proj0.record()
        if self._fp8:
            qkvz = self.in_proj_qkvz.forward(hidden_states)
            conv_in, z = torch.split(qkvz, [self.conv_dim, self.value_dim], dim=-1)
            ba = self.in_proj_ba.forward(hidden_states)
            b, a = torch.split(ba, [self.num_v_heads, self.num_v_heads], dim=-1)
        else:
            proj = self.in_proj.forward(hidden_states)
            conv_in, z, b, a = torch.split(proj, self._in_proj_split, dim=-1)
        z = z.reshape(total, self.num_v_heads, self.head_v_dim)
        li = pool.local_index(self.layer_id)
        if _t_proj0 is not None:
            from .model import _PENDING_EVENTS
            _t_proj1 = torch.cuda.Event(enable_timing=True); _t_proj1.record()
            _PENDING_EVENTS.append(("gdn.in_proj", _t_proj0, _t_proj1))

        # The decode kernel takes one token per sequence: its `q` is [1, num_seqs, ...] and it
        # indexes state by sequence. A verification forward carries a whole drafted block for
        # one sequence, so it must go down the chunk path even though the phase says decode.
        one_token_each = total == fla.cu_seqlens.numel() - 1
        if batch.is_decode and one_token_each:
            # Fused fla decode kernel: gating + in-kernel l2norm + recurrent update +
            # per-request state read/write-by-index, all in one kernel (no gather/scatter,
            # no clone, no external l2norm). q/k stay at num_k_heads (kernel handles GQA).
            mixed = self._conv_decode(conv_in, fla.cache_indices, pool)  # [B, conv_dim]
            B = mixed.shape[0]
            qf, kf, vf = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
            q = qf.reshape(1, B, self.num_k_heads, self.head_k_dim).to(dtype)
            k = kf.reshape(1, B, self.num_k_heads, self.head_k_dim).to(dtype)
            v = vf.reshape(1, B, self.num_v_heads, self.head_v_dim).to(dtype)
            core_out = gdn_decode_fla(
                q, k, v, a, b, A_log=self.A_log, dt_bias=self.dt_bias,
                state_source=pool.recurrent_states[li], indices=fla.cache_indices,
                cu_seqlens=fla.cu_seqlens, scale=self.head_k_dim ** -0.5,
            )
        else:
            timing = _FORWARD_TIMING and not torch.cuda.is_current_stream_capturing()
            if timing:
                from .model import _PENDING_EVENTS
                ev = lambda: (lambda e: (e.record(), e)[1])(torch.cuda.Event(enable_timing=True))
                t_conv0 = ev()
            mixed = self._conv_prefill(
                conv_in, pool, fla.cu_seqlens, fla.cache_indices, fla.has_initial_state)
            if timing:
                t_conv1 = ev()
            # fla chunk handles GQA in-kernel: q/k stay at num_k_heads, v at num_v_heads.
            qf, kf, vf = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
            q = qf.reshape(1, total, self.num_k_heads, self.head_k_dim).to(dtype)
            k = kf.reshape(1, total, self.num_k_heads, self.head_k_dim).to(dtype)
            v = vf.reshape(1, total, self.num_v_heads, self.head_v_dim).to(dtype)
            # The 8-token verification block stays on the chunk path. Routing it through the
            # fused recurrent decode kernel (sglang's target_verify) was tried and measured:
            # 17.9 ms against the chunk path's 2.3 ms, and 0 of 4 outputs identical. That
            # kernel is one warp per (sequence, head) walking T tokens sequentially -- built for
            # many sequences of one token, not one sequence of eight. Do not re-test without
            # a different kernel.
            g, beta = self._gate_params(a, b)
            g = g.reshape(1, total, self.num_v_heads)
            beta = beta.float().reshape(1, total, self.num_v_heads)
            # The chunk kernel reads + writes back initial_state[cache_indices] in place;
            # fresh sequences (cached_len==0) must start from a zeroed slot.
            if fla.fresh_state_indices is not None:
                pool.recurrent_states[li].index_fill_(0, fla.fresh_state_indices, 0.0)
            track = fla.track_dst is not None
            if timing:
                t_chunk0 = ev()
            result = gdn_prefill_chunk_fla(
                q, k, v, g, beta,
                state_source=pool.recurrent_states[li], indices=fla.cache_indices,
                cu_seqlens=fla.cu_seqlens, scale=self.head_k_dim ** -0.5,
                return_h=track,
                chunk_indices=fla.chunk_indices, chunk_indices_o=fla.chunk_indices_o,
                chunk_offsets=fla.chunk_offsets,
            )
            if timing:
                t_chunk1 = ev()
                _PENDING_EVENTS.append(("gdn.conv", t_conv0, t_conv1))
                _PENDING_EVENTS.append(("gdn.gate+reshape", t_conv1, t_chunk0))
                _PENDING_EVENTS.append(("gdn.chunk", t_chunk0, t_chunk1))
            rollback = ctx.gdn_rollback
            if rollback is not None and rollback.recording:
                # Kept for a possible rewind: a rejected block needs these rows to walk the
                # state forward again, and recomputing them would mean re-reading the weights.
                rollback.stash(
                    self.layer_id, self.rescan_prefix, q, k, v, g, beta, conv_in,
                    local_index=li, head_k_dim=self.head_k_dim,
                    fused=rescan_prefix_fused,
                )
            if track:
                core_out, h = result
                self._write_track_snapshot(pool, li, conv_in, h, fla)
            else:
                core_out = result

        core_out = core_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        if _t_proj0 is not None:
            _t_out0 = torch.cuda.Event(enable_timing=True); _t_out0.record()
        out = self.norm.forward(core_out, z)
        if self.out_proj_v_tiled:
            out = v_grouped_to_tiled(out, total, self.num_k_heads, self.num_v_heads, self.head_v_dim)
        out = out.reshape(total, -1)
        result_out = self.out_proj.forward(out)
        if _t_proj0 is not None:
            _t_out1 = torch.cuda.Event(enable_timing=True); _t_out1.record()
            _PENDING_EVENTS.append(("gdn.norm+out_proj", _t_out0, _t_out1))
        return result_out


__all__ = ["Qwen3_5GatedDeltaNet"]
