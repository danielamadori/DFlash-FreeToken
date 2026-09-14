from __future__ import annotations

import os
from functools import partial
from typing import Callable, Generic, TypeVar


class BaseEnv:
    def _init(self, name: str) -> None:
        raise NotImplementedError


T = TypeVar("T")


class EnvVar(BaseEnv, Generic[T]):
    def __init__(self, default_value: T, fn: Callable[[str], T]):
        self.value = default_value
        self.fn = fn
        super().__init__()

    def _init(self, name: str) -> None:
        env_value = os.getenv(name)
        if env_value is not None:
            try:
                self.value = self.fn(env_value)
            except Exception:
                pass

    def __bool__(self):
        return self.value

    def __str__(self):
        return str(self.value)


_TO_BOOL = lambda x: x.lower() in ("1", "true", "yes")


def _PARSE_MEM_BYTES(mem: str) -> int:
    mem = mem.strip().upper()
    if not mem[-1].isalpha():
        return int(mem)
    if mem.endswith("B"):
        mem = mem[:-1]
    UNIT_MAP = {"K": 1024, "M": 1024**2, "G": 1024**3}
    return int(float(mem[:-1]) * UNIT_MAP[mem[-1]])


ENV_PREFIX = "FREETOKEN_"
EnvInt = partial(EnvVar[int], fn=int)
EnvFloat = partial(EnvVar[float], fn=float)
EnvBool = partial(EnvVar[bool], fn=_TO_BOOL)
EnvOption = partial(EnvVar[bool | None], fn=_TO_BOOL, default_value=None)
EnvMem = partial(EnvVar[int], fn=_PARSE_MEM_BYTES)
EnvStr = partial(EnvVar[str], fn=str)


class EnvClassSingleton:
    _instance: EnvClassSingleton | None = None

    # shell
    SHELL_MAX_TOKENS = EnvInt(2048)
    # None = unset -> resolved from the model's generation_config.json sampling defaults
    # (sglang's sampling_defaults='model'); set the env var to override.
    SHELL_TOP_K = EnvInt(None)
    SHELL_TOP_P = EnvFloat(None)
    SHELL_TEMPERATURE = EnvFloat(None)

    # backend runtime
    FLASHINFER_USE_TENSOR_CORES = EnvOption()
    DISABLE_OVERLAP_SCHEDULING = EnvBool(False)
    # Let speculative decoding run on hybrid linear-attention models by rewinding the GDN
    # recurrent state to a block's accepted prefix. A wrong rewind reads as fluent text, not
    # as an error, so this asked to be shown equal to a non-speculative run before being
    # trusted. It now is, for the eager path: tests/engine/test_gdn_rollback_numerics.py runs
    # the same tokens twice -- once straight, once speculate-then-rewind -- and the recurrent
    # state comes back bit-identical, with the convolution window inside one bf16 ulp (the
    # floor is the input projection tiling over W rows instead of the committed rows, not the
    # rewind). Held across eight consecutive partly-rejected blocks, so drift would show.
    # The graph path is covered now as well: the same file captures the forward, replays it on
    # a block's rows, adopts the capture-time stash and rewinds, and lands on the eager
    # control. That case discriminates -- replaying WITHOUT copying the block's rows into the
    # baked address fails it by a factor of 968 -- so the adopted entries are shown to carry
    # the replayed rows and not the capture's.
    SPEC_GDN_ROLLBACK = EnvBool(True)
    # Draft greedily with a plain per-position argmax even when the model carries a DFlash2
    # selector. Not a speed knob: it is the A/B that says whether the selector is earning its
    # place. Its job is to make a block whose tokens follow one another, so switching it off
    # must cost acceptance -- and if it costs nothing, it is wired in but doing nothing, which
    # the totals alone cannot reveal.
    SPEC_NO_SELECTOR = EnvBool(False)
    # Replay the K+1-row speculative verify forward as a CUDA graph instead of launching its
    # kernels one by one. This asked to be shown bitwise equal to the eager verify before
    # being trusted, in both the state it leaves and the logits it returns, because a stale
    # captured buffer reads as fluent text and never as an error. Both are now shown:
    #
    #   state   tests/engine/test_gdn_rollback_numerics.py, the adopt case. Replaying without
    #           copying the block's rows into the baked address fails it by a factor of 968,
    #           so the case can tell a stale stash from a fresh one.
    #   logits  SPEC_VERIFY_GRAPH_SHADOW below, 2700 blocks over two runs of ten prompts on
    #           Qwen3.8-27B: zero rows with a different argmax, and a worst logit gap of
    #           exactly 0. Adding 1.0 to one logit of one path moved that gap to 1.031, so
    #           the comparison measures something.
    #
    # What it is worth: the vendored fla chunk kernel costs 157 us per layer eager and 23 us
    # replayed, on the same inputs, unchanged from 1 row to 128 -- 134 us per layer of launch
    # and dispatch overhead, 6.4 ms across 48 GDN layers. End to end the verify goes from 32.7
    # to 22.2 ms and the block from 38.8 to 28.3: 73 to 100 tokens a second, at identical
    # acceptance.
    SPEC_VERIFY_GRAPH = EnvBool(True)
    # Replay the verify block, then run it again eagerly on the same rows and report where the
    # two disagree. The only way to check the replayed logits: this engine does not reproduce
    # its own greedy output run to run, so comparing generated text between the two paths says
    # nothing -- it says nothing between eager and eager either. Costs a whole extra verify per
    # block, and returns the eager logits, so it is a diagnostic and never a serving mode.
    SPEC_VERIFY_GRAPH_SHADOW = EnvBool(False)
    # Replay the 8-row DFlash2 draft forward as one CUDA graph per context-row count. This
    # asked for the replayed draft tokens to be shown equal to the eager static-cache draft
    # before being trusted, since a graph baking a stale ring or hidden address reads as lost
    # acceptance and never as an error. Shown by SPEC_DRAFT_GRAPH_SHADOW below: 1400 blocks at
    # temperature 0, zero mismatching tokens.
    #
    # The graph stops at the logits. It used to capture the whole block, sampling included,
    # with temperature 0 baked in -- so a request above 0 could not replay it at all without
    # the rejection sampler dividing by one-hot probabilities, and every request production
    # serves is above 0. Splitting the forward from the choosing is what made it apply where
    # it pays: the draft goes from 5.4 to 4.7 ms of a 28.7 ms block at temperature 1.
    SPEC_DRAFT_GRAPH = EnvBool(True)
    # Also run the eager draft on every replayed block, log a token mismatch, return the
    # eager pair. One host sync per block: the only way to see what rejection hides.
    #
    # It compares TOKENS, so it can only prove anything at temperature 0. Above it the two
    # paths sample independently from the same distribution and disagree by construction --
    # measured at 1329 mismatching blocks of 1419, which looks like a broken graph and is
    # nothing but two honest draws.
    SPEC_DRAFT_GRAPH_SHADOW = EnvBool(False)

    # Name each step of the speculative block for a profiler (nsys -t cuda,nvtx). Off by
    # default because the ranges are only useful under a profiler; they never synchronise, so
    # unlike the phase timers they do not change what they measure.
    SPEC_NVTX = EnvBool(False)

    # Decompose the time to first token at its boundaries (sent, tokenization, the engine's
    # first acks, the first formatted event) and log one line per request. Off by default.
    TTFT_MARKS = EnvBool(False)
    PYNCCL_MAX_BUFFER_SIZE = EnvMem(1024**3)
    # GatedDeltaNet recurrent (SSM) state dtype: float32 (default) | bfloat16 | float16.
    # fp32 matches the Qwen3.x configs (mamba_ssm_dtype); fp16/bf16 halves the GDN state
    # pool at some precision cost on the long recurrence (mirrors SGLang's mamba_ssm_dtype).
    MAMBA_SSM_DTYPE = EnvStr("float32")
    # Paged KV cache dtype: auto (default, = the model dtype) | bfloat16 | float8_e4m3fn |
    # float8_e5m2. The KV cache is the largest single allocation at long context and it does
    # NOT have to match the model: 16 full-attention layers x 2 slabs x 4 kv heads x 256 dims
    # is 64 KiB per token on this 27B, so 2 slots of 65536 tokens cost 8 GiB in bf16 and 4 in
    # fp8 -- on a 24 GB card that is the difference between fitting production's context and
    # not (llama.cpp serves the same shape at q4_0, 2.25 GiB).
    #
    # USE e5m2. Measured 2026-09-10 against a bf16 control in the same window, needle probe,
    # nine recalls in haystacks of 1153-4353 tokens: bfloat16 18234 tokens 9/9, float8_e5m2
    # 35626 tokens 9/9 (1.99x the context per GiB, no loss the probe can see), float8_e4m3fn
    # 8/9 -- missing the SAME recall in three separate runs. Not the expected order: e4m3 keeps
    # one more mantissa bit, but it saturates at 448, and attention leans on the outliers that
    # clips. Default stays auto because this is a deliberate trade, not a free win.
    KV_CACHE_DTYPE = EnvStr("auto")

    def __new__(cls):
        # single instance
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            attr_value = getattr(self, attr_name)
            assert isinstance(attr_value, BaseEnv)
            attr_value._init(f"{ENV_PREFIX}{attr_name}")


ENV = EnvClassSingleton()
