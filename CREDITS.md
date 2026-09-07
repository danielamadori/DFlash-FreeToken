# Credits

This repository is a fork of **FreeToken** that adds DFlash speculative decoding to the
engine. Neither of the two projects it stands on is its own work, and both remain under
their own licences.

## Upstream projects

### FreeToken — the serving engine
- Source: <https://github.com/FlashML-org/FreeToken>
- Authors: FlashML and the FreeToken contributors
- Licence: Apache License 2.0 (see [`LICENSE`](LICENSE))

Everything outside the files listed below is upstream work. The engine, its MoE
offloading, the CUDA/Triton kernels, the scheduler, the API server and the desktop app
are FreeToken's.

### DFlash — the speculative decoding method and draft models
- Source: <https://github.com/z-lab/dflash> (vendored here as the `dflash` submodule)
- Authors: Z Lab
- Licence: MIT, Copyright (c) 2026 Z Lab (see [`dflash/LICENSE`](dflash/LICENSE))

The draft model classes, the block-diffusion drafting loop, the rejection sampler this
fork's verification mirrors, and the `DFlash`/`DFlash2` checkpoints are Z Lab's. The
reference implementation in `dflash/dflash/model.py::dflash_generate` is what this fork's
verification pass was checked against, position by position.

Draft checkpoints used with this fork are published by Z Lab on Hugging Face
(<https://huggingface.co/z-lab>), under their own terms.

### GGUF support -- FlashML-org/FreeToken#131, by vcruz305

- Source: <https://github.com/FlashML-org/FreeToken/pull/131> (branch `feat/generic-gguf`)
- Author: vcruz305
- Licence: Apache License 2.0, as part of FreeToken

Merged here, not written here: 36 commits and roughly 5,900 lines that fill the Python
tables for all 21 ggml quant types (the CUDA side already dispatched 19; the tables
described 6, so K-quants were unreachable), add the qwen35 / qwen35moe / qwen3moe /
deepseek4 GGUF model specs, multi-shard reading, and CPU MoE kernels for Q4_K and Q6_K.
It also gives the five `switch (type)` blocks in `gguf_kernel.cu` a `default:` -- without
one an unsupported type returned uninitialised memory from `torch::empty` instead of
raising.

That PR is what makes this fork able to read the same GGUF llama.cpp serves, which is the
only way to compare the two engines on one model. The work below is separate from it.

## What this fork adds

Confined to the DFlash integration; the surrounding engine is untouched upstream code.

| Area | Files |
|---|---|
| Draft runner (loads a DFlash draft, drafts a block, rejection sampling) | `python/freetoken/engine/draft_runner.py` |
| Speculative block bookkeeping (which positions survive, which KV pages are freed) | `python/freetoken/engine/speculative.py` |
| Hidden-state capture the draft is conditioned on | `python/freetoken/models/blocks.py`, `models/muse_glimmer/model.py`, `models/qwen3/model.py` |
| Draft, verify and rollback inside the scheduler, where KV allocation lives | `python/freetoken/scheduler/scheduler.py`, `scheduler/cache.py` |
| All-position logits for a verification forward | `python/freetoken/core.py`, `layers/embedding.py`, `engine/engine.py` |
| Multi-query decode routed to the append attention wrapper | `python/freetoken/attention/fi.py` |
| Detokenisation of several tokens committed in one round | `python/freetoken/tokenizer/detokenize.py` |
| Logit-margin trace, to tell arithmetic from a defect | `python/freetoken/engine/spec_trace.py` |
| Kernel-package probes that import rather than locate | `python/freetoken/kernel/backend.py` |
| GGUF build: verified host compiler, per-device arch, nvcc version check | `python/freetoken/kernel/gguf.py` |
| Tests for the above | `tests/engine/test_dflash_speculative.py`, `tests/engine/test_speculative_bookkeeping.py`, `tests/tokenizer/test_detokenize.py` |

## Honest status

Measured on an RTX 4090 with Qwen3-8B and `z-lab/Qwen3-8B-DFlash-b16`, at equal settings
in one window: 57.7 t/s without the draft against 87.4 with it at 128 tokens, and 57.1
against 102.1 at 512 — 1.5x to 1.8x, with 1.27 of 14.5 drafted candidates accepted per
block.

Equivalence at temperature 0 is **not** demonstrated. Greedy speculative decoding should
reproduce greedy decoding token for token; after fixing a detokenisation defect that
duplicated words, two of six prompts match byte for byte and four diverge at one isolated
word while staying coherent. The likely cause is that bf16 kernels are not
batch-invariant, but nobody has looked at the logits at a divergence point, so that is a
hypothesis and not a result.

Greedy divergence has since been measured rather than guessed: where the two first
differ, the same two candidates are in play and the winner's logit comes out 0.25 lower
from the verification forward -- exactly one bfloat16 ULP at that magnitude (exponent 5,
7 mantissa bits). A near-tie becomes an exact tie and the tie goes by index order. The
verification and the KV rollback are sound.

Supported configuration is narrow and refuses to start outside it: page size 1, no
sliding-window KV, no hybrid GDN state, tensor parallel 1, one request in flight,
non-overlap scheduling, the `fi` attention backend, and hidden-state capture implemented
for the `qwen3` and `muse_glimmer` families only.
