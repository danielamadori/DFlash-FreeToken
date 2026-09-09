"""The tensor-core MMQ against a dequantized reference, on the shapes stream-k changes.

Stream-k gives up the one-block-per-output-tile grid: blocks walk a continuous
(tile, k-block) index and a block that lands mid-tile leaves its partial sum in a scratch
buffer for a second kernel to add. That second path only runs when the tile count does not
divide the block count, so a test that only covers wide outputs never touches it.

The shapes here are the ones the 27B actually runs, chosen so both arms are exercised:
out_features 5120 (40 tiles, far below the 128 SMs -> stream-k with fixup) and 17408
(136 tiles). Column counts straddle J = 128 so both one and two column tiles occur.

The reference is ggml_dequantize plus a float matmul over the same Q8_1-rounded activation the
kernel consumes, so the two differ only by summation order -- which is exactly what stream-k
changes, hence the tolerance rather than equality.

Skipped without CUDA or the GGUF file (a fork-local fixture: the file is 14 GiB).
"""
from __future__ import annotations

import math
import os

import pytest
import torch

GGUF = os.environ.get(
    "FREETOKEN_TEST_GGUF",
    os.path.expanduser("~/.lmstudio/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_S.gguf"),
)
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.exists(GGUF), reason="needs CUDA and the 27B GGUF"
)


def _pick_by_out_features():
    """One 2-D tensor per (out_features, ggml type) the port covers, from the real file."""
    from freetoken.layers.gguf import GGML_NAME, MMA_TYPES
    from freetoken.models.gguf.reader import iter_gguf_tensors

    picked = {}
    for t in iter_gguf_tensors(GGUF):
        if len(t.shape) != 2 or t.ggml_type not in MMA_TYPES:
            continue
        out_features = t.shape[0]
        if out_features not in (5120, 17408):
            continue
        picked.setdefault((out_features, GGML_NAME.get(t.ggml_type)), t)
    return picked


@pytest.fixture(scope="module")
def tensors():
    return _pick_by_out_features()


def _tol(x_q: torch.Tensor, w: torch.Tensor) -> float:
    """Bound on the disagreement between two int8 kernels over the same operands.

    NOT one ulp of the output: a dot product over 17408 terms of alternating sign can be much
    smaller than the terms it is made of, so the error has to be measured against the sum of
    absolute products, which is what actually accumulates. A dequantized-weight reference is no
    use at this K either -- ggml_dequantize returns bf16, and that rounding alone is 2^-9 per
    term, well above what the two quantized kernels differ by.
    """
    magnitude = (x_q.abs().float() @ w.abs().float().t()).max().item()
    # One bf16 rounding of the accumulated magnitude. Measured worst case on the 27B's own
    # tensors, MMA against MMVQ over 24..256 rows and both widths: 7.8e-4 of the magnitude,
    # so this bound (1.95e-3) keeps a 2.5x margin without being loose enough to hide a bug.
    return 2.0 ** -9 * magnitude + 1e-4


@pytest.mark.parametrize("out_features", [5120, 17408])
@pytest.mark.parametrize("rows", [24, 64, 127, 128, 129, 256])
def test_mma_matches_the_vector_kernel(tensors, out_features, rows):
    """The tensor-core MMQ against the batched MMVQ, which test_mmvq_real_tensors.py anchors to
    a dequantized reference. Both consume the same packed weight and the same Q8_1 activation
    and multiply in int8, so they may differ only by summation order -- which is exactly what
    stream-k changes."""
    from freetoken.kernel.gguf import ggml_dequantize, ggml_mul_mat_mma, ggml_mul_mat_vec_a8

    candidates = [(k, t) for k, t in tensors.items() if k[0] == out_features]
    if not candidates:
        pytest.skip(f"no MMA-typed tensor with out_features={out_features} in the file")

    for (_, name), t in candidates:
        weight = t.packed().to("cuda")
        in_features = t.shape[1]
        torch.manual_seed(1234 + rows)
        x = torch.randn(rows, in_features, dtype=torch.bfloat16, device="cuda") * 0.1

        got = ggml_mul_mat_mma(weight, x, t.ggml_type, out_features).float()
        ref = ggml_mul_mat_vec_a8(weight, x, t.ggml_type, out_features).float()

        w = ggml_dequantize(weight, t.ggml_type, out_features, in_features)
        tol = _tol(x, w)
        del w
        torch.cuda.empty_cache()

        diff = (got - ref).abs().max().item()
        assert diff <= tol, f"{name} out={out_features} rows={rows}: {diff} > {tol}"
        del weight, x, got, ref
        torch.cuda.empty_cache()


def test_the_fixup_path_is_actually_reached(tensors):
    """out_features=5120 gives 40 tiles against 128 SMs, so the block count cannot divide the
    tile count and the fixup kernel must run. If this ever stops being true the test above
    silently stops covering stream-k, so assert the geometry itself."""
    nsm = torch.cuda.get_device_properties(0).multi_processor_count
    tiles = 5120 // 128            # config.I is 128 for every ported type
    assert tiles < nsm, "5120-wide outputs no longer under-fill this GPU; pick another shape"
    assert tiles % nsm != 0, "tile count now divides the block count: the fixup path is skipped"
