"""What a request at temperature 0 is entitled to: one answer, the same one every time.

The engine used to hand such a request to the stochastic sampler. Not through any decision
about temperature -- through top_p. ``sampling_defaults="model"`` fills an unspecified top_p
from the checkpoint, this model's GGUF carries ``general.sampling.top_p = 0.95``, and
``is_greedy`` insisted on ``top_p == 1.0``. So "temperature 0" became temperature 1e-6 with
nucleus sampling: argmax in all but name, decided by an RNG whose state depended on how many
draws the path had taken.

It showed only at an exact tie between the top two logits -- four positions in 298 on one
300-token run -- and there it forked the rest of the answer. That is where the block size came
to change the generated text, and where greedy speculative decoding came to disagree with
greedy decoding.
"""

from __future__ import annotations

from freetoken.core import SamplingParams


def test_temperature_zero_is_greedy_whatever_the_checkpoint_recommends():
    """0.95 is this model's recommended top_p, and it must not make the request stochastic.

    Nucleus sampling keeps the smallest set of tokens whose mass reaches p, and that set
    contains the argmax for any p > 0. At temperature 0 the distribution IS the argmax, so no
    filter can take it away: the request has exactly one answer.
    """
    assert SamplingParams(temperature=0.0, top_p=0.95, top_k=20).is_greedy
    assert SamplingParams(temperature=0.0, top_p=0.5).is_greedy
    assert SamplingParams(temperature=0.0, top_p=1.0).is_greedy


def test_top_k_one_is_greedy_too_for_the_same_reason():
    """One survivor is the argmax, whatever the temperature does to the rest."""
    assert SamplingParams(temperature=0.7, top_k=1, top_p=0.95).is_greedy


def test_a_real_temperature_is_not_greedy():
    """The fix must not swallow requests that asked to be sampled."""
    assert not SamplingParams(temperature=0.7, top_p=1.0).is_greedy
    assert not SamplingParams(temperature=0.7, top_p=0.95, top_k=20).is_greedy
    assert not SamplingParams(temperature=1e-3).is_greedy, "small is not zero"
