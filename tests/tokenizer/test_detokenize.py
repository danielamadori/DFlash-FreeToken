"""DetokenizeManager must not duplicate text when one uid gets several messages in a
single call.

Every caller before speculative decoding sent at most one message per uid per call: one
token generated per request per scheduler round. The two-pass implementation this
replaced built every message's read/surr id slices against the round's STARTING
offsets, so a second message for the same uid re-included the first message's text. A
speculative round can legitimately commit several tokens for one request in one round
(the accepted candidates plus the bonus token), which is exactly what exposed it: a
live run produced "user user is user is asking" instead of "user is asking".
"""
from __future__ import annotations

import pytest

from freetoken.message import DetokenizeMsg
from freetoken.tokenizer.detokenize import DetokenizeManager


class _WordTokenizer:
    """One token id per whole word, joined by spaces -- enough to show duplication."""

    eos_token_id = -1
    VOCAB = {1: "user", 2: "is", 3: "asking", 4: "a", 5: "question"}

    def batch_decode(self, sequences: list[list[int]]) -> list[str]:
        return [" ".join(self.VOCAB[i] for i in seq) for seq in sequences]


def _msg(uid: int, token: int, *, finished: bool = False) -> DetokenizeMsg:
    return DetokenizeMsg(
        uid=uid, next_token=token, finished=finished, finish_reason=None,
        matched_stop=None, stop_strs=None,
    )


def _manager() -> DetokenizeManager:
    return DetokenizeManager(_WordTokenizer(), eos_token_ids=frozenset({-1}))


def test_one_message_per_uid_per_call_is_unaffected() -> None:
    """The pre-existing shape: every prior caller matches this, must stay correct."""
    mgr = _manager()
    out = []
    for token in (1, 2, 3):
        out += mgr.detokenize([_msg(1, token)])
    assert "".join(out) == "user is asking"


def test_several_messages_for_one_uid_in_one_call_do_not_duplicate() -> None:
    """The speculative-decoding shape that exposed the bug: N tokens, one round."""
    mgr = _manager()
    out = mgr.detokenize([_msg(1, 1), _msg(1, 2), _msg(1, 3)])
    assert "".join(out) == "user is asking"


def test_result_matches_calling_one_message_at_a_time() -> None:
    """The batched call must be indistinguishable from unrolling it into single calls."""
    batched = _manager()
    batched_out = "".join(batched.detokenize([_msg(7, t) for t in (1, 2, 3, 4, 5)]))

    sequential = _manager()
    sequential_out = "".join(
        "".join(sequential.detokenize([_msg(7, t)])) for t in (1, 2, 3, 4, 5)
    )
    assert batched_out == sequential_out == "user is asking a question"


def test_concurrent_uids_stay_independent_and_correctly_ordered() -> None:
    """A multi-request serve keeps batching across uids; only same-uid order matters."""
    mgr = _manager()
    out = mgr.detokenize([_msg(1, 1), _msg(2, 3), _msg(1, 2), _msg(2, 4)])
    assert out[0] + out[2] == "user is"
    assert out[1] + out[3] == "asking a"


def test_a_growing_block_size_does_not_grow_the_duplication() -> None:
    """Characterises the failure mode directly: output length must not depend on how
    many tokens land in one round, only on how many distinct tokens were generated."""
    two_at_once = _manager()
    two_out = "".join(two_at_once.detokenize([_msg(1, 1), _msg(1, 2)]))

    five_at_once = _manager()
    five_out = "".join(
        five_at_once.detokenize([_msg(1, 1), _msg(1, 2), _msg(1, 3), _msg(1, 4), _msg(1, 5)])
    )
    assert two_out == "user is"
    assert five_out == "user is asking a question"
    # the bug's signature: with 5 messages in one call the old code emitted the first
    # word 5 times over (once per message's overlapping slice)
    assert five_out.count("user") == 1


def test_finished_request_is_dropped_from_decode_map() -> None:
    mgr = _manager()
    mgr.detokenize([_msg(1, 1), _msg(1, 2, finished=True)])
    assert 1 not in mgr.decode_map


@pytest.mark.parametrize("token_count", [1, 2, 3, 5])
def test_a_finished_multi_token_round_still_flushes_once(token_count: int) -> None:
    """The last message of a multi-token speculative commit can itself be `finished`."""
    mgr = _manager()
    tokens = list(range(1, token_count + 1))
    msgs = [_msg(1, t) for t in tokens[:-1]] + [_msg(1, tokens[-1], finished=True)]
    out = "".join(mgr.detokenize(msgs))
    assert out == " ".join(_WordTokenizer.VOCAB[t] for t in tokens)
    assert 1 not in mgr.decode_map
