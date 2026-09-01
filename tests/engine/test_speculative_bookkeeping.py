import pytest
import torch

from freetoken.core import Req
from freetoken.engine.speculative import (
    commit_verified,
    open_draft_block,
    unsupported_reason,
)


def _decoding_req(prompt_len: int = 10, output_len: int = 8) -> Req:
    """A request mid-decode: one sampled token appended, its KV not yet computed."""
    ids = torch.arange(prompt_len + 1, dtype=torch.int32)
    return Req(
        input_ids=ids,
        table_idx=0,
        cached_len=prompt_len,
        output_len=output_len,
        uid=1,
        sampling_params=None,
        cache_handle=None,
    )


def test_open_draft_block_extends_only_the_device_window():
    req = _decoding_req()
    before = req.input_ids.tolist()
    block = open_draft_block(req, 4)

    assert block.size == 4
    assert block.first_position == 11
    assert list(block.positions) == [11, 12, 13, 14]
    assert req.device_len == 15
    assert req.extend_len == 5  # the pending token plus the four candidates
    # the candidates live in the token pool, not in the record of what was generated
    assert req.input_ids.tolist() == before


def test_open_draft_block_leaves_room_for_the_bonus_token():
    req = _decoding_req(output_len=3)  # max_device_len = 14, device_len = 11
    block = open_draft_block(req, 5)

    assert block.size == 2  # 3 free slots, one reserved for the target's own token
    assert req.device_len == 13
    assert req.max_device_len - req.device_len == 1


def test_open_draft_block_declines_when_the_budget_is_spent():
    req = _decoding_req(output_len=1)  # only the pending token fits
    assert open_draft_block(req, 4).size == 0


def test_commit_verified_keeps_the_accepted_prefix_and_frees_the_rest():
    req = _decoding_req()
    block = open_draft_block(req, 4)

    freed = commit_verified(req, block, accepted=2)

    assert list(freed) == [13, 14]  # the two rejected candidates
    assert req.cached_len == 13     # prompt, the pending token, two accepted candidates
    assert req.device_len == 14     # one position pending for the bonus token
    assert req.extend_len == 1      # the next forward is a plain one-token step


def test_commit_verified_with_everything_accepted_frees_nothing():
    req = _decoding_req()
    block = open_draft_block(req, 2)

    assert list(commit_verified(req, block, accepted=2)) == []
    assert req.cached_len == 13
    assert req.device_len == 14


def test_commit_verified_with_nothing_accepted_matches_a_plain_decode_step():
    req = _decoding_req()
    block = open_draft_block(req, 3)

    assert list(commit_verified(req, block, accepted=0)) == [11, 12, 13]
    assert req.cached_len == 11
    assert req.device_len == 12


def test_commit_verified_rejects_an_impossible_acceptance():
    req = _decoding_req()
    block = open_draft_block(req, 2)
    with pytest.raises(ValueError, match="outside the drafted block"):
        commit_verified(req, block, accepted=3)


def test_unsupported_reason_names_every_configuration_it_cannot_roll_back():
    ok = dict(page_size=1, is_swa=False, is_hybrid=False, tp_size=1, overlap_scheduling=False)
    assert unsupported_reason(**ok) is None
    assert "page_size" in unsupported_reason(**{**ok, "page_size": 16})
    assert "sliding-window" in unsupported_reason(**{**ok, "is_swa": True})
    assert "GDN" in unsupported_reason(**{**ok, "is_hybrid": True})
    assert "tensor parallel" in unsupported_reason(**{**ok, "tp_size": 2})
    assert "overlap" in unsupported_reason(**{**ok, "overlap_scheduling": True})
