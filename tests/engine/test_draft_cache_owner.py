"""The draft's KV cache belongs to one request.

It holds that request's context keys, built up block by block. Carried across requests it is
not empty for the next one: it is the previous request's keys, cropped to the new request's
position and used as its context, so every block after the first request drafts against the
wrong text. The target still verifies, so output stays correct -- what drops is acceptance,
silently. On Qwen3.8-27B this fork accepted 27.9% from the identical draft file that llama.cpp
accepts 35.4% from.

This exercises only the ownership bookkeeping; it needs no weights and no GPU.
"""

from __future__ import annotations

from types import SimpleNamespace

from freetoken.engine.draft_runner import DFlashRunner


def _runner_with_fake_cache():
    """A DFlashRunner shell: only the fields the ownership logic in draft() touches."""
    r = DFlashRunner.__new__(DFlashRunner)
    r._draft_cache = None
    r._cache_owner = None
    r.block_size = 4
    made: list[object] = []

    def make_cache(config):
        made.append(object())
        return made[-1]

    r._make_cache = make_cache
    r.draft_model = SimpleNamespace(config=None)
    return r, made


def _open_block(r: DFlashRunner, request_uid: int) -> None:
    """The lines of draft() that decide whether the cache is reused, lifted out so they can run
    without hidden states, a target, or a device."""
    if r._draft_cache is None or request_uid != r._cache_owner:
        r.reset_cache()
        r._cache_owner = request_uid


def test_first_block_creates_the_cache():
    r, made = _runner_with_fake_cache()
    _open_block(r, request_uid=7)
    assert len(made) == 1
    assert r._draft_cache is made[0]
    assert r._cache_owner == 7


def test_later_blocks_of_the_same_request_keep_the_cache():
    """That is the whole point of the cache: the context accumulates across blocks."""
    r, made = _runner_with_fake_cache()
    _open_block(r, request_uid=7)
    for _ in range(5):
        _open_block(r, request_uid=7)
    assert len(made) == 1, "one cache for the request, however many blocks it takes"


def test_a_new_request_gets_a_fresh_cache():
    """The bug: the second request drafted against the first request's keys."""
    r, made = _runner_with_fake_cache()
    _open_block(r, request_uid=7)
    first = r._draft_cache
    _open_block(r, request_uid=8)
    assert len(made) == 2
    assert r._draft_cache is not first, "request 8 must not see request 7's context keys"
    assert r._cache_owner == 8


def test_returning_to_an_earlier_uid_is_still_a_new_request():
    """uids are not recycled within a server's life, but the rule must not assume it."""
    r, made = _runner_with_fake_cache()
    _open_block(r, request_uid=7)
    _open_block(r, request_uid=8)
    _open_block(r, request_uid=7)
    assert len(made) == 3
