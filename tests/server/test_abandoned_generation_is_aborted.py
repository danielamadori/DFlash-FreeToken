"""A caller that hangs up mid-generation must not leave the work running.

Measured on thething, 2026-09-26: the engine held 1725 admitted requests, a p95 latency of 18.7
hours, and a one-token request took over 45 seconds. The cause was here. stream_with_cancellation
polls request.is_disconnected() between chunks and aborts, and it was wired into the three
streaming branches only; the whole-answer branch right below each of them was a bare
`await generate_full`. A whole answer never yields, so nothing checked, and the admitted count
grew by 68 an hour for 26 hours without one hour of decline.

Both halves are covered here: the abort, which stops one abandoned generation, and the cap, which
stops the accumulation whatever the path -- including /v1/completions, whose non-streamed branch
consumes wait_for_ack directly and is NOT covered by the abort.

Seen red first: with generate_full_watching reverted to generate_full the abort case fails with
abort_user never called, and with max_waiting_req forced to 0 the cap case raises nothing.
"""
from __future__ import annotations

import asyncio
import os
import sys
from types import SimpleNamespace

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PY = os.path.join(_ROOT, "python")
if _PY not in sys.path:
    sys.path.insert(0, _PY)

import inspect  # noqa: E402

from freetoken.server import anthropic_api, openai_api, responses_api  # noqa: E402
from freetoken.server.api_server import FrontendManager  # noqa: E402
from freetoken.server.generation import (  # noqa: E402
    GenSpec,
    QueueFullError,
    submit_generation,
)


class FakeRequest:
    """Starlette's Request as the cancellation poll sees it: one question, answered from a flag."""

    def __init__(self, disconnected: bool = False) -> None:
        self.disconnected = disconnected

    async def is_disconnected(self) -> bool:
        return self.disconnected


def _manager() -> FrontendManager:
    """A FrontendManager with nothing running: await_with_cancellation reads self.abort_user
    and nothing else, so __init__ and its scheduler are not needed."""
    manager = FrontendManager.__new__(FrontendManager)
    aborted: list[int] = []

    async def abort_user(uid: int) -> None:
        aborted.append(uid)

    manager.abort_user = abort_user  # type: ignore[method-assign]
    manager.aborted = aborted  # type: ignore[attr-defined]
    return manager


# --- the abort: one abandoned generation is stopped ------------------------

def test_a_caller_that_hangs_up_gets_its_generation_aborted() -> None:
    manager = _manager()

    async def run() -> None:
        async def long_generation() -> str:
            await asyncio.sleep(30)
            return "never read"

        with pytest.raises(asyncio.CancelledError):
            await manager.await_with_cancellation(
                long_generation(), FakeRequest(disconnected=True), uid=7)
        await asyncio.sleep(0.05)  # abort_user is dispatched as a task

    asyncio.run(run())
    assert manager.aborted == [7]


def test_a_caller_that_stays_gets_its_answer_and_no_abort() -> None:
    """The honest half: the poll must not disturb the ordinary case."""
    manager = _manager()

    async def run() -> str:
        async def short_generation() -> str:
            return "answer"

        return await manager.await_with_cancellation(
            short_generation(), FakeRequest(disconnected=False), uid=8)

    assert asyncio.run(run()) == "answer"
    assert manager.aborted == []


def test_a_generation_failure_reaches_the_caller_unchanged() -> None:
    """A real failure must not be turned into a cancellation: the adapter maps it to a 400."""
    manager = _manager()

    async def run() -> None:
        async def broken_generation() -> str:
            raise ValueError("chat template cannot render this conversation")

        with pytest.raises(ValueError, match="cannot render"):
            await manager.await_with_cancellation(
                broken_generation(), FakeRequest(disconnected=False), uid=9)

    asyncio.run(run())
    assert manager.aborted == []


# --- the cap: accumulation becomes impossible -----------------------------

def _state(active: int, cap: int) -> SimpleNamespace:
    return SimpleNamespace(config=SimpleNamespace(max_waiting_req=cap),
                           stats=SimpleNamespace(active=active))


def _spec() -> GenSpec:
    return GenSpec(messages=[{"role": "user", "content": "hello"}],
                   sampling_params=SimpleNamespace())


def test_past_the_cap_a_request_is_refused() -> None:
    with pytest.raises(QueueFullError, match="64"):
        asyncio.run(submit_generation(_spec(), _state(active=64, cap=64)))


def test_the_refusal_is_not_a_ValueError_because_it_is_not_a_400() -> None:
    """The request is well formed: reading it as a 400 would tell the caller to change it."""
    assert not issubclass(QueueFullError, ValueError)


def test_a_cap_of_zero_means_no_cap() -> None:
    """Whoever wants the old behaviour writes 0 rather than removing the field. The AttributeError
    is the fake state running out of attributes AFTER the cap let the request through."""
    with pytest.raises(AttributeError):
        asyncio.run(submit_generation(_spec(), _state(active=10_000, cap=0)))


# --- the wiring: every whole-answer branch goes through the wrapper -------

def test_every_non_streamed_adapter_goes_through_the_watching_wrapper() -> None:
    """The invariant the original defect broke, and the one a fourth adapter would break again.

    stream_with_cancellation was wired into three streaming branches and the three whole-answer
    branches beside them were bare. Checking the source is the only way to state "no adapter
    awaits a generation without watching its caller" for adapters that do not exist yet.
    """
    modules = (openai_api, anthropic_api, responses_api)
    scoperti = []
    for module in modules:
        source = inspect.getsource(module)
        # `generate_full(` still appears inside generate_full_watching's own name, so the bare
        # call is the one not preceded by `_watching`.
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("result = await generate_full(") or \
               stripped.startswith("return await generate_full("):
                scoperti.append(f"{module.__name__}: {stripped}")
    assert scoperti == [], "whole-answer branches that do not watch their caller: " + str(scoperti)


def test_the_wrapper_passes_through_when_there_is_no_request_to_watch() -> None:
    """A caller with no Request (an internal call) must still get its answer."""
    from freetoken.server.generation import generate_full_watching

    called = {}

    async def run() -> None:
        async def fake_generate_full(uid, spec, state, *, source):
            called["source"] = source
            return "answer"

        import freetoken.server.generation as generation
        real = generation.generate_full
        generation.generate_full = fake_generate_full
        try:
            assert await generate_full_watching(
                1, _spec(), object(), source="/v1/chat/completions", request=None) == "answer"
        finally:
            generation.generate_full = real

    asyncio.run(run())
    assert called["source"] == "/v1/chat/completions"
