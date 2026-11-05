"""The bearer middleware: a Bedrock API key on every attempt, read off the event loop."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from conduit.core.errors import ProviderError, ProviderFailure
from conduit.providers.bedrock_mantle.auth import bearer_middleware

MARKER = "SECRET-MARKER-7f3a"


class FakeRequest:
    """Shaped like the SDK's APIRequest: read-only headers and a ``copy``."""

    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers

    def copy(self, *, headers: dict[str, str]) -> FakeRequest:
        return FakeRequest(headers)


class RotatingKeys:
    """Hands out a new key on every call and records the thread it ran on."""

    def __init__(self) -> None:
        self.calls = 0
        self.threads: list[int] = []

    def current(self) -> str:
        self.calls += 1
        self.threads.append(threading.get_ident())
        return f"bedrock-api-key-synthetic-{self.calls}"


class FailingKeys:
    def current(self) -> str:
        raise ValueError(f"credentials expired for {MARKER}")


class Sent:
    def __init__(self) -> None:
        self.requests: list[FakeRequest] = []

    async def __call__(self, request: FakeRequest) -> str:
        self.requests.append(request)
        return "response"


async def test_each_attempt_carries_the_current_key_as_a_bearer_token() -> None:
    keys = RotatingKeys()
    middleware = bearer_middleware(keys)
    sent = Sent()
    original = FakeRequest({"anthropic-version": "2023-06-01"})
    assert await middleware(original, sent) == "response"
    assert await middleware(original, sent) == "response"  # a retry: asked again
    assert [r.headers["Authorization"] for r in sent.requests] == [
        "Bearer bedrock-api-key-synthetic-1",
        "Bearer bedrock-api-key-synthetic-2",
    ]
    assert all(r.headers["anthropic-version"] == "2023-06-01" for r in sent.requests)
    assert original.headers == {"anthropic-version": "2023-06-01"}  # never mutated


async def test_the_key_is_read_off_the_event_loop_thread() -> None:
    # Refreshing a short-term key can refresh AWS credentials, which may block.
    keys = RotatingKeys()
    await bearer_middleware(keys)(FakeRequest({}), Sent())
    assert keys.threads
    assert threading.get_ident() not in keys.threads


@pytest.mark.parametrize("name", ["Authorization", "authorization", "X-Api-Key", "x-api-key"])
async def test_any_other_credential_header_is_replaced(name: str) -> None:
    sent = Sent()
    await bearer_middleware(RotatingKeys())(FakeRequest({name: MARKER, "other": "kept"}), sent)
    headers = sent.requests[0].headers
    assert MARKER not in headers.values()
    assert headers["Authorization"] == "Bearer bedrock-api-key-synthetic-1"
    assert headers["other"] == "kept"
    assert {k.lower() for k in headers} == {"authorization", "other"}


async def test_a_key_that_cannot_be_had_is_rejected_with_nothing_chained() -> None:
    sent = Sent()
    with pytest.raises(ProviderError) as caught:
        await bearer_middleware(FailingKeys())(FakeRequest({}), sent)
    error = caught.value
    assert error.failure is ProviderFailure.REJECTED
    assert error.__cause__ is None
    assert error.__context__ is None
    for text in (str(error), repr(error), *(repr(a) for a in error.args)):
        assert MARKER not in text
    assert sent.requests == []  # nothing was sent


async def test_cancellation_propagates_unchanged() -> None:
    async def cancelled(request: Any) -> Any:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await bearer_middleware(RotatingKeys())(FakeRequest({}), cancelled)


async def test_a_failure_after_sending_is_not_touched() -> None:
    class Boom(Exception):
        pass

    async def failing(request: Any) -> Any:
        raise Boom

    with pytest.raises(Boom):
        await bearer_middleware(RotatingKeys())(FakeRequest({}), failing)
