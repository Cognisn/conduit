"""Bearer keys set at send time, read off the event loop, and only through the hook."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from conduit.core.errors import ProviderError, ProviderFailure
from conduit.providers.bedrock_mantle.auth import bearer_client_class, set_bearer

MARKER = "SECRET-MARKER-7f3a"


class FakeRequest:
    """Shaped like an httpx request: mutable headers."""

    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers


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


async def test_each_call_sets_the_current_key_as_a_bearer_token() -> None:
    keys = RotatingKeys()
    first = FakeRequest({"anthropic-version": "2023-06-01"})
    second = FakeRequest({})
    await set_bearer(first, keys)
    await set_bearer(second, keys)  # a retry: asked again
    assert first.headers == {
        "anthropic-version": "2023-06-01",
        "Authorization": "Bearer bedrock-api-key-synthetic-1",
    }
    assert second.headers == {"Authorization": "Bearer bedrock-api-key-synthetic-2"}


async def test_the_key_is_read_off_the_event_loop_thread() -> None:
    # Refreshing a short-term key can refresh AWS credentials, which may block.
    keys = RotatingKeys()
    await set_bearer(FakeRequest({}), keys)
    assert keys.threads
    assert threading.get_ident() not in keys.threads


@pytest.mark.parametrize("name", ["Authorization", "authorization", "X-Api-Key", "x-api-key"])
async def test_any_other_credential_header_is_replaced(name: str) -> None:
    request = FakeRequest({name: MARKER, "other": "kept"})
    await set_bearer(request, RotatingKeys())
    assert MARKER not in request.headers.values()
    assert request.headers["Authorization"] == "Bearer bedrock-api-key-synthetic-1"
    assert request.headers["other"] == "kept"
    assert {k.lower() for k in request.headers} == {"authorization", "other"}


async def test_a_key_that_cannot_be_had_is_rejected_with_nothing_chained() -> None:
    request = FakeRequest({"other": "kept"})
    with pytest.raises(ProviderError) as caught:
        await set_bearer(request, FailingKeys())
    error = caught.value
    assert error.failure is ProviderFailure.REJECTED
    assert error.__cause__ is None
    assert error.__context__ is None
    for text in (str(error), repr(error), *(repr(a) for a in error.args)):
        assert MARKER not in text
    assert request.headers == {"other": "kept"}  # nothing set


async def test_cancellation_propagates_unchanged() -> None:
    release = threading.Event()

    class Blocking:
        def current(self) -> str:
            release.wait(5)
            return "bedrock-api-key-late"

    task = asyncio.create_task(set_bearer(FakeRequest({}), Blocking()))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()


class FakeClient:
    """Stands in for the SDK client: its own hook runs first, and records that it did."""

    def __init__(self, **options: Any) -> None:
        self.options = options
        self.prepared: list[Any] = []

    async def _prepare_request(self, request: Any) -> None:
        self.prepared.append(request)


async def test_the_client_class_sets_the_key_in_its_send_time_hook() -> None:
    cls = bearer_client_class(FakeClient, RotatingKeys())
    assert issubclass(cls, FakeClient)
    client = cls(skip_auth=True)
    assert client.options == {"skip_auth": True}
    request = FakeRequest({})
    await client._prepare_request(request)
    assert client.prepared == [request]  # the SDK's own hook still runs
    assert request.headers == {"Authorization": "Bearer bedrock-api-key-synthetic-1"}
