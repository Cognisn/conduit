"""A stand-in for the anthropic SDK's async client, answering from recordings or scripts.

The adapter reads stream events as mappings (it calls ``model_dump`` on the SDK's event
models), so these fakes yield plain dictionaries in the API's wire format. Errors are
raised as exceptions named like the SDK's, which is how the adapter classifies them
without importing the SDK.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

from conduit.testing.replay import Replay

Event = dict[str, Any]


class APIError(Exception):
    """Mirrors the SDK's base error, which carries the provider's own message."""


class APIConnectionError(APIError):
    pass


class APITimeoutError(APIConnectionError):
    pass


class APIStatusError(APIError):
    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class RateLimitError(APIStatusError):
    pass


class BadRequestError(APIStatusError):
    pass


class AuthenticationError(APIStatusError):
    pass


class InternalServerError(APIStatusError):
    pass


class OverloadedError(APIStatusError):
    pass


class FakeStream:
    def __init__(
        self,
        events: Sequence[Event | BaseException],
        *,
        wrap: bool = False,
        close_error: BaseException | None = None,
    ) -> None:
        self._events = list(events)
        self._wrap = wrap
        self._close_error = close_error
        self.closed = False

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[Any]:
        for event in self._events:
            if isinstance(event, BaseException):
                raise event
            yield _Dumpable(event) if self._wrap else event

    async def close(self) -> None:
        self.closed = True
        if self._close_error is not None:
            raise self._close_error


def assert_clean(error: BaseException, marker: str) -> None:
    """Nothing is chained to ``error``, and ``marker`` appears nowhere in it."""
    assert error.__cause__ is None
    assert error.__context__ is None
    for text in (str(error), repr(error), *(repr(a) for a in error.args)):
        assert marker not in text


class _Dumpable:
    """Looks like an SDK event model: only ``model_dump`` gives its fields."""

    def __init__(self, data: Event) -> None:
        self._data = data

    def model_dump(self, **_: Any) -> Event:
        return self._data


Answer = Callable[[dict[str, Any]], "FakeStream | BaseException"]


class FakeMessages:
    def __init__(self, answer: Answer) -> None:
        self._answer = answer
        self.calls: list[dict[str, Any]] = []
        self.streams: list[FakeStream] = []

    async def create(self, **params: Any) -> FakeStream:
        self.calls.append(params)
        answer = self._answer(params)
        if isinstance(answer, BaseException):
            raise answer
        self.streams.append(answer)
        return answer


class FakeClient:
    def __init__(self, answer: Answer, *, close_error: BaseException | None = None) -> None:
        self.messages = FakeMessages(answer)
        self.closed = False
        self._close_error = close_error

    async def close(self) -> None:
        self.closed = True
        if self._close_error is not None:
            raise self._close_error


def scripted_client(*streams: Sequence[Event | BaseException] | BaseException) -> FakeClient:
    """A client answering successive calls with these event lists (or raising)."""
    pending = list(streams)

    def answer(params: dict[str, Any]) -> FakeStream | BaseException:
        step = pending.pop(0)
        if isinstance(step, BaseException):
            return step
        return FakeStream(step)

    return FakeClient(answer)


def replay_client(replay: Replay) -> FakeClient:
    """A client whose answers come from a recording; a changed request fails the test."""

    def answer(params: dict[str, Any]) -> FakeStream | BaseException:
        return FakeStream(replay.next(params), wrap=True)

    return FakeClient(answer)


def usage(
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read: int | None = 0,
    cache_write: int | None = 0,
) -> dict[str, Any]:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_write,
    }


def message_start(start_usage: dict[str, Any], *, model: str = "claude-sonnet-5-5") -> Event:
    return {
        "type": "message_start",
        "message": {
            "id": "msg_synthetic_01",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": start_usage,
        },
    }


def text_block(index: int, *pieces: str) -> list[Event]:
    events: list[Event] = [
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "text", "text": ""},
        }
    ]
    events += [
        {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": p}}
        for p in pieces
    ]
    events.append({"type": "content_block_stop", "index": index})
    return events


def tool_block(index: int, call_id: str, name: str, *json_pieces: str) -> list[Event]:
    events: list[Event] = [
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "tool_use", "id": call_id, "name": name, "input": {}},
        }
    ]
    events += [
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": p},
        }
        for p in json_pieces
    ]
    events.append({"type": "content_block_stop", "index": index})
    return events


def thinking_block(index: int) -> list[Event]:
    return [
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "signature_delta", "signature": "c2lnbmF0dXJl"},
        },
        {"type": "content_block_stop", "index": index},
    ]


def message_end(stop_reason: str | None, delta_usage: dict[str, Any]) -> list[Event]:
    return [
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": delta_usage,
        },
        {"type": "message_stop"},
    ]
