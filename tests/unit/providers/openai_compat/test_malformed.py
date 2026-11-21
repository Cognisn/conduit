"""Malformed chunks: always ``ProviderError(PROTOCOL)``, never a provider value in an error.

The SDK builds chunk objects without validation, so a chunk can carry any shape. Every
case here must end the call with a protocol failure whose message, representation and
exception chain hold none of the chunk's values.
"""

from __future__ import annotations

import asyncio
import warnings
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, cast

import pytest

from conduit.core.catalogue import default_catalogue
from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.types import Message, ModelRequest
from conduit.providers.openai_compat import OpenAICompatProvider
from conduit.providers.openai_compat.mapping import StreamMapper

MARK = "SECRET-PROVIDER-TEXT"
USAGE = {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
STOP = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}


def _choice(delta: Any, finish: Any = None) -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


class _Stream:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[Any]:
        for chunk in self._chunks:
            yield chunk

    async def close(self) -> None:
        return None


class _Client:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks
        self.chat = self
        self.completions = self

    async def create(self, **kwargs: Any) -> _Stream:
        return _Stream(self._chunks)


class _BadDump:
    """A chunk object whose dump fails with provider text in the error."""

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        raise ValueError(MARK)


CASES: dict[str, list[Any]] = {
    "content is an int": [_choice({"content": 1234567}), STOP, USAGE],
    "content is a mapping": [_choice({"content": {"x": MARK}}), STOP, USAGE],
    "content is a list": [_choice({"content": [MARK]}), STOP, USAGE],
    "refusal is a mapping": [_choice({"refusal": {"x": MARK}}), STOP, USAGE],
    "a choice is a string": [{"choices": [MARK]}, STOP, USAGE],
    "choices is a string": [{"choices": MARK}, STOP, USAGE],
    "choices is a mapping": [{"choices": {"x": MARK}}, STOP, USAGE],
    "choices is an empty mapping": [{"choices": {}}, STOP, USAGE],
    "a delta is a string": [_choice(MARK), STOP, USAGE],
    "usage is a list": [STOP, {"choices": [], "usage": [MARK]}],
    "usage is a string": [STOP, {"choices": [], "usage": MARK}],
    "prompt details are a string": [
        STOP,
        {
            "choices": [],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "prompt_tokens_details": MARK},
        },
    ],
    "finish reason is a mapping": [_choice({}, {"x": MARK}), USAGE],
    "tool calls is a string": [_choice({"tool_calls": MARK}), STOP, USAGE],
    "tool calls is an empty mapping": [_choice({"tool_calls": {}}), STOP, USAGE],
    "a tool call is a string": [_choice({"tool_calls": [MARK]}), STOP, USAGE],
    "a tool function is a string": [
        _choice({"tool_calls": [{"index": 0, "id": "call_a", "function": MARK}]}),
        STOP,
        USAGE,
    ],
    "tool arguments are a mapping": [
        _choice(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_a",
                        "function": {"name": "f", "arguments": {"x": MARK}},
                    }
                ]
            }
        ),
        STOP,
        USAGE,
    ],
    "tool arguments are not JSON": [
        _choice(
            {
                "tool_calls": [
                    {"index": 0, "id": "call_a", "function": {"name": "f", "arguments": "{" + MARK}}
                ]
            }
        ),
        STOP,
        USAGE,
    ],
    "the chunk is a string": [MARK],
    "the chunk dump fails": [_BadDump()],
}


def _chain(error: BaseException) -> list[BaseException]:
    found: list[BaseException] = []
    current: BaseException | None = error
    while current is not None and current not in found:
        found.append(current)
        current = current.__cause__ or current.__context__
    return found


@pytest.mark.parametrize("chunks", list(CASES.values()), ids=list(CASES))
async def test_malformed_chunks_are_protocol_failures_without_provider_text(
    chunks: list[Any],
) -> None:
    provider = OpenAICompatProvider(
        _Client(chunks), catalogue_provider="openai", catalogue=default_catalogue()
    )
    request = ModelRequest(model="gpt-5", messages=(Message.user("hi"),), max_output_tokens=10)
    with pytest.raises(ProviderError) as caught:
        async for _ in provider.stream(request):
            pass
    error = caught.value
    assert error.failure is ProviderFailure.PROTOCOL
    assert _chain(error) == [error]  # no cause, no context
    for text in (str(error), repr(error), repr(error.args)):
        assert MARK not in text
        assert "1234567" not in text


MAPPING_CASES = {
    name: chunks
    for name, chunks in CASES.items()
    if all(isinstance(chunk, dict) for chunk in chunks)
}


@pytest.mark.parametrize("chunks", list(MAPPING_CASES.values()), ids=list(MAPPING_CASES))
def test_the_mapper_itself_refuses_each_shape(chunks: list[dict[str, Any]]) -> None:
    """The mapper checks every shape, not only the provider's catch-all around it."""
    mapper = StreamMapper()
    with pytest.raises(ProviderError) as caught:
        for chunk in chunks:
            mapper.feed(chunk)
        mapper.finish()
    assert caught.value.failure is ProviderFailure.PROTOCOL
    assert _chain(caught.value) == [caught.value]


async def test_an_unvalidated_sdk_chunk_is_refused_without_warnings() -> None:
    models = pytest.importorskip("openai._models")
    chat = pytest.importorskip("openai.types.chat")
    chunk = models.construct_type(
        type_=chat.ChatCompletionChunk,
        value={
            "id": "chatcmpl-example",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-5",
            "choices": [{"index": 0, "delta": {"content": {"x": MARK}}, "finish_reason": None}],
        },
    )
    provider = OpenAICompatProvider(
        _Client([chunk]), catalogue_provider="openai", catalogue=default_catalogue()
    )
    request = ModelRequest(model="gpt-5", messages=(Message.user("hi"),), max_output_tokens=10)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        with pytest.raises(ProviderError) as caught:
            async for _ in provider.stream(request):
                pass
    assert caught.value.failure is ProviderFailure.PROTOCOL
    assert _chain(caught.value) == [caught.value]
    assert not any(MARK in str(w.message) for w in seen)


async def test_a_result_that_is_not_a_stream_is_a_protocol_failure() -> None:
    class _NotAStream:
        def __repr__(self) -> str:
            return MARK

    class _OddClient(_Client):
        async def create(self, **kwargs: Any) -> Any:
            return _NotAStream()

    provider = OpenAICompatProvider(
        _OddClient([]), catalogue_provider="openai", catalogue=default_catalogue()
    )
    request = ModelRequest(model="gpt-5", messages=(Message.user("hi"),), max_output_tokens=10)
    with pytest.raises(ProviderError) as caught:
        async for _ in provider.stream(request):
            pass
    assert caught.value.failure is ProviderFailure.PROTOCOL
    assert _chain(caught.value) == [caught.value]


class _ClosingStream(_Stream):
    """A stream whose close fails with provider text."""

    async def close(self) -> None:
        raise RuntimeError(MARK + "-close")


class _ClosingClient(_Client):
    async def create(self, **kwargs: Any) -> _Stream:
        return _ClosingStream(self._chunks)


def _request() -> ModelRequest:
    return ModelRequest(model="gpt-5", messages=(Message.user("hi"),), max_output_tokens=10)


async def test_a_failing_close_on_its_own_is_a_provider_error_without_a_chain() -> None:
    chunks = [_choice({"content": "hi"}), STOP, USAGE]
    provider = OpenAICompatProvider(
        _ClosingClient(chunks), catalogue_provider="openai", catalogue=default_catalogue()
    )
    with pytest.raises(ProviderError) as caught:
        async for _ in provider.stream(_request()):
            pass
    assert caught.value.failure is ProviderFailure.UNAVAILABLE
    assert _chain(caught.value) == [caught.value]
    assert MARK not in str(caught.value)
    assert MARK not in repr(caught.value)


async def test_a_failing_close_never_replaces_the_error_already_leaving() -> None:
    chunks = [_choice({"content": "hi"}), {"choices": MARK}]
    provider = OpenAICompatProvider(
        _ClosingClient(chunks), catalogue_provider="openai", catalogue=default_catalogue()
    )
    with pytest.raises(ProviderError) as caught:
        async for _ in provider.stream(_request()):
            pass
    assert caught.value.failure is ProviderFailure.PROTOCOL
    assert _chain(caught.value) == [caught.value]
    assert MARK not in str(caught.value)
    assert MARK not in repr(caught.value)


async def test_a_failing_close_on_an_abandoned_stream_is_quiet() -> None:
    chunks = [_choice({"content": "hi"}), _choice({"content": "there"}), STOP, USAGE]
    provider = OpenAICompatProvider(
        _ClosingClient(chunks), catalogue_provider="openai", catalogue=default_catalogue()
    )
    stream = cast(AsyncGenerator[Any, None], provider.stream(_request()))
    async for _ in stream:
        break
    await stream.aclose()


class _CancelledOnClose(_Stream):
    """A stream whose close is cancelled."""

    def __init__(self, chunks: list[Any], cancelled: asyncio.CancelledError) -> None:
        super().__init__(chunks)
        self.cancelled = cancelled

    async def close(self) -> None:
        raise self.cancelled


class _CancellingClient(_Client):
    def __init__(self, chunks: list[Any]) -> None:
        super().__init__(chunks)
        self.cancelled = asyncio.CancelledError()

    async def create(self, **kwargs: Any) -> _Stream:
        return _CancelledOnClose(self._chunks, self.cancelled)


@pytest.mark.parametrize(
    "chunks",
    [
        [_choice({"content": "hi"}), STOP, USAGE],  # after a whole stream
        [_choice({"content": "hi"}), {"choices": MARK}],  # while a ProviderError is leaving
    ],
    ids=["after a whole stream", "while a provider error leaves"],
)
async def test_a_cancellation_during_close_propagates_unchanged(chunks: list[Any]) -> None:
    client = _CancellingClient(chunks)
    provider = OpenAICompatProvider(
        client, catalogue_provider="openai", catalogue=default_catalogue()
    )
    with pytest.raises(asyncio.CancelledError) as caught:
        async for _ in provider.stream(_request()):
            pass
    assert caught.value is client.cancelled
