"""Every Ollama error path keeps nothing of the failing exception (the wave-wide rule).

A conduit error raised while handling another would carry it as ``__context__`` even
with ``from None``, where an error reporter could read Ollama's text. Each path here
asserts no cause, no context, and no secret in the error's text or args. Nothing
reaches a network: httpx's mock transport answers every request.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.registry import ProviderRegistry
from conduit.core.secrets import LiteralSecretResolver
from conduit.core.types import Message, ModelRequest, ToolResultBlock
from conduit.providers.ollama import OllamaFactory
from conduit.providers.ollama.mapping import StreamMapper
from conduit.providers.ollama.provider import _lines
from tests.unit.providers.ollama.chains import assert_nothing_attached

httpx = pytest.importorskip("httpx", reason="needs the ollama extra")

SECRET = "secret-provider-text-0451"
NEWLINE = chr(10)


def _request(*messages: Message) -> ModelRequest:
    return ModelRequest(
        model="llama3.1:8b", messages=messages or (Message.user("Hi"),), max_output_tokens=64
    )


async def _failure(handler: Any, request: ModelRequest | None = None) -> ProviderError:
    factory = OllamaFactory(transport=httpx.MockTransport(handler))
    registry = ProviderRegistry(secrets=LiteralSecretResolver({}), factories=[factory])
    provider = await registry.build("local", {"kind": "ollama"})
    with pytest.raises(ProviderError) as caught:
        await provider.complete(request or _request())
    return caught.value


def _body(*lines: str) -> Any:
    return httpx.Response(200, content=(NEWLINE.join(lines) + NEWLINE).encode("utf-8"))


def _text(text: str) -> str:
    return json.dumps({"message": {"role": "assistant", "content": text}, "done": False})


@pytest.mark.parametrize(
    ("error", "failure"),
    [
        ("ConnectError", ProviderFailure.UNAVAILABLE),
        ("ReadTimeout", ProviderFailure.TIMEOUT),
        ("RemoteProtocolError", ProviderFailure.UNAVAILABLE),
    ],
)
async def test_a_transport_error(error: str, failure: ProviderFailure) -> None:
    def handle(request: Any) -> Any:
        raise getattr(httpx, error)(SECRET, request=request)

    caught = await _failure(handle)
    assert caught.failure is failure
    assert_nothing_attached(caught, SECRET)


async def test_an_http_error_status() -> None:
    caught = await _failure(lambda request: httpx.Response(404, json={"error": SECRET}))
    assert caught.failure is ProviderFailure.REJECTED
    assert_nothing_attached(caught, SECRET)


@pytest.mark.parametrize(
    "line",
    [
        "{not json " + SECRET,
        json.dumps({"error": SECRET}),
        json.dumps(
            {"message": {"tool_calls": [{"function": {"name": "f", "arguments": "{" + SECRET}}]}}
        ),
        json.dumps({"message": {"tool_calls": [{"function": {"name": f"not a name {SECRET}"}}]}}),
        json.dumps(
            {"message": {"tool_calls": [{"function": {"name": "f", "arguments": [SECRET]}}]}}
        ),
    ],
)
async def test_a_line_the_mapper_cannot_read(line: str) -> None:
    caught = await _failure(lambda request: _body(_text("a"), line))
    assert caught.failure in (ProviderFailure.PROTOCOL, ProviderFailure.UNAVAILABLE)
    assert_nothing_attached(caught, SECRET)


async def test_a_line_that_is_not_utf8() -> None:
    raw = bytes([0xFF, 0xFE]) + SECRET.encode() + NEWLINE.encode()
    caught = await _failure(lambda request: httpx.Response(200, content=raw))
    assert caught.failure is ProviderFailure.PROTOCOL
    assert_nothing_attached(caught, SECRET)


async def test_a_result_for_an_unknown_call() -> None:
    result = ToolResultBlock(tool_use_id=f"toolu_{SECRET}", content=())
    caught = await _failure(
        lambda request: _body(), _request(Message(role="user", content=(result,)))
    )
    assert caught.failure is ProviderFailure.REJECTED
    assert_nothing_attached(caught, SECRET)


async def test_a_close_that_fails_during_a_failure_neither_replaces_nor_chains() -> None:
    closed: list[bool] = []

    class Failing(httpx.AsyncByteStream):  # type: ignore[misc,name-defined]
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield (_text("a") + NEWLINE + json.dumps({"error": SECRET}) + NEWLINE).encode()

        async def aclose(self) -> None:
            closed.append(True)
            raise httpx.ReadError(f"close {SECRET}")

    caught = await _failure(lambda request: httpx.Response(200, stream=Failing()))
    assert closed == [True]
    # The error line's failure survives; the close's error neither replaces it nor
    # rides along as its context.
    assert caught.failure is ProviderFailure.UNAVAILABLE
    assert_nothing_attached(caught, SECRET)


async def test_a_close_that_fails_after_a_protocol_failure_keeps_protocol() -> None:
    class Failing(httpx.AsyncByteStream):  # type: ignore[misc,name-defined]
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield ("{not json" + NEWLINE).encode()

        async def aclose(self) -> None:
            raise httpx.ReadError(f"close {SECRET}")

    caught = await _failure(lambda request: httpx.Response(200, stream=Failing()))
    assert caught.failure is ProviderFailure.PROTOCOL
    assert_nothing_attached(caught, SECRET)


async def test_a_transport_error_mid_body() -> None:
    class Breaking(httpx.AsyncByteStream):  # type: ignore[misc,name-defined]
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield (_text("a") + NEWLINE).encode()
            raise httpx.ReadError(SECRET)

    caught = await _failure(lambda request: httpx.Response(200, stream=Breaking()))
    assert caught.failure is ProviderFailure.UNAVAILABLE
    assert_nothing_attached(caught, SECRET)


@pytest.mark.parametrize(
    "line",
    [
        "{not json " + SECRET,
        json.dumps(
            {"message": {"tool_calls": [{"function": {"name": "f", "arguments": "{" + SECRET}}]}}
        ),
        json.dumps({"message": {"tool_calls": [{"function": {"name": f"not a name {SECRET}"}}]}}),
    ],
)
def test_the_mapper_itself_attaches_nothing(line: str) -> None:
    with pytest.raises(ProviderError) as caught:
        StreamMapper().feed(line)
    assert caught.value.failure is ProviderFailure.PROTOCOL
    assert_nothing_attached(caught.value, SECRET)


async def test_the_line_reader_itself_attaches_nothing() -> None:
    # The provider re-creates the reader's errors, so the reader is checked directly.
    async def body() -> AsyncIterator[bytes]:
        yield bytes([0xFF, 0xFE]) + SECRET.encode() + NEWLINE.encode()

    with pytest.raises(ProviderError) as caught:
        async for _ in _lines(body(), 1024):
            pass
    assert caught.value.failure is ProviderFailure.PROTOCOL
    assert_nothing_attached(caught.value, SECRET)
