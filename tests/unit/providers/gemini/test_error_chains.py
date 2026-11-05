"""Every Gemini error path keeps nothing of the failing exception (the wave-wide rule).

A conduit error raised while handling another would carry it as ``__context__`` even
with ``from None``, where an error reporter could read the provider's text. Each path
here asserts no cause, no context, and no secret in the error's text or args.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

from conduit.core.errors import ConfigurationError, ProviderError, ProviderFailure
from conduit.core.registry import MeteredProvider, ProviderRegistry
from conduit.core.secrets import LiteralSecretResolver
from conduit.core.types import Message, ModelRequest, ToolResultBlock
from conduit.providers.gemini import GeminiFactory
from conduit.providers.gemini.mapping import StreamMapper
from tests.unit.providers.gemini.chains import assert_nothing_attached
from tests.unit.providers.gemini.recordings import FakeClient

SECRET = "secret-provider-text-0451"
SETTINGS = {"kind": "gemini", "api_key": "secret://gemini/key"}


def _request(*messages: Message, timeout_s: float = 5) -> ModelRequest:
    return ModelRequest(
        model="gemini-3.8-flash",
        messages=messages or (Message.user("Hi"),),
        max_output_tokens=64,
        timeout_s=timeout_s,
    )


async def _build(client: Any) -> MeteredProvider:
    registry = ProviderRegistry(secrets=LiteralSecretResolver({}), factories=[GeminiFactory()])
    return await registry.build("gemini", SETTINGS, client=client)


async def _failure(client: Any, request: ModelRequest | None = None) -> ProviderError:
    provider = await _build(client)
    with pytest.raises(ProviderError) as caught:
        await provider.complete(request or _request())
    return caught.value


def _chunk(*parts: dict[str, Any]) -> dict[str, Any]:
    return {"candidates": [{"content": {"role": "model", "parts": list(parts)}}]}


class _Models:
    """``client.aio.models`` driven by a coroutine factory for each call."""

    def __init__(self, open_stream: Callable[[], Any]) -> None:
        self.open_stream = open_stream

    async def generate_content_stream(self, **params: Any) -> Any:
        return await self.open_stream()


def _client(open_stream: Callable[[], Any]) -> Any:
    client = FakeClient(lambda params: [])
    client.aio.models = _Models(open_stream)  # type: ignore[assignment]
    return client


async def test_an_error_opening_the_stream() -> None:
    async def open_stream() -> Any:
        raise RuntimeError(SECRET)

    error = await _failure(_client(open_stream))
    assert error.failure is ProviderFailure.UNAVAILABLE
    assert_nothing_attached(error, SECRET)


async def test_a_timeout_opening_the_stream() -> None:
    async def open_stream() -> Any:
        await asyncio.sleep(5)

    error = await _failure(_client(open_stream), _request(timeout_s=0.05))
    assert error.failure is ProviderFailure.TIMEOUT
    assert_nothing_attached(error)


async def test_an_error_mid_stream() -> None:
    error = await _failure(FakeClient(lambda params: [_chunk({"text": "a"}), ValueError(SECRET)]))
    assert error.failure is ProviderFailure.UNAVAILABLE
    assert_nothing_attached(error, SECRET)


async def test_a_timeout_mid_stream() -> None:
    async def open_stream() -> Any:
        async def stream() -> AsyncIterator[Any]:
            await asyncio.sleep(5)
            yield {}

        return stream()

    error = await _failure(_client(open_stream), _request(timeout_s=0.05))
    assert error.failure is ProviderFailure.TIMEOUT
    assert_nothing_attached(error)


@pytest.mark.parametrize(
    "chunk",
    [
        {"candidates": [{"content": [SECRET]}]},
        _chunk({"function_call": {"name": "lookup_host"}, "thought_signature": f"%%{SECRET}%%"}),
        _chunk({"function_call": {"name": f"not a name {SECRET}", "args": {}}}),
        _chunk({"function_call": {"name": "lookup_host", "args": [SECRET]}}),
    ],
)
async def test_a_chunk_the_mapper_cannot_read(chunk: dict[str, Any]) -> None:
    error = await _failure(FakeClient(lambda params: [chunk]))
    assert error.failure is ProviderFailure.PROTOCOL
    assert_nothing_attached(error, SECRET)


async def test_a_result_for_an_unknown_call() -> None:
    result = ToolResultBlock(tool_use_id=f"toolu_{SECRET}", content=())
    error = await _failure(
        FakeClient(lambda params: []), _request(Message(role="user", content=(result,)))
    )
    assert error.failure is ProviderFailure.REJECTED
    assert_nothing_attached(error, SECRET)


async def test_a_close_that_fails_during_a_failure_neither_replaces_nor_chains() -> None:
    closed: list[bool] = []

    class Stream:
        def __aiter__(self) -> Stream:
            return self

        async def __anext__(self) -> Any:
            raise ValueError(SECRET)

        async def aclose(self) -> None:
            closed.append(True)
            raise RuntimeError(f"close {SECRET}")

    async def open_stream() -> Any:
        return Stream()

    error = await _failure(_client(open_stream))
    assert closed == [True]
    assert error.failure is ProviderFailure.UNAVAILABLE
    assert_nothing_attached(error, SECRET)


def test_a_missing_api_key_secret() -> None:
    factory = GeminiFactory()
    settings = factory.settings_model.model_validate(SETTINGS)
    with pytest.raises(ConfigurationError) as caught:
        factory.create_client(settings, {})
    assert_nothing_attached(caught.value)


@pytest.mark.parametrize(
    "part",
    [
        {"function_call": {"name": "lookup_host"}, "thought_signature": f"%%{SECRET}%%"},
        {"function_call": {"name": f"not a name {SECRET}", "args": {}}},
        {"function_call": {"name": "lookup_host", "args": [SECRET]}},
    ],
)
def test_the_mapper_itself_attaches_nothing(part: dict[str, Any]) -> None:
    # The provider re-creates the mapper's errors, so the mapper is checked directly:
    # it is usable on its own, and its errors must be clean too.
    with pytest.raises(ProviderError) as caught:
        StreamMapper().feed(_chunk(part))
    assert caught.value.failure is ProviderFailure.PROTOCOL
    assert_nothing_attached(caught.value, SECRET)
