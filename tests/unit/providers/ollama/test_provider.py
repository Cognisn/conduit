"""The Ollama provider end to end through the registry, over httpx's mock transport.

Nothing here reaches a network or a local Ollama: every request is answered by a
handler in the test, from a recording or a scripted reply.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.registry import MeteredProvider, ProviderRegistry
from conduit.core.secrets import LiteralSecretResolver
from conduit.core.streaming import StreamStopped, TextDelta
from conduit.core.types import (
    Message,
    ModelRequest,
    StopReason,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)
from conduit.providers.ollama import OllamaFactory
from conduit.testing.replay import Replay, load_recording
from tests.unit.providers.ollama import recordings

httpx = pytest.importorskip("httpx", reason="needs the ollama extra")

Handler = Callable[[Any], Any]

TOOL = ToolSpec(
    name="lookup_host",
    description="Look a host up in the inventory.",
    input_schema={
        "type": "object",
        "properties": {"host": {"type": "string"}},
        "required": ["host"],
    },
)


def _ndjson(lines: list[str], status: int = 200) -> Any:
    return httpx.Response(
        status,
        content=("\n".join(lines) + "\n").encode("utf-8"),
        headers={"content-type": "application/x-ndjson"},
    )


def replaying(replay: Replay) -> Handler:
    def handle(request: Any) -> Any:
        recorded = replay.next(
            {
                "method": request.method,
                "path": request.url.path,
                "body": json.loads(request.content),
            }
        )
        return _ndjson(recorded["lines"], recorded["status"])

    return handle


async def _build(handler: Handler, settings: dict[str, Any] | None = None) -> MeteredProvider:
    factory = OllamaFactory(transport=httpx.MockTransport(handler))
    registry = ProviderRegistry(secrets=LiteralSecretResolver({}), factories=[factory])
    return await registry.build("local", {"kind": "ollama", **(settings or {})})


def _request(*messages: Message, **fields: Any) -> ModelRequest:
    values: dict[str, Any] = {
        "model": recordings.MODEL,
        "messages": messages or (Message.user("Hi"),),
        "max_output_tokens": 512,
    }
    values.update(fields)
    return ModelRequest(**values)


class TestRecordings:
    @pytest.mark.parametrize("filename", sorted(recordings.RECORDINGS))
    def test_committed_recordings_match_their_builders(self, filename: str) -> None:
        committed = load_recording(recordings.DIRECTORY / filename)
        assert committed == recordings.RECORDINGS[filename]()

    async def test_a_tool_round_trip(self) -> None:
        replay = Replay(load_recording(recordings.DIRECTORY / "tool-round-trip.json"))
        provider = await _build(replaying(replay))
        system = (TextBlock(text=recordings.SYSTEM),)
        first = _request(Message.user(recordings.QUESTION), system=system, tools=(TOOL,))

        response = await provider.complete(first)
        assert response.text == "Checking the inventory."
        assert response.stop_reason is StopReason.TOOL_USE
        assert response.usage == Usage(
            input_tokens=84, output_tokens=31, cache_read_tokens=0, cache_write_tokens=0
        )
        (call,) = response.tool_uses
        assert call.name == "lookup_host"
        assert call.input == {"host": "host-a.example.com"}

        second = first.model_copy(
            update={
                "messages": (
                    *first.messages,
                    Message(role="assistant", content=response.content),
                    Message(
                        role="user",
                        content=(
                            ToolResultBlock(
                                tool_use_id=call.id, content=(TextBlock(text=recordings.RESULT),)
                            ),
                        ),
                    ),
                )
            }
        )
        answer = await provider.complete(second)
        assert answer.text == "Yes: host-a.example.com is up."
        assert answer.stop_reason is StopReason.END_TURN
        assert answer.usage == Usage(
            input_tokens=47, output_tokens=12, cache_read_tokens=84, cache_write_tokens=0
        )
        replay.assert_exhausted()

    async def test_a_reply_cut_short_streams_its_text(self) -> None:
        replay = Replay(load_recording(recordings.DIRECTORY / "cut-short.json"))
        provider = await _build(replaying(replay))
        request = _request(
            Message.user("Describe the example.com network."),
            system=(TextBlock(text=recordings.SYSTEM),),
            max_output_tokens=8,
        )
        events = [event async for event in provider.stream(request)]
        assert events[:-1] == [
            TextDelta(text="The example.com network "),
            TextDelta(text="has two"),
        ]
        stop = events[-1]
        assert isinstance(stop, StreamStopped)
        assert stop.stop_reason is StopReason.MAX_TOKENS
        assert stop.usage == Usage(
            input_tokens=30, output_tokens=8, cache_read_tokens=0, cache_write_tokens=0
        )
        replay.assert_exhausted()


def _done_lines(text: str = "Hi.") -> list[str]:
    return [
        json.dumps({"message": {"role": "assistant", "content": text}, "done": False}),
        json.dumps(
            {
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3,
                "eval_count": 2,
            }
        ),
    ]


class TestRequests:
    async def test_the_request_goes_to_the_loopback_default_with_the_call_timeout(self) -> None:
        seen: list[Any] = []

        def handle(request: Any) -> Any:
            seen.append(request)
            return _ndjson(_done_lines())

        provider = await _build(handle)
        await provider.complete(_request(timeout_s=42))
        (request,) = seen
        assert str(request.url) == "http://127.0.0.1:11434/api/chat"
        assert request.extensions["timeout"]["read"] == 42
        assert request.extensions["timeout"]["connect"] == 42
        assert "authorization" not in request.headers

    async def test_a_base_url_with_a_path_keeps_it(self) -> None:
        seen: list[Any] = []

        def handle(request: Any) -> Any:
            seen.append(str(request.url))
            return _ndjson(_done_lines())

        provider = await _build(handle, {"base_url": "https://ollama.example.com/proxy/"})
        await provider.complete(_request())
        assert seen == ["https://ollama.example.com/proxy/api/chat"]

    async def test_num_ctx_is_the_catalogue_context_window_by_default(self) -> None:
        bodies: list[dict[str, Any]] = []

        def handle(request: Any) -> Any:
            bodies.append(json.loads(request.content))
            return _ndjson(_done_lines())

        provider = await _build(handle)
        await provider.complete(_request())
        assert bodies[0]["options"]["num_ctx"] == 131072

    async def test_the_num_ctx_setting_wins(self) -> None:
        bodies: list[dict[str, Any]] = []

        def handle(request: Any) -> Any:
            bodies.append(json.loads(request.content))
            return _ndjson(_done_lines())

        provider = await _build(handle, {"num_ctx": 16384})
        await provider.complete(_request())
        assert bodies[0]["options"]["num_ctx"] == 16384

    async def test_a_round_trip_through_tool_use_blocks(self) -> None:
        bodies: list[dict[str, Any]] = []

        def handle(request: Any) -> Any:
            bodies.append(json.loads(request.content))
            return _ndjson(_done_lines())

        provider = await _build(handle)
        call = ToolUseBlock(id="toolu_x", name="lookup_host", input={"host": "a.example.com"})
        await provider.complete(
            _request(
                Message.user("Go"),
                Message(role="assistant", content=(call,)),
                Message(
                    role="user",
                    content=(ToolResultBlock(tool_use_id="toolu_x", content=()),),
                ),
                tools=(TOOL,),
            )
        )
        assert bodies[0]["messages"][-1] == {
            "role": "tool",
            "tool_name": "lookup_host",
            "content": "",
        }


class TestFailures:
    @pytest.mark.parametrize(
        ("status", "failure"),
        [
            (404, ProviderFailure.REJECTED),
            (400, ProviderFailure.REJECTED),
            (500, ProviderFailure.UNAVAILABLE),
            (503, ProviderFailure.UNAVAILABLE),
        ],
    )
    async def test_an_http_error_carries_no_provider_text(
        self, status: int, failure: ProviderFailure
    ) -> None:
        def handle(request: Any) -> Any:
            return httpx.Response(status, json={"error": "model 'secret-model' not found"})

        provider = await _build(handle)
        with pytest.raises(ProviderError) as caught:
            await provider.complete(_request())
        assert caught.value.failure is failure
        assert "secret-model" not in str(caught.value)
        assert caught.value.__cause__ is None

    @pytest.mark.parametrize(
        ("error", "failure"),
        [
            ("ConnectError", ProviderFailure.UNAVAILABLE),
            ("RemoteProtocolError", ProviderFailure.UNAVAILABLE),
            ("ReadTimeout", ProviderFailure.TIMEOUT),
            ("ConnectTimeout", ProviderFailure.TIMEOUT),
        ],
    )
    async def test_transport_errors(self, error: str, failure: ProviderFailure) -> None:
        def handle(request: Any) -> Any:
            raise getattr(httpx, error)("secret detail", request=request)

        provider = await _build(handle)
        with pytest.raises(ProviderError) as caught:
            await provider.complete(_request())
        assert caught.value.failure is failure
        assert "secret detail" not in str(caught.value)
        assert caught.value.__cause__ is None
        assert caught.value.__suppress_context__

    async def test_an_error_line_mid_stream(self) -> None:
        def handle(request: Any) -> Any:
            return _ndjson([_done_lines()[0], json.dumps({"error": "out of memory"})])

        provider = await _build(handle)
        with pytest.raises(ProviderError) as caught:
            await provider.complete(_request())
        assert caught.value.failure is ProviderFailure.UNAVAILABLE

    async def test_a_stream_without_done_is_a_protocol_failure(self) -> None:
        def handle(request: Any) -> Any:
            return _ndjson(_done_lines()[:1])

        provider = await _build(handle)
        with pytest.raises(ProviderError) as caught:
            await provider.complete(_request())
        assert caught.value.failure is ProviderFailure.PROTOCOL


class TestLifecycle:
    async def test_stopping_early_closes_the_response(self) -> None:
        class TrackedStream(httpx.AsyncByteStream):  # type: ignore[misc,name-defined]
            """An NDJSON body that records whether it was closed."""

            def __init__(self, lines: list[str]) -> None:
                self.lines = lines
                self.closed = False

            async def __aiter__(self) -> AsyncIterator[bytes]:
                for line in self.lines:
                    yield (line + "\n").encode("utf-8")

            async def aclose(self) -> None:
                self.closed = True

        body = TrackedStream(_done_lines("one") + _done_lines("two"))

        def handle(request: Any) -> Any:
            return httpx.Response(200, stream=body)

        provider = await _build(handle)
        stream = provider.stream(_request())
        first = await anext(stream)
        assert first == TextDelta(text="one")
        await stream.aclose()
        assert body.closed

    async def test_aclose_closes_the_client(self) -> None:
        factory = OllamaFactory(transport=httpx.MockTransport(lambda r: _ndjson(_done_lines())))
        settings = factory.settings_model.model_validate({"kind": "ollama"})
        client = factory.create_client(settings, {})
        registry = ProviderRegistry(secrets=LiteralSecretResolver({}), factories=[factory])
        provider = await registry.build("local", settings, client=client)
        await provider.complete(_request())
        await provider.aclose()
        assert client.is_closed


class TestClient:
    def test_create_client_builds_an_async_client_that_ignores_the_environment(self) -> None:
        factory = OllamaFactory()
        settings = factory.settings_model.model_validate(
            {"kind": "ollama", "base_url": "http://192.0.2.10:11434"}
        )
        client = factory.create_client(settings, {})
        try:
            assert isinstance(client, httpx.AsyncClient)
            assert str(client.base_url) == "http://192.0.2.10:11434"
            assert client.trust_env is False
            assert client.follow_redirects is False
        finally:
            import asyncio

            asyncio.run(client.aclose())
