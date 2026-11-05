"""The Ollama provider end to end through the registry, over httpx's mock transport.

Nothing here reaches a network or a local Ollama: every request is answered by a
handler in the test, from a recording or a scripted reply.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

from conduit.core.catalogue import default_catalogue
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
from conduit.providers.ollama import OllamaFactory, OllamaProvider
from conduit.providers.ollama.provider import MAX_LINE_BYTES
from conduit.testing.replay import Replay, load_recording
from tests.unit.providers.ollama import recordings
from tests.unit.providers.ollama.chains import assert_nothing_attached

httpx = pytest.importorskip("httpx", reason="needs the ollama extra")

Handler = Callable[[Any], Any]
NEWLINE = chr(10)

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


def _direct(handler: Handler, *, max_line_bytes: int) -> OllamaProvider:
    """The adapter itself, with a small line limit (the registry builds the default)."""
    factory = OllamaFactory(transport=httpx.MockTransport(handler))
    settings = factory.settings_model.model_validate({"kind": "ollama"})
    return OllamaProvider(
        "local",
        settings,
        factory.create_client(settings, {}),
        catalogue=default_catalogue(),
        max_line_bytes=max_line_bytes,
    )


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
        assert_nothing_attached(caught.value, "secret-model")

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
        assert_nothing_attached(caught.value, "secret detail")

    @pytest.mark.parametrize("status", [301, 302, 307])
    async def test_a_redirect_is_rejected_and_not_followed(self, status: int) -> None:
        seen: list[str] = []

        def handle(request: Any) -> Any:
            seen.append(str(request.url))
            return httpx.Response(status, headers={"location": "https://elsewhere.example.net/"})

        provider = await _build(handle)
        with pytest.raises(ProviderError) as caught:
            await provider.complete(_request())
        assert caught.value.failure is ProviderFailure.REJECTED
        assert seen == ["http://127.0.0.1:11434/api/chat"]

    @pytest.mark.parametrize("terminated", [True, False])
    async def test_an_overlong_line_is_a_protocol_failure(self, terminated: bool) -> None:
        # A line longer than the limit fails rather than growing without bound,
        # whether or not its newline ever arrives. The stream is otherwise valid
        # (it ends with a done line), so only the limit can fail it.
        text, done = _done_lines("x" * 200)
        if terminated:
            body = (text + NEWLINE + done + NEWLINE).encode("utf-8")
        else:
            final = json.loads(done)
            final["message"]["content"] = "x" * 200
            body = json.dumps(final).encode("utf-8")
        assert len(body) > 200

        def handle(request: Any) -> Any:
            return httpx.Response(200, content=body)

        within = _direct(handle, max_line_bytes=len(body))
        assert (await within.complete(_request())).text == "x" * 200
        provider = _direct(handle, max_line_bytes=100)
        with pytest.raises(ProviderError) as caught:
            await provider.complete(_request())
        assert caught.value.failure is ProviderFailure.PROTOCOL

    @pytest.mark.parametrize("size", [1, 7, 1 << 16])
    async def test_lines_within_the_limit_and_split_across_chunks_are_read(self, size: int) -> None:
        # One-byte chunks put a newline at the start of a chunk, so a reader that
        # searches only the newly received part must still find it.
        payload = (NEWLINE.join(_done_lines("Hello there.")) + NEWLINE).encode("utf-8")

        class Trickle(httpx.AsyncByteStream):  # type: ignore[misc,name-defined]
            async def __aiter__(self) -> AsyncIterator[bytes]:
                for i in range(0, len(payload), size):
                    yield payload[i : i + size]

        def handle(request: Any) -> Any:
            return httpx.Response(200, stream=Trickle())

        provider = _direct(handle, max_line_bytes=200)
        response = await provider.complete(_request())
        assert response.text == "Hello there."

    async def test_a_short_line_after_a_split_line_is_found(self) -> None:
        # The first line arrives in two chunks; the second chunk also carries a short
        # line ("{}") whose newline lies before where the previous search ended. A
        # reader that kept its old search offset would miss it and merge two lines.
        text, done = _done_lines("Hello there.")
        chunks = [
            text[:30].encode("utf-8"),
            (text[30:] + NEWLINE + "{}" + NEWLINE).encode("utf-8"),
            (done + NEWLINE).encode("utf-8"),
        ]

        class Split(httpx.AsyncByteStream):  # type: ignore[misc,name-defined]
            async def __aiter__(self) -> AsyncIterator[bytes]:
                for chunk in chunks:
                    yield chunk

        provider = await _build(lambda request: httpx.Response(200, stream=Split()))
        assert (await provider.complete(_request())).text == "Hello there."

    @pytest.mark.parametrize("terminated", [True, False])
    async def test_the_line_bound_is_exact(self, terminated: bool) -> None:
        # A line of exactly the limit (not counting its newline) is read; one byte
        # more is a protocol failure.
        text, done = _done_lines("x" * 40)
        body = (text + NEWLINE + done + (NEWLINE if terminated else "")).encode("utf-8")
        longest = max(len(text.encode("utf-8")), len(done.encode("utf-8")))

        def handle(request: Any) -> Any:
            return httpx.Response(200, content=body)

        exact = _direct(handle, max_line_bytes=longest)
        assert (await exact.complete(_request())).text == "x" * 40
        under = _direct(handle, max_line_bytes=longest - 1)
        with pytest.raises(ProviderError) as caught:
            await under.complete(_request())
        assert caught.value.failure is ProviderFailure.PROTOCOL

    async def test_a_failing_body_close_after_the_done_line_is_a_success(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # httpx closes the body when it is exhausted; once the done line has been
        # read, an error there cannot change the outcome. Only its class is logged.
        class ClosesBadly(httpx.AsyncByteStream):  # type: ignore[misc,name-defined]
            async def __aiter__(self) -> AsyncIterator[bytes]:
                yield (NEWLINE.join(_done_lines("Hi.")) + NEWLINE).encode("utf-8")

            async def aclose(self) -> None:
                raise httpx.ReadError("secret close detail")

        provider = await _build(lambda request: httpx.Response(200, stream=ClosesBadly()))
        with caplog.at_level("WARNING", logger="conduit.providers.ollama"):
            response = await provider.complete(_request())
        assert response.text == "Hi."
        assert response.stop_reason is StopReason.END_TURN
        assert "ReadError" in caplog.text
        assert "secret close detail" not in caplog.text

    async def test_a_failing_body_close_before_the_done_line_fails_cleanly(self) -> None:
        class ClosesBadly(httpx.AsyncByteStream):  # type: ignore[misc,name-defined]
            async def __aiter__(self) -> AsyncIterator[bytes]:
                yield (_done_lines("Hi.")[0] + NEWLINE).encode("utf-8")

            async def aclose(self) -> None:
                raise httpx.ReadError("secret close detail")

        provider = await _build(lambda request: httpx.Response(200, stream=ClosesBadly()))
        with pytest.raises(ProviderError) as caught:
            await provider.complete(_request())
        assert caught.value.failure is ProviderFailure.UNAVAILABLE
        assert_nothing_attached(caught.value, "secret close detail")

    async def test_a_line_that_is_not_utf8_is_a_protocol_failure(self) -> None:
        def handle(request: Any) -> Any:
            return httpx.Response(200, content=bytes([0xFF, 0xFE]) + b"{}" + NEWLINE.encode())

        provider = await _build(handle)
        with pytest.raises(ProviderError) as caught:
            await provider.complete(_request())
        assert caught.value.failure is ProviderFailure.PROTOCOL
        assert_nothing_attached(caught.value)

    def test_the_default_line_limit_is_generous_but_bounded(self) -> None:
        assert 1 << 20 <= MAX_LINE_BYTES <= 64 << 20

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
