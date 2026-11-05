"""A real google-genai client from the Gemini factory, answered by httpx's mock transport.

These tests need the gemini extra and are skipped without it. None reaches a network:
the factory passes a mock transport to the SDK's httpx client, which answers with
server-sent events in the Gemini API's own JSON shape.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest

from conduit.core.registry import MeteredProvider, ProviderRegistry
from conduit.core.secrets import LiteralSecretResolver
from conduit.core.streaming import (
    ProviderStateReported,
    StreamStopped,
    TextDelta,
    ToolCallFinished,
    ToolCallStarted,
)
from conduit.core.types import Message, ModelRequest, StopReason, ToolResultBlock
from conduit.providers.gemini import GeminiFactory
from tests.unit.providers.gemini import recordings
from tests.unit.providers.gemini.recordings import FakeClient

genai = pytest.importorskip("google.genai", reason="needs the gemini extra")
httpx = pytest.importorskip("httpx", reason="needs the gemini extra")

# Shaped like a Gemini API key, never a real one.
API_KEY = "AIzaSyD-SYNTHETIC-EXAMPLE-KEY-000000000"
SIGNATURE = base64.b64encode(b"synthetic-signature").decode("ascii")


def _sse(*chunks: dict[str, Any]) -> Any:
    body = "".join(f"data: {json.dumps(chunk)}\r\n\r\n" for chunk in chunks)
    return httpx.Response(
        200, content=body.encode("utf-8"), headers={"content-type": "text/event-stream"}
    )


CALL_CHUNKS = (
    {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "Checking."},
                        {
                            "functionCall": {
                                "id": "fc_1",
                                "name": "lookup_host",
                                "args": {"host": "a.example.com"},
                            },
                            "thoughtSignature": SIGNATURE,
                        },
                    ],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 20,
            "cachedContentTokenCount": 8,
            "candidatesTokenCount": 5,
            "thoughtsTokenCount": 7,
        },
        "modelVersion": "gemini-3.8-flash",
    },
)


class Recorder:
    def __init__(self, *responses: Any) -> None:
        self.requests: list[Any] = []
        self.responses = list(responses)

    def __call__(self, request: Any) -> Any:
        self.requests.append(request)
        return self.responses.pop(0)


async def _build(recorder: Recorder, **settings: Any) -> MeteredProvider:
    factory = GeminiFactory(async_client_args={"transport": httpx.MockTransport(recorder)})
    registry = ProviderRegistry(
        secrets=LiteralSecretResolver({"gemini/key": API_KEY}), factories=[factory]
    )
    return await registry.build(
        "gemini", {"kind": "gemini", "api_key": "secret://gemini/key", **settings}
    )


def _request(*messages: Message) -> ModelRequest:
    return ModelRequest(
        model="gemini-3.8-flash",
        messages=messages or (Message.user("Is a.example.com up?"),),
        max_output_tokens=64,
    )


@pytest.fixture(autouse=True)
def _hostile_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # Every variable the SDK reads to pick a backend, endpoint or key is set to
    # something else; the factory's explicit settings must win over each.
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", "https://elsewhere.example.net")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "example-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    monkeypatch.setenv("GOOGLE_API_KEY", "AIzaSyD-ENVIRONMENT-KEY-NOT-TO-BE-USED0")
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyD-ENVIRONMENT-KEY-NOT-TO-BE-USED1")


async def test_a_streamed_call_goes_to_the_gemini_api_with_the_resolved_key() -> None:
    recorder = Recorder(_sse(*CALL_CHUNKS))
    provider = await _build(recorder)
    events = [event async for event in provider.stream(_request())]

    (request,) = recorder.requests
    assert request.url.host == "generativelanguage.googleapis.com"
    # The Gemini API path, not Vertex AI's (/v1beta1/publishers/google/models/...).
    assert request.url.path == "/v1beta/models/gemini-3.8-flash:streamGenerateContent"
    assert request.headers["x-goog-api-key"] == API_KEY
    body = json.loads(request.content)
    assert body["contents"] == [{"role": "user", "parts": [{"text": "Is a.example.com up?"}]}]
    assert body["generationConfig"]["maxOutputTokens"] == 64

    # Re-pointed for the provider-state block: the signature is reported before its call.
    signed = events[1]
    assert isinstance(signed, ProviderStateReported)
    assert json.loads(signed.state.payload) == {"call_id": "fc_1", "signature": SIGNATURE}
    assert [events[0], *events[2:4]] == [
        TextDelta(text="Checking."),
        ToolCallStarted(id="fc_1", name="lookup_host"),
        ToolCallFinished(id="fc_1", name="lookup_host", input={"host": "a.example.com"}),
    ]
    stop = events[-1]
    assert isinstance(stop, StreamStopped)
    assert stop.stop_reason is StopReason.TOOL_USE
    assert stop.usage.input_tokens == 12
    assert stop.usage.cache_read_tokens == 8
    assert stop.usage.output_tokens == 12
    assert stop.usage.cache_write_tokens == 0


async def test_the_signature_and_call_id_go_back_on_the_wire() -> None:
    final = {
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": "Up."}]}, "finishReason": "STOP"}
        ],
        "usageMetadata": {"promptTokenCount": 30, "candidatesTokenCount": 1},
    }
    recorder = Recorder(_sse(*CALL_CHUNKS), _sse(final))
    provider = await _build(recorder)
    first = await provider.complete(_request())
    call = first.tool_uses[0]
    second = _request(
        Message.user("Is a.example.com up?"),
        Message(role="assistant", content=first.content),
        Message(role="user", content=(ToolResultBlock(tool_use_id=call.id, content=()),)),
    )
    assert (await provider.complete(second)).text == "Up."

    body = json.loads(recorder.requests[1].content)
    model_parts = body["contents"][1]["parts"]
    assert model_parts[1]["functionCall"]["id"] == "fc_1"
    assert model_parts[1]["thoughtSignature"] == SIGNATURE
    assert body["contents"][2]["parts"][0]["functionResponse"] == {
        "id": "fc_1",
        "name": "lookup_host",
        "response": {"output": ""},
    }


async def test_a_proxy_base_url_is_used() -> None:
    recorder = Recorder(_sse(*CALL_CHUNKS))
    provider = await _build(recorder, base_url="https://gemini.example.com/proxy")
    await provider.complete(_request())
    url = recorder.requests[0].url
    assert url.host == "gemini.example.com"
    assert url.path.startswith("/proxy/")


async def test_sdk_chunk_objects_map_like_their_json_form() -> None:
    # The recordings hold chunks in JSON form; the SDK yields objects. Both must
    # give the same events.
    types = pytest.importorskip("google.genai.types")
    recording = recordings.tool_round_trip()
    chunks = recording.interactions[0].response

    as_json = FakeClient(lambda params: chunks)
    as_objects = FakeClient(
        lambda params: [types.GenerateContentResponse.model_validate(c) for c in chunks]
    )
    registry = ProviderRegistry(secrets=LiteralSecretResolver({}), factories=[GeminiFactory()])
    settings = {"kind": "gemini", "api_key": "secret://gemini/key"}
    request = _request()
    a = [e async for e in (await registry.build("a", settings, client=as_json)).stream(request)]
    b = [e async for e in (await registry.build("b", settings, client=as_objects)).stream(request)]
    assert a == b
    assert any(isinstance(e, ToolCallFinished) for e in a)
