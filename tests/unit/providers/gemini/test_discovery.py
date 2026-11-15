"""Gemini discovery: models.list through the registry-built client, from recorded pages.

The factory's real google-genai client answers from httpx's mock transport, so the
SDK's own paging (pageToken) runs against the recorded pages. Only chat-capable models
(those whose supported actions include generateContent) are listed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from conduit.core.discovery import DiscoveryOutcome, DiscoveryResult
from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.registry import DiscoveringFactory, ProviderRegistry
from conduit.core.secrets import LiteralSecretResolver
from conduit.providers.gemini import GeminiFactory
from conduit.testing.replay import Replay, load_recording
from tests.unit.providers import discovery_recordings as recordings

genai = pytest.importorskip("google.genai", reason="needs the gemini extra")
httpx = pytest.importorskip("httpx", reason="needs the gemini extra")

# Shaped like a Gemini API key, never a real one.
API_KEY = "AIzaSyD-SYNTHETIC-EXAMPLE-KEY-000000000"
SETTINGS = {"kind": "gemini", "api_key": "secret://gemini/key", "max_attempts": 1}
Handler = Callable[[Any], Any]


def _replaying(name: str) -> tuple[Handler, Replay, list[Any]]:
    replay = Replay(load_recording(recordings.ROOT / name))
    seen: list[Any] = []

    def handle(request: Any) -> Any:
        seen.append(request)
        query = dict(sorted(request.url.params.items()))
        recorded = replay.next({"method": request.method, "path": request.url.path, "query": query})
        return httpx.Response(recorded["status"], json=recorded["body"])

    return handle, replay, seen


async def _discover(handler: Handler) -> DiscoveryResult:
    factory = GeminiFactory(async_client_args={"transport": httpx.MockTransport(handler)})
    registry = ProviderRegistry(
        secrets=LiteralSecretResolver({"gemini/key": API_KEY}), factories=[factory]
    )
    return await registry.discover("gemini", SETTINGS)


def test_the_gemini_factory_discovers() -> None:
    assert isinstance(GeminiFactory(), DiscoveringFactory)


async def test_chat_capable_models_on_every_page_are_listed_and_joined() -> None:
    handler, replay, seen = _replaying("gemini/discovery-models.json")
    result = await _discover(handler)
    replay.assert_exhausted()  # the second page was asked for with its token
    assert {m.model_id: (m.outcome, m.label) for m in result.models} == {
        "gemini-4-argon": (DiscoveryOutcome.ALLOWED, None),
        "example-gemini-preview": (DiscoveryOutcome.UNDECLARED, None),
    }
    assert all(r.url.host == "generativelanguage.googleapis.com" for r in seen)
    assert all(r.headers["x-goog-api-key"] == API_KEY for r in seen)


async def test_an_empty_listing() -> None:
    handler, replay, _ = _replaying("gemini/discovery-empty.json")
    result = await _discover(handler)
    replay.assert_exhausted()
    assert result.supported and result.models == ()


@pytest.mark.parametrize(
    ("status", "failure"),
    [
        (400, ProviderFailure.REJECTED),
        (403, ProviderFailure.REJECTED),
        (429, ProviderFailure.UNAVAILABLE),
        (500, ProviderFailure.UNAVAILABLE),
    ],
)
async def test_a_refused_listing_is_a_provider_failure(
    status: int, failure: ProviderFailure
) -> None:
    def handle(request: Any) -> Any:
        body = {"error": {"code": status, "message": "API key not valid", "status": "X"}}
        return httpx.Response(status, json=body)

    with pytest.raises(ProviderError) as raised:
        await _discover(handle)
    assert raised.value.failure is failure
    assert raised.value.__cause__ is None and raised.value.__context__ is None


async def test_a_listing_that_never_ends_is_a_protocol_failure() -> None:
    def handle(request: Any) -> Any:
        return httpx.Response(200, json={"models": [], "nextPageToken": "again"})

    with pytest.raises(ProviderError) as raised:
        await _discover(handle)
    assert raised.value.failure is ProviderFailure.PROTOCOL
