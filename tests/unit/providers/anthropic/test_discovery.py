"""Anthropic discovery: the Models API for the direct endpoint, from recorded pages.

The SDK's real async client lists through a local transport, so its own paging is used
against the recorded pages. Claude in Foundry documents no models listing, and is
reported unsupported.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from conduit.core.catalogue import Capabilities
from conduit.core.discovery import DiscoveryOutcome, DiscoveryResult
from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.registry import DiscoveringFactory, ProviderRegistry
from conduit.core.residency import ResidencyPolicy, ResidencyScope
from conduit.core.secrets import LiteralSecretResolver
from conduit.providers.anthropic import AnthropicFactory
from conduit.testing.replay import Replay, load_recording
from tests.unit.providers import discovery_recordings as recordings

anthropic = pytest.importorskip("anthropic", reason="needs the anthropic extra")
httpx2 = pytest.importorskip("httpx2", reason="needs the anthropic extra")

KEY = "synthetic-anthropic-key-not-real"
DIRECT = {"kind": "anthropic", "api_key": "secret://anthropic/key", "max_retries": 0}
Handler = Callable[[Any], Any]


def _replaying(name: str) -> tuple[Handler, Replay]:
    replay = Replay(load_recording(recordings.ROOT / name))

    def handle(request: Any) -> Any:
        query = dict(sorted(request.url.params.items()))
        recorded = replay.next({"method": request.method, "path": request.url.path, "query": query})
        return httpx2.Response(recorded["status"], json=recorded["body"])

    return handle, replay


def _client(handler: Handler) -> Any:
    return anthropic.AsyncAnthropic(
        api_key=KEY,
        base_url="https://api.anthropic.com",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )


async def _discover(handler: Handler, residency: ResidencyPolicy | None = None) -> DiscoveryResult:
    registry = ProviderRegistry(
        secrets=LiteralSecretResolver({"anthropic/key": KEY}),
        factories=[AnthropicFactory()],
        residency=residency,
    )
    return await registry.discover("claude", DIRECT, client=_client(handler))


def test_the_anthropic_factory_discovers() -> None:
    assert isinstance(AnthropicFactory(), DiscoveringFactory)


async def test_every_page_is_listed_and_joined() -> None:
    handler, replay = _replaying("anthropic/discovery-models.json")
    result = await _discover(handler)
    replay.assert_exhausted()  # the second page was asked for with after_id
    assert {m.model_id: (m.outcome, m.label, m.catalogued) for m in result.models} == {
        "claude-opus-5-5": (DiscoveryOutcome.ALLOWED, "model", True),
        "example-claude-preview": (DiscoveryOutcome.ALLOWED, "model", False),
        "claude-haiku-4-5": (DiscoveryOutcome.ALLOWED, "model", True),
    }


async def test_each_model_is_global_with_its_reported_limits_and_capabilities() -> None:
    handler, _ = _replaying("anthropic/discovery-models.json")
    result = await _discover(handler)
    assert {m.scope for m in result.models} == {ResidencyScope.GLOBAL}
    preview = result.get("example-claude-preview")
    assert preview is not None and preview.entry is not None
    assert (preview.entry.context_window, preview.entry.output_limit) == (1_000_000, 128_000)
    assert preview.entry.capabilities == Capabilities(vision=True, structured_output=True)


async def test_a_region_restricted_policy_refuses_the_api() -> None:
    handler, _ = _replaying("anthropic/discovery-models.json")
    policy = ResidencyPolicy(regions=frozenset({"xx-home-1"}))
    result = await _discover(handler, residency=policy)
    assert result.allowed == ()


async def test_an_empty_listing() -> None:
    handler, replay = _replaying("anthropic/discovery-empty.json")
    result = await _discover(handler)
    replay.assert_exhausted()
    assert result.supported and result.models == ()


@pytest.mark.parametrize(
    ("status", "failure"),
    [
        (401, ProviderFailure.REJECTED),
        (403, ProviderFailure.REJECTED),
        (429, ProviderFailure.UNAVAILABLE),
        (500, ProviderFailure.UNAVAILABLE),
    ],
)
async def test_a_refused_listing_is_a_provider_failure(
    status: int, failure: ProviderFailure
) -> None:
    def handle(request: Any) -> Any:
        body = {"type": "error", "error": {"type": "authentication_error", "message": "bad"}}
        return httpx2.Response(status, json=body)

    with pytest.raises(ProviderError) as raised:
        await _discover(handle)
    assert raised.value.failure is failure
    assert raised.value.__cause__ is None and raised.value.__context__ is None


async def test_a_listing_that_never_ends_is_a_protocol_failure() -> None:
    def handle(request: Any) -> Any:
        body = {"data": [], "has_more": True, "first_id": "x", "last_id": "x"}
        return httpx2.Response(200, json=body)

    with pytest.raises(ProviderError) as raised:
        await _discover(handle)
    assert raised.value.failure is ProviderFailure.PROTOCOL


async def test_the_key_is_sent_and_the_listing_goes_to_the_models_path() -> None:
    seen: list[Any] = []

    def handle(request: Any) -> Any:
        seen.append(request)
        return httpx2.Response(200, content=json.dumps({"data": [], "has_more": False}))

    await _discover(handle)
    [request] = seen
    assert request.url.host == "api.anthropic.com"
    assert request.headers["authorization"] == f"Bearer {KEY}"


async def test_claude_in_foundry_is_reported_unsupported() -> None:
    registry = ProviderRegistry(
        secrets=LiteralSecretResolver({"anthropic/key": KEY}), factories=[AnthropicFactory()]
    )
    result = await registry.discover(
        "claude_foundry",
        {
            "kind": "anthropic",
            "endpoint": "foundry",
            "api_key": "secret://anthropic/key",
            "base_url": "https://example-resource.services.ai.azure.com/anthropic/",
        },
    )
    assert not result.supported
    assert result.reason is not None and "Foundry" in result.reason
