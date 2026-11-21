"""bedrock-mantle discovery: the endpoint's documented OpenAI-compatible model listing.

The factory's real mantle client, re-pointed at a local transport (the SDK's public
``with_options(http_client=...)``), lists ``GET /v1/models`` on the configured Region's
mantle host with conduit's bearer key. Only Claude models (``anthropic.`` ids), which
the Messages route this adapter calls serves, are kept.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import SecretStr

from conduit.core.discovery import DiscoveryOutcome, DiscoveryResult
from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.registry import DiscoveringFactory, ProviderRegistry
from conduit.core.residency import ResidencyScope
from conduit.core.secrets import LiteralSecretResolver
from conduit.providers.bedrock_mantle import BedrockMantleFactory, BedrockMantleSettings
from conduit.testing.replay import Replay, load_recording
from tests.unit.providers import discovery_recordings as recordings

pytest.importorskip("anthropic.lib.bedrock", reason="needs the bedrock-mantle extra")
httpx2 = pytest.importorskip("httpx2", reason="needs the bedrock-mantle extra")

MELBOURNE = "ap-southeast-4"
LONG_TERM_KEY = "synthetic-long-term-key-not-real"
SETTINGS = {
    "kind": "bedrock_mantle",
    "region": MELBOURNE,
    "auth": {"mode": "long_term_api_key", "api_key": "secret://mantle/key"},
    "allow_long_term_api_keys": True,
    "max_retries": 0,
}
Handler = Callable[[Any], Any]


def _replaying(name: str) -> tuple[Handler, Replay, list[Any]]:
    replay = Replay(load_recording(recordings.ROOT / name))
    seen: list[Any] = []

    def handle(request: Any) -> Any:
        seen.append(request)
        query = dict(sorted(request.url.params.items()))
        recorded = replay.next({"method": request.method, "path": request.url.path, "query": query})
        return httpx2.Response(recorded["status"], json=recorded["body"])

    return handle, replay, seen


async def _discover(handler: Handler, settings: dict[str, Any] = SETTINGS) -> DiscoveryResult:
    validated = BedrockMantleSettings.model_validate(settings)
    client = BedrockMantleFactory().create_client(
        validated, {"auth.api_key": SecretStr(LONG_TERM_KEY)}
    )
    local = client.with_options(
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    )
    registry = ProviderRegistry(
        secrets=LiteralSecretResolver({"mantle/key": LONG_TERM_KEY}),
        factories=[BedrockMantleFactory()],
    )
    return await registry.discover("mantle_melbourne", settings, client=local)


def test_the_mantle_factory_discovers() -> None:
    assert isinstance(BedrockMantleFactory(), DiscoveringFactory)


async def test_claude_models_are_listed_and_joined() -> None:
    handler, replay, seen = _replaying("bedrock_mantle/discovery-models.json")
    result = await _discover(handler)
    replay.assert_exhausted()
    assert {m.model_id: (m.outcome, m.label) for m in result.models} == {
        "anthropic.claude-opus-5-5": (DiscoveryOutcome.ALLOWED, "model"),
        "anthropic.claude-sonnet-5": (DiscoveryOutcome.ALLOWED, "model"),
        "anthropic.example-claude-preview": (DiscoveryOutcome.ALLOWED, "model"),
    }
    [request] = seen
    assert str(request.url) == f"https://bedrock-mantle.{MELBOURNE}.api.aws/v1/models"
    assert request.headers["authorization"] == f"Bearer {LONG_TERM_KEY}"


async def test_a_model_the_messages_route_does_not_serve_is_left_out() -> None:
    handler, _, _ = _replaying("bedrock_mantle/discovery-models.json")
    result = await _discover(handler)
    assert result.supported and result.models
    assert "openai.gpt-oss-120b" not in {m.model_id for m in result.models}


async def test_each_model_is_processed_in_the_endpoint_region() -> None:
    handler, _, _ = _replaying("bedrock_mantle/discovery-models.json")
    result = await _discover(handler)
    assert {(m.scope, m.regions) for m in result.models} == {
        (ResidencyScope.IN_REGION, frozenset({MELBOURNE}))
    }
    assert [m.model_id for m in result.uncatalogued] == ["anthropic.example-claude-preview"]


async def test_an_empty_listing() -> None:
    handler, replay, _ = _replaying("bedrock_mantle/discovery-empty.json")
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
        return httpx2.Response(status, json={"message": "Authentication failed"})

    with pytest.raises(ProviderError) as raised:
        await _discover(handle)
    assert raised.value.failure is failure
    assert raised.value.__cause__ is None and raised.value.__context__ is None


@pytest.mark.parametrize("body", [{}, {"data": "x"}, {"data": [{"object": "model"}]}, [1]])
async def test_an_answer_that_is_not_the_documented_shape_is_a_protocol_failure(
    body: Any,
) -> None:
    def handle(request: Any) -> Any:
        return httpx2.Response(200, json=body)

    with pytest.raises(ProviderError) as raised:
        await _discover(handle)
    assert raised.value.failure is ProviderFailure.PROTOCOL


async def test_a_replaced_endpoint_is_reported_unsupported() -> None:
    registry = ProviderRegistry(
        secrets=LiteralSecretResolver({"mantle/key": LONG_TERM_KEY}),
        factories=[BedrockMantleFactory()],
    )
    result = await registry.discover(
        "mantle_vpc", {**SETTINGS, "endpoint_url": "https://vpce.example.com/anthropic"}
    )
    assert not result.supported
    assert result.reason is not None and "endpoint_url" in result.reason
