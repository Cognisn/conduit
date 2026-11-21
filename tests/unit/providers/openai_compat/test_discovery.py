"""OpenAI-compatible discovery: the Models API for OpenAI and X.AI, from recordings.

The SDK's real async client lists through a local transport. Foundry's model ids are
deployment names, which its OpenAI v1 endpoint does not list, so Foundry is reported
unsupported.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pytest

from conduit.core.discovery import DiscoveryOutcome, DiscoveryResult
from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.registry import DiscoveringFactory, ProviderRegistry
from conduit.core.residency import ResidencyScope
from conduit.core.secrets import LiteralSecretResolver
from conduit.providers.openai_compat import OpenAICompatFactory
from conduit.providers.openai_compat.factory import OpenAICompatClient
from conduit.testing.replay import Replay, load_recording
from tests.unit.providers import discovery_recordings as recordings

openai = pytest.importorskip("openai", reason="needs the openai extra")
httpx2 = pytest.importorskip("httpx2", reason="needs the openai extra")

KEY = "synthetic-openai-key-not-real"
SECRETS = LiteralSecretResolver({"openai/key": KEY})
OPENAI = {
    "kind": "openai_compat",
    "service": {"name": "openai", "api_key": "secret://openai/key"},
    "max_retries": 0,
}
XAI = {
    "kind": "openai_compat",
    "service": {"name": "xai", "api_key": "secret://openai/key"},
    "max_retries": 0,
}
Handler = Callable[[Any], Any]


def _replaying(name: str) -> tuple[Handler, Replay]:
    replay = Replay(load_recording(recordings.ROOT / name))

    def handle(request: Any) -> Any:
        query = dict(sorted(request.url.params.items()))
        recorded = replay.next({"method": request.method, "path": request.url.path, "query": query})
        return httpx2.Response(recorded["status"], json=recorded["body"])

    return handle, replay


async def _discover(
    handler: Handler,
    settings: Mapping[str, Any] = OPENAI,
    base_url: str = "https://api.openai.com/v1",
) -> DiscoveryResult:
    sdk = openai.AsyncOpenAI(
        api_key=KEY,
        base_url=base_url,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    registry = ProviderRegistry(secrets=SECRETS, factories=[OpenAICompatFactory()])
    return await registry.discover("gpt", settings, client=OpenAICompatClient(sdk=sdk))


def test_the_openai_compatible_factory_discovers() -> None:
    assert isinstance(OpenAICompatFactory(), DiscoveringFactory)


async def test_openai_models_are_listed_and_joined() -> None:
    handler, replay = _replaying("openai_compat/discovery-openai.json")
    result = await _discover(handler)
    replay.assert_exhausted()
    assert result.catalogue_provider == "openai"
    assert {m.model_id: (m.outcome, m.label) for m in result.models} == {
        "gpt-5.6-sol": (DiscoveryOutcome.ALLOWED, "model"),
        "example-gpt-tuned": (DiscoveryOutcome.ALLOWED, "model"),
        "text-embedding-3-small": (DiscoveryOutcome.ALLOWED, "model"),
    }


async def test_xai_models_join_the_xai_catalogue() -> None:
    handler, replay = _replaying("openai_compat/discovery-xai.json")
    result = await _discover(handler, XAI, base_url="https://api.x.ai/v1")
    replay.assert_exhausted()
    assert result.catalogue_provider == "xai"
    assert {m.model_id: m.outcome for m in result.models} == {
        "grok-4.7": DiscoveryOutcome.ALLOWED,
        "example-grok-preview": DiscoveryOutcome.ALLOWED,
    }
    assert [m.model_id for m in result.uncatalogued] == ["example-grok-preview"]
    assert {m.scope for m in result.models} == {ResidencyScope.GLOBAL}
    preview = result.get("example-grok-preview")
    assert preview is not None and preview.entry is not None
    assert preview.entry.context_window == 131_072  # X.AI's context_length


async def test_an_empty_listing() -> None:
    handler, replay = _replaying("openai_compat/discovery-empty.json")
    result = await _discover(handler)
    replay.assert_exhausted()
    assert result.supported and result.models == ()


@pytest.mark.parametrize(
    ("status", "failure"),
    [
        (401, ProviderFailure.REJECTED),
        (403, ProviderFailure.REJECTED),
        (429, ProviderFailure.UNAVAILABLE),
        (503, ProviderFailure.UNAVAILABLE),
    ],
)
async def test_a_refused_listing_is_a_provider_failure(
    status: int, failure: ProviderFailure
) -> None:
    def handle(request: Any) -> Any:
        return httpx2.Response(status, json={"error": {"message": "Incorrect API key"}})

    with pytest.raises(ProviderError) as raised:
        await _discover(handle)
    assert raised.value.failure is failure
    assert raised.value.__cause__ is None and raised.value.__context__ is None


async def test_an_entry_without_an_id_is_a_protocol_failure() -> None:
    def handle(request: Any) -> Any:
        return httpx2.Response(200, json={"object": "list", "data": [{"object": "model"}]})

    with pytest.raises(ProviderError) as raised:
        await _discover(handle)
    assert raised.value.failure is ProviderFailure.PROTOCOL


async def test_foundry_is_reported_unsupported() -> None:
    registry = ProviderRegistry(secrets=SECRETS, factories=[OpenAICompatFactory()])
    result = await registry.discover(
        "foundry",
        {
            "kind": "openai_compat",
            "service": {
                "name": "foundry",
                "endpoint": "https://example-resource.openai.azure.com/openai/v1/",
                "auth": {"mode": "api_key", "api_key": "secret://openai/key"},
            },
        },
    )
    assert not result.supported
    assert result.reason is not None and "deployment" in result.reason
