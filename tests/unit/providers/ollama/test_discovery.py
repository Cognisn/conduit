"""Ollama discovery: the server's local models, from recorded /api/tags answers."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from conduit.core.discovery import DiscoveryOutcome, DiscoveryResult
from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.registry import DiscoveringFactory, ProviderRegistry
from conduit.core.residency import ResidencyPolicy, ResidencyScope
from conduit.core.secrets import LiteralSecretResolver
from conduit.providers.ollama import OllamaFactory
from conduit.providers.ollama.discovery import parse_tags
from conduit.testing.replay import Replay, load_recording
from tests.unit.providers import discovery_recordings as recordings

httpx = pytest.importorskip("httpx", reason="needs the ollama extra")

Handler = Callable[[Any], Any]


def _replaying(name: str) -> tuple[Handler, Replay]:
    replay = Replay(load_recording(recordings.ROOT / name))

    def handle(request: Any) -> Any:
        recorded = replay.next({"method": request.method, "path": request.url.path})
        return httpx.Response(recorded["status"], json=recorded["body"])

    return handle, replay


async def _discover(handler: Handler, residency: ResidencyPolicy | None = None) -> DiscoveryResult:
    factory = OllamaFactory(transport=httpx.MockTransport(handler))
    registry = ProviderRegistry(
        secrets=LiteralSecretResolver({}), factories=[factory], residency=residency
    )
    return await registry.discover("local", {"kind": "ollama"})


def test_the_ollama_factory_discovers() -> None:
    assert isinstance(OllamaFactory(), DiscoveringFactory)


async def test_the_local_models_are_listed_and_joined() -> None:
    handler, replay = _replaying("ollama/discovery-tags.json")
    result = await _discover(handler)
    replay.assert_exhausted()
    assert result.supported
    assert {m.model_id: m.outcome for m in result.models} == {
        "llama3.1:8b": DiscoveryOutcome.ALLOWED,
        "example-local:3b": DiscoveryOutcome.ALLOWED,
        # Listed as nomic-embed-text:latest; Ollama resolves the bare name to it.
        "nomic-embed-text": DiscoveryOutcome.ALLOWED,
        "example-cloud:120b": DiscoveryOutcome.ALLOWED,
    }
    assert all(m.label is None for m in result.models)


async def test_a_local_model_is_local_and_a_remote_one_is_global() -> None:
    handler, _ = _replaying("ollama/discovery-tags.json")
    result = await _discover(handler)
    scopes = {m.model_id: m.scope for m in result.models}
    assert scopes["llama3.1:8b"] is ResidencyScope.LOCAL
    assert scopes["example-cloud:120b"] is ResidencyScope.GLOBAL


async def test_a_local_only_policy_refuses_the_remote_model() -> None:
    handler, _ = _replaying("ollama/discovery-tags.json")
    policy = ResidencyPolicy(scopes=frozenset({ResidencyScope.LOCAL}))
    result = await _discover(handler, residency=policy)
    assert [m.model_id for m in result.refused] == ["example-cloud:120b"]


async def test_an_empty_server() -> None:
    handler, replay = _replaying("ollama/discovery-empty.json")
    result = await _discover(handler)
    replay.assert_exhausted()
    assert result.supported and result.models == ()


@pytest.mark.parametrize(
    ("status", "failure"),
    [
        (401, ProviderFailure.REJECTED),
        (403, ProviderFailure.REJECTED),
        (500, ProviderFailure.UNAVAILABLE),
        (504, ProviderFailure.TIMEOUT),
    ],
)
async def test_a_refused_listing_is_a_provider_failure(
    status: int, failure: ProviderFailure
) -> None:
    def handle(request: Any) -> Any:
        return httpx.Response(status, json={"error": "denied for example.com"})

    with pytest.raises(ProviderError) as raised:
        await _discover(handle)
    assert raised.value.failure is failure
    assert raised.value.__cause__ is None and raised.value.__context__ is None


async def test_a_connection_failure_is_unavailable() -> None:
    def handle(request: Any) -> Any:
        raise httpx.ConnectError("no route to 192.0.2.10", request=request)

    with pytest.raises(ProviderError) as raised:
        await _discover(handle)
    assert raised.value.failure is ProviderFailure.UNAVAILABLE
    assert raised.value.__context__ is None


@pytest.mark.parametrize(
    "body",
    [b"not json", json.dumps({"models": "x"}).encode(), json.dumps({"models": [{}]}).encode()],
)
async def test_a_body_that_is_not_the_documented_shape_is_a_protocol_failure(
    body: bytes,
) -> None:
    def handle(request: Any) -> Any:
        return httpx.Response(200, content=body)

    with pytest.raises(ProviderError) as raised:
        await _discover(handle)
    assert raised.value.failure is ProviderFailure.PROTOCOL


@pytest.mark.parametrize(
    ("listed", "reported"),
    [
        ("nomic-embed-text:latest", "nomic-embed-text"),
        ("example/model:latest", "example/model"),
        ("llama3.1:8b", "llama3.1:8b"),
        ("example-local:latest-q4", "example-local:latest-q4"),
        ("example-local", "example-local"),
        (
            "registry.example.com:5000/example/model:latest",
            "registry.example.com:5000/example/model",
        ),
        (
            "registry.example.com:5000/example/model:3b",
            "registry.example.com:5000/example/model:3b",
        ),
    ],
)
def test_the_latest_tag_is_reported_under_the_bare_name(listed: str, reported: str) -> None:
    [model] = parse_tags({"models": [{"name": listed, "model": listed}]})
    assert model.model_id == reported
