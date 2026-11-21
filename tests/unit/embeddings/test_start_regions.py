"""Embedding start-up and calls held to a connection's Region choice.

``start_embeddings`` lists each assigned provider and resolves on the listing, so a
choice narrows it as it narrows chat start-up; ``build_embedder`` and an unassigned
``MeteredEmbedder`` are held to the choice too. Every listing and client is synthetic:
the OpenAI listing is a recording behind a mock transport, and the fake embedding
adapter has no client at all.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import pytest

from conduit.core.catalogue import Catalogue, ModelEntry
from conduit.core.errors import ResidencyRefused
from conduit.core.provider import EmbeddingRequest
from conduit.core.registry import ProviderRegistry
from conduit.core.residency import RegionChoice, Residency, ResidencyScope
from conduit.core.secrets import LiteralSecretResolver
from tests.unit.embeddings.fakes import FakeEmbedFactory, FakeEmbedSettings
from tests.unit.embeddings.test_start_listed import _ListedOpenAI, _settings, _start

HOME, AWAY = "xx-home-1", "xx-away-1"


class TestOpenAIChoice:
    SERVICE: ClassVar[dict[str, Any]] = {"name": "openai", "api_key": "secret://openai/key"}

    def _provider(self, code: str) -> dict[str, Any]:
        return {"kind": "openai_compat", "service": {**self.SERVICE, "data_residency": code}}

    async def test_a_model_processed_in_region_starts_with_its_declared_residency(self) -> None:
        runtime = await _start(
            _settings(self._provider("eu"), "text-embedding-3-small"), _ListedOpenAI()
        )
        try:
            entry = runtime.for_task("embed").assignment.entry
            assert (entry.residency_scope, entry.regions) == (
                ResidencyScope.CROSS_REGION,
                frozenset({"openai:eu"}),
            )
            assert runtime.discovery["embedder"].choice is not None
        finally:
            await runtime.aclose()

    async def test_a_model_the_region_does_not_process_in_region_is_refused(self) -> None:
        with pytest.raises(ResidencyRefused, match="global processing is outside"):
            await _start(_settings(self._provider("au"), "text-embedding-3-small"), _ListedOpenAI())


class ChoosingEmbedSettings(FakeEmbedSettings):
    processing: RegionChoice | None = None
    declares: Literal["nothing", "away"] = "nothing"

    def region_choice(self) -> RegionChoice | None:
        return self.processing

    def declared_residency(self, model_id: str) -> Residency | None:
        if self.declares == "away":
            return Residency(ResidencyScope.CROSS_REGION, frozenset({HOME, AWAY}))
        return None


class ChoosingEmbedFactory(FakeEmbedFactory):
    settings_model = ChoosingEmbedSettings


CATALOGUE = Catalogue(
    [
        ModelEntry(
            model_id="fake-embed-home",
            provider="fakeembed",
            kind="embedding",
            residency_scope=ResidencyScope.IN_REGION,
            regions=frozenset({HOME}),
            context_window=8_192,
            embedding_dimensions=8,
        )
    ],
    version="embed-choice-1",
)
SETTINGS = {
    "kind": "fakeembed",
    "api_key": "secret://fake/key",
    "region": HOME,
    "processing": {"regions": [HOME]},
}


def _registry(factory: ChoosingEmbedFactory) -> ProviderRegistry:
    return ProviderRegistry(
        secrets=LiteralSecretResolver({"fake/key": "not-a-real-key"}),
        catalogue=CATALOGUE,
        embedding_factories=[factory],
    )


class TestEmbedderChoice:
    async def test_build_embedder_refuses_an_assignment_outside_the_choice(self) -> None:
        registry = _registry(ChoosingEmbedFactory())
        with pytest.raises(ResidencyRefused, match="outside the connection"):
            await registry.build_embedder(
                "embedder", {**SETTINGS, "declares": "away"}, assigned={"embed": "fake-embed-home"}
            )
        built = await registry.build_embedder(
            "embedder", SETTINGS, assigned={"embed": "fake-embed-home"}
        )
        assert built.assigned_models == frozenset({"fake-embed-home"})
        await registry.aclose()

    async def test_an_unassigned_embedder_refuses_each_call_outside_the_choice(self) -> None:
        factory = ChoosingEmbedFactory()
        registry = _registry(factory)
        built = await registry.build_embedder("embedder", {**SETTINGS, "declares": "away"})
        with pytest.raises(ResidencyRefused, match="outside the connection"):
            await built.embed(EmbeddingRequest(model="fake-embed-home", texts=("hello",)))
        assert factory.embedders[-1].requests == []  # refused before any client call
        await registry.aclose()

    async def test_an_unassigned_embedder_inside_the_choice_embeds(self) -> None:
        registry = _registry(ChoosingEmbedFactory())
        built = await registry.build_embedder("embedder", SETTINGS)
        response = await built.embed(EmbeddingRequest(model="fake-embed-home", texts=("hello",)))
        assert len(response.vectors) == 1
        await registry.aclose()
