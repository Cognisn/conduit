"""Start-up for embeddings: task classes, assignments, dimensions and residency, checked once."""

from __future__ import annotations

from typing import Any

import pytest

from conduit.core.catalogue import CatalogueOverlay, ModelEntry
from conduit.core.errors import (
    AssignmentRefused,
    CapabilityRefused,
    ConfigurationError,
    ResidencyRefused,
    UndeclaredModel,
    UnknownTaskClass,
)
from conduit.core.events import MemoryEventSink
from conduit.core.residency import ResidencyPolicy, ResidencyScope
from conduit.core.runtime import ConduitSettings
from conduit.core.secrets import LiteralSecretResolver
from conduit.core.tasks import CapabilityFloor, TaskClass
from conduit.embeddings import (
    EmbeddingBinding,
    EmbeddingRuntime,
    StubEmbeddingFactory,
    builtin_embedding_factories,
    start_embeddings,
)
from conduit.embeddings.stub import STUB_EMBED_MODEL
from tests.unit.embeddings.fakes import CATALOGUE, FAKE, FakeEmbedFactory

SECRETS = LiteralSecretResolver({"fake/key": "not-a-real-key"})
SEARCH = TaskClass(name="search", floor=CapabilityFloor(kind="embedding"))
CHAT = TaskClass(name="chat", floor=CapabilityFloor(kind="chat"))
STUB = {"kind": "stub", "not_for_production": True}


def _settings(**fields: Any) -> ConduitSettings:
    data: dict[str, Any] = {
        "providers": {"embed": FAKE},
        "assignments": {"search": {"provider": "embed", "model": "fake-embed"}},
    }
    data.update(fields)
    return ConduitSettings.model_validate(data)


async def _start(
    settings: ConduitSettings,
    task_classes: tuple[TaskClass, ...] = (SEARCH,),
    **kwargs: Any,
) -> tuple[EmbeddingRuntime, FakeEmbedFactory]:
    factory = FakeEmbedFactory()
    runtime = await start_embeddings(
        settings,
        task_classes=task_classes,
        factories=[factory],
        secrets=SECRETS,
        catalogue=CATALOGUE,
        **kwargs,
    )
    return runtime, factory


async def test_start_resolves_and_builds_an_embedding_task_class() -> None:
    runtime, factory = await _start(_settings())
    binding = runtime.for_task("search")
    assert isinstance(binding, EmbeddingBinding)
    assert binding.assignment.model == "fake-embed"
    assert binding.dimensions == 8
    response = await binding.embed(["A note about example.com"])
    assert response.dimensions == 8
    assert factory.embedders[0].requests[0].task_class == "search"
    await runtime.aclose()
    assert factory.embedders[0].closed


async def test_a_chat_model_cannot_satisfy_an_embedding_task_class() -> None:
    settings = _settings(assignments={"search": {"provider": "embed", "model": "fake-chat"}})
    with pytest.raises(CapabilityRefused, match="chat model, not embedding"):
        await _start(settings)


async def test_an_embedding_model_cannot_satisfy_a_chat_task_class() -> None:
    # The reverse, through the core's own start-up resolution: the floor's kind.
    from conduit.core.tasks import resolve_assignments
    from tests.unit.core.fakes import FakeSettings

    with pytest.raises(CapabilityRefused, match="embedding model, not chat"):
        resolve_assignments(
            [CHAT],
            {"chat": _settings().assignments["search"]},
            {"embed": FakeSettings.model_validate(FAKE | {"kind": "fake"})},
            CATALOGUE.apply(
                CatalogueOverlay(
                    add=(
                        ModelEntry(
                            model_id="fake-embed",
                            provider="fake",
                            kind="embedding",
                            residency_scope=ResidencyScope.LOCAL,
                            context_window=512,
                            embedding_dimensions=8,
                        ),
                    )
                )
            ),
        )


async def test_start_refuses_a_chat_task_class() -> None:
    with pytest.raises(ConfigurationError, match="not an embedding task class"):
        await _start(_settings(), task_classes=(SEARCH, CHAT))


async def test_start_refuses_an_undeclared_model() -> None:
    settings = _settings(assignments={"search": {"provider": "embed", "model": "fake-missing"}})
    with pytest.raises(UndeclaredModel):
        await _start(settings)


async def test_start_refuses_a_residency_breach_before_any_client() -> None:
    settings = _settings(
        assignments={"search": {"provider": "embed", "model": "fake-embed-global"}},
        residency=ResidencyPolicy(scopes=frozenset({ResidencyScope.LOCAL})),
    )
    factory = FakeEmbedFactory()
    with pytest.raises(ResidencyRefused):
        await start_embeddings(
            settings,
            task_classes=[SEARCH],
            factories=[factory],
            secrets=SECRETS,
            catalogue=CATALOGUE,
        )
    assert factory.clients == []


async def test_start_refuses_a_floor_on_dimensions() -> None:
    floor = CapabilityFloor(kind="embedding", min_embedding_dimensions=16)
    with pytest.raises(CapabilityRefused, match="embedding dimensions"):
        await _start(_settings(), task_classes=(TaskClass(name="search", floor=floor),))


async def test_reduced_dimensions_are_resolved_at_start() -> None:
    runtime, factory = await _start(_settings(), dimensions={"search": 4})
    binding = runtime.for_task("search")
    assert binding.dimensions == 4
    response = await binding.embed(["A note about example.com"])
    assert response.dimensions == 4
    assert factory.embedders[0].requests[0].dimensions == 4
    await runtime.aclose()


async def test_reduced_dimensions_below_the_floor_are_refused() -> None:
    floor = CapabilityFloor(kind="embedding", min_embedding_dimensions=6)
    with pytest.raises(CapabilityRefused, match="below"):
        await _start(
            _settings(),
            task_classes=(TaskClass(name="search", floor=floor),),
            dimensions={"search": 4},
        )


async def test_dimensions_a_model_does_not_offer_are_refused_and_closed() -> None:
    factory = FakeEmbedFactory()
    with pytest.raises(CapabilityRefused, match="does not offer"):
        await start_embeddings(
            _settings(),
            task_classes=[SEARCH],
            factories=[factory],
            secrets=SECRETS,
            catalogue=CATALOGUE,
            dimensions={"search": 6},
        )
    assert factory.embedders[0].closed


async def test_dimensions_for_an_unknown_task_class_are_refused() -> None:
    with pytest.raises(AssignmentRefused, match="other"):
        await _start(_settings(), dimensions={"other": 4})


async def test_an_assignment_with_an_output_budget_is_refused() -> None:
    settings = _settings(
        assignments={"search": {"provider": "embed", "model": "fake-embed", "max_output_tokens": 5}}
    )
    with pytest.raises(CapabilityRefused):
        await _start(settings)


async def test_assignments_for_other_task_classes_are_left_alone() -> None:
    settings = _settings(
        providers={"embed": FAKE, "chat": {"kind": "fake", "api_key": "secret://fake/key"}},
        assignments={
            "search": {"provider": "embed", "model": "fake-embed"},
            "chat": {"provider": "chat", "model": "fake-large"},
        },
    )
    runtime, _ = await _start(settings)
    assert set(runtime.assignments) == {"search"}
    assert set(runtime.embedders) == {"embed"}
    await runtime.aclose()


async def test_every_declared_task_class_needs_an_assignment() -> None:
    settings = _settings(assignments={})
    with pytest.raises(AssignmentRefused, match="no assignment"):
        await _start(settings)


async def test_an_unknown_task_class_is_refused_at_lookup() -> None:
    runtime, _ = await _start(_settings())
    with pytest.raises(UnknownTaskClass):
        runtime.for_task("other")
    await runtime.aclose()


async def test_overlays_apply_to_embedding_entries() -> None:
    overlay = CatalogueOverlay(
        name="product",
        override=(
            ModelEntry(
                model_id="fake-embed",
                provider="fakeembed",
                kind="embedding",
                residency_scope=ResidencyScope.LOCAL,
                context_window=512,
                embedding_dimensions=4,
            ),
        ),
    )
    runtime, _ = await _start(_settings(catalogue=(overlay,)))
    assert runtime.for_task("search").dimensions == 4
    await runtime.aclose()


async def test_events_reach_the_product_sink() -> None:
    sink = MemoryEventSink()
    runtime, _ = await _start(_settings(), events=sink)
    await runtime.for_task("search").embed(["A note"])
    assert [e.type for e in sink.events] == ["embedding_call_started", "embedding_call_finished"]
    assert all(e.task_class == "search" for e in sink.events)
    await runtime.aclose()


async def test_the_stub_embedder_serves_the_default_catalogue() -> None:
    settings = ConduitSettings.model_validate(
        {
            "providers": {"stub": STUB},
            "assignments": {"search": {"provider": "stub", "model": STUB_EMBED_MODEL}},
        }
    )
    runtime = await start_embeddings(
        settings,
        task_classes=[SEARCH],
        factories=builtin_embedding_factories(),
        secrets=SECRETS,
    )
    first = await runtime.for_task("search").embed(["A note about example.com", "Another"])
    second = await runtime.for_task("search").embed(["A note about example.com"])
    assert first.dimensions == 8
    assert first.vectors[0] == second.vectors[0]
    assert first.vectors[0] != first.vectors[1]
    assert first.usage is not None and first.usage.input_tokens == 5
    await runtime.aclose()


def test_the_builtin_factories_cover_every_remote_kind() -> None:
    kinds = {factory.kind for factory in builtin_embedding_factories()}
    assert kinds == {
        "stub",
        "bedrock",
        "openai_compat",
        "gemini",
        "ollama",
        "sentence_transformers",
    }
    assert isinstance(builtin_embedding_factories()[0], StubEmbeddingFactory)


def _stub_settings() -> ConduitSettings:
    return ConduitSettings.model_validate(
        {
            "providers": {"stub": STUB},
            "assignments": {"search": {"provider": "stub", "model": STUB_EMBED_MODEL}},
        }
    )


@pytest.mark.parametrize("length", [0, -3])
async def test_dimensions_below_one_are_refused_at_start(length: int) -> None:
    with pytest.raises(CapabilityRefused, match="fewer than 1"):
        await start_embeddings(
            _stub_settings(),
            task_classes=[SEARCH],
            factories=builtin_embedding_factories(),
            secrets=SECRETS,
            dimensions={"search": length},
        )


async def test_the_stub_serves_reduced_dimensions_end_to_end() -> None:
    from conduit.embeddings.stub import stub_vector

    runtime = await start_embeddings(
        _stub_settings(),
        task_classes=[SEARCH],
        factories=builtin_embedding_factories(),
        secrets=SECRETS,
        dimensions={"search": 3},
    )
    response = await runtime.for_task("search").embed(["A note about example.com"])
    assert response.dimensions == 3
    assert response.vectors[0] == stub_vector("A note about example.com", 8)[:3]
    await runtime.aclose()
