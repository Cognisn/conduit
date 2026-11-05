"""The embedding funnel: the registry builds embedders, and every call is checked and metered."""

from __future__ import annotations

import asyncio
import json
import sys
import warnings
from typing import Any

import pytest

from conduit.core.errors import (
    CapabilityRefused,
    ConfigurationError,
    ModelNotAssigned,
    ProviderError,
    ProviderFailure,
    ResidencyRefused,
    SettingsRefused,
    UndeclaredModel,
)
from conduit.core.events import CapturePolicy, ContentKind, FanOutSink, MemoryEventSink
from conduit.core.provider import (
    EmbeddingProvider,
    EmbeddingRequest,
    EmbeddingResponse,
    embedding_usage,
)
from conduit.core.registry import (
    EmbeddingCallFinishedEvent,
    EmbeddingCallStartedEvent,
    EmbeddingProviderFactory,
    MeteredEmbedder,
    ProviderRegistry,
)
from conduit.core.residency import ResidencyPolicy, ResidencyScope
from conduit.core.secrets import LiteralSecretResolver
from tests.unit.core.fakes import FakeFactory
from tests.unit.embeddings.fakes import CATALOGUE, FAKE, FakeEmbedder, FakeEmbedFactory

SECRETS = LiteralSecretResolver({"fake/key": "not-a-real-key"})
TEXTS = ("Inventory note for host-a.example.com", "Second note about example.org")


def _registry(
    sink: MemoryEventSink | None = None,
    capture: CapturePolicy | None = None,
    residency: ResidencyPolicy | None = None,
) -> tuple[ProviderRegistry, FakeEmbedFactory]:
    factory = FakeEmbedFactory()
    registry = ProviderRegistry(
        secrets=SECRETS,
        catalogue=CATALOGUE,
        events=sink,
        capture=capture,
        residency=residency,
        embedding_factories=[factory],
    )
    return registry, factory


def _request(model: str = "fake-embed", **fields: Any) -> EmbeddingRequest:
    return EmbeddingRequest(model=model, texts=TEXTS, run_id="run_1", **fields)


def _inner(factory: FakeEmbedFactory) -> FakeEmbedder:
    return factory.embedders[-1]


def _clean(error: BaseException) -> None:
    assert error.__cause__ is None
    assert error.__context__ is None
    for text in TEXTS:
        assert text not in str(error)
        assert text not in repr(error)
        assert all(text not in str(arg) for arg in error.args)


# The request -------------------------------------------------------------------------


def test_a_request_needs_at_least_one_text() -> None:
    with pytest.raises(ValueError):
        EmbeddingRequest(model="fake-embed", texts=())


def test_a_request_refuses_an_empty_text_without_quoting_the_others() -> None:
    with pytest.raises(ValueError) as caught:
        EmbeddingRequest(model="fake-embed", texts=(TEXTS[0], ""))
    assert TEXTS[0] not in str(caught.value)


def test_a_request_refuses_non_positive_dimensions() -> None:
    with pytest.raises(ValueError):
        EmbeddingRequest(model="fake-embed", texts=TEXTS, dimensions=0)


def test_embedding_usage_counts_input_tokens_only() -> None:
    usage = embedding_usage(12)
    assert (usage.input_tokens, usage.output_tokens) == (12, 0)
    assert (usage.cache_read_tokens, usage.cache_write_tokens) == (0, 0)


# Registration ------------------------------------------------------------------------


def test_the_fake_factory_meets_the_protocol() -> None:
    factory = FakeEmbedFactory()
    assert isinstance(factory, EmbeddingProviderFactory)


def test_embedding_kinds_are_listed_apart_from_chat_kinds() -> None:
    registry, _ = _registry()
    assert registry.embedding_kinds == frozenset({"fakeembed"})
    assert registry.kinds == frozenset()


def test_an_embedding_kind_registered_twice_is_refused() -> None:
    registry, _ = _registry()
    with pytest.raises(ConfigurationError, match="twice"):
        registry.register_embedding(FakeEmbedFactory())


def test_an_embedding_kind_must_be_a_valid_kind() -> None:
    registry, _ = _registry()
    factory = FakeEmbedFactory()
    factory.kind = "Not A Kind"
    with pytest.raises(ConfigurationError, match="not a valid kind"):
        registry.register_embedding(factory)


def test_registering_an_embedding_factory_draws_no_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _registry()


def test_a_chat_kind_and_an_embedding_kind_may_share_a_name() -> None:
    registry, _ = _registry()
    registry.register(FakeFactory())
    assert "fake" in registry.kinds


# Building ----------------------------------------------------------------------------


async def test_build_embedder_validates_resolves_and_wraps() -> None:
    registry, factory = _registry()
    embedder = await registry.build_embedder("embed", FAKE)
    assert isinstance(embedder, MeteredEmbedder)
    assert isinstance(embedder, EmbeddingProvider)
    assert embedder.name == "embed"
    assert embedder.kind == "fakeembed"
    assert factory.clients[0].key.get_secret_value() == "not-a-real-key"
    assert factory.catalogues == [CATALOGUE]


async def test_build_embedder_refuses_a_kind_with_no_embedding_adapter() -> None:
    registry, _ = _registry()
    registry.register(FakeFactory())
    with pytest.raises(ConfigurationError, match="no embedding adapter"):
        await registry.build_embedder("chat", {"kind": "fake", "api_key": "secret://fake/key"})


async def test_build_embedder_refuses_bad_settings_by_field_only() -> None:
    registry, _ = _registry()
    with pytest.raises(SettingsRefused) as caught:
        await registry.build_embedder("embed", {**FAKE, "behaviour": "sk-not-a-behaviour"})
    assert "sk-not-a-behaviour" not in str(caught.value)
    assert caught.value.fields == ("behaviour",)


async def test_an_assigned_build_refuses_an_undeclared_model() -> None:
    registry, factory = _registry()
    with pytest.raises(UndeclaredModel):
        await registry.build_embedder("embed", FAKE, assigned={"search": "fake-missing"})
    assert factory.clients == []


async def test_an_assigned_build_refuses_a_chat_model() -> None:
    registry, factory = _registry()
    with pytest.raises(CapabilityRefused, match="chat model"):
        await registry.build_embedder("embed", FAKE, assigned={"search": "fake-chat"})
    assert factory.clients == []


async def test_an_assigned_build_refuses_a_residency_breach() -> None:
    policy = ResidencyPolicy(scopes=frozenset({ResidencyScope.LOCAL}))
    registry, factory = _registry(residency=policy)
    with pytest.raises(ResidencyRefused):
        await registry.build_embedder("embed", FAKE, assigned={"search": "fake-embed-global"})
    assert factory.clients == []


async def test_an_assigned_build_refuses_no_assignment() -> None:
    registry, _ = _registry()
    with pytest.raises(ConfigurationError, match="no assignment"):
        await registry.build_embedder("embed", FAKE, assigned={})


async def test_an_assigned_build_refuses_a_model_the_adapter_does_not_serve() -> None:
    registry, factory = _registry()
    factory.unserved.add("fake-embed")
    with pytest.raises(ConfigurationError, match="cannot call"):
        await registry.build_embedder("embed", FAKE, assigned={"search": "fake-embed"})
    assert _inner(factory).closed


async def test_a_call_on_a_model_the_adapter_does_not_serve_is_refused() -> None:
    registry, factory = _registry()
    embedder = await registry.build_embedder("probe", FAKE)
    _inner(factory).unserved.add("fake-embed")
    with pytest.raises(ConfigurationError, match="cannot call"):
        await embedder.embed(_request())
    assert _inner(factory).requests == []


async def test_assigned_dimensions_must_be_accepted_by_the_model() -> None:
    registry, _ = _registry()
    with pytest.raises(CapabilityRefused, match="dimensions"):
        await registry.build_embedder(
            "embed", FAKE, assigned={"search": "fake-embed"}, dimensions={"search": 6}
        )


async def test_assigned_dimensions_above_the_model_are_refused() -> None:
    registry, _ = _registry()
    with pytest.raises(CapabilityRefused, match="dimensions"):
        await registry.build_embedder(
            "embed", FAKE, assigned={"search": "fake-embed-small"}, dimensions={"search": 8}
        )


async def test_dimensions_for_a_task_class_not_assigned_are_refused() -> None:
    registry, _ = _registry()
    with pytest.raises(ConfigurationError, match="not assigned"):
        await registry.build_embedder(
            "embed", FAKE, assigned={"search": "fake-embed"}, dimensions={"other": 4}
        )


def test_a_metered_embedder_cannot_be_made_outside_the_registry() -> None:
    registry, _ = _registry()
    settings = registry.validate_embedding("embed", FAKE)
    with pytest.raises(TypeError, match="only by ProviderRegistry"):
        MeteredEmbedder(
            FakeEmbedder(settings, CATALOGUE),  # type: ignore[arg-type]
            token=object(),
            name="embed",
            settings=settings,
            catalogue=CATALOGUE,
            events=FanOutSink([]),
        )


# Calls -------------------------------------------------------------------------------


async def test_a_call_returns_one_vector_per_text_and_meters_it() -> None:
    sink = MemoryEventSink()
    registry, _ = _registry(sink)
    embedder = await registry.build_embedder("embed", FAKE)
    response = await embedder.embed(_request(correlation_id="corr_1"))
    assert len(response.vectors) == 2
    assert response.dimensions == 8
    started, finished = sink.events
    assert isinstance(started, EmbeddingCallStartedEvent)
    assert isinstance(finished, EmbeddingCallFinishedEvent)
    assert started.call_id == finished.call_id
    assert (started.provider, started.kind, started.model) == ("embed", "fakeembed", "fake-embed")
    assert (started.run_id, started.correlation_id) == ("run_1", "corr_1")
    assert started.text_count == 2
    assert started.dimensions == 8
    assert finished.outcome == "ok"
    assert finished.vector_count == 2
    assert finished.dimensions == 8
    assert finished.usage == embedding_usage(8)
    assert finished.duration_ms >= 0


async def test_usage_maps_to_input_tokens_on_the_event() -> None:
    sink = MemoryEventSink()
    registry, _ = _registry(sink)
    embedder = await registry.build_embedder("embed", FAKE)
    await embedder.embed(_request())
    finished = sink.events[-1]
    assert isinstance(finished, EmbeddingCallFinishedEvent)
    assert finished.usage is not None
    assert finished.usage.input_tokens == 8
    assert finished.usage.output_tokens == 0


async def test_a_provider_that_reports_no_usage_is_metered_without_it() -> None:
    sink = MemoryEventSink()
    registry, _ = _registry(sink)
    embedder = await registry.build_embedder("embed", {**FAKE, "behaviour": "no_usage"})
    response = await embedder.embed(_request())
    assert response.usage is None
    finished = sink.events[-1]
    assert isinstance(finished, EmbeddingCallFinishedEvent)
    assert finished.outcome == "ok"
    assert finished.usage is None


async def test_usage_with_anything_but_input_tokens_is_a_protocol_failure() -> None:
    registry, _ = _registry()
    embedder = await registry.build_embedder("embed", {**FAKE, "behaviour": "wrong_usage"})
    with pytest.raises(ProviderError) as caught:
        await embedder.embed(_request())
    assert caught.value.failure is ProviderFailure.PROTOCOL
    _clean(caught.value)


async def test_texts_are_not_captured_by_default() -> None:
    sink = MemoryEventSink()
    registry, _ = _registry(sink)
    embedder = await registry.build_embedder("embed", FAKE)
    await embedder.embed(_request())
    started = sink.events[0]
    assert isinstance(started, EmbeddingCallStartedEvent)
    assert started.input.text is None
    assert started.input.kind is ContentKind.PROMPT
    assert started.input.size > 0
    dumped = json.dumps([e.model_dump(mode="json") for e in sink.events])
    for text in TEXTS:
        assert text not in dumped


async def test_texts_are_captured_when_prompts_are() -> None:
    sink = MemoryEventSink()
    registry, _ = _registry(sink, capture=CapturePolicy(prompts=True))
    embedder = await registry.build_embedder("embed", FAKE)
    await embedder.embed(_request())
    started = sink.events[0]
    assert isinstance(started, EmbeddingCallStartedEvent)
    assert started.input.text is not None
    assert json.loads(started.input.text) == list(TEXTS)


async def test_texts_are_not_captured_under_tool_content_alone() -> None:
    sink = MemoryEventSink()
    registry, _ = _registry(sink, capture=CapturePolicy(tool_content=True))
    embedder = await registry.build_embedder("embed", FAKE)
    await embedder.embed(_request())
    started = sink.events[0]
    assert isinstance(started, EmbeddingCallStartedEvent)
    assert started.input.text is None


async def test_an_undeclared_model_is_refused_before_any_call() -> None:
    sink = MemoryEventSink()
    registry, factory = _registry(sink)
    embedder = await registry.build_embedder("embed", FAKE)
    with pytest.raises(UndeclaredModel):
        await embedder.embed(_request("fake-missing"))
    assert _inner(factory).requests == []
    assert sink.events == []


async def test_a_chat_model_is_refused_before_any_call() -> None:
    sink = MemoryEventSink()
    registry, factory = _registry(sink)
    embedder = await registry.build_embedder("embed", FAKE)
    with pytest.raises(CapabilityRefused, match="chat model"):
        await embedder.embed(_request("fake-chat"))
    assert _inner(factory).requests == []
    assert sink.events == []


async def test_an_unassigned_call_is_held_to_the_residency_policy() -> None:
    policy = ResidencyPolicy(scopes=frozenset({ResidencyScope.LOCAL}))
    registry, factory = _registry(residency=policy)
    embedder = await registry.build_embedder("embed", FAKE)
    with pytest.raises(ResidencyRefused):
        await embedder.embed(_request("fake-embed-global"))
    assert _inner(factory).requests == []
    await embedder.embed(_request("fake-embed"))
    assert len(_inner(factory).requests) == 1


async def test_dimensions_above_the_model_are_refused_before_any_call() -> None:
    registry, factory = _registry()
    embedder = await registry.build_embedder("embed", FAKE)
    with pytest.raises(ConfigurationError, match="dimensions"):
        await embedder.embed(_request("fake-embed-small", dimensions=8))
    assert _inner(factory).requests == []


async def test_dimensions_the_model_does_not_accept_are_refused_before_any_call() -> None:
    registry, factory = _registry()
    embedder = await registry.build_embedder("embed", FAKE)
    with pytest.raises(ConfigurationError, match="dimensions"):
        await embedder.embed(_request(dimensions=6))
    assert _inner(factory).requests == []


async def test_reduced_dimensions_are_sent_and_checked() -> None:
    registry, factory = _registry()
    embedder = await registry.build_embedder("embed", FAKE)
    response = await embedder.embed(_request(dimensions=4))
    assert response.dimensions == 4
    assert _inner(factory).requests[-1].dimensions == 4


async def test_the_model_s_own_dimensions_are_sent_as_none() -> None:
    registry, factory = _registry()
    embedder = await registry.build_embedder("embed", FAKE)
    await embedder.embed(_request(dimensions=8))
    assert _inner(factory).requests[-1].dimensions is None


async def test_the_model_s_own_dimensions_need_no_acceptance() -> None:
    registry, factory = _registry()
    embedder = await registry.build_embedder("embed", FAKE)
    response = await embedder.embed(_request("fake-embed-small", dimensions=4))
    assert response.dimensions == 4
    assert _inner(factory).requests[-1].dimensions is None


@pytest.mark.parametrize(
    "behaviour", ["short_vector", "long_vector", "missing_vector", "extra_vector", "nan"]
)
async def test_a_response_of_the_wrong_shape_is_a_protocol_failure(behaviour: str) -> None:
    sink = MemoryEventSink()
    registry, _ = _registry(sink)
    embedder = await registry.build_embedder("embed", {**FAKE, "behaviour": behaviour})
    with pytest.raises(ProviderError) as caught:
        await embedder.embed(_request())
    assert caught.value.failure is ProviderFailure.PROTOCOL
    _clean(caught.value)
    finished = sink.events[-1]
    assert isinstance(finished, EmbeddingCallFinishedEvent)
    assert finished.outcome == "failed"
    assert finished.failure is ProviderFailure.PROTOCOL


async def test_a_reduced_dimension_response_of_full_length_is_refused() -> None:
    registry, factory = _registry()
    embedder = await registry.build_embedder("embed", FAKE)
    inner = _inner(factory)
    original = inner.embed

    async def full_length(request: EmbeddingRequest) -> Any:
        return await original(request.model_copy(update={"dimensions": None}))

    inner.embed = full_length  # type: ignore[method-assign]
    with pytest.raises(ProviderError) as caught:
        await embedder.embed(_request(dimensions=4))
    assert caught.value.failure is ProviderFailure.PROTOCOL


async def test_a_provider_failure_is_metered_and_raised_unchanged() -> None:
    sink = MemoryEventSink()
    registry, _ = _registry(sink)
    embedder = await registry.build_embedder("embed", {**FAKE, "behaviour": "fail"})
    with pytest.raises(ProviderError) as caught:
        await embedder.embed(_request())
    assert caught.value.failure is ProviderFailure.UNAVAILABLE
    finished = sink.events[-1]
    assert isinstance(finished, EmbeddingCallFinishedEvent)
    assert (finished.outcome, finished.failure) == ("failed", ProviderFailure.UNAVAILABLE)
    assert finished.usage is None


async def test_another_error_is_metered_by_class_name() -> None:
    sink = MemoryEventSink()
    registry, _ = _registry(sink)
    embedder = await registry.build_embedder("embed", {**FAKE, "behaviour": "raise"})
    with pytest.raises(RuntimeError):
        await embedder.embed(_request())
    finished = sink.events[-1]
    assert isinstance(finished, EmbeddingCallFinishedEvent)
    assert finished.error_type == "RuntimeError"
    assert "example.com" not in json.dumps(finished.model_dump(mode="json"))


async def test_cancellation_propagates_unchanged_and_is_metered() -> None:
    sink = MemoryEventSink()
    registry, factory = _registry(sink)
    embedder = await registry.build_embedder("embed", {**FAKE, "behaviour": "slow"})
    task = asyncio.create_task(embedder.embed(_request()))
    await _inner(factory).waiting.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    finished = sink.events[-1]
    assert isinstance(finished, EmbeddingCallFinishedEvent)
    assert (finished.outcome, finished.error_type) == ("failed", "cancelled")


async def test_concurrent_calls_are_metered_separately() -> None:
    sink = MemoryEventSink()
    registry, _ = _registry(sink)
    embedder = await registry.build_embedder("embed", FAKE)
    await asyncio.gather(*(embedder.embed(_request()) for _ in range(5)))
    finished = [e for e in sink.events if isinstance(e, EmbeddingCallFinishedEvent)]
    assert len({e.call_id for e in finished}) == 5


# Assignments -------------------------------------------------------------------------


async def test_an_assigned_embedder_serves_only_its_assignments() -> None:
    registry, factory = _registry()
    embedder = await registry.build_embedder(
        "embed", FAKE, assigned={"search": "fake-embed", "tags": "fake-embed-small"}
    )
    assert embedder.assigned_models == frozenset({"fake-embed", "fake-embed-small"})
    await embedder.embed(_request(task_class="search"))
    await embedder.embed(_request("fake-embed-small"))
    with pytest.raises(ModelNotAssigned):
        await embedder.embed(_request("fake-embed-small", task_class="search"))
    with pytest.raises(ModelNotAssigned):
        await embedder.embed(_request("fake-embed-global"))
    assert len(_inner(factory).requests) == 2


async def test_an_assigned_task_class_gets_its_dimensions() -> None:
    registry, factory = _registry()
    embedder = await registry.build_embedder(
        "embed", FAKE, assigned={"search": "fake-embed"}, dimensions={"search": 4}
    )
    response = await embedder.embed(_request(task_class="search"))
    assert response.dimensions == 4
    assert _inner(factory).requests[-1].dimensions == 4
    with pytest.raises(ConfigurationError, match="dimensions"):
        await embedder.embed(_request(task_class="search", dimensions=8))


# Closing -----------------------------------------------------------------------------


async def test_a_failing_close_is_logged_and_never_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry, factory = _registry()
    embedder = await registry.build_embedder("embed", FAKE)
    _inner(factory).close_error = RuntimeError("close failed for example.com")
    await embedder.aclose()
    assert "RuntimeError" in caplog.text
    assert "example.com" not in caplog.text


async def test_cancellation_while_closing_propagates() -> None:
    registry, factory = _registry()
    embedder = await registry.build_embedder("embed", FAKE)
    _inner(factory).close_error = asyncio.CancelledError()  # type: ignore[assignment]
    with pytest.raises(asyncio.CancelledError):
        await embedder.aclose()


async def test_the_registry_closes_its_embedders() -> None:
    registry, factory = _registry()
    await registry.build_embedder("one", FAKE)
    await registry.build_embedder("two", FAKE)
    await registry.aclose()
    assert all(e.closed for e in factory.embedders)


# Vector values (review B1, m2) -------------------------------------------------------


async def _answering(vector: tuple[Any, ...]) -> tuple[Any, MemoryEventSink]:
    """An embedder whose adapter answers every text with ``vector``."""
    sink = MemoryEventSink()
    registry, factory = _registry(sink)
    embedder = await registry.build_embedder("embed", FAKE)
    inner = _inner(factory)

    async def answer(request: EmbeddingRequest) -> EmbeddingResponse:
        return EmbeddingResponse(
            model=request.model,
            vectors=tuple(vector for _ in request.texts),
            usage=embedding_usage(1),
        )

    inner.embed = answer  # type: ignore[method-assign]
    return embedder, sink


def _finished(sink: MemoryEventSink) -> list[EmbeddingCallFinishedEvent]:
    return [e for e in sink.events if isinstance(e, EmbeddingCallFinishedEvent)]


@pytest.mark.parametrize(
    "vector",
    [
        (float("inf"), float("-inf"), *([0.0] * 6)),
        (10**400, *([0.0] * 7)),
        (-(10**400), *([0.0] * 7)),
        (True, *([0.0] * 7)),
        (0.0, False, *([0.0] * 6)),
        ("0.5", *([0.0] * 7)),
        (None, *([0.0] * 7)),
        (float("nan"), *([0.0] * 7)),
    ],
)
async def test_a_bad_value_is_a_protocol_failure_and_metered_once(vector: tuple[Any, ...]) -> None:
    embedder, sink = await _answering(vector)
    with pytest.raises(ProviderError) as caught:
        await embedder.embed(_request())
    assert caught.value.failure is ProviderFailure.PROTOCOL
    _clean(caught.value)
    (finished,) = _finished(sink)
    assert (finished.outcome, finished.failure) == ("failed", ProviderFailure.PROTOCOL)


@pytest.mark.parametrize(
    "vector",
    [
        (1e308,) * 8,
        (-1e308,) * 8,
        (sys.float_info.max,) * 8,
        (1, 2, 3, 4, 5, 6, 7, 8),
        (int(sys.float_info.max),) * 8,
    ],
)
async def test_large_finite_values_are_accepted_and_metered_once(vector: tuple[Any, ...]) -> None:
    embedder, sink = await _answering(vector)
    response = await embedder.embed(_request())
    assert response.vectors[0] == vector
    (finished,) = _finished(sink)
    assert finished.outcome == "ok"


# Dimensions of zero (review m1) ------------------------------------------------------


async def test_zero_assigned_dimensions_are_refused_at_build() -> None:
    from conduit.embeddings.stub import StubEmbeddingFactory

    registry = ProviderRegistry(secrets=SECRETS, embedding_factories=[StubEmbeddingFactory()])
    with pytest.raises(CapabilityRefused, match="at least 1"):
        await registry.build_embedder(
            "stub",
            {"kind": "stub", "not_for_production": True},
            assigned={"search": "stub-embed"},
            dimensions={"search": 0},
        )


# The request's own errors (review m2) ------------------------------------------------


class _Marked:
    def __repr__(self) -> str:
        return "zz-marker-text-4417"


def test_a_request_error_never_quotes_the_input() -> None:
    with pytest.raises(ValueError) as caught:
        EmbeddingRequest(model="fake-embed", texts=(_Marked(),))  # type: ignore[arg-type]
    assert "zz-marker-text-4417" not in str(caught.value)
    assert "zz-marker-text-4417" not in repr(caught.value)
