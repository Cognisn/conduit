"""The provider registry: the one place provider clients are made, and the metering point."""

from __future__ import annotations

import asyncio
from contextlib import aclosing

import pytest

from conduit.core.catalogue import Catalogue, ModelEntry
from conduit.core.errors import (
    ConfigurationError,
    ProviderError,
    ProviderFailure,
    ResidencyRefused,
    SecretUnavailable,
    UndeclaredModel,
)
from conduit.core.events import (
    CapturePolicy,
    MemoryEventSink,
    ModelCallFinishedEvent,
    ModelCallStartedEvent,
)
from conduit.core.registry import ModelNotAssigned, ProviderRegistry
from conduit.core.residency import ResidencyPolicy, ResidencyScope
from conduit.core.secrets import LiteralSecretResolver
from conduit.core.streaming import StreamStopped, TextDelta
from conduit.core.types import Message, ModelRequest, Usage
from tests.unit.core.fakes import CATALOGUE, PARTIAL, USAGE, FakeFactory, FakeSettings

SECRETS = LiteralSecretResolver({"fake/key": "not-a-real-key"})
PROMPT = "Draft a note for the team at example.org"


def _registry(
    sink: MemoryEventSink | None = None, capture: CapturePolicy | None = None
) -> tuple[ProviderRegistry, FakeFactory]:
    factory = FakeFactory()
    registry = ProviderRegistry(
        secrets=SECRETS, catalogue=CATALOGUE, events=sink, capture=capture or CapturePolicy()
    )
    registry.register(factory)
    return registry, factory


def _request(model: str = "fake-large", task_class: str | None = "chat") -> ModelRequest:
    return ModelRequest(
        model=model,
        messages=(Message.user(PROMPT),),
        max_output_tokens=100,
        run_id="run_1",
        task_class=task_class,
    )


FAKE = {"kind": "fake", "api_key": "secret://fake/key"}


class TestRegistration:
    def test_a_kind_registers_once(self) -> None:
        registry, factory = _registry()
        assert registry.kinds == frozenset({"fake"})
        with pytest.raises(ConfigurationError):
            registry.register(factory)

    async def test_an_unregistered_kind_is_refused(self) -> None:
        registry, _ = _registry()
        with pytest.raises(ConfigurationError, match="no provider adapter"):
            await registry.build("p", {"kind": "absent"})


class TestBuilding:
    async def test_settings_are_validated_by_the_adapter_and_secrets_resolved(self) -> None:
        registry, factory = _registry()
        provider = await registry.build("primary", {"kind": "fake", "api_key": "secret://fake/key"})
        assert provider.name == "primary"
        assert provider.kind == "fake"
        assert len(factory.clients) == 1
        assert factory.clients[0].key.get_secret_value() == "not-a-real-key"
        assert isinstance(factory.providers[0].settings, FakeSettings)

    async def test_unknown_adapter_fields_are_refused(self) -> None:
        registry, factory = _registry()
        with pytest.raises(ConfigurationError):
            await registry.build("p", {"kind": "fake", "api_key": "secret://fake/key", "x": 1})
        assert factory.clients == []

    async def test_an_unresolvable_secret_stops_construction(self) -> None:
        registry, factory = _registry()
        with pytest.raises(SecretUnavailable):
            await registry.build("p", {"kind": "fake", "api_key": "secret://other"})
        assert factory.clients == []

    async def test_a_test_client_replaces_construction(self) -> None:
        registry, factory = _registry()
        stand_in = object()
        await registry.build("p", {"kind": "fake", "api_key": "secret://fake/key"}, client=stand_in)
        assert factory.clients == []
        assert factory.providers[0].client is stand_in

    async def test_aclose_closes_every_built_provider(self) -> None:
        registry, factory = _registry()
        await registry.build("a", {"kind": "fake", "api_key": "secret://fake/key"})
        await registry.build("b", {"kind": "fake", "api_key": "secret://fake/key"})
        await registry.aclose()
        assert all(p.closed for p in factory.providers)


class TestMetering:
    async def test_every_call_emits_started_and_finished_with_final_usage(self) -> None:
        sink = MemoryEventSink()
        registry, _ = _registry(sink)
        provider = await registry.build("primary", {"kind": "fake", "api_key": "secret://fake/key"})
        response = await provider.complete(_request())
        assert response.text == f"echo: {PROMPT}"
        started, finished = sink.events
        assert isinstance(started, ModelCallStartedEvent)
        assert isinstance(finished, ModelCallFinishedEvent)
        assert started.call_id == finished.call_id
        assert (started.provider, started.kind, started.model) == ("primary", "fake", "fake-large")
        assert started.run_id == "run_1" and started.task_class == "chat"
        assert finished.outcome == "ok"
        assert finished.usage == USAGE
        assert finished.usage == response.usage

    async def test_streaming_yields_the_adapter_events_unchanged(self) -> None:
        registry, _ = _registry(MemoryEventSink())
        provider = await registry.build("p", {"kind": "fake", "api_key": "secret://fake/key"})
        events = [e async for e in provider.stream(_request())]
        assert isinstance(events[-1], StreamStopped)

    async def test_content_is_not_captured_by_default(self) -> None:
        sink = MemoryEventSink()
        registry, _ = _registry(sink)
        provider = await registry.build("p", {"kind": "fake", "api_key": "secret://fake/key"})
        await provider.complete(_request())
        for event in sink.events:
            assert PROMPT not in event.model_dump_json()
        started = sink.events[0]
        assert isinstance(started, ModelCallStartedEvent)
        assert started.request.text is None and started.request.size > len(PROMPT)

    async def test_content_is_captured_when_the_product_opts_in(self) -> None:
        sink = MemoryEventSink()
        registry, _ = _registry(sink, CapturePolicy(prompts=True))
        provider = await registry.build("p", {"kind": "fake", "api_key": "secret://fake/key"})
        await provider.complete(_request())
        started, finished = sink.events
        assert isinstance(started, ModelCallStartedEvent)
        assert isinstance(finished, ModelCallFinishedEvent)
        assert started.request.text is not None and PROMPT in started.request.text
        assert finished.response is not None and finished.response.text == f"echo: {PROMPT}"

    async def test_an_undeclared_model_is_refused_before_the_adapter_is_called(self) -> None:
        sink = MemoryEventSink()
        registry, _ = _registry(sink)
        provider = await registry.build("p", {"kind": "fake", "api_key": "secret://fake/key"})
        with pytest.raises(UndeclaredModel):
            await provider.complete(_request("fake-unknown"))
        assert sink.events == []

    async def test_a_provider_failure_is_recorded_and_raised(self) -> None:
        sink = MemoryEventSink()
        registry, _ = _registry(sink)
        provider = await registry.build(
            "p", {"kind": "fake", "api_key": "secret://fake/key", "behaviour": "fail"}
        )
        with pytest.raises(ProviderError):
            await provider.complete(_request())
        finished = sink.events[-1]
        assert isinstance(finished, ModelCallFinishedEvent)
        assert finished.outcome == "failed"
        assert finished.failure is ProviderFailure.UNAVAILABLE
        assert finished.usage is None

    async def test_a_stream_without_a_stop_is_a_protocol_failure(self) -> None:
        sink = MemoryEventSink()
        registry, _ = _registry(sink)
        provider = await registry.build(
            "p", {"kind": "fake", "api_key": "secret://fake/key", "behaviour": "no_stop"}
        )
        with pytest.raises(ProviderError) as raised:
            await provider.complete(_request())
        assert raised.value.failure is ProviderFailure.PROTOCOL
        finished = sink.events[-1]
        assert isinstance(finished, ModelCallFinishedEvent)
        assert finished.failure is ProviderFailure.PROTOCOL


class TestAssignmentScope:
    """A provider built for resolved assignments calls only the models resolved for it."""

    async def test_the_resolved_models_are_exposed(self) -> None:
        registry, _ = _registry()
        provider = await registry.build(
            "primary", FAKE, assigned={"chat": "fake-large", "summary": "fake-small"}
        )
        assert provider.assigned_models == frozenset({"fake-large", "fake-small"})
        unbound = await registry.build("spare", FAKE)
        assert unbound.assigned_models is None

    async def test_another_declared_model_is_refused_before_the_adapter_is_called(self) -> None:
        sink = MemoryEventSink()
        registry, factory = _registry(sink)
        provider = await registry.build("primary", FAKE, assigned={"chat": "fake-large"})
        for task_class in ("chat", None):
            with pytest.raises(ModelNotAssigned):
                await provider.complete(_request("fake-small", task_class))
        assert sink.events == []
        assert factory.providers[0].requests == []

    async def test_a_task_class_is_held_to_its_own_model(self) -> None:
        sink = MemoryEventSink()
        registry, factory = _registry(sink)
        provider = await registry.build(
            "primary", FAKE, assigned={"chat": "fake-large", "summary": "fake-small"}
        )
        with pytest.raises(ModelNotAssigned):
            await provider.complete(_request("fake-small", "chat"))
        with pytest.raises(ModelNotAssigned):
            await provider.complete(_request("fake-large", "elsewhere"))
        assert sink.events == []
        assert factory.providers[0].requests == []

    async def test_calls_within_the_assignment_are_served(self) -> None:
        sink = MemoryEventSink()
        registry, _ = _registry(sink)
        provider = await registry.build(
            "primary", FAKE, assigned={"chat": "fake-large", "summary": "fake-small"}
        )
        await provider.complete(_request("fake-large", "chat"))
        await provider.complete(_request("fake-small", "summary"))
        await provider.complete(_request("fake-small", None))
        assert [e.type for e in sink.events].count("model_call_finished") == 3

    async def test_an_empty_assignment_is_refused_when_built(self) -> None:
        registry, factory = _registry()
        with pytest.raises(ConfigurationError):
            await registry.build("primary", FAKE, assigned={})
        assert factory.clients == []

    async def test_an_undeclared_assigned_model_is_refused_when_built(self) -> None:
        registry, factory = _registry()
        with pytest.raises(UndeclaredModel):
            await registry.build("primary", FAKE, assigned={"chat": "fake-unknown"})
        assert factory.clients == []


def _scoped_catalogue() -> Catalogue:
    return Catalogue(
        [
            *CATALOGUE,
            ModelEntry(
                model_id="fake-global",
                provider="fake",
                residency_scope=ResidencyScope.GLOBAL,
                context_window=100_000,
                output_limit=8_000,
            ),
        ],
        version="fake-scoped-1",
    )


class TestRegistryResidency:
    """A provider built without assignments is held to the registry's residency policy."""

    async def test_a_model_outside_the_policy_is_refused_on_every_call(self) -> None:
        sink = MemoryEventSink()
        registry = ProviderRegistry(
            secrets=SECRETS,
            catalogue=_scoped_catalogue(),
            events=sink,
            residency=ResidencyPolicy(scopes=frozenset({ResidencyScope.LOCAL})),
            factories=[FakeFactory()],
        )
        provider = await registry.build("p", FAKE)
        with pytest.raises(ResidencyRefused):
            await provider.complete(_request("fake-global"))
        assert sink.events == []
        await provider.complete(_request("fake-large"))
        assert len(sink.events) == 2

    async def test_without_a_policy_every_declared_model_is_allowed(self) -> None:
        registry = ProviderRegistry(
            secrets=SECRETS, catalogue=_scoped_catalogue(), factories=[FakeFactory()]
        )
        assert registry.residency == ResidencyPolicy()
        provider = await registry.build("p", FAKE)
        response = await provider.complete(_request("fake-global"))
        assert response.text == f"echo: {PROMPT}"


def _finished(sink: MemoryEventSink) -> list[ModelCallFinishedEvent]:
    return [e for e in sink.events if isinstance(e, ModelCallFinishedEvent)]


class TestEndedEarly:
    """Every call is metered once, however it ends, and its inner stream is closed."""

    async def test_a_consumer_that_stops_early_is_metered_as_abandoned(self) -> None:
        sink = MemoryEventSink()
        registry, factory = _registry(sink)
        provider = await registry.build("p", FAKE)
        async with aclosing(provider.stream(_request())) as events:
            async for event in events:
                if isinstance(event, TextDelta):
                    break
        (finished,) = _finished(sink)
        assert finished.outcome == "failed"
        assert finished.error_type == "abandoned"
        assert finished.usage == Usage.zero()
        assert factory.providers[0].streams_closed == 1

    async def test_a_cancelled_call_is_metered_with_the_last_usage_seen(self) -> None:
        sink = MemoryEventSink()
        registry, factory = _registry(sink)
        provider = await registry.build("p", {**FAKE, "behaviour": "slow"})
        call = asyncio.create_task(provider.complete(_request()))
        await asyncio.wait_for(factory.providers[0].waiting.wait(), 5)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        (finished,) = _finished(sink)
        assert finished.outcome == "failed"
        assert finished.error_type == "cancelled"
        assert finished.usage == PARTIAL
        assert factory.providers[0].streams_closed == 1

    async def test_collect_refusing_an_event_after_the_stop_still_meters_the_call(self) -> None:
        sink = MemoryEventSink()
        registry, factory = _registry(sink)
        provider = await registry.build("p", {**FAKE, "behaviour": "after_stop"})
        with pytest.raises(ProviderError) as raised:
            await provider.complete(_request())
        assert raised.value.failure is ProviderFailure.PROTOCOL
        (finished,) = _finished(sink)
        assert finished.outcome == "failed"
        assert finished.error_type == "abandoned"
        assert finished.usage == USAGE
        assert factory.providers[0].streams_closed == 1

    async def test_a_finished_stream_is_metered_once_and_closed(self) -> None:
        sink = MemoryEventSink()
        registry, factory = _registry(sink)
        provider = await registry.build("p", FAKE)
        await provider.complete(_request())
        (finished,) = _finished(sink)
        assert finished.outcome == "ok"
        assert factory.providers[0].streams_closed == 1

    async def test_a_provider_failure_after_usage_carries_that_usage(self) -> None:
        sink = MemoryEventSink()
        registry, factory = _registry(sink)
        provider = await registry.build("p", {**FAKE, "behaviour": "no_stop"})
        with pytest.raises(ProviderError):
            await provider.complete(_request())
        (finished,) = _finished(sink)
        assert finished.failure is ProviderFailure.PROTOCOL
        assert finished.usage == PARTIAL
        assert factory.providers[0].streams_closed == 1
