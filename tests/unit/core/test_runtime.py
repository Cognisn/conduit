"""Start-up: settings from a mapping, assignments resolved before any client is made."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from conduit.core.errors import CapabilityRefused, ConfigurationError, UndeclaredModel
from conduit.core.events import MemoryEventSink, ModelCallFinishedEvent
from conduit.core.registry import ModelNotAssigned
from conduit.core.runtime import ConduitSettings, start
from conduit.core.secrets import LiteralSecretResolver
from conduit.core.tasks import CapabilityFloor, TaskClass
from conduit.core.types import Message, ModelRequest
from tests.unit.core.fakes import CATALOGUE, FakeFactory

SECRETS = LiteralSecretResolver({"fake/key": "not-a-real-key"})
CHAT = TaskClass(name="chat", floor=CapabilityFloor(tools=True))
SUMMARY = TaskClass(name="summary")


def _settings(**overrides: Any) -> ConduitSettings:
    data: dict[str, Any] = {
        "providers": {
            "primary": {"kind": "fake", "api_key": "secret://fake/key"},
            "spare": {"kind": "fake", "api_key": "secret://fake/key"},
        },
        "assignments": {
            "chat": {"provider": "primary", "model": "fake-large", "max_output_tokens": 500},
            "summary": {"provider": "primary", "model": "fake-small"},
        },
    }
    data.update(overrides)
    return ConduitSettings.model_validate(data)


async def test_start_builds_only_assigned_providers() -> None:
    factory = FakeFactory()
    runtime = await start(
        _settings(),
        task_classes=[CHAT, SUMMARY],
        factories=[factory],
        secrets=SECRETS,
        catalogue=CATALOGUE,
    )
    assert set(runtime.providers) == {"primary"}
    assert len(factory.clients) == 1
    await runtime.aclose()
    assert factory.providers[0].closed


async def test_a_refused_assignment_constructs_no_client() -> None:
    factory = FakeFactory()
    settings = _settings(
        assignments={
            "chat": {"provider": "primary", "model": "fake-small"},  # lacks tools
            "summary": {"provider": "primary", "model": "fake-small"},
        }
    )
    with pytest.raises(CapabilityRefused):
        await start(
            settings,
            task_classes=[CHAT, SUMMARY],
            factories=[factory],
            secrets=SECRETS,
            catalogue=CATALOGUE,
        )
    assert factory.clients == []


async def test_every_configured_provider_is_validated_at_start_up() -> None:
    settings = _settings(
        providers={
            "primary": {"kind": "fake", "api_key": "secret://fake/key"},
            "spare": {"kind": "fake", "api_key": "secret://fake/key", "unknown": True},
        }
    )
    with pytest.raises(ConfigurationError):
        await start(
            settings,
            task_classes=[CHAT, SUMMARY],
            factories=[FakeFactory()],
            secrets=SECRETS,
            catalogue=CATALOGUE,
        )


async def test_catalogue_overlays_from_settings_are_applied() -> None:
    settings = _settings(
        catalogue=[
            {
                "name": "product",
                "withdraw": [{"provider": "fake", "model_id": "fake-small"}],
            }
        ]
    )
    with pytest.raises(UndeclaredModel):
        await start(
            settings,
            task_classes=[CHAT, SUMMARY],
            factories=[FakeFactory()],
            secrets=SECRETS,
            catalogue=CATALOGUE,
        )


async def test_a_binding_builds_requests_for_its_task_class() -> None:
    sink = MemoryEventSink()
    runtime = await start(
        _settings(),
        task_classes=[CHAT, SUMMARY],
        factories=[FakeFactory()],
        secrets=SECRETS,
        events=sink,
        catalogue=CATALOGUE,
    )
    binding = runtime.for_task("chat")
    request = binding.request(messages=(Message.user("hello"),), run_id="run_9")
    assert request.model == "fake-large"
    assert request.max_output_tokens == 500
    assert request.task_class == "chat"
    response = await binding.provider.complete(request)
    assert response.text == "echo: hello"
    finished = sink.events[-1]
    assert isinstance(finished, ModelCallFinishedEvent)
    assert finished.task_class == "chat" and finished.run_id == "run_9"
    await runtime.aclose()


def test_provider_names_are_identifiers() -> None:
    with pytest.raises(ValidationError):
        _settings(providers={"has space": {"kind": "fake", "api_key": "secret://fake/key"}})


async def test_a_runtime_provider_calls_only_its_resolved_models() -> None:
    sink = MemoryEventSink()
    factory = FakeFactory()
    settings = _settings(
        assignments={"chat": {"provider": "primary", "model": "fake-large"}},
        require_all_assignments=False,
    )
    runtime = await start(
        settings,
        task_classes=[CHAT, SUMMARY],
        factories=[factory],
        secrets=SECRETS,
        events=sink,
        catalogue=CATALOGUE,
    )
    assert runtime.providers["primary"].assigned_models == frozenset({"fake-large"})
    stray = ModelRequest(
        model="fake-small", messages=(Message.user("hello"),), max_output_tokens=100
    )
    with pytest.raises(ModelNotAssigned):
        await runtime.providers["primary"].complete(stray)
    with pytest.raises(ModelNotAssigned):
        await runtime.for_task("chat").provider.complete(stray)
    assert sink.events == []
    assert factory.providers[0].requests == []
    await runtime.aclose()


async def test_a_binding_cannot_borrow_another_task_classes_model() -> None:
    sink = MemoryEventSink()
    runtime = await start(
        _settings(),
        task_classes=[CHAT, SUMMARY],
        factories=[FakeFactory()],
        secrets=SECRETS,
        events=sink,
        catalogue=CATALOGUE,
    )
    summary = runtime.for_task("summary")
    borrowed = summary.request(messages=(Message.user("hello"),)).model_copy(
        update={"task_class": "chat"}
    )
    with pytest.raises(ModelNotAssigned):
        await summary.provider.complete(borrowed)
    assert sink.events == []
    await runtime.aclose()


async def test_the_runtime_registry_holds_the_settings_residency_policy() -> None:
    settings = _settings(residency={"scopes": ["local"]})
    runtime = await start(
        settings,
        task_classes=[CHAT, SUMMARY],
        factories=[FakeFactory()],
        secrets=SECRETS,
        catalogue=CATALOGUE,
    )
    assert runtime.registry.residency == settings.residency
    await runtime.aclose()
