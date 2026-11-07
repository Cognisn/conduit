"""Start-up follow-ups: the runtime's sink and capture policy (Q0027), and embedding task
classes given to ``start`` (Q0030)."""

from __future__ import annotations

from typing import Any

import pytest

from conduit.core.catalogue import Catalogue, ModelEntry
from conduit.core.errors import AssignmentRefused, CapabilityRefused, ConfigurationError
from conduit.core.events import (
    CapturePolicy,
    ContentKind,
    EventSink,
    MemoryEventSink,
    ToolCallEvent,
)
from conduit.core.facade import LoopThread, start_sync
from conduit.core.runtime import ConduitSettings, Runtime, start
from conduit.core.secrets import LiteralSecretResolver
from conduit.core.tasks import CapabilityFloor, TaskClass
from tests.unit.core.fakes import FakeFactory

SECRETS = LiteralSecretResolver({"fake/key": "not-a-real-key"})
MARKER = "zz synthetic runtime marker 7310 zz"
CHAT = TaskClass(name="chat")
SEARCH = TaskClass(name="search", floor=CapabilityFloor(kind="embedding"))

CATALOGUE = Catalogue(
    [
        ModelEntry(
            model_id="fake-large",
            provider="fake",
            residency_scope="local",  # type: ignore[arg-type]
            context_window=100_000,
            output_limit=8_000,
        ),
        ModelEntry(
            model_id="fake-embed",
            provider="fake",
            kind="embedding",
            residency_scope="local",  # type: ignore[arg-type]
            context_window=512,
            embedding_dimensions=8,
        ),
        ModelEntry(
            model_id="local-embed",
            provider="fakeembed",
            kind="embedding",
            residency_scope="local",  # type: ignore[arg-type]
            context_window=512,
            embedding_dimensions=8,
        ),
    ],
    version="runtime-follow-up-1",
)


def _settings(**overrides: Any) -> ConduitSettings:
    data: dict[str, Any] = {
        "providers": {"primary": {"kind": "fake", "api_key": "secret://fake/key"}},
        "assignments": {"chat": {"provider": "primary", "model": "fake-large"}},
    }
    data.update(overrides)
    return ConduitSettings.model_validate(data)


async def _start(settings: ConduitSettings, **kwargs: Any) -> Runtime:
    return await start(
        settings,
        task_classes=kwargs.pop("task_classes", [CHAT]),
        factories=kwargs.pop("factories", [FakeFactory()]),
        secrets=SECRETS,
        catalogue=CATALOGUE,
        **kwargs,
    )


def _tool_event() -> ToolCallEvent:
    loose = CapturePolicy(prompts=True, tool_content=True)
    return ToolCallEvent(
        call_id="toolu_1",
        tool="lookup",
        outcome="ok",
        arguments=loose.capture(ContentKind.TOOL, MARKER),
        duration_ms=1.0,
    )


class TestSinkAndCapture:
    async def test_the_capture_policy_is_the_settings_policy(self) -> None:
        policy = CapturePolicy(prompts=True)
        runtime = await _start(_settings(capture=policy))
        assert runtime.capture == policy
        await runtime.aclose()

    async def test_the_sink_delivers_to_the_products_sink(self) -> None:
        sink = MemoryEventSink()
        runtime = await _start(_settings(capture=CapturePolicy(tool_content=True)), events=sink)
        assert isinstance(runtime.events, EventSink)
        await runtime.events.emit(_tool_event())
        (received,) = sink.events
        assert isinstance(received, ToolCallEvent)
        assert received.arguments.text == MARKER
        await runtime.aclose()

    async def test_the_sink_holds_events_to_the_runtimes_capture_policy(self) -> None:
        sink = MemoryEventSink()
        runtime = await _start(_settings(), events=sink)
        await runtime.events.emit(_tool_event())  # emitted under a looser policy
        (received,) = sink.events
        assert isinstance(received, ToolCallEvent)
        assert received.arguments.text is None
        assert MARKER not in received.model_dump_json()
        await runtime.aclose()

    async def test_without_a_product_sink_emitting_does_nothing(self) -> None:
        runtime = await _start(_settings())
        await runtime.events.emit(_tool_event())
        await runtime.aclose()

    async def test_capture_cannot_be_loosened_through_the_sink(self) -> None:
        sink = MemoryEventSink()
        runtime = await _start(_settings(), events=sink)
        with pytest.raises(AttributeError):
            setattr(runtime.events, "policy", CapturePolicy(prompts=True, tool_content=True))  # noqa: B010
        await runtime.events.emit(_tool_event())
        (received,) = sink.events
        assert isinstance(received, ToolCallEvent)
        assert received.arguments.text is None
        assert runtime.capture == CapturePolicy()
        await runtime.aclose()

    @pytest.mark.parametrize("name", ["events", "capture"])
    async def test_both_are_read_only(self, name: str) -> None:
        runtime = await _start(_settings())
        with pytest.raises(AttributeError):
            setattr(runtime, name, None)
        await runtime.aclose()


class TestEmbeddingTaskClasses:
    async def test_an_embedding_task_class_is_left_for_start_embeddings(self) -> None:
        factory = FakeFactory()
        settings = _settings(
            assignments={
                "chat": {"provider": "primary", "model": "fake-large"},
                "search": {"provider": "primary", "model": "fake-embed"},
            }
        )
        runtime = await _start(settings, task_classes=[CHAT, SEARCH], factories=[factory])
        assert set(runtime.assignments) == {"chat"}
        assert runtime.providers["primary"].assigned_models == frozenset({"fake-large"})
        await runtime.aclose()

    async def test_a_provider_only_embedding_task_classes_use_is_not_built(self) -> None:
        factory = FakeFactory()
        settings = _settings(
            providers={
                "primary": {"kind": "fake", "api_key": "secret://fake/key"},
                "vectors": {"kind": "fake", "api_key": "secret://fake/key"},
            },
            assignments={
                "chat": {"provider": "primary", "model": "fake-large"},
                "search": {"provider": "vectors", "model": "fake-embed"},
            },
        )
        runtime = await _start(settings, task_classes=[CHAT, SEARCH], factories=[factory])
        assert set(runtime.providers) == {"primary"}
        assert len(factory.clients) == 1
        await runtime.aclose()

    async def test_an_embedding_only_provider_kind_needs_no_chat_adapter(self) -> None:
        settings = _settings(
            providers={
                "primary": {"kind": "fake", "api_key": "secret://fake/key"},
                "vectors": {"kind": "fakeembed", "api_key": "secret://fake/key"},
            },
            assignments={
                "chat": {"provider": "primary", "model": "fake-large"},
                "search": {"provider": "vectors", "model": "local-embed"},
            },
        )
        runtime = await _start(settings, task_classes=[CHAT, SEARCH])
        assert set(runtime.assignments) == {"chat"}
        assert set(runtime.providers) == {"primary"}
        await runtime.aclose()

    def test_start_sync_leaves_them_too(self) -> None:
        settings = _settings(
            assignments={
                "chat": {"provider": "primary", "model": "fake-large"},
                "search": {"provider": "primary", "model": "fake-embed"},
            }
        )
        loop = LoopThread(name="conduit-test-runtime")
        try:
            with start_sync(
                settings,
                task_classes=[CHAT, SEARCH],
                factories=[FakeFactory()],
                secrets=SECRETS,
                catalogue=CATALOGUE,
                loop=loop,
            ) as runtime:
                assert set(runtime.assignments) == {"chat"}
        finally:
            loop.close()

    async def test_a_task_class_declared_twice_is_still_refused(self) -> None:
        twin = TaskClass(name="chat", floor=CapabilityFloor(kind="embedding"))
        with pytest.raises(ConfigurationError, match="declared twice"):
            await _start(_settings(), task_classes=[CHAT, twin])

    async def test_an_embedding_task_class_without_an_assignment_is_not_refused(self) -> None:
        runtime = await _start(_settings(), task_classes=[CHAT, SEARCH])
        assert set(runtime.assignments) == {"chat"}
        await runtime.aclose()

    async def test_a_chat_task_class_on_an_embedding_model_is_still_refused(self) -> None:
        settings = _settings(assignments={"chat": {"provider": "primary", "model": "fake-embed"}})
        with pytest.raises(CapabilityRefused):
            await _start(settings)

    async def test_an_assignment_for_an_undeclared_task_class_is_still_refused(self) -> None:
        settings = _settings(
            assignments={
                "chat": {"provider": "primary", "model": "fake-large"},
                "search": {"provider": "primary", "model": "fake-embed"},
            }
        )
        with pytest.raises(AssignmentRefused):
            await _start(settings, task_classes=[CHAT])

    async def test_an_unused_provider_is_still_validated(self) -> None:
        settings = _settings(
            providers={
                "primary": {"kind": "fake", "api_key": "secret://fake/key"},
                "spare": {"kind": "fakeembed", "api_key": "secret://fake/key"},
            }
        )
        with pytest.raises(ConfigurationError):
            await _start(settings, task_classes=[CHAT, SEARCH])

    async def test_a_provider_chat_and_embedding_task_classes_share_is_validated(
        self,
    ) -> None:
        settings = _settings(
            providers={
                "primary": {
                    "kind": "fake",
                    "api_key": "secret://fake/key",
                    "behaviour": "no-such-behaviour",
                }
            },
            assignments={
                "chat": {"provider": "primary", "model": "fake-large"},
                "search": {"provider": "primary", "model": "fake-embed"},
            },
        )
        with pytest.raises(ConfigurationError):
            await _start(settings, task_classes=[CHAT, SEARCH])
