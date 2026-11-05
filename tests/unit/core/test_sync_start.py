"""The sync start helper: conduit started and used from synchronous code (Q0015).

``start_sync`` runs ``start`` on a conduit loop and returns a ``SyncRuntime``. Its task
bindings and providers are sync views: a sync user never holds an async object, and
every call runs on the loop that built the providers.
"""

from __future__ import annotations

import inspect
import threading
from collections.abc import Iterator, Mapping
from typing import Any

import pytest
from pydantic import SecretStr

from conduit.core.errors import FacadeError, ModelNotAssigned, UndeclaredModel
from conduit.core.facade import (
    LoopThread,
    SyncProvider,
    SyncRuntime,
    SyncTaskBinding,
    start_sync,
)
from conduit.core.registry import MeteredProvider
from conduit.core.runtime import ConduitSettings, Runtime, TaskBinding
from conduit.core.secrets import LiteralSecretResolver
from conduit.core.settings import ProviderSettings
from conduit.core.streaming import StreamStopped, TextDelta
from conduit.core.tasks import TaskClass
from conduit.core.types import Message, ModelResponse
from tests.unit.core.fakes import CATALOGUE, FakeClient, FakeFactory

SECRETS = LiteralSecretResolver({"fake/key": "not-a-real-key"})
CHAT = TaskClass(name="chat")
SUMMARY = TaskClass(name="summary")


def _settings(model: str = "fake-large") -> ConduitSettings:
    return ConduitSettings.model_validate(
        {
            "providers": {"primary": {"kind": "fake", "api_key": "secret://fake/key"}},
            "assignments": {
                "chat": {"provider": "primary", "model": model},
                "summary": {"provider": "primary", "model": "fake-small"},
            },
        }
    )


class _ThreadRecorder(FakeFactory):
    def __init__(self) -> None:
        super().__init__()
        self.threads: list[int] = []

    def create_client(self, settings: ProviderSettings, secrets: Mapping[str, SecretStr]) -> Any:
        self.threads.append(threading.get_ident())
        return FakeClient(secrets["api_key"])


@pytest.fixture
def loop() -> Iterator[LoopThread]:
    owned = LoopThread(name="conduit-sync-start-loop")
    yield owned
    owned.close(join_timeout=5)


def _start(loop: LoopThread, factory: FakeFactory | None = None) -> SyncRuntime:
    return start_sync(
        _settings(),
        task_classes=[CHAT, SUMMARY],
        factories=[factory or FakeFactory()],
        secrets=SECRETS,
        catalogue=CATALOGUE,
        loop=loop,
    )


def _async_surface(obj: object) -> list[str]:
    """Public attributes of ``obj`` that are, or return without calling, async objects."""
    found = []
    for name in dir(obj):
        if name.startswith("_"):
            continue
        value = getattr(obj, name)
        if (
            inspect.iscoroutinefunction(value)
            or inspect.isasyncgenfunction(value)
            or isinstance(value, MeteredProvider | Runtime | TaskBinding)
            or (
                isinstance(value, Mapping)
                and any(isinstance(v, MeteredProvider) for v in value.values())
            )
        ):
            found.append(name)
    return found


class TestStartSync:
    def test_a_task_runs_end_to_end_from_sync_code(self, loop: LoopThread) -> None:
        runtime = _start(loop)
        try:
            binding = runtime.for_task("chat")
            response = binding.complete(binding.request(messages=[Message.user("hello")]))
            assert isinstance(response, ModelResponse)
            assert response.text == "echo: hello"
        finally:
            runtime.close()

    def test_streaming_is_a_plain_iterator(self, loop: LoopThread) -> None:
        with _start(loop) as runtime:
            binding = runtime.for_task("chat")
            events = list(binding.stream(binding.request(messages=[Message.user("hi")])))
        assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "echo: hi"
        assert isinstance(events[-1], StreamStopped)

    def test_no_sync_view_hands_out_an_async_object(self, loop: LoopThread) -> None:
        with _start(loop) as runtime:
            binding = runtime.for_task("chat")
            assert isinstance(binding, SyncTaskBinding)
            assert isinstance(binding.provider, SyncProvider)
            assert set(runtime.providers) == {"primary"}
            assert all(isinstance(p, SyncProvider) for p in runtime.providers.values())
            for view in (runtime, binding, binding.provider, runtime.providers["primary"]):
                assert _async_surface(view) == [], type(view).__name__

    def test_the_providers_are_built_and_called_on_the_loop(self, loop: LoopThread) -> None:
        factory = _ThreadRecorder()
        with _start(loop, factory) as runtime:
            binding = runtime.for_task("summary")
            binding.complete(binding.request(messages=[Message.user("x")]))
        assert factory.threads == [loop.thread_id]
        assert factory.threads[0] != threading.get_ident()

    def test_the_assignment_still_holds(self, loop: LoopThread) -> None:
        with _start(loop) as runtime:
            binding = runtime.for_task("summary")
            assert binding.assignment.model == "fake-small"
            other = binding.request(messages=[Message.user("x")]).model_copy(
                update={"model": "fake-large"}
            )
            with pytest.raises(ModelNotAssigned):
                binding.complete(other)

    def test_a_refused_start_raises_and_builds_nothing(self, loop: LoopThread) -> None:
        factory = FakeFactory()
        with pytest.raises(UndeclaredModel):
            start_sync(
                _settings(model="fake-absent"),
                task_classes=[CHAT, SUMMARY],
                factories=[factory],
                secrets=SECRETS,
                catalogue=CATALOGUE,
                loop=loop,
            )
        assert factory.clients == []

    def test_close_closes_every_provider(self, loop: LoopThread) -> None:
        factory = FakeFactory()
        runtime = _start(loop, factory)
        runtime.close()
        assert factory.providers and all(p.closed for p in factory.providers)
        runtime.close()  # a second close does nothing

    def test_it_is_refused_on_the_loops_own_thread(self, loop: LoopThread) -> None:
        async def nested() -> None:
            _start(loop)

        with pytest.raises(FacadeError):
            loop.run(nested())

    def test_timeouts_apply_to_the_views(self, loop: LoopThread) -> None:
        with _start(loop) as runtime:
            quick = runtime.with_timeout(5)
            assert quick.timeout == 5
            assert quick.for_task("chat").provider.timeout == 5
