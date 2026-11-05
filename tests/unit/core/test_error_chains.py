"""No conduit error keeps the exception it was raised for (D0040).

Every error the core, the stub and the replay helpers raise while handling another
exception is raised after the handler, so ``__cause__`` and ``__context__`` are both
``None`` and nothing the original carried (a key, pydantic's text, an input value) is
reachable from the error a product catches.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from conduit.core import facade
from conduit.core.catalogue import load_catalogue
from conduit.core.errors import (
    ConfigurationError,
    FacadeError,
    SecretUnavailable,
    UnknownTaskClass,
    UnknownTool,
)
from conduit.core.facade import LoopThread
from conduit.core.registry import ProviderRegistry
from conduit.core.secrets import LiteralSecretResolver, SecretRef
from conduit.core.tasks import Assignments
from conduit.core.tools import ToolCatalogue
from conduit.providers.stub import StubFactory, StubSettings
from conduit.testing.replay import load_recording
from tests.unit.core.test_facade import Service, _stubborn

SECRET_TEXT = "zz-chain-marker-sk-0000zz"


def _unchained(error: BaseException) -> None:
    assert error.__cause__ is None
    assert error.__context__ is None


def _caught(kind: type[BaseException], call: Callable[[], Any]) -> BaseException:
    with pytest.raises(kind) as caught:
        call()
    return caught.value


@pytest.fixture
def loop() -> Iterator[LoopThread]:
    owned = LoopThread(name="conduit-chain-loop")
    yield owned
    owned.close(join_timeout=5)


class TestLookups:
    def test_an_unknown_task_class(self) -> None:
        _unchained(_caught(UnknownTaskClass, lambda: Assignments({}).for_task("triage")))

    def test_an_unknown_tool(self) -> None:
        registry = ToolCatalogue({}).registry_for("triage")
        _unchained(_caught(UnknownTool, lambda: registry.resolve("absent")))

    async def test_an_unheld_secret(self) -> None:
        resolver = LiteralSecretResolver({"held": SECRET_TEXT})
        with pytest.raises(SecretUnavailable) as caught:
            await resolver.resolve(SecretRef("absent"))
        _unchained(caught.value)

    async def test_an_unregistered_provider_kind(self) -> None:
        registry = ProviderRegistry(secrets=LiteralSecretResolver({}))
        with pytest.raises(ConfigurationError) as caught:
            await registry.build("p", {"kind": "absent"})
        _unchained(caught.value)

    def test_an_unknown_stub_script(self) -> None:
        settings = StubSettings(script="absent", not_for_production=True)
        error = _caught(
            ConfigurationError, lambda: StubFactory().create_provider("p", settings, None)
        )
        _unchained(error)


class TestLoading:
    def test_a_catalogue_that_does_not_validate(self) -> None:
        error = _caught(
            ConfigurationError,
            lambda: load_catalogue({"version": "1", "models": [{"model_id": SECRET_TEXT}]}),
        )
        _unchained(error)
        assert SECRET_TEXT not in str(error)
        assert "models.0" in str(error)  # where it failed, never what it held
        assert "type=" not in str(error) and "validation error" not in str(error)

    def test_a_catalogue_file_that_is_not_json(self, tmp_path: Path) -> None:
        path = tmp_path / "catalogue.json"
        path.write_text('{"version": "' + SECRET_TEXT, encoding="utf-8")
        error = _caught(ConfigurationError, lambda: load_catalogue(path))
        _unchained(error)
        assert SECRET_TEXT not in str(error)

    def test_a_recording_that_does_not_validate(self, tmp_path: Path) -> None:
        path = tmp_path / "recording.json"
        path.write_text(json.dumps({"name": SECRET_TEXT, "nonsense": SECRET_TEXT}), "utf-8")
        error = _caught(ConfigurationError, lambda: load_recording(path))
        _unchained(error)
        assert SECRET_TEXT not in str(error)


class TestFacade:
    def test_a_timeout(self, loop: LoopThread) -> None:
        error = _caught(TimeoutError, lambda: loop.run(Service().slow(30), timeout=0.05))
        _unchained(error)

    def test_a_stranded_step(self, loop: LoopThread, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(facade, "UNWIND_GRACE_S", 0.1)
        iterator = loop.iterate(_stubborn([]), timeout=0.1)
        assert next(iterator) == 1
        _unchained(_caught(TimeoutError, lambda: next(iterator)))

    def test_a_close_while_a_caller_waits(self) -> None:
        owned = LoopThread()
        outcome: list[BaseException] = []

        def wait() -> None:
            try:
                owned.run(Service().slow(30), timeout=None)
            except BaseException as error:
                outcome.append(error)

        caller = threading.Thread(target=wait, daemon=True)
        caller.start()
        deadline = time.monotonic() + 5
        while owned.thread_id is None and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.1)
        owned.close(join_timeout=5)
        caller.join(5)
        assert len(outcome) == 1 and isinstance(outcome[0], FacadeError)
        _unchained(outcome[0])

    async def test_a_timeout_on_the_loop(self) -> None:
        """The deadline's own TimeoutError is not kept by the one conduit raises."""
        with pytest.raises(TimeoutError, match="did not finish") as caught:
            await facade._within(Service().slow(30), 0.05)
        _unchained(caught.value)
