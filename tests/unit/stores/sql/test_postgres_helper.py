"""The Postgres test helper: the Postgres CI job must never pass by skipping.

These tests never reach Docker: testcontainers' Postgres class is replaced by a
stand-in, so they behave the same on machines with and without a Docker engine.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("testcontainers", reason="the dev extra is not installed")

import testcontainers.community.postgres as tc_postgres

from tests.unit.stores.sql import postgres


class _Container:
    """Records what the helper asks for; ``start`` fails as a missing Docker engine does."""

    made: list[_Container]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.ports: dict[str, Any] = {"5432": None}
        self.stopped = False
        type(self).made.append(self)

    def start(self) -> _Container:
        raise ConnectionError("no Docker engine")

    def stop(self) -> None:
        self.stopped = True


class _NoClient:
    """Fails as soon as it is made, as testcontainers does on a machine with no Docker."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ConnectionError("no Docker socket")


@pytest.fixture
def container(monkeypatch: pytest.MonkeyPatch) -> type[_Container]:
    _Container.made = []
    monkeypatch.setattr(tc_postgres, "PostgresContainer", _Container)
    monkeypatch.delenv(postgres.MODE_VARIABLE, raising=False)
    monkeypatch.delenv(postgres.PORT_VARIABLE, raising=False)
    return _Container


def _enter() -> None:
    with postgres.postgres_server():
        pass


def test_require_fails_when_no_container_starts(
    container: type[_Container], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(postgres.MODE_VARIABLE, "require")
    # Catch every outcome: a skip must count as this test failing, not as it skipping.
    with pytest.raises(BaseException) as raised:
        _enter()
    assert raised.type is pytest.fail.Exception
    assert "require" in str(raised.value) and "ConnectionError" in str(raised.value)
    assert container.made[0].stopped


def test_require_fails_when_no_client_can_be_made(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tc_postgres, "PostgresContainer", _NoClient)
    monkeypatch.setenv(postgres.MODE_VARIABLE, "require")
    with pytest.raises(BaseException) as raised:
        _enter()
    assert raised.type is pytest.fail.Exception


def test_auto_skips_with_the_reason(container: type[_Container]) -> None:
    with pytest.raises(pytest.skip.Exception, match="ConnectionError"):
        _enter()


def test_a_client_that_cannot_reach_docker_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    # On macOS runners the Docker client fails as the container object is made.
    monkeypatch.setattr(tc_postgres, "PostgresContainer", _NoClient)
    monkeypatch.delenv(postgres.MODE_VARIABLE, raising=False)
    with pytest.raises(pytest.skip.Exception, match="ConnectionError"):
        _enter()


def test_skip_skips_without_making_a_container(
    container: type[_Container], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(postgres.MODE_VARIABLE, "skip")
    with pytest.raises(pytest.skip.Exception, match="Postgres CI job"):
        _enter()
    assert container.made == []


def test_an_unknown_mode_is_a_usage_error(
    container: type[_Container], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(postgres.MODE_VARIABLE, "sometimes")
    with pytest.raises(pytest.UsageError):
        _enter()


def test_the_port_is_bound_to_loopback_only(
    container: type[_Container], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(postgres.PORT_VARIABLE, "22233")
    with pytest.raises(pytest.skip.Exception):
        _enter()
    monkeypatch.delenv(postgres.PORT_VARIABLE)
    with pytest.raises(pytest.skip.Exception):
        _enter()
    assert [c.ports for c in container.made] == [
        {"5432": ("127.0.0.1", 22233)},
        {"5432": ("127.0.0.1",)},
    ]


def test_the_watchdog_ends_a_hung_process() -> None:
    script = (
        "import time\n"
        "from tests.unit.stores.sql.postgres import watchdog\n"
        "with watchdog(1):\n"
        "    time.sleep(60)\n"
    )
    started = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        cwd=Path(__file__).resolve().parents[4],
    )
    assert result.returncode != 0
    assert time.monotonic() - started < 30
    assert "Timeout" in result.stderr


def test_the_watchdog_is_cancelled_after_the_test() -> None:
    with postgres.watchdog(1):
        pass
    time.sleep(1.5)  # the process is still alive
