"""The Postgres test helper: the Postgres CI job must never pass by skipping."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("testcontainers", reason="the dev extra is not installed")

from testcontainers.community.postgres import PostgresContainer

from tests.unit.stores.sql import postgres


def _no_docker(self: PostgresContainer) -> PostgresContainer:
    raise ConnectionError("no Docker engine")


def _enter() -> None:
    with postgres.postgres_server():
        pass


def test_require_fails_when_no_container_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(postgres.MODE_VARIABLE, "require")
    monkeypatch.setattr(PostgresContainer, "start", _no_docker)
    # Catch every outcome: a skip must count as this test failing, not as it skipping.
    with pytest.raises(BaseException) as raised:
        _enter()
    assert raised.type is pytest.fail.Exception
    assert "require" in str(raised.value) and "ConnectionError" in str(raised.value)


def test_auto_skips_with_the_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(postgres.MODE_VARIABLE, raising=False)
    monkeypatch.setattr(PostgresContainer, "start", _no_docker)
    with pytest.raises(pytest.skip.Exception, match="ConnectionError"):
        _enter()


def test_skip_skips_without_starting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(postgres.MODE_VARIABLE, "skip")
    monkeypatch.setattr(PostgresContainer, "start", _no_docker)
    with pytest.raises(pytest.skip.Exception, match="Postgres CI job"):
        _enter()


def test_an_unknown_mode_is_a_usage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(postgres.MODE_VARIABLE, "sometimes")
    with pytest.raises(pytest.UsageError):
        _enter()


def test_the_port_is_bound_to_loopback_only(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[object] = []

    def capture(self: PostgresContainer) -> PostgresContainer:
        seen.append(dict(self.ports))
        raise ConnectionError("stop here")

    monkeypatch.delenv(postgres.MODE_VARIABLE, raising=False)
    monkeypatch.setenv(postgres.PORT_VARIABLE, "22233")
    monkeypatch.setattr(PostgresContainer, "start", capture)
    with pytest.raises(pytest.skip.Exception):
        _enter()
    monkeypatch.delenv(postgres.PORT_VARIABLE)
    with pytest.raises(pytest.skip.Exception):
        _enter()
    assert seen == [{"5432": ("127.0.0.1", 22233)}, {"5432": ("127.0.0.1",)}]


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
