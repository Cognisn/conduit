"""A throwaway Postgres server for the sql extra's tests, through testcontainers.

``CONDUIT_POSTGRES_TESTS`` chooses what happens:

- ``auto`` (the default): run against a container when Docker can start one, else
  skip with the reason;
- ``skip``: skip (the CI test matrix, where the Postgres job runs these tests);
- ``require``: run, and fail rather than skip when no container starts (the Postgres
  CI job, so it can never pass by skipping).

The published port is bound to 127.0.0.1 only. ``CONDUIT_POSTGRES_TEST_PORT`` fixes
the host port (a workstation's own slot port); otherwise Docker picks a free one.
Credentials are fixed, development-only values.
"""

from __future__ import annotations

import contextlib
import faulthandler
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

MODE_VARIABLE = "CONDUIT_POSTGRES_TESTS"
PORT_VARIABLE = "CONDUIT_POSTGRES_TEST_PORT"
IMAGE = "postgres:17-alpine"
# Not "conduit": Postgres puts a schema named after the user ("$user") first on the
# search path, which would make conduit's schema the default one.
USER = "conduit_tester"
PASSWORD = "conduit-dev-only"
DATABASE = "conduit"
#: Seconds one test may take before the watchdog ends the run.
WATCHDOG_SECONDS = 300


def mode() -> str:
    value = os.environ.get(MODE_VARIABLE, "auto").strip().lower() or "auto"
    if value not in {"auto", "skip", "require"}:
        raise pytest.UsageError(f"{MODE_VARIABLE} must be auto, skip or require")
    return value


def _unavailable(reason: str) -> None:
    if mode() == "require":
        pytest.fail(f"{MODE_VARIABLE}=require, but {reason}", pytrace=False)
    pytest.skip(reason)


@contextmanager
def postgres_server() -> Iterator[str]:
    """Start a Postgres container and yield its asyncpg URL; stop it afterwards."""
    if mode() == "skip":
        pytest.skip(f"{MODE_VARIABLE}=skip: the Postgres CI job runs these tests")
    try:
        from testcontainers.community.postgres import PostgresContainer
    except ImportError:
        _unavailable("testcontainers is not installed (the dev extra)")
        raise  # pragma: no cover - _unavailable raises
    try:
        # Making the container object already reaches Docker (it fails here on macOS
        # runners, which have no Docker).
        container = PostgresContainer(
            IMAGE, username=USER, password=PASSWORD, dbname=DATABASE, driver="asyncpg"
        )
    except Exception as error:
        _unavailable(f"no Docker client could be made ({type(error).__name__})")
        raise  # pragma: no cover - _unavailable raises
    port = os.environ.get(PORT_VARIABLE)
    # testcontainers binds every interface; docker-py takes (address, port), or
    # (address,) for a free port on that address only.
    ports: dict[str, Any] = container.ports
    ports["5432"] = ("127.0.0.1", int(port)) if port else ("127.0.0.1",)
    try:
        container.start()
    except Exception as error:
        # Docker is absent, or runs Windows containers (GitHub's Windows runners).
        with contextlib.suppress(Exception):  # nothing may have started to stop
            container.stop()
        _unavailable(f"no Postgres container could be started ({type(error).__name__})")
    try:
        yield container.get_connection_url(host="127.0.0.1")
    finally:
        container.stop()


@contextmanager
def watchdog(seconds: int = WATCHDOG_SECONDS) -> Iterator[None]:
    """End the test process, with every thread's traceback, if a test outlives ``seconds``.

    An asyncio timeout cannot fire when the event loop itself is blocked, so a hang in
    a database test would otherwise wait for the CI job's own limit.
    """
    faulthandler.dump_traceback_later(seconds, exit=True)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()
