"""The sync facade leaves no pending task to be garbage-collected after its loop closes.

asyncio logs "Task was destroyed but it is pending!" when a task still pending is
collected. Closing a ``LoopThread`` cancels and unwinds every call it can; a call that
ignores cancellation for good is kept referenced, deliberately, so it is never collected
while pending.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from conduit.core import facade
from conduit.core.errors import FacadeError
from conduit.core.facade import LoopThread

DESTROYED = "Task was destroyed but it is pending"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _destroyed(caplog: pytest.LogCaptureFixture) -> list[str]:
    gc.collect()
    return [r.getMessage() for r in caplog.records if DESTROYED in r.getMessage()]


def _wait_in_thread(owned: LoopThread, awaitable: Any) -> tuple[threading.Thread, list[object]]:
    outcome: list[object] = []

    def wait() -> None:
        try:
            outcome.append(owned.run(awaitable, timeout=None))
        except BaseException as error:
            outcome.append(error)

    caller = threading.Thread(target=wait, name="caller", daemon=True)
    caller.start()
    return caller, outcome


class TestNothingIsLeftPending:
    def test_a_call_ignoring_cancellation_is_kept_not_destroyed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.ERROR, logger="asyncio")
        owned = LoopThread()
        started = threading.Event()

        async def stubborn() -> None:
            started.set()
            while True:
                try:
                    await asyncio.sleep(3600)
                except asyncio.CancelledError:
                    continue  # ignores every cancellation

        before = set(facade._ABANDONED)
        caller, outcome = _wait_in_thread(owned, stubborn())
        assert started.wait(5)
        owned.close(join_timeout=0.3)
        caller.join(5)
        assert len(outcome) == 1 and isinstance(outcome[0], FacadeError)
        kept = facade._ABANDONED - before
        assert kept and all(not task.done() for task in kept), "the stubborn call was let go"
        del owned, caller, outcome
        assert _destroyed(caplog) == []

    def test_a_task_started_during_shutdown_is_cancelled_and_unwound(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A call's cleanup starts a task while the shutdown runs, after the shutdown
        # took its list of tasks to cancel: only the sweep after the loop stops reaches it.
        caplog.set_level(logging.ERROR, logger="asyncio")
        owned = LoopThread()
        started = threading.Event()
        cleaned: list[str] = []
        spawned: list[asyncio.Task[None]] = []

        async def helper() -> None:
            try:
                await asyncio.sleep(3600)
            finally:
                cleaned.append("helper cleaned")

        async def call() -> None:
            started.set()
            try:
                await asyncio.sleep(3600)
            finally:
                spawned.append(asyncio.get_running_loop().create_task(helper()))
                cleaned.append("call cleaned")

        caller, outcome = _wait_in_thread(owned, call())
        assert started.wait(5)
        before = set(facade._ABANDONED)
        owned.close(join_timeout=1)
        caller.join(5)
        assert len(outcome) == 1 and isinstance(outcome[0], FacadeError)
        assert sorted(cleaned) == ["call cleaned", "helper cleaned"], "a task was not unwound"
        assert before == facade._ABANDONED  # nothing needed keeping
        del owned, caller, outcome
        assert _destroyed(caplog) == []


def test_the_facade_tests_log_no_destroyed_task_under_dev_mode() -> None:
    env = {**os.environ, "PYTHONDEVMODE": "1", "PYTHONWARNINGS": "default"}
    env.pop("PYTEST_ADDOPTS", None)
    run = subprocess.run(
        [
            sys.executable,
            "-X",
            "dev",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            # Without pytest's logging plugin, asyncio's records reach stderr, which -s
            # passes through, including those logged as the interpreter exits.
            "-p",
            "no:logging",
            "-s",
            "tests/unit/core/test_facade.py",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    output = run.stdout + run.stderr
    assert run.returncode == 0, output[-4000:]
    assert DESTROYED not in output, output[-4000:]
