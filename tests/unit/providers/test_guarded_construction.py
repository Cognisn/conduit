"""Every chat adapter builds its client in one guarded step.

A constructor that raises surfaces as a fixed ``ConfigurationError`` naming the adapter
and the exception's class only, from ``create_client`` and through the registry, with
nothing chained. The resolved key (a placeholder) never reaches the error's text, a
long traceback with locals, or a log record. A child pytest run under
``-vvv -l --tb=long`` proves the same for every chat and embedding case at once, both
from ``create_client`` and through the registry (whose own guard would otherwise hide
an adapter that is not guarded).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.unit.providers.construction_cases import (
    ALL_CASES,
    CHAT_CASES,
    Case,
    check_hidden,
    failure_of,
    ids,
    leaks,
    watch_loggers,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("case", CHAT_CASES, ids=ids(CHAT_CASES))
async def test_the_factory_refuses_with_a_fixed_error(
    case: Case, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    watch_loggers(caplog)
    case.patch(monkeypatch)
    check_hidden(case, await failure_of(case.create_client()), caplog)


@pytest.mark.parametrize("case", CHAT_CASES, ids=ids(CHAT_CASES))
async def test_the_registry_refuses_with_a_fixed_error(
    case: Case, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    watch_loggers(caplog)
    case.patch(monkeypatch)
    check_hidden(case, await failure_of(case.build()), caplog)


CHILD = """
import asyncio

import pytest

from tests.unit.providers.construction_cases import ALL_CASES, ids


@pytest.mark.parametrize("case", ALL_CASES, ids=ids(ALL_CASES))
def test_construction_through_the_registry_fails(case, monkeypatch):
    case.patch(monkeypatch)
    asyncio.run(case.build())


@pytest.mark.parametrize("case", ALL_CASES, ids=ids(ALL_CASES))
def test_construction_by_the_factory_fails(case, monkeypatch):
    case.patch(monkeypatch)
    asyncio.run(case.create_client())
"""


def test_no_key_reaches_a_long_traceback_with_locals(tmp_path: Path) -> None:
    test_file = tmp_path / "test_synthetic_construction.py"
    test_file.write_text(CHILD, encoding="utf-8", newline="\n")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            "-vvv",
            "-l",
            "--tb=long",
            "-rA",
            "--log-level=DEBUG",
            f"--rootdir={tmp_path}",
            str(test_file),
        ],
        cwd=REPO_ROOT,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    # Reduced to booleans first: the output itself never reaches this test's report.
    output = result.stdout + result.stderr
    leaked = leaks(output)
    all_failed = f"{2 * len(ALL_CASES)} failed" in output
    refused = output.count("could not be built") >= 2 * len(ALL_CASES)
    locals_shown = "case = Case(" in output
    del output
    assert all_failed, "not every synthetic construction failed"
    assert refused, "not every failure was the fixed refusal"
    assert locals_shown, "pytest printed no frame locals"
    assert not leaked, "the key or the SDK's text reached pytest's failure output"
