"""A pytest plugin for the skip guard, loaded in a child pytest by ``test_skip_reasons.py``.

- With ``CONDUIT_TEST_COLLECT_OUTPUT`` set (and ``--collect-only``), it writes each
  collected item's node id, whether it is marked ``integration``, and the needs its
  ``live_needs`` markers state, as JSON. Nothing runs.
- With ``CONDUIT_TEST_OUTCOME_OUTPUT`` set, it writes each test's outcome (``passed``,
  ``skipped``, ``error`` in set-up, ``failed`` in the call) and its message, as JSON.
  The guard runs it only on a synthetic test file, never on a live test.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.live.conftest import needs_of

COLLECT_OUTPUT = "CONDUIT_TEST_COLLECT_OUTPUT"
OUTCOME_OUTPUT = "CONDUIT_TEST_OUTCOME_OUTPUT"

_outcomes: dict[str, dict[str, str]] = {}


def pytest_collection_finish(session: pytest.Session) -> None:
    target = os.environ.get(COLLECT_OUTPUT)
    if not target:
        return
    found = [
        {
            "nodeid": item.nodeid,
            "integration": item.get_closest_marker("integration") is not None,
            "needs": [
                {"label": need.label, "groups": [list(group) for group in need.groups]}
                for need in needs_of(item)
            ],
        }
        for item in session.items
    ]
    Path(target).write_text(json.dumps(found), encoding="utf-8", newline="\n")


def _message(report: pytest.TestReport) -> str:
    longrepr = report.longrepr
    if isinstance(longrepr, tuple):  # a skip: (path, line, "Skipped: reason")
        return str(longrepr[2])
    return str(longrepr) if longrepr is not None else ""


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if not os.environ.get(OUTCOME_OUTPUT):
        return
    name = report.nodeid.split("::")[-1]
    if report.skipped:
        _outcomes[name] = {"outcome": "skipped", "message": _message(report)}
    elif report.failed:
        outcome = "error" if report.when != "call" else "failed"
        _outcomes[name] = {"outcome": outcome, "message": _message(report)}
    elif report.when == "call":
        _outcomes.setdefault(name, {"outcome": "passed", "message": ""})


def pytest_sessionfinish(session: pytest.Session) -> None:
    target = os.environ.get(OUTCOME_OUTPUT)
    if target:
        Path(target).write_text(json.dumps(_outcomes), encoding="utf-8", newline="\n")
