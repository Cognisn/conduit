"""The live suite's gate and report.

Every live test is marked ``integration`` (deselected by the default run's
``-m 'not integration'``) and states its needs with ``live_needs``. The autouse gate
below checks those needs against the process environment before the test body runs and
skips, naming the variables, when one is unmet. A live test without ``live_needs`` fails
here rather than reaching a live service.

At the end of a run the report (``tests/live/report.py``) is printed, and written as
JSON to the file ``CONDUIT_LIVE_REPORT`` names, if it is set.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest

from tests.live.needs import Need, check, value
from tests.live.report import REPORT

LIVE_NEEDS = "live_needs"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"{LIVE_NEEDS}(*needs): the CONDUIT_LIVE_ variables a live test needs (tests/live)",
    )


def needs_of(node: pytest.Item) -> tuple[Need, ...]:
    """Every ``Need`` the item's ``live_needs`` markers name, closest first."""
    found: list[Need] = []
    for marker in node.iter_markers(LIVE_NEEDS):
        found.extend(need for need in marker.args if isinstance(need, Need))
    return tuple(found)


def gate(node: pytest.Item, environ: Mapping[str, str]) -> None:
    """Skip ``node`` unless its needs are met; fail an integration test with none."""
    if node.get_closest_marker("integration") is None:
        return
    check(needs_of(node), environ)


@pytest.fixture(autouse=True)
def live_gate(request: pytest.FixtureRequest) -> Iterator[None]:
    """Applied to every test under tests/live, before any function-scoped fixture it uses.

    Live tests take no fixture of a wider scope that calls a service, so nothing live
    runs before the gate."""
    item = request.node
    assert isinstance(item, pytest.Item)
    gate(item, os.environ)
    yield


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    if not REPORT.lines:
        return
    terminalreporter.section("conduit live report")
    for line in REPORT.render():
        terminalreporter.write_line(line)
    target = value(os.environ, "REPORT")
    if target:
        REPORT.write_json(Path(target))
        terminalreporter.write_line(f"report written to {target}")
