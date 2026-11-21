"""Every embedding adapter builds its client in one guarded step.

The remote embedding factories share their chat factory's ``create_client``; the local
factory has its own. A constructor that raises surfaces as a fixed
``ConfigurationError`` naming the adapter and the exception's class only, with nothing
chained, and the resolved key (a placeholder) reaches no traceback, frame or log
record. ``tests/unit/providers/test_guarded_construction.py`` also runs these cases in a
child pytest under ``-vvv -l --tb=long``.
"""

from __future__ import annotations

import pytest

from tests.unit.providers.construction_cases import (
    EMBEDDING_CASES,
    Case,
    check_hidden,
    failure_of,
    ids,
    watch_loggers,
)


@pytest.mark.parametrize("case", EMBEDDING_CASES, ids=ids(EMBEDDING_CASES))
async def test_the_factory_refuses_with_a_fixed_error(
    case: Case, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    watch_loggers(caplog)
    case.patch(monkeypatch)
    check_hidden(case, await failure_of(case.create_client()), caplog)


@pytest.mark.parametrize("case", EMBEDDING_CASES, ids=ids(EMBEDDING_CASES))
async def test_the_registry_refuses_with_a_fixed_error(
    case: Case, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    watch_loggers(caplog)
    case.patch(monkeypatch)
    check_hidden(case, await failure_of(case.build()), caplog)
