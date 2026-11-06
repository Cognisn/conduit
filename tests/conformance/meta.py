"""Helpers for the suites' own tests (meta-tests).

A conformance scenario can skip (``unittest.SkipTest``) as well as pass or fail.
``pytest.raises`` lets a skip through, and pytest then reports the meta-test as skipped,
so a meta-test expecting a failure would hide a scenario that skips instead. These
helpers turn any skip into a failure of the meta-test.
"""

from __future__ import annotations

import re
import unittest
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import pytest


@contextmanager
def fails(*kinds: type[BaseException], match: str | None = None) -> Iterator[None]:
    """The block must fail with ``AssertionError`` (or one of ``kinds``), never skip."""
    expected = kinds or (AssertionError,)
    skipped: str | None = None
    raised: BaseException | None = None
    try:
        yield
    except unittest.SkipTest as skip:
        skipped = str(skip)
    except expected as error:
        raised = error
    if skipped is not None:
        pytest.fail(f"the scenario skipped instead of failing: {skipped}")
    if raised is None:
        pytest.fail(f"the scenario did not fail with {', '.join(k.__name__ for k in expected)}")
    if match is not None and not re.search(match, str(raised)):
        pytest.fail(f"the failure {str(raised)!r} does not match {match!r}")


def passes(scenario: Callable[[], None]) -> None:
    """Run ``scenario``; it must pass, never skip."""
    skipped: str | None = None
    try:
        scenario()
    except unittest.SkipTest as skip:
        skipped = str(skip)
    if skipped is not None:
        pytest.fail(f"the scenario skipped instead of passing: {skipped}")
