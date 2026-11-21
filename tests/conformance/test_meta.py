"""The meta-test helpers themselves: each must judge a pass, a skip, a failure and a
failure's message the right way, or the suites' own tests prove nothing."""

from __future__ import annotations

import unittest

import pytest

from tests.conformance.meta import fails, passes

Failed = pytest.fail.Exception


def _passing() -> None:
    return None


def _skipping() -> None:
    raise unittest.SkipTest("the scenario skipped")


def _failing() -> None:
    raise AssertionError("the expected failure: run 3 out of order")


def test_fails_accepts_a_failure() -> None:
    with fails():
        _failing()


def test_fails_refuses_a_skip() -> None:
    with pytest.raises(Failed, match="skipped instead of failing"), fails():
        _skipping()


def test_fails_refuses_a_block_that_passes() -> None:
    with pytest.raises(Failed, match="did not fail with AssertionError"), fails():
        _passing()


def test_fails_accepts_a_matching_message() -> None:
    with fails(match="out of order"):
        _failing()


def test_fails_refuses_a_message_that_does_not_match() -> None:
    with pytest.raises(Failed, match="does not match"), fails(match="keeps none"):
        _failing()


def test_fails_accepts_the_kinds_it_is_given() -> None:
    with fails(AssertionError, KeyError):
        raise KeyError("absent")


def test_fails_lets_an_unexpected_kind_through() -> None:
    with pytest.raises(ValueError), fails():
        raise ValueError("not an assertion")


def test_passes_accepts_a_pass() -> None:
    passes(_passing)


def test_passes_refuses_a_skip() -> None:
    with pytest.raises(Failed, match="skipped instead of passing"):
        passes(_skipping)


def test_passes_lets_a_failure_through() -> None:
    with pytest.raises(AssertionError, match="expected failure"):
        passes(_failing)
