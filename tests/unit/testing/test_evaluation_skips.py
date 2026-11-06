"""The live evaluation test skips, naming the variables it needs, when they are unset:
proved here in the default run against an empty environment, without calling a model."""

from __future__ import annotations

import pytest

from tests.live import needs
from tests.live.needs import LiveEnv, check


def test_the_live_evaluation_skips_with_a_stated_reason_when_unconfigured() -> None:
    with pytest.raises(pytest.skip.Exception) as skipped:
        check((needs.ANTHROPIC_DIRECT,), LiveEnv({}))
    assert "CONDUIT_LIVE_ANTHROPIC_API_KEY" in str(skipped.value)


def test_the_live_evaluation_is_marked_integration() -> None:
    from tests.unit.testing import test_evaluation_live

    assert test_evaluation_live.pytestmark.name == "integration"
