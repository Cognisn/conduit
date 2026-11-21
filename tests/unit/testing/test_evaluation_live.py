"""The seed evaluation cases against a real model. Marked integration: never in the
default run, and skipped, naming the variables, when the live variables are unset.

Run only on the maintainer's word, with the live suite's variables (``tests/live``):

    CONDUIT_LIVE_ANTHROPIC_API_KEY=... pytest -m integration tests/unit/testing

``CONDUIT_LIVE_ANTHROPIC_MODEL`` chooses the model (the cheapest Claude by default).
Both task classes the cases use are assigned to it.
"""

from __future__ import annotations

import pytest

from conduit.core.runtime import ConduitSettings, start
from conduit.core.tasks import CapabilityFloor, TaskClass
from conduit.providers import builtin_factories
from conduit.testing.evaluation import LiveHarness
from tests.live import needs
from tests.live.needs import check, live_env, value
from tests.live.targets import anthropic_direct, resolver
from tests.unit.testing.evaluation_cases import CASES, ApproveReadsOnly

pytestmark = pytest.mark.integration

#: The cheapest Claude in the default catalogue.
DEFAULT_MODEL = "claude-haiku-4-5"
TASK_CLASSES = [
    TaskClass(name="chat", floor=CapabilityFloor(tools=True)),
    TaskClass(name="review", floor=CapabilityFloor(tools=True)),
]


async def test_the_live_cases_pass_on_a_real_model() -> None:
    environ = live_env()
    check((needs.ANTHROPIC_DIRECT,), environ)
    model = value(environ, "ANTHROPIC_MODEL", DEFAULT_MODEL)
    settings = ConduitSettings.model_validate(
        {
            "providers": {"live": anthropic_direct()},
            "assignments": {
                task.name: {"provider": "live", "model": model, "max_output_tokens": 1024}
                for task in TASK_CLASSES
            },
        }
    )
    runtime = await start(
        settings,
        task_classes=TASK_CLASSES,
        factories=builtin_factories(),
        secrets=resolver(environ),
    )
    try:
        report = await LiveHarness(runtime, approvals=ApproveReadsOnly()).run(CASES)
    finally:
        await runtime.aclose()
    print(report.summary())
    assert report.failures() == {}
    assert report.ran == sum(1 for case in CASES if case.live)
