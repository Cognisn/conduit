"""The seed evaluation cases in the default run: every case against the stub provider."""

from __future__ import annotations

from conduit.core.tasks import CapabilityFloor, TaskClass
from conduit.testing.evaluation import StubHarness
from tests.unit.testing.evaluation_cases import CASES, ApproveReadsOnly

TASK_CLASSES = [
    TaskClass(name="chat", floor=CapabilityFloor(tools=True)),
    TaskClass(name="review", floor=CapabilityFloor(tools=True)),
]


async def test_every_seed_case_passes_against_the_stub() -> None:
    harness = StubHarness(task_classes=TASK_CLASSES, approvals=ApproveReadsOnly())
    report = await harness.run(CASES)
    assert report.failures() == {}
    assert (report.ran, report.skipped, report.score) == (len(CASES), 0, 1.0)


def test_case_names_are_unique() -> None:
    names = [case.name for case in CASES]
    assert len(names) == len(set(names))


def test_some_seed_cases_are_live_and_some_are_not() -> None:
    assert {case.live for case in CASES} == {True, False}
