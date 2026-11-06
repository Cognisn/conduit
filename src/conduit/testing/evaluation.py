"""The evaluation harness: cases run through the agent runner, judged by checks.

A case is a task, an optional stub script and tools, and checks over what happened:
the run's result, every event it streamed and every event delivered to its sink, and the
tools the executor was asked to run. The same cases run two ways:

- ``StubHarness``: the stub provider plays each case's script (the model behaviour the
  case is about), in the default test run, with no network;
- ``LiveHarness``: on a runtime the product started with its configured assignments,
  only in tests marked ``integration``, so only when someone asks for a live run. It
  runs the cases marked ``live`` and skips the others, saying why.

A check is any callable taking an ``Observation`` and returning ``None`` when it holds,
or a short reason when it does not. The shipped checks cover completion, failure,
hand-off, an output field, the tools executed, and text that must never reach an event.
Failure reasons name identifiers, counts and outcomes only: a report never repeats
content, so it is safe to print or keep.

The harness supplies mechanism only. The task (its system prompt, contract and
hand-offs), the tools, the approval handler and the capture policy are the product's.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from conduit.agent.events import RunEvent
from conduit.agent.results import RunFailure, RunResult, RunStatus
from conduit.agent.runner import AgentRunner, AgentTask, Bindings
from conduit.core.backends import ApprovalHandler
from conduit.core.events import CapturePolicy, Event, MemoryEventSink
from conduit.core.tasks import CapabilityFloor, TaskClass
from conduit.core.tools import TaskTools, ToolCatalogue
from conduit.providers.stub import StubScript, StubTurn, scripted
from conduit.testing._stub import (
    DeclaredTool,
    RecordingExecutor,
    check_tool_name,
    start_stub_runtime,
)

__all__ = [
    "Check",
    "EvalCase",
    "EvalHarness",
    "EvalOutcome",
    "EvalReport",
    "EvalTool",
    "LiveHarness",
    "Observation",
    "StubHarness",
    "completed",
    "executed",
    "failed",
    "handed_off",
    "never_executed",
    "no_text_in_events",
    "output_field",
]

#: Why the live harness skips a case not marked live.
NOT_LIVE = "not marked live: its checks hold only for the scripted stub"
_OK = '{"status": "ok"}'


@dataclass(frozen=True, slots=True)
class EvalTool:
    """A tool a case's task may call, answered with a canned ``response``.

    ``task_classes`` defaults to every task class the case's task can run on (its own and
    its hand-off targets). ``requires_approval`` sends the call to the harness's approval
    handler first, as the runner does for a product's marked tool.
    """

    name: str
    description: str = "A tool for an evaluation case."
    response: str = _OK
    task_classes: frozenset[str] = frozenset()
    input_schema: Mapping[str, Any] = field(default_factory=lambda: {"type": "object"})
    requires_approval: bool = False

    def __post_init__(self) -> None:
        check_tool_name(self.name)


@dataclass(frozen=True, slots=True)
class Observation:
    """What a case's run did.

    ``events`` are the run events it streamed, in order (text deltas included);
    ``delivered`` the events its sink received (the runner's and, on the stub, the
    provider registry's), after the capture policy; ``executed`` the tools the executor
    ran, in order.
    """

    result: RunResult
    events: tuple[RunEvent, ...]
    delivered: tuple[Event, ...]
    executed: tuple[str, ...]


#: A check: ``None`` when it holds, otherwise why not, naming no content.
Check = Callable[[Observation], str | None]


@dataclass(frozen=True)
class EvalCase:
    """One case. ``script`` is the stub's behaviour (turns played in order, or a
    ``StubScript``); ``stub_checks`` hold only for that script and are not applied live;
    ``live`` marks a case whose ``checks`` also hold for a real model."""

    name: str
    task: AgentTask
    checks: tuple[Check, ...]
    script: StubScript | Sequence[StubTurn] | None = None
    tools: tuple[EvalTool, ...] = ()
    stub_checks: tuple[Check, ...] = ()
    live: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("an evaluation case needs a name")
        if not self.checks:
            raise ValueError(f"case {self.name} needs at least one check")

    def task_classes(self) -> list[str]:
        """The task classes the case's run may use: its own, then its hand-off targets."""
        return [self.task.task_class, *sorted(self.task.hand_offs - {self.task.task_class})]


@dataclass(frozen=True, slots=True)
class EvalOutcome:
    """One case's outcome. ``skipped`` says why a case did not run; a skipped case has
    not passed. ``result`` is ``None`` when the run raised or did not run."""

    case: str
    passed: bool
    failures: tuple[str, ...] = ()
    result: RunResult | None = None
    skipped: str | None = None


@dataclass(frozen=True, slots=True)
class EvalReport:
    outcomes: tuple[EvalOutcome, ...]

    @property
    def passed(self) -> int:
        return sum(1 for o in self.outcomes if o.passed)

    @property
    def skipped(self) -> int:
        return sum(1 for o in self.outcomes if o.skipped is not None)

    @property
    def ran(self) -> int:
        return len(self.outcomes) - self.skipped

    @property
    def failed(self) -> int:
        return self.ran - self.passed

    @property
    def score(self) -> float:
        """The share of the cases that ran which passed; 0.0 when none ran."""
        return self.passed / self.ran if self.ran else 0.0

    def failures(self) -> dict[str, tuple[str, ...]]:
        """Each failed case's reasons, by case name."""
        return {o.case: o.failures for o in self.outcomes if not o.passed and o.skipped is None}

    def summary(self) -> str:
        """One line per case and a total. Names and reasons only, never content."""
        lines = []
        for o in self.outcomes:
            if o.skipped is not None:
                lines.append(f"{o.case}: skipped ({o.skipped})")
            elif o.passed:
                lines.append(f"{o.case}: passed")
            else:
                lines.append(f"{o.case}: failed ({'; '.join(o.failures)})")
        lines.append(
            f"{self.passed}/{self.ran} passed, {self.skipped} skipped, score {self.score:.2f}"
        )
        return "\n".join(lines)


class EvalHarness:
    """Runs cases one at a time and judges them. ``StubHarness`` and ``LiveHarness``
    say where each case's run goes; ``approvals`` and ``capture`` are passed to the
    runner as a product would pass its own."""

    stub: bool = False

    def __init__(
        self,
        *,
        approvals: ApprovalHandler | None = None,
        capture: CapturePolicy | None = None,
    ) -> None:
        self._approvals = approvals
        self._capture = capture

    async def run(self, cases: Iterable[EvalCase]) -> EvalReport:
        chosen = list(cases)
        names = [case.name for case in chosen]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f"case names must be unique: {', '.join(repeated)}")
        return EvalReport(tuple([await self.run_case(case) for case in chosen]))

    async def run_case(self, case: EvalCase) -> EvalOutcome:
        raise NotImplementedError  # pragma: no cover - each harness defines it

    async def _observe(
        self,
        case: EvalCase,
        bindings: Bindings,
        sink: MemoryEventSink,
    ) -> EvalOutcome:
        executor = RecordingExecutor({tool.name: tool.response for tool in case.tools})
        everywhere = frozenset(case.task_classes())
        catalogue = ToolCatalogue.from_tools(
            DeclaredTool(
                name=tool.name,
                description=tool.description,
                task_classes=tool.task_classes or everywhere,
                input_schema=tool.input_schema,
                requires_approval=tool.requires_approval,
            )
            for tool in case.tools
        )
        tools = {
            task_class: TaskTools(catalogue.registry_for(task_class), executor)
            for task_class in everywhere
        }
        runner = AgentRunner(
            bindings,
            tools=tools if case.tools else None,
            approvals=self._approvals,
            events=sink,
            capture=self._capture,
        )
        raised: str | None = None
        streamed: list[RunEvent] = []
        result: RunResult | None = None
        try:
            run = runner.start(case.task)
            async for event in run:
                streamed.append(event)
            result = run.result
        except Exception as error:  # a run that raises fails its case; cancellation does not
            raised = type(error).__name__
        if raised is not None or result is None:
            return EvalOutcome(case.name, False, (f"the run raised {raised}",))
        observation = Observation(
            result=result,
            events=tuple(streamed),
            delivered=tuple(sink.events),
            executed=tuple(name for name, _ in executor.executed),
        )
        checks = case.checks + (case.stub_checks if self.stub else ())
        failures = tuple(f for f in (_judge(check, observation) for check in checks) if f)
        return EvalOutcome(case.name, not failures, failures, result)


def _judge(check: Check, observation: Observation) -> str | None:
    raised: str | None = None
    verdict: str | None = None
    try:
        verdict = check(observation)
    except Exception as error:  # a broken check fails its case, naming only the type
        raised = type(error).__name__
    if raised is not None:
        return f"a check raised {raised}"
    return verdict


class StubHarness(EvalHarness):
    """Each case on the stub provider playing its script, in the default test run.

    ``task_classes`` gives the product's own task classes (with their floors) by name;
    any class a case uses that is not given is declared with a tools floor.
    """

    stub = True

    def __init__(
        self,
        *,
        task_classes: Iterable[TaskClass] = (),
        approvals: ApprovalHandler | None = None,
        capture: CapturePolicy | None = None,
    ) -> None:
        super().__init__(approvals=approvals, capture=capture)
        self._classes = {task.name: task for task in task_classes}

    async def run_case(self, case: EvalCase) -> EvalOutcome:
        if case.script is None:
            raise ValueError(f"case {case.name} has no stub script")
        script = case.script if callable(case.script) else scripted(list(case.script))
        if self._classes:
            classes = list(self._classes.values())
        else:
            floor = CapabilityFloor(tools=True)
            classes = [TaskClass(name=name, floor=floor) for name in case.task_classes()]
        sink = MemoryEventSink()
        runtime = await start_stub_runtime(script, classes, events=sink)
        try:
            return await self._observe(case, runtime, sink)
        finally:
            await runtime.aclose()


class LiveHarness(EvalHarness):
    """The cases marked ``live``, on ``bindings``: a runtime the product started with its
    configured assignments and providers. Use it only in tests marked ``integration``.

    A case not marked live, or whose task class or a hand-off target has no assignment
    in the runtime, is skipped with the reason. The harness never closes the runtime.
    """

    stub = False

    def __init__(
        self,
        bindings: Bindings,
        *,
        approvals: ApprovalHandler | None = None,
        capture: CapturePolicy | None = None,
    ) -> None:
        super().__init__(approvals=approvals, capture=capture)
        self._bindings = bindings

    async def run_case(self, case: EvalCase) -> EvalOutcome:
        if not case.live:
            return EvalOutcome(case.name, False, skipped=NOT_LIVE)
        assigned = self._bindings.assignments
        for task_class in case.task_classes():
            if task_class not in assigned:
                return EvalOutcome(
                    case.name,
                    False,
                    skipped=f"task class {task_class} has no assignment in the runtime",
                )
        return await self._observe(case, self._bindings, MemoryEventSink())


# --- Checks -----------------------------------------------------------------------------


def completed() -> Check:
    """The run completed."""

    def check(o: Observation) -> str | None:
        if o.result.status is RunStatus.COMPLETED:
            return None
        failure = o.result.failure.value if o.result.failure is not None else "unknown"
        return f"expected completion, got {o.result.status.value} ({failure})"

    return check


def failed(*failures: RunFailure) -> Check:
    """The run failed, for one of ``failures`` if any are given."""

    def check(o: Observation) -> str | None:
        if o.result.status is not RunStatus.FAILED:
            return f"expected a failure, got {o.result.status.value}"
        if failures and o.result.failure not in failures:
            wanted = ", ".join(f.value for f in failures)
            got = o.result.failure.value if o.result.failure is not None else "unknown"
            return f"expected a failure for {wanted}, got {got}"
        return None

    return check


def handed_off(*task_classes: str) -> Check:
    """The run handed the task off, finally to one of ``task_classes`` if any are given."""

    def check(o: Observation) -> str | None:
        if not o.result.hand_offs:
            return "expected a hand-off, got none"
        if task_classes and o.result.hand_offs[-1] not in task_classes:
            wanted = ", ".join(task_classes)
            return f"expected a hand-off to {wanted}, got {o.result.hand_offs[-1]}"
        return None

    return check


def output_field(name: str, value: object) -> Check:
    """The run's structured output has field ``name`` equal to ``value``."""

    def check(o: Observation) -> str | None:
        if o.result.output is None:
            return "the run has no output"
        missing = object()
        actual = getattr(o.result.output, name, missing)
        return (
            None
            if actual is not missing and actual == value
            else (f"output field {name} was not {value!r}")
        )

    return check


def never_executed(tool: str) -> Check:
    """The executor never ran ``tool`` (refused, unapproved or never asked for)."""

    def check(o: Observation) -> str | None:
        return f"tool {tool} was executed" if tool in o.executed else None

    return check


def executed(tool: str) -> Check:
    """The executor ran ``tool`` at least once."""

    def check(o: Observation) -> str | None:
        return None if tool in o.executed else f"tool {tool} was not executed"

    return check


def no_text_in_events(text: str) -> Check:
    """``text`` appears in no event the run streamed or delivered to its sink. The
    failure gives only the text's length, never the text."""
    if not text:
        raise ValueError("no_text_in_events needs a non-empty text")

    def check(o: Observation) -> str | None:
        for event in (*o.events, *o.delivered):
            dumped = event.model_dump(mode="json")
            if text in json.dumps(dumped, ensure_ascii=False) or text in repr(event):
                return f"a {len(text)}-character text reached an event"
        return None

    return check
