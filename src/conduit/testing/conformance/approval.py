"""The approval handler conformance suite.

Each scenario states one thing conduit relies on of an ``ApprovalHandler``, or one thing
a product may rely on of conduit around its handler. The agent runner sends a tool call
the product marks as needing approval to the handler before the tool runs, and runs the
tool only on an ``ApprovalDecision`` whose ``approved`` is true. So a handler must
answer each request with a decision of its own (never another request's, however many
are pending at once). In return conduit never runs a tool its handler refused, and
treats a handler that raises, or that is still deciding when the run's wall-time budget
runs out, as a refusal.

The product says which calls its handler approves and which it refuses, through
``approved_call`` and ``refused_call``. The runner scenarios run a real ``AgentRunner``
on the stub provider, which asks for that call; nothing reaches a network.
"""

# Suites are test code shipped for products to run under pytest: assertions are the point.
# ruff: noqa: S101

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from conduit.agent.budgets import RunBudget
from conduit.agent.events import ApprovalDecidedEvent, ApprovalOutcome
from conduit.agent.runner import AgentRunner, AgentTask, Bindings
from conduit.core.backends import ApprovalDecision, ApprovalHandler, ApprovalRequest
from conduit.core.events import ToolCallEvent
from conduit.core.tasks import CapabilityFloor, TaskClass
from conduit.core.tools import TaskTools, ToolCatalogue
from conduit.core.types import Message
from conduit.providers.stub import StubToolCall, StubTurn, answer_tool_results
from conduit.testing._stub import (
    DeclaredTool,
    RecordingExecutor,
    check_tool_name,
    start_stub_runtime,
)
from conduit.testing.conformance.base import ConformanceSuite

#: The task class the runner scenarios run under.
SUITE_TASK_CLASS = "conformance"


@dataclass(frozen=True, slots=True)
class ApprovalCall:
    """A tool call as the model makes it: the tool's name and its arguments."""

    tool: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        check_tool_name(self.tool)


class _Raises:
    """Asks the product's handler, then fails as a broken handler would."""

    def __init__(self, handler: ApprovalHandler) -> None:
        self._handler = handler

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        await self._handler.decide(request)
        raise RuntimeError("the approval handler failed")


class _Stalls:
    """Asks the product's handler, then never answers, as a stuck queue would."""

    def __init__(self, handler: ApprovalHandler) -> None:
        self._handler = handler

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        await self._handler.decide(request)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")  # pragma: no cover


@dataclass(frozen=True, slots=True)
class _Ran:
    executed: list[tuple[str, dict[str, Any]]]
    outcomes: list[ApprovalOutcome]
    tool_error: str | None


class ApprovalHandlerSuite(ConformanceSuite):
    """Decisions, the runner honouring them, failures as refusals, and concurrency."""

    interface: ClassVar[str] = "approval_handler"
    #: Requests decided at once in the concurrency scenario.
    concurrent_requests: ClassVar[int] = 12
    #: The run's wall-time budget in the stalled-handler scenario.
    stalled_budget_s: ClassVar[float] = 0.5

    async def make_handler(self) -> ApprovalHandler:
        """A new handler for one scenario. Products override this; a handler that asks a
        person is built here with a stand-in that answers as the person would."""
        raise NotImplementedError("override make_handler to return your approval handler")

    def approved_call(self, n: int = 0) -> ApprovalCall:
        """A call your handler approves. ``n`` numbers calls in the concurrency scenario;
        calls with different numbers should differ (put ``n`` in an argument)."""
        raise NotImplementedError("override approved_call with a call your handler approves")

    def refused_call(self, n: int = 0) -> ApprovalCall:
        """A call your handler refuses, numbered as ``approved_call``'s are."""
        raise NotImplementedError("override refused_call with a call your handler refuses")

    async def close_handler(self, handler: ApprovalHandler) -> None:
        """Release a handler after a scenario. Override if it needs closing."""
        closer = getattr(handler, "aclose", None)
        if closer is not None:
            await closer()

    def _scenario(self, body: Callable[[ApprovalHandler], Awaitable[None]]) -> None:
        async def scenario() -> None:
            handler = await self.make_handler()
            try:
                await body(handler)
            finally:
                await self.close_handler(handler)

        self.run(scenario)

    @staticmethod
    def _request(call: ApprovalCall, n: int) -> ApprovalRequest:
        return ApprovalRequest(
            request_id=f"apr_suite_{n:04d}",
            run_id=f"run_suite_{n:04d}",
            task_class=SUITE_TASK_CLASS,
            tool=call.tool,
            arguments=dict(call.arguments),
        )

    @staticmethod
    async def _decided(handler: ApprovalHandler, request: ApprovalRequest) -> bool:
        decision: Any = await handler.decide(request)
        assert isinstance(decision, ApprovalDecision), (
            f"decide returned a {type(decision).__name__}, not an ApprovalDecision"
        )
        return decision.approved

    def build_runner(
        self,
        runtime: Bindings,
        tools: Mapping[str, TaskTools],
        handler: ApprovalHandler,
    ) -> AgentRunner:
        """The runner the runner scenarios use: conduit's own, with ``handler`` as its
        approvals. Products leave this alone: overriding it voids the raise and stall
        scenarios, which guard conduit's runner, not the handler. The suite's own tests
        replace it with a runner that forgives a failing handler, to prove those two
        scenarios would fail it."""
        return AgentRunner(runtime, tools=dict(tools), approvals=handler)

    async def _run(
        self, handler: ApprovalHandler, call: ApprovalCall, budget: RunBudget | None = None
    ) -> _Ran:
        """One run in which the model asks for ``call`` once, then answers."""
        script = answer_tool_results(
            StubTurn(tool_calls=(StubToolCall(name=call.tool, input=dict(call.arguments)),)),
            lambda results: StubTurn(text="finished"),
        )
        task_class = TaskClass(name=SUITE_TASK_CLASS, floor=CapabilityFloor(tools=True))
        runtime = await start_stub_runtime(script, [task_class])
        try:
            tool = DeclaredTool(
                name=call.tool,
                description="A tool the suite's run asks for.",
                task_classes=frozenset({SUITE_TASK_CLASS}),
                requires_approval=True,
            )
            executor = RecordingExecutor()
            registry = ToolCatalogue.from_tools([tool]).registry_for(SUITE_TASK_CLASS)
            runner = self.build_runner(
                runtime, {SUITE_TASK_CLASS: TaskTools(registry, executor)}, handler
            )
            task = AgentTask(
                task_class=SUITE_TASK_CLASS,
                messages=(Message.user("Carry out the change."),),
                budget=budget or RunBudget(),
            )
            events = [event async for event in runner.start(task)]
        finally:
            await runtime.aclose()
        outcomes = [e.outcome for e in events if isinstance(e, ApprovalDecidedEvent)]
        finished = [e for e in events if isinstance(e, ToolCallEvent) and e.tool == call.tool]
        return _Ran(
            executed=executor.executed,
            outcomes=outcomes,
            tool_error=finished[0].error_code if finished else None,
        )

    def test_each_call_gets_its_decision(self) -> None:
        async def body(handler: ApprovalHandler) -> None:
            approved = await self._decided(handler, self._request(self.approved_call(0), 0))
            refused = await self._decided(handler, self._request(self.refused_call(1), 1))
            assert approved is True, "the handler did not approve the approved call"
            assert refused is False, "the handler did not refuse the refused call"

        self._scenario(body)

    def test_an_approved_call_runs_its_tool(self) -> None:
        call = self.approved_call(0)

        async def body(handler: ApprovalHandler) -> None:
            ran = await self._run(handler, call)
            assert ran.outcomes == ["approved"], ran.outcomes
            assert [name for name, _ in ran.executed] == [call.tool]
            assert ran.executed[0][1] == dict(call.arguments)

        self._scenario(body)

    def test_a_refused_call_never_runs_its_tool(self) -> None:
        call = self.refused_call(0)

        async def body(handler: ApprovalHandler) -> None:
            ran = await self._run(handler, call)
            assert ran.executed == [], "a refused call's tool was executed"
            assert ran.outcomes == ["refused"], ran.outcomes
            assert ran.tool_error == "not_approved"

        self._scenario(body)

    def test_a_handler_that_raises_is_a_refusal(self) -> None:
        """Guards conduit's runner around your handler: a decision that raises is a
        refusal, and the tool never runs."""
        call = self.approved_call(0)

        async def body(handler: ApprovalHandler) -> None:
            ran = await self._run(_Raises(handler), call)
            assert ran.executed == [], "a tool ran although its approval raised"
            assert ran.outcomes == ["failed"], ran.outcomes

        self._scenario(body)

    def test_a_handler_still_deciding_when_the_budget_ends_is_a_refusal(self) -> None:
        """Guards conduit's runner around your handler: a decision still pending when the
        run's wall-time budget ends is a refusal, and the tool never runs."""
        call = self.approved_call(0)
        budget = RunBudget(max_wall_time_s=self.stalled_budget_s)

        async def body(handler: ApprovalHandler) -> None:
            ran = await self._run(_Stalls(handler), call, budget)
            assert ran.executed == [], "a tool ran although its approval never came"
            assert ran.outcomes == ["failed"], ran.outcomes

        self._scenario(body)

    def test_concurrent_requests_get_their_own_decisions(self) -> None:
        count = self.concurrent_requests
        # Approved and refused calls interleaved, all pending at once.
        calls = [
            (self.approved_call(n), True) if n % 2 == 0 else (self.refused_call(n), False)
            for n in range(count)
        ]

        async def body(handler: ApprovalHandler) -> None:
            decided = await asyncio.gather(
                *(
                    self._decided(handler, self._request(call, n))
                    for n, (call, _) in enumerate(calls)
                )
            )
            assert list(decided) == [wanted for _, wanted in calls]

        self._scenario(body)
