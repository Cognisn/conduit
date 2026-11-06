"""The approval handler conformance suite, against handlers written as a product would
write them (conduit ships none: approvals are the product's interface), and the suite's
own checks: it must fail a handler that breaks the contract."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Mapping
from typing import Any, cast

import pytest

from conduit.agent import AgentRunner, Bindings
from conduit.core.backends import ApprovalDecision, ApprovalHandler, ApprovalRequest
from conduit.core.tools import TaskTools
from conduit.testing.conformance import ApprovalHandlerSuite
from conduit.testing.conformance.approval import ApprovalCall
from tests.conformance.meta import fails, passes


class AllowListApprovals:
    """Approves changes only to hosts on an allow list (the backends guide's example)."""

    def __init__(self, allowed_hosts: set[str]) -> None:
        self._allowed = allowed_hosts

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        host = str(request.arguments.get("host", ""))
        if host in self._allowed:
            return ApprovalDecision(approved=True, decided_by="change-window@example.com")
        return ApprovalDecision(approved=False, reason="outside the change window")


class TestAllowListApprovals(ApprovalHandlerSuite):
    async def make_handler(self) -> ApprovalHandler:
        return AllowListApprovals({"192.0.2.20"})

    def approved_call(self, n: int = 0) -> ApprovalCall:
        return ApprovalCall(tool="restart_service", arguments={"host": "192.0.2.20", "n": n})

    def refused_call(self, n: int = 0) -> ApprovalCall:
        return ApprovalCall(tool="restart_service", arguments={"host": "192.0.2.21", "n": n})


class QueuedApprovals:
    """Puts each request on a queue for a reviewer task and waits for its answer, as a
    product asking a person through a web interface would."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[tuple[ApprovalRequest, asyncio.Future[bool]]] = asyncio.Queue()
        self._reviewer = asyncio.create_task(self._review())

    async def _review(self) -> None:
        while True:
            request, answer = await self._queue.get()
            await asyncio.sleep(0.001)  # a person takes a moment
            answer.set_result(request.arguments.get("ticket") == "CHG-1001")

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        answer: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        await self._queue.put((request, answer))
        approved = await answer
        return ApprovalDecision(approved=approved, decided_by="reviewer@example.org")

    async def aclose(self) -> None:
        self._reviewer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._reviewer


class TestQueuedApprovals(ApprovalHandlerSuite):
    async def make_handler(self) -> ApprovalHandler:
        return QueuedApprovals()

    def approved_call(self, n: int = 0) -> ApprovalCall:
        return ApprovalCall(tool="close_ticket", arguments={"ticket": "CHG-1001", "n": n})

    def refused_call(self, n: int = 0) -> ApprovalCall:
        return ApprovalCall(tool="close_ticket", arguments={"ticket": "CHG-2002", "n": n})


# --- The suite's own checks: each broken handler fails the scenario it breaks. -----------


class _Base:
    def wanted(self, request: ApprovalRequest) -> bool:
        return request.arguments.get("host") == "192.0.2.20"


class _ApprovesEverything(_Base):
    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(approved=True)


class _ReturnsAMapping(_Base):
    async def decide(self, request: ApprovalRequest) -> Any:
        return {"approved": self.wanted(request)}


class _SharedAnswer(_Base):
    """Keeps the request under review on the instance: concurrent requests collide."""

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        self.current = request
        await asyncio.sleep(0.001)
        return ApprovalDecision(approved=self.wanted(self.current))


class _Sound(_Base):
    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(approved=self.wanted(request))


def _suite(handler: type[_Base]) -> ApprovalHandlerSuite:
    class Suite(ApprovalHandlerSuite):
        async def make_handler(self) -> ApprovalHandler:
            return cast(ApprovalHandler, handler())  # broken on purpose

        def approved_call(self, n: int = 0) -> ApprovalCall:
            return ApprovalCall(tool="restart_service", arguments={"host": "192.0.2.20", "n": n})

        def refused_call(self, n: int = 0) -> ApprovalCall:
            return ApprovalCall(tool="restart_service", arguments={"host": "192.0.2.21", "n": n})

    return Suite()


@pytest.mark.parametrize(
    ("broken", "scenario"),
    [
        (_ApprovesEverything, "test_each_call_gets_its_decision"),
        (_ApprovesEverything, "test_a_refused_call_never_runs_its_tool"),
        (_ReturnsAMapping, "test_each_call_gets_its_decision"),
        (_SharedAnswer, "test_concurrent_requests_get_their_own_decisions"),
    ],
)
def test_the_suite_fails_a_broken_handler(broken: type[_Base], scenario: str) -> None:
    with fails():
        getattr(_suite(broken), scenario)()


@pytest.mark.parametrize(
    "scenario",
    [name for name in dir(ApprovalHandlerSuite) if name.startswith("test_")],
)
def test_every_scenario_passes_a_sound_handler(scenario: str) -> None:
    passes(getattr(_suite(_Sound), scenario))


def test_a_call_must_be_a_tool_name_and_its_arguments() -> None:
    with pytest.raises(ValueError, match="snake_case"):
        ApprovalCall(tool="Restart Service", arguments={})


def test_an_unconfigured_suite_says_what_to_override() -> None:
    with pytest.raises(NotImplementedError, match="make_handler"):
        ApprovalHandlerSuite().test_each_call_gets_its_decision()


class _AnswersTheWrongCaller(_Base):
    """Queues each request, then answers from the newest: the right mix of approvals and
    refusals, given to the wrong callers when requests overlap."""

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        self.pending: list[ApprovalRequest] = getattr(self, "pending", [])
        self.pending.append(request)
        await asyncio.sleep(0.001)
        return ApprovalDecision(approved=self.wanted(self.pending.pop()))


def test_the_suite_fails_a_handler_that_answers_the_wrong_caller() -> None:
    suite = _suite(_AnswersTheWrongCaller)
    suite.test_each_call_gets_its_decision()  # sound one request at a time
    with fails():
        suite.test_concurrent_requests_get_their_own_decisions()


# The raise and stall scenarios guard conduit's runner around the product's handler. A
# runner that forgave a failing handler (approving on a raise, or on its own deadline
# before the run's budget ran out) must fail them.


class _ForgivesARaise:
    def __init__(self, handler: ApprovalHandler) -> None:
        self._handler = handler

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        try:
            return await self._handler.decide(request)
        except Exception:
            return ApprovalDecision(approved=True)


class _ForgivesAStall:
    def __init__(self, handler: ApprovalHandler) -> None:
        self._handler = handler

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        try:
            async with asyncio.timeout(0.05):
                return await self._handler.decide(request)
        except TimeoutError:
            return ApprovalDecision(approved=True)


def _forgiving(wrapper: type[_ForgivesARaise] | type[_ForgivesAStall]) -> ApprovalHandlerSuite:
    base = type(_suite(_Sound))

    class Suite(base):  # type: ignore[valid-type, misc]
        def build_runner(
            self,
            runtime: Bindings,
            tools: Mapping[str, TaskTools],
            handler: ApprovalHandler,
        ) -> AgentRunner:
            return AgentRunner(runtime, tools=tools, approvals=wrapper(handler))

    return cast(ApprovalHandlerSuite, Suite())


def test_the_raise_scenario_fails_a_runner_that_approves_on_a_raise() -> None:
    with fails():
        _forgiving(_ForgivesARaise).test_a_handler_that_raises_is_a_refusal()


def test_the_stall_scenario_fails_a_runner_that_approves_on_a_stall() -> None:
    with fails():
        _forgiving(
            _ForgivesAStall
        ).test_a_handler_still_deciding_when_the_budget_ends_is_a_refusal()


def test_the_runner_scenarios_pass_with_conduits_runner_through_the_hook() -> None:
    class Suite(type(_suite(_Sound))):  # type: ignore[misc]
        def build_runner(
            self,
            runtime: Bindings,
            tools: Mapping[str, TaskTools],
            handler: ApprovalHandler,
        ) -> AgentRunner:
            runner: AgentRunner = super().build_runner(runtime, tools, handler)
            return runner

    Suite().test_a_handler_that_raises_is_a_refusal()
