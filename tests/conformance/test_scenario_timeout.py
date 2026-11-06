"""Every conformance scenario is bounded: a backend that hangs fails its scenario with a
stated reason within ``scenario_timeout_s``, rather than hanging the test run."""

from __future__ import annotations

import asyncio
import contextlib
import time
import unittest
from collections.abc import Mapping, Sequence
from typing import ClassVar

import pytest
from pydantic import SecretStr

from conduit.core.backends import (
    ApprovalDecision,
    ApprovalHandler,
    ApprovalRequest,
    Skill,
    SkillSource,
    SkillSummary,
    TokenSet,
    TokenStore,
)
from conduit.core.events import Event, EventSink, MemoryEventSink
from conduit.core.secrets import SecretRef, SecretResolver
from conduit.core.transcript import TranscriptEntry, TranscriptStore
from conduit.stores.memory import MemoryTranscriptStore
from conduit.testing.conformance import (
    ApprovalCall,
    ApprovalHandlerSuite,
    ConformanceSuite,
    EventSinkSuite,
    SecretResolverSuite,
    SkillSourceSuite,
    TokenStoreSuite,
    TranscriptStoreSuite,
)

BOUND = 0.3
GRACE = 0.3


async def _forever() -> None:
    await asyncio.Event().wait()


def _fails_within_the_bound(scenario: object) -> None:
    began = time.monotonic()
    raised: BaseException | None = None
    try:
        scenario()  # type: ignore[operator]
    except unittest.SkipTest as skip:
        pytest.fail(f"the scenario skipped instead of failing: {skip}")
    except AssertionError as error:
        raised = error
    assert raised is not None, "the scenario did not fail"
    assert f"did not finish within {BOUND} s" in str(raised)
    assert time.monotonic() - began < BOUND + GRACE + 5.0
    assert raised.__cause__ is None
    assert raised.__context__ is None


def test_the_default_bound_is_set_for_every_suite() -> None:
    assert ConformanceSuite.scenario_timeout_s == 60.0


def test_a_hanging_scenario_fails_with_a_stated_reason_within_the_bound() -> None:
    class Suite(ConformanceSuite):
        scenario_timeout_s: ClassVar[float] = BOUND
        scenario_cancel_grace_s: ClassVar[float] = GRACE

    _fails_within_the_bound(lambda: Suite().run(_forever))


def test_a_scenario_within_the_bound_is_unchanged() -> None:
    ran: list[bool] = []

    async def quick() -> None:
        ran.append(True)

    ConformanceSuite().run(quick)
    assert ran == [True]


def test_an_assertion_inside_the_bound_is_reported_as_itself() -> None:
    async def wrong() -> None:
        raise AssertionError("the scenario's own failure")

    with pytest.raises(AssertionError, match="the scenario's own failure"):
        ConformanceSuite().run(wrong)


class _HangingResolver:
    async def resolve(self, ref: SecretRef) -> SecretStr:
        await _forever()
        raise AssertionError("unreachable")


class _HangingSink:
    async def emit(self, event: Event) -> None:
        await _forever()


class _HangingHandler:
    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        await _forever()
        raise AssertionError("unreachable")


class _HangingTokens:
    async def get(self, key: str) -> TokenSet | None:
        await _forever()
        return None

    async def put(self, key: str, tokens: TokenSet) -> None:
        await _forever()

    async def delete(self, key: str) -> None:
        await _forever()


class _HangingTranscripts(MemoryTranscriptStore):
    async def list_sessions(self) -> Sequence[str]:
        await _forever()
        return []

    async def append(
        self, session_id: str, entries: Sequence[TranscriptEntry]
    ) -> tuple[TranscriptEntry, ...]:
        await _forever()
        return ()


class _HangingSkills:
    async def list_skills(self) -> Sequence[SkillSummary]:
        await _forever()
        return []

    async def load_skill(self, name: str) -> Skill:
        await _forever()
        raise LookupError(name)


class _ResolverSuite(SecretResolverSuite):
    scenario_timeout_s: ClassVar[float] = BOUND
    scenario_cancel_grace_s: ClassVar[float] = GRACE

    async def make_resolver(self, secrets: Mapping[str, str]) -> SecretResolver:
        return _HangingResolver()


class _SinkSuite(EventSinkSuite):
    scenario_timeout_s: ClassVar[float] = BOUND
    scenario_cancel_grace_s: ClassVar[float] = GRACE
    emit_deadline_s: ClassVar[float] = 30.0

    async def make_sink(self) -> EventSink:
        return _HangingSink()


class _RecordedHangs(EventSinkSuite):
    scenario_timeout_s: ClassVar[float] = BOUND
    scenario_cancel_grace_s: ClassVar[float] = GRACE

    async def make_sink(self) -> EventSink:
        return MemoryEventSink()

    async def recorded(self, sink: EventSink) -> Sequence[Mapping[str, object]]:
        await _forever()
        return []


class _HandlerSuite(ApprovalHandlerSuite):
    scenario_timeout_s: ClassVar[float] = BOUND
    scenario_cancel_grace_s: ClassVar[float] = GRACE

    async def make_handler(self) -> ApprovalHandler:
        return _HangingHandler()

    def approved_call(self, n: int = 0) -> ApprovalCall:
        return ApprovalCall("restart_service", {"n": n})

    def refused_call(self, n: int = 0) -> ApprovalCall:
        return ApprovalCall("restart_service", {"n": -n - 1})


class _TokenSuite(TokenStoreSuite):
    scenario_timeout_s: ClassVar[float] = BOUND
    scenario_cancel_grace_s: ClassVar[float] = GRACE

    async def make_store(self) -> TokenStore:
        return _HangingTokens()


class _TranscriptSuite(TranscriptStoreSuite):
    scenario_timeout_s: ClassVar[float] = BOUND
    scenario_cancel_grace_s: ClassVar[float] = GRACE

    async def make_store(self) -> TranscriptStore:
        return _HangingTranscripts()


class _SkillSuite(SkillSourceSuite):
    scenario_timeout_s: ClassVar[float] = BOUND
    scenario_cancel_grace_s: ClassVar[float] = GRACE

    async def make_source(self, skills: Sequence[Skill]) -> SkillSource:
        return _HangingSkills()


@pytest.mark.parametrize(
    ("suite", "scenario"),
    [
        (_ResolverSuite, "test_a_known_reference_resolves_to_its_value"),
        (_SinkSuite, "test_every_event_is_recorded_in_order"),
        (_RecordedHangs, "test_every_event_is_recorded_in_order"),
        (_HandlerSuite, "test_each_call_gets_its_decision"),
        (_TokenSuite, "test_an_unknown_key_is_none"),
        (_TranscriptSuite, "test_concurrent_appends_to_one_session_stay_whole"),
        (_SkillSuite, "test_an_absent_skill_is_a_lookup_error"),
    ],
)
def test_a_hanging_backend_fails_every_suite_within_the_bound(
    suite: type[ConformanceSuite], scenario: str
) -> None:
    _fails_within_the_bound(getattr(suite(), scenario))


def test_a_scenarios_own_timeout_is_not_mistaken_for_the_bound() -> None:
    async def times_out_itself() -> None:
        async with asyncio.timeout(0.01):
            await _forever()

    with pytest.raises(TimeoutError):
        ConformanceSuite().run(times_out_itself)


def test_the_default_grace_after_cancelling_is_set() -> None:
    assert ConformanceSuite.scenario_cancel_grace_s == 5.0


class _Bounded(ConformanceSuite):
    scenario_timeout_s: ClassVar[float] = BOUND
    scenario_cancel_grace_s: ClassVar[float] = GRACE


def test_a_close_that_hangs_after_the_bound_still_fails_within_the_grace() -> None:
    async def hangs_and_hangs_on_close() -> None:
        try:
            await _forever()
        finally:
            await _forever()  # a close that hangs too

    _fails_within_the_bound(lambda: _Bounded().run(hangs_and_hangs_on_close))


def test_a_backend_that_swallows_cancellation_still_fails_within_the_grace() -> None:
    async def swallows() -> None:
        while True:
            with contextlib.suppress(asyncio.CancelledError):  # the defect
                await asyncio.sleep(10)

    _fails_within_the_bound(lambda: _Bounded().run(swallows))


class _TokensThatHangOnClose(_HangingTokens):
    async def aclose(self) -> None:
        await _forever()


class _TokenSuiteHangingOnClose(TokenStoreSuite):
    scenario_timeout_s: ClassVar[float] = BOUND
    scenario_cancel_grace_s: ClassVar[float] = GRACE

    async def make_store(self) -> TokenStore:
        return _TokensThatHangOnClose()


def test_a_suite_whose_backend_hangs_in_close_fails_within_the_grace() -> None:
    _fails_within_the_bound(_TokenSuiteHangingOnClose().test_an_unknown_key_is_none)
