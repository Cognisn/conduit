"""FanOutSink and cancellation: a cancellation always propagates, unchanged.

Every test runs genuinely concurrent sink tasks: each sink waits on the others or on a
gate, so the sinks are all in flight when the cancellation happens.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from conduit.core.events import CapturePolicy, Event, FanOutSink


def _event() -> Event:
    return Event(type="synthetic_event")


class Gate:
    """A sink that records it started, then waits for the gate to open."""

    def __init__(self, started: asyncio.Barrier | None = None) -> None:
        self.started = started
        self.entered = asyncio.Event()
        self.gate = asyncio.Event()
        self.received: list[Event] = []
        self.cancelled = False

    async def emit(self, event: Event) -> None:
        self.entered.set()
        try:
            if self.started is not None:
                await self.started.wait()
            await self.gate.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        self.received.append(event)


class CancellingSink:
    """A sink whose emit raises CancelledError of its own, once every sink has started."""

    def __init__(self, started: asyncio.Barrier) -> None:
        self.started = started
        self.raised = asyncio.CancelledError("synthetic sink cancellation")

    async def emit(self, event: Event) -> None:
        await self.started.wait()
        raise self.raised


class FailingSink:
    async def emit(self, event: Event) -> None:
        await asyncio.sleep(0)
        raise ValueError("synthetic sink failure")


class TestASinksOwnCancellation:
    async def test_propagates_unchanged(self) -> None:
        started = asyncio.Barrier(2)
        cancelling = CancellingSink(started)
        other = Gate(started)
        other.gate.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await FanOutSink([cancelling, other]).emit(_event())
        assert raised.value is cancelling.raised
        assert raised.value.args == ("synthetic sink cancellation",)
        assert raised.value.__cause__ is None

    async def test_the_other_sinks_still_receive_the_event(self) -> None:
        started = asyncio.Barrier(3)
        first, second = Gate(started), Gate(started)
        first.gate.set()
        second.gate.set()
        with pytest.raises(asyncio.CancelledError):
            await FanOutSink([first, CancellingSink(started), second]).emit(_event())
        assert len(first.received) == 1 and len(second.received) == 1

    async def test_is_not_logged_as_a_sink_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.WARNING, logger="conduit.events")
        with pytest.raises(asyncio.CancelledError):
            await FanOutSink([CancellingSink(asyncio.Barrier(1))]).emit(_event())
        assert not [r for r in caplog.records if "CancelledError" in r.getMessage()]

    async def test_other_failures_are_still_logged_alongside_it(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING, logger="conduit.events")
        with pytest.raises(asyncio.CancelledError):
            await FanOutSink([FailingSink(), CancellingSink(asyncio.Barrier(1))]).emit(_event())
        messages = [r.getMessage() for r in caplog.records]
        assert any("FailingSink" in m and "ValueError" in m for m in messages)
        assert all("synthetic sink failure" not in m for m in messages)


class TestCancellingTheFanOut:
    async def test_propagates_and_cancels_every_sink(self) -> None:
        started = asyncio.Barrier(3)  # both sinks and the test
        first, second = Gate(started), Gate(started)
        emitting = asyncio.create_task(FanOutSink([first, second]).emit(_event()))
        await asyncio.wait_for(started.wait(), 5)
        emitting.cancel("synthetic outer cancellation")
        with pytest.raises(asyncio.CancelledError) as raised:
            await emitting
        assert raised.value.args == ("synthetic outer cancellation",)
        assert first.cancelled and second.cancelled
        assert first.received == [] and second.received == []

    async def test_a_timeout_around_the_fan_out_is_a_timeout(self) -> None:
        sink = Gate()
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.05):
                await FanOutSink([sink]).emit(_event())
        assert sink.cancelled


def test_the_fan_outs_policy_is_read_only() -> None:
    bus = FanOutSink([], policy=CapturePolicy())
    with pytest.raises(AttributeError):
        setattr(bus, "policy", CapturePolicy(prompts=True))  # noqa: B010
    assert bus.policy == CapturePolicy()


class TestOtherFailuresKeepTodaysBehaviour:
    async def test_a_failing_sink_is_logged_and_does_not_raise(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING, logger="conduit.events")
        good = Gate()
        good.gate.set()
        await FanOutSink([FailingSink(), good]).emit(_event())
        assert len(good.received) == 1
        assert any("FailingSink" in r.getMessage() for r in caplog.records)
