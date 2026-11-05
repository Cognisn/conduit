"""The sync facade: calls run on a loop conduit owns, never on that loop's own thread."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

from conduit.core import facade
from conduit.core.errors import FacadeError
from conduit.core.events import MemoryEventSink, ModelCallFinishedEvent
from conduit.core.facade import LoopThread, SyncFacade, default_loop, sync
from conduit.core.registry import ProviderRegistry
from conduit.core.secrets import LiteralSecretResolver
from conduit.core.types import Message, ModelRequest
from tests.unit.core.fakes import CATALOGUE, PARTIAL, FakeFactory


@pytest.fixture
def loop() -> Iterator[LoopThread]:
    owned = LoopThread(name="conduit-test-loop")
    yield owned
    owned.close(join_timeout=5)


class Service:
    """An async API, as conduit's layers expose them."""

    label = "service"

    def __init__(self) -> None:
        self.thread: int | None = None
        self.cancelled = False
        self.closed_stream = False

    async def add(self, a: int, b: int) -> int:
        self.thread = threading.get_ident()
        await asyncio.sleep(0)
        return a + b

    async def fail(self) -> None:
        raise ValueError("expected")

    async def slow(self, seconds: float) -> str:
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return "done"

    async def numbers(self, count: int) -> AsyncIterator[int]:
        try:
            for i in range(count):
                await asyncio.sleep(0)
                yield i
        finally:
            self.closed_stream = True

    def describe(self) -> str:
        return "a plain method"


class TestLoopThread:
    def test_runs_on_its_own_thread_and_returns(self, loop: LoopThread) -> None:
        service = Service()
        assert loop.run(service.add(2, 3)) == 5
        assert service.thread == loop.thread_id != threading.get_ident()

    def test_exceptions_propagate(self, loop: LoopThread) -> None:
        with pytest.raises(ValueError, match="expected"):
            loop.run(Service().fail())

    def test_a_timeout_cancels_the_call(self, loop: LoopThread) -> None:
        service = Service()
        with pytest.raises(TimeoutError):
            loop.run(service.slow(5), timeout=0.05)
        deadline = time.monotonic() + 2
        while not service.cancelled and time.monotonic() < deadline:
            time.sleep(0.01)
        assert service.cancelled

    def test_refuses_to_run_on_the_loops_own_thread(self, loop: LoopThread) -> None:
        # Without the guard this would block the loop thread waiting on itself; the
        # outer timeout turns that into a failure rather than a hang.
        async def reenter() -> str:
            try:
                loop.run(Service().add(1, 1), timeout=1)
            except FacadeError as error:
                return f"refused: {error}"
            return "ran"

        result = loop.run(reenter(), timeout=3)
        assert result.startswith("refused")
        assert "deadlock" in result

    def test_a_closed_loop_refuses(self) -> None:
        owned = LoopThread()
        owned.run(Service().add(1, 1))
        owned.close()
        coroutine = Service().add(1, 1)
        with pytest.raises(FacadeError):
            owned.run(coroutine)

    def test_concurrent_callers_share_the_loop_concurrently(self, loop: LoopThread) -> None:
        service = Service()
        results: list[str] = []

        def call() -> None:
            results.append(loop.run(service.slow(0.2), timeout=5))

        threads = [threading.Thread(target=call) for _ in range(10)]
        began = time.monotonic()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert results == ["done"] * 10
        assert time.monotonic() - began < 1.5  # ten 0.2 s sleeps overlapped


class TestFacade:
    def test_coroutine_methods_become_blocking_calls(self, loop: LoopThread) -> None:
        facade = sync(Service(), loop=loop)
        assert facade.add(4, 5) == 9

    def test_async_generators_become_iterators(self, loop: LoopThread) -> None:
        service = Service()
        facade = sync(service, loop=loop)
        assert list(facade.numbers(3)) == [0, 1, 2]
        assert service.closed_stream

    def test_abandoning_an_iterator_closes_the_generator(self, loop: LoopThread) -> None:
        service = Service()
        iterator = sync(service, loop=loop).numbers(10)
        assert next(iterator) == 0
        iterator.close()
        assert service.closed_stream

    def test_plain_attributes_and_methods_pass_through(self, loop: LoopThread) -> None:
        facade = sync(Service(), loop=loop)
        assert facade.label == "service"
        assert facade.describe() == "a plain method"

    def test_timeouts_apply_per_facade(self, loop: LoopThread) -> None:
        facade = SyncFacade(Service(), loop=loop, timeout=0.05)
        with pytest.raises(TimeoutError):
            facade.slow(5)
        assert facade.with_timeout(5).slow(0.01) == "done"

    def test_the_default_loop_is_shared(self) -> None:
        assert default_loop() is default_loop()
        assert sync(Service()).add(1, 2) == 3


async def _careful(started: threading.Event, cleaned: list[str]) -> AsyncIterator[int]:
    """A generator whose cleanup awaits, as an SDK stream's does."""
    try:
        yield 1
        started.set()
        await asyncio.sleep(30)
        yield 2
    finally:
        await asyncio.sleep(0.05)
        cleaned.append("cleaned")


class TestIteratorTimeout:
    def test_a_step_that_times_out_raises_timeout_after_cleanup(self, loop: LoopThread) -> None:
        cleaned: list[str] = []
        iterator = loop.iterate(_careful(threading.Event(), cleaned), timeout=0.2)
        assert next(iterator) == 1
        with pytest.raises(TimeoutError):
            next(iterator)
        assert cleaned == ["cleaned"]  # unwound before the caller saw the timeout

    def test_a_timed_out_call_has_unwound_when_run_returns(self, loop: LoopThread) -> None:
        service = Service()
        with pytest.raises(TimeoutError):
            loop.run(service.slow(5), timeout=0.05)
        assert service.cancelled  # no polling: the cancellation finished first

    def test_a_facade_timeout_on_a_metered_stream_is_metered_as_cancelled(
        self, loop: LoopThread
    ) -> None:
        sink = MemoryEventSink()
        registry = ProviderRegistry(
            secrets=LiteralSecretResolver({"fake/key": "not-a-real-key"}),
            catalogue=CATALOGUE,
            events=sink,
            factories=[FakeFactory()],
        )
        provider = loop.run(
            registry.build(
                "p", {"kind": "fake", "api_key": "secret://fake/key", "behaviour": "slow"}
            )
        )
        request = ModelRequest(
            model="fake-large", messages=(Message.user("hello"),), max_output_tokens=10
        )
        with pytest.raises(TimeoutError):
            list(sync(provider, loop=loop, timeout=0.3).stream(request))
        with pytest.raises(TimeoutError):
            sync(provider, loop=loop, timeout=0.3).complete(request)
        finished = [e for e in sink.events if isinstance(e, ModelCallFinishedEvent)]
        assert [e.error_type for e in finished] == ["cancelled", "cancelled"]
        assert all(e.usage == PARTIAL for e in finished)


async def _stubborn(cleaned: list[str]) -> AsyncIterator[int]:
    """A generator whose cleanup takes far longer than the grace period."""
    try:
        yield 1
        await asyncio.sleep(30)
        yield 2
    finally:
        await asyncio.sleep(0.5)
        cleaned.append("cleaned")


class TestStrandedSteps:
    def test_a_cleanup_that_overruns_the_grace_period_still_releases_the_caller(
        self, loop: LoopThread, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(facade, "UNWIND_GRACE_S", 0.1)
        cleaned: list[str] = []
        iterator = loop.iterate(_stubborn(cleaned), timeout=0.1)
        assert next(iterator) == 1
        began = time.monotonic()
        with pytest.raises(TimeoutError):
            next(iterator)
        assert time.monotonic() - began < 0.45  # released before the cleanup finished
        assert cleaned == []
        deadline = time.monotonic() + 3
        while not cleaned and time.monotonic() < deadline:
            time.sleep(0.01)
        assert cleaned == ["cleaned"]  # the cancelled step closed the generator itself

    def test_a_timeout_the_call_raises_itself_passes_through(self, loop: LoopThread) -> None:
        async def own() -> None:
            raise TimeoutError("the call's own deadline")

        with pytest.raises(TimeoutError, match="own deadline"):
            loop.run(own(), timeout=5)


class TestClose:
    def test_close_releases_a_caller_waiting_without_a_timeout(self) -> None:
        owned = LoopThread()
        service = Service()
        outcome: list[BaseException] = []

        def wait() -> None:
            try:
                owned.run(service.slow(30), timeout=None)
            except BaseException as error:
                outcome.append(error)

        caller = threading.Thread(target=wait, daemon=True)  # a regression must not hang the run
        caller.start()
        deadline = time.monotonic() + 5
        while owned.thread_id is None and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.1)
        owned.close(join_timeout=5)
        caller.join(5)
        assert not caller.is_alive()
        assert len(outcome) == 1 and isinstance(outcome[0], FacadeError)
        assert service.cancelled

    def test_close_finalises_suspended_generators(self) -> None:
        owned = LoopThread()
        service = Service()
        iterator = owned.iterate(service.numbers(10))
        assert next(iterator) == 0
        owned.close(join_timeout=5)
        assert service.closed_stream
        iterator.close()  # the loop is closed: nothing more to do, and no error

    def test_close_is_refused_on_the_loops_own_thread(self, loop: LoopThread) -> None:
        async def close_from_inside() -> str:
            try:
                loop.close()
            except FacadeError as error:
                return f"refused: {error}"
            return "closed"

        assert loop.run(close_from_inside(), timeout=3).startswith("refused")
        assert loop.run(Service().add(1, 1)) == 2  # still serving


def _wait_in_thread(owned: LoopThread, awaitable: Any) -> tuple[threading.Thread, list[object]]:
    outcome: list[object] = []

    def wait() -> None:
        try:
            outcome.append(owned.run(awaitable, timeout=None))
        except BaseException as error:
            outcome.append(error)

    caller = threading.Thread(target=wait, name="caller", daemon=True)
    caller.start()
    return caller, outcome


class TestCloseRaces:
    def test_a_call_submitted_as_close_begins_is_failed_not_stranded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The reviewer's interleaving: the caller has passed the closed check, close()
        # runs its shutdown, and only then does the caller's submission land.
        owned = LoopThread()
        owned.run(asyncio.sleep(0))
        shut_down = threading.Event()
        real_shutdown = facade._shutdown
        real_submit = asyncio.run_coroutine_threadsafe

        async def tracked(loop: asyncio.AbstractEventLoop) -> None:
            await real_shutdown(loop)
            shut_down.set()

        def late_submit(coroutine: Any, loop: asyncio.AbstractEventLoop) -> Any:
            if threading.current_thread().name == "caller":
                closer = threading.Thread(target=owned.close, daemon=True)
                closer.start()
                shut_down.wait(1)  # close cannot get this far while a submission is pending
            return real_submit(coroutine, loop)

        monkeypatch.setattr(facade, "_shutdown", tracked)
        monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", late_submit)
        caller, outcome = _wait_in_thread(owned, asyncio.sleep(3600))
        caller.join(10)
        assert not caller.is_alive(), "the caller was left waiting on a closed loop"
        assert len(outcome) == 1 and isinstance(outcome[0], FacadeError)

    def test_a_close_completing_before_the_submission_lands_still_gives_facade_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The closed check and the submission are one step: a close() that ran to the
        # end in between would leave the caller submitting to a closed loop.
        owned = LoopThread()
        owned.run(asyncio.sleep(0))
        real_submit = asyncio.run_coroutine_threadsafe

        def late_submit(coroutine: Any, loop: asyncio.AbstractEventLoop) -> Any:
            if threading.current_thread().name == "caller":
                closer = threading.Thread(target=owned.close, daemon=True)
                closer.start()
                closer.join(1)  # close cannot finish while a submission is under way
            return real_submit(coroutine, loop)

        monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", late_submit)
        caller, outcome = _wait_in_thread(owned, asyncio.sleep(3600))
        caller.join(10)
        assert not caller.is_alive(), "the caller was left waiting on a closed loop"
        assert len(outcome) == 1 and isinstance(outcome[0], FacadeError), outcome

    def test_a_call_ignoring_cancellation_is_failed_when_close_gives_up(self) -> None:
        owned = LoopThread()
        started = threading.Event()

        async def stubborn() -> None:
            started.set()
            while True:
                try:
                    await asyncio.sleep(3600)
                except asyncio.CancelledError:
                    continue  # ignores every cancellation

        caller, outcome = _wait_in_thread(owned, stubborn())
        assert started.wait(5)
        owned.close(join_timeout=0.3)
        caller.join(5)
        assert not caller.is_alive(), "the caller was left waiting on a closed loop"
        assert len(outcome) == 1 and isinstance(outcome[0], FacadeError)

    def test_every_call_after_close_is_refused_with_facade_error(self) -> None:
        owned = LoopThread()
        owned.run(asyncio.sleep(0))
        owned.close()
        coroutine = asyncio.sleep(0)
        with pytest.raises(FacadeError):
            owned.run(coroutine)
