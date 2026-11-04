"""The sync facade: calls run on a loop conduit owns, never on that loop's own thread."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Iterator

import pytest

from conduit.core.errors import FacadeError
from conduit.core.facade import LoopThread, SyncFacade, default_loop, sync


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
