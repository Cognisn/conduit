"""The sync facade: conduit's async API, callable from synchronous code (C3).

A ``LoopThread`` owns an event loop on a daemon thread. ``LoopThread.run`` submits a
coroutine to it and waits, with a timeout applied on the loop: on timeout the call is
cancelled, and has unwound, before ``TimeoutError`` reaches the caller. Calling
``run`` from the loop's own thread would wait on itself forever, so it is refused with
``FacadeError`` (the guard SPARK's database bridge uses), as is ``close``.
``LoopThread.close`` cancels every pending call (a caller still waiting gets
``FacadeError``) and finalises suspended async generators before the loop stops.

``sync(obj)`` wraps any object with an async API: each coroutine method becomes a
blocking call and each async-generator method (``stream``, for example) becomes a
plain iterator, both running on the loop thread. Everything else passes through. The
facade is generated from the object at call time, so every async interface conduit
has, now or later, gets a sync form without a hand-written twin.

What a facade method returns comes back as it is, never wrapped: a ``TaskBinding``
from ``Runtime.for_task`` is a plain object whose ``provider`` is async. A sync caller
wraps each async object it is handed, on the same loop::

    loop = default_loop()
    runtime = sync(loop.run(start(settings, task_classes=..., factories=..., secrets=...)))
    binding = runtime.for_task("triage")  # returned as is
    provider = sync(binding.provider, loop=loop)
    response = provider.complete(binding.request(messages=[Message.user("...")]))
    runtime.aclose()

``start`` must run on the loop the facade uses (``loop.run(start(...))``, never
``asyncio.run``): providers make their SDK clients, connection pools and locks on the
loop that builds them, and must be called on that loop.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import inspect
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Generator
from typing import Any

from conduit.core.errors import FacadeError

DEFAULT_TIMEOUT_S: float = 300.0
#: How long a timed-out call may take to unwind before its caller is released anyway.
UNWIND_GRACE_S: float = 10.0


class LoopThread:
    """An event loop on a thread conduit owns."""

    def __init__(self, *, name: str = "conduit-loop") -> None:
        self._name = name
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._thread_id: int | None = None
        self._closed = False

    def _ensure(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._closed:
                raise FacadeError("the conduit loop is closed")
            if self._loop is None:
                loop = asyncio.new_event_loop()
                ready = threading.Event()

                def serve() -> None:
                    self._thread_id = threading.get_ident()
                    asyncio.set_event_loop(loop)
                    ready.set()
                    loop.run_forever()

                thread = threading.Thread(target=serve, name=self._name, daemon=True)
                thread.start()
                ready.wait()
                self._loop, self._thread = loop, thread
            return self._loop

    @property
    def thread_id(self) -> int | None:
        return self._thread_id

    def on_loop_thread(self) -> bool:
        return self._thread_id is not None and threading.get_ident() == self._thread_id

    def run[T](self, awaitable: Awaitable[T], *, timeout: float | None = DEFAULT_TIMEOUT_S) -> T:
        """Run ``awaitable`` on the loop and wait for it. Refused on the loop's own thread.

        The timeout is applied on the loop: on expiry the call is cancelled and has
        finished unwinding (its ``finally`` blocks have run) before ``TimeoutError``
        reaches the caller. Should the unwinding itself hang for longer than
        ``UNWIND_GRACE_S``, the caller is released with ``TimeoutError`` regardless.
        A call still waiting when the loop is closed raises ``FacadeError``.
        """
        if self.on_loop_thread():
            _discard(awaitable)
            raise FacadeError(
                "the sync facade was called on conduit's own loop thread, which would "
                "deadlock; await the async API there instead"
            )
        try:
            loop = self._ensure()
        except FacadeError:
            _discard(awaitable)
            raise
        future = asyncio.run_coroutine_threadsafe(_within(awaitable, timeout), loop)
        # Wait first, then read the result: a TimeoutError the call raises itself is the
        # same class as the wait expiring, and must not be mistaken for it.
        concurrent.futures.wait([future], None if timeout is None else timeout + UNWIND_GRACE_S)
        if not future.done():
            # Cancelled on the loop at its deadline, but still unwinding after the grace
            # period. It is not cancelled again, which would cut its cleanup short.
            raise _Stranded(f"conduit call did not finish within {timeout} s")
        try:
            return future.result()
        except concurrent.futures.CancelledError:
            if self._closed:
                raise FacadeError("the conduit loop was closed while the call waited") from None
            raise

    def iterate[T](
        self, iterator: AsyncIterator[T], *, timeout: float | None = DEFAULT_TIMEOUT_S
    ) -> Generator[T, None, None]:
        """A blocking iterator over ``iterator``; each step has the timeout.

        A step that times out has been cancelled and unwound on the loop (closing the
        generator) before ``TimeoutError`` is raised. An iterator abandoned part-way
        closes its generator.
        """
        stranded = False
        try:
            while True:
                try:
                    yield self.run(_next(iterator), timeout=timeout)
                except StopAsyncIteration:
                    return
                except _Stranded:
                    stranded = True
                    raise
        finally:
            closer = getattr(iterator, "aclose", None)
            # A stranded step is still running on the loop; closing it there now would
            # fail, and its own cancellation closes the generator.
            if closer is not None and not stranded and not self._closed:
                self.run(closer(), timeout=timeout)

    def close(self, *, join_timeout: float | None = 5) -> None:
        """Cancel every pending call, finalise suspended async generators, then stop.

        A caller waiting on a cancelled call gets ``FacadeError``. Refused on the loop's
        own thread, where it would wait on itself.
        """
        if self.on_loop_thread():
            raise FacadeError(
                "the conduit loop cannot be closed from its own thread; close it from "
                "another thread"
            )
        with self._lock:
            if self._closed:
                return
            self._closed = True
            loop, thread = self._loop, self._thread
        if loop is None or thread is None:
            return
        shutdown = asyncio.run_coroutine_threadsafe(_shutdown(loop), loop)
        try:
            shutdown.result(join_timeout)
        except (concurrent.futures.TimeoutError, concurrent.futures.CancelledError):
            shutdown.cancel()
        loop.call_soon_threadsafe(loop.stop)
        thread.join(join_timeout)
        if not thread.is_alive():
            loop.close()

    def __enter__(self) -> LoopThread:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class _Stranded(TimeoutError):
    """A timed-out call that had not finished unwinding when the caller was released."""


async def _within[T](awaitable: Awaitable[T], timeout: float | None) -> T:
    """``awaitable`` under ``timeout``, on the loop: on expiry it is cancelled and has
    unwound before ``TimeoutError`` is raised."""
    deadline = asyncio.timeout(timeout)
    try:
        async with deadline:
            return await awaitable
    except TimeoutError:
        if deadline.expired():
            raise TimeoutError(f"conduit call did not finish within {timeout} s") from None
        raise


async def _next[T](iterator: AsyncIterator[T]) -> T:
    return await anext(iterator)


async def _shutdown(loop: asyncio.AbstractEventLoop) -> None:
    current = asyncio.current_task()
    pending = [task for task in asyncio.all_tasks(loop) if task is not current]
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    await loop.shutdown_asyncgens()


def _discard(awaitable: Awaitable[Any]) -> None:
    if inspect.iscoroutine(awaitable):
        awaitable.close()


_default: LoopThread | None = None
_default_lock = threading.Lock()


def default_loop() -> LoopThread:
    """The process-wide loop the facade uses unless given another."""
    global _default
    with _default_lock:
        if _default is None:
            _default = LoopThread()
            atexit.register(_default.close)
        return _default


class SyncFacade:
    """``target``'s async API as blocking calls on a conduit loop thread."""

    def __init__(
        self,
        target: Any,
        *,
        loop: LoopThread | None = None,
        timeout: float | None = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._target = target
        self._loop = loop or default_loop()
        self._timeout = timeout

    def with_timeout(self, timeout: float | None) -> SyncFacade:
        return SyncFacade(self._target, loop=self._loop, timeout=timeout)

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._target, name)
        if not callable(attribute):
            return attribute
        return self._wrap(attribute)

    def _wrap(self, function: Callable[..., Any]) -> Callable[..., Any]:
        loop, timeout = self._loop, self._timeout

        def call(*args: Any, **kwargs: Any) -> Any:
            result = function(*args, **kwargs)
            if isinstance(result, AsyncIterator):
                return loop.iterate(result, timeout=timeout)
            if inspect.isawaitable(result):
                return loop.run(result, timeout=timeout)
            return result

        call.__name__ = getattr(function, "__name__", "call")
        call.__doc__ = getattr(function, "__doc__", None)
        return call

    def __repr__(self) -> str:
        return f"SyncFacade({self._target!r})"


def sync(
    target: Any, *, loop: LoopThread | None = None, timeout: float | None = DEFAULT_TIMEOUT_S
) -> Any:
    """The sync facade over ``target``."""
    return SyncFacade(target, loop=loop, timeout=timeout)


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "LoopThread",
    "SyncFacade",
    "default_loop",
    "sync",
]
