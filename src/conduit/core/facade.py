"""The sync facade: conduit's async API, callable from synchronous code (C3).

A ``LoopThread`` owns an event loop on a daemon thread. ``LoopThread.run`` submits a
coroutine to it and waits, with a timeout; on timeout the call is cancelled. Calling
``run`` from the loop's own thread would wait on itself forever, so it is refused with
``FacadeError`` (the guard SPARK's database bridge uses).

``sync(obj)`` wraps any object with an async API: each coroutine method becomes a
blocking call and each async-generator method (``stream``, for example) becomes a
plain iterator, both running on the loop thread. Everything else passes through. The
facade is generated from the object at call time, so every async interface conduit
has, now or later, gets a sync form without a hand-written twin.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import inspect
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any

from conduit.core.errors import FacadeError

DEFAULT_TIMEOUT_S: float = 300.0


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
        """Run ``awaitable`` on the loop and wait for it. Refused on the loop's own thread."""
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
        future = asyncio.run_coroutine_threadsafe(_as_coroutine(awaitable), loop)
        try:
            return future.result(timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError(f"conduit call did not finish within {timeout} s") from None

    def iterate[T](
        self, iterator: AsyncIterator[T], *, timeout: float | None = DEFAULT_TIMEOUT_S
    ) -> Iterator[T]:
        """A blocking iterator over ``iterator``; each step has the timeout."""
        try:
            while True:
                try:
                    yield self.run(iterator.__anext__(), timeout=timeout)
                except StopAsyncIteration:
                    return
        finally:
            closer = getattr(iterator, "aclose", None)
            if closer is not None and not self._closed:
                self.run(closer(), timeout=timeout)

    def close(self, *, join_timeout: float | None = 5) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            loop, thread = self._loop, self._thread
        if loop is not None and thread is not None:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(join_timeout)
            if not thread.is_alive():
                loop.close()

    def __enter__(self) -> LoopThread:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


async def _as_coroutine[T](awaitable: Awaitable[T]) -> T:
    return await awaitable


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
