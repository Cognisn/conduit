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

Starting conduit from sync code: ``start_sync`` takes ``start``'s arguments, runs
``start`` on a conduit loop and returns a ``SyncRuntime``. Nothing it hands out is
async: ``for_task`` gives a ``SyncTaskBinding`` and ``providers`` maps each name to a
``SyncProvider``, whose ``complete`` blocks and whose ``stream`` is a plain iterator,
each running on the loop that built the provider::

    with start_sync(settings, task_classes=..., factories=..., secrets=...) as runtime:
        binding = runtime.for_task("triage")
        response = binding.complete(binding.request(messages=[Message.user("...")]))
        for event in binding.stream(binding.request(messages=[...])):
            ...

``close`` (or leaving the ``with`` block) closes every provider. ``with_timeout`` gives
the same runtime with another per-call timeout. ``start_sync`` is refused on the
loop's own thread, like every facade call.

``sync(obj)`` is the general form, for any other async object. What a ``sync`` facade
method returns comes back as it is, never wrapped (a ``TaskBinding`` from
``Runtime.for_task``, say, whose ``provider`` is async), so a caller using it directly
wraps each async object it is handed, on the same loop. Whatever starts conduit must
run on the loop the facade uses (``loop.run(start(...))``, never ``asyncio.run``):
providers make their SDK clients, connection pools and locks on the loop that builds
them, and must be called on that loop. ``start_sync`` does this for you.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import inspect
import logging
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Generator, Iterable, Iterator
from types import MappingProxyType, TracebackType
from typing import Any

from conduit.core.catalogue import Catalogue
from conduit.core.errors import FacadeError
from conduit.core.events import EventSink
from conduit.core.registry import MeteredProvider, ProviderFactory
from conduit.core.runtime import ConduitSettings, Runtime, start, task_request
from conduit.core.secrets import SecretResolver
from conduit.core.streaming import StreamEvent
from conduit.core.tasks import Assignments, ResolvedAssignment, TaskClass
from conduit.core.types import Message, ModelRequest, ModelResponse, SystemBlock, ToolSpec

logger = logging.getLogger("conduit.facade")

DEFAULT_TIMEOUT_S: float = 300.0
#: How long a timed-out call may take to unwind before its caller is released anyway.
UNWIND_GRACE_S: float = 10.0
#: How long ``close(join_timeout=None)`` lets calls still pending after the shutdown unwind.
SETTLE_S: float = 5.0


class LoopThread:
    """An event loop on a thread conduit owns."""

    def __init__(self, *, name: str = "conduit-loop") -> None:
        self._name = name
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._thread_id: int | None = None
        self._closed = False
        # Every call submitted and not yet finished. Registered under the lock, with the
        # closed check, so close() fails whatever its shutdown did not reach.
        self._pending: set[concurrent.futures.Future[Any]] = set()

    def _ensure_locked(self) -> asyncio.AbstractEventLoop:
        """The loop, started on first use. The caller holds ``_lock``."""
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

    def _forget(self, future: concurrent.futures.Future[Any]) -> None:
        with self._lock:
            self._pending.discard(future)

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
        with self._lock:
            try:
                loop = self._ensure_locked()
            except FacadeError:
                _discard(awaitable)
                raise
            future = asyncio.run_coroutine_threadsafe(_within(awaitable, timeout), loop)
            self._pending.add(future)
        future.add_done_callback(self._forget)
        # Wait with exception(), then read the result: it raises TimeoutError only when the
        # wait expires, never for a TimeoutError the call raised itself, and it wakes when
        # close() cancels the future (concurrent.futures.wait would not).
        # conduit's own errors are raised after the handler, so neither keeps the
        # exception it replaces (no __cause__, no __context__).
        stranded = False
        try:
            future.exception(None if timeout is None else timeout + UNWIND_GRACE_S)
            return future.result()
        except concurrent.futures.CancelledError:
            if not self._closed:
                raise
        except TimeoutError:
            if future.done():
                raise  # the call's own TimeoutError, from result()
            # Cancelled on the loop at its deadline, but still unwinding after the grace
            # period. It is not cancelled again, which would cut its cleanup short.
            stranded = True
        if stranded:
            raise _Stranded(f"conduit call did not finish within {timeout} s")
        raise FacadeError("the conduit loop was closed while the call waited")

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

        A caller waiting on a cancelled call gets ``FacadeError``, as does one whose call
        ignored cancellation past ``join_timeout``. Refused on the loop's own thread,
        where it would wait on itself. Each of its three steps (the shutdown, the thread
        join, and the settle that unwinds whatever the shutdown did not reach) may take
        up to ``join_timeout`` (the settle, ``SETTLE_S`` when that is ``None``), so
        ``close`` can take up to three times that.
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
        # Fail every call the shutdown did not finish (one that ignored cancellation, or
        # one submitted as close began), so no caller is left waiting on a stopped loop.
        with self._lock:
            stranded = list(self._pending)
        for future in stranded:
            future.cancel()
        if not thread.is_alive():
            _settle(loop, SETTLE_S if join_timeout is None else join_timeout)
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
        if not deadline.expired():
            raise  # the call's own TimeoutError
    # Raised after the handler, so it keeps neither the deadline's error nor its context.
    raise TimeoutError(f"conduit call did not finish within {timeout} s")


async def _next[T](iterator: AsyncIterator[T]) -> T:
    return await anext(iterator)


async def _shutdown(loop: asyncio.AbstractEventLoop) -> None:
    current = asyncio.current_task()
    pending = [task for task in asyncio.all_tasks(loop) if task is not current]
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    await loop.shutdown_asyncgens()


#: Tasks a closed loop could not finish: each ignored cancellation past its bound. They
#: are kept referenced, deliberately, so none is garbage-collected while pending (which
#: asyncio would log as "Task was destroyed but it is pending").
_ABANDONED: set[asyncio.Task[Any]] = set()


async def _sweep(timeout: float) -> None:
    """Cancel every other task on the running loop and give them ``timeout`` to unwind."""
    current = asyncio.current_task()
    pending = {task for task in asyncio.all_tasks() if task is not current}
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.wait(pending, timeout=timeout)


def _settle(loop: asyncio.AbstractEventLoop, timeout: float) -> None:
    """On ``loop``, stopped and with its thread gone, finish what the shutdown did not.

    A call submitted as close began, or the shutdown itself when it gave up on a call
    ignoring cancellation, can still be pending. The loop is run here, on the closing
    thread, until each is cancelled and unwound or ``timeout`` passes; any still pending
    then is kept in ``_ABANDONED``. A thread already running another loop cannot run
    this one, so there they are kept without the sweep.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        if any(not task.done() for task in asyncio.all_tasks(loop)):
            loop.run_until_complete(_sweep(timeout))
    _ABANDONED.update(task for task in asyncio.all_tasks(loop) if not task.done())


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


class SyncProvider:
    """A built provider as sync code uses it: blocking calls on the loop that built it."""

    def __init__(self, provider: MeteredProvider, loop: LoopThread, timeout: float | None) -> None:
        self._provider = provider
        self._loop = loop
        self.timeout = timeout

    @property
    def name(self) -> str:
        return self._provider.name

    @property
    def kind(self) -> str:
        return self._provider.kind

    @property
    def assigned_models(self) -> frozenset[str] | None:
        return self._provider.assigned_models

    def complete(self, request: ModelRequest) -> ModelResponse:
        """The collected response. Blocks; raises ``TimeoutError`` past the timeout."""
        return self._loop.run(self._provider.complete(request), timeout=self.timeout)

    def stream(self, request: ModelRequest) -> Iterator[StreamEvent]:
        """The call's events as a plain iterator; each step has the timeout."""
        return self._loop.iterate(self._provider.stream(request), timeout=self.timeout)

    def __repr__(self) -> str:
        return f"SyncProvider(name={self.name!r}, kind={self.kind!r})"


class SyncTaskBinding:
    """A task class's provider and resolved assignment, for sync code."""

    def __init__(self, provider: SyncProvider, assignment: ResolvedAssignment) -> None:
        self.provider = provider
        self.assignment = assignment

    def request(
        self,
        *,
        messages: Iterable[Message],
        system: Iterable[SystemBlock] = (),
        tools: Iterable[ToolSpec] = (),
        **fields: Any,
    ) -> ModelRequest:
        """A request on this task's model, within its output budget (as
        ``TaskBinding.request``)."""
        return task_request(
            self.assignment, messages=messages, system=system, tools=tools, **fields
        )

    def complete(self, request: ModelRequest) -> ModelResponse:
        return self.provider.complete(request)

    def stream(self, request: ModelRequest) -> Iterator[StreamEvent]:
        return self.provider.stream(request)

    def __repr__(self) -> str:
        return f"SyncTaskBinding(task_class={self.assignment.task_class.name!r})"


class SyncRuntime:
    """A started ``Runtime`` for sync code. Nothing it hands out is async."""

    def __init__(
        self, runtime: Runtime, loop: LoopThread, timeout: float | None = DEFAULT_TIMEOUT_S
    ) -> None:
        self._runtime = runtime
        self._loop = loop
        self.timeout = timeout

    @property
    def catalogue(self) -> Catalogue:
        return self._runtime.catalogue

    @property
    def assignments(self) -> Assignments:
        return self._runtime.assignments

    @property
    def providers(self) -> MappingProxyType[str, SyncProvider]:
        """Every built provider, by configured name, as a sync view."""
        return MappingProxyType(
            {
                name: SyncProvider(provider, self._loop, self.timeout)
                for name, provider in self._runtime.providers.items()
            }
        )

    def for_task(self, task_class: str) -> SyncTaskBinding:
        """The task class's binding. Raises ``UnknownTaskClass`` for one not assigned."""
        binding = self._runtime.for_task(task_class)
        return SyncTaskBinding(
            SyncProvider(binding.provider, self._loop, self.timeout), binding.assignment
        )

    def with_timeout(self, timeout: float | None) -> SyncRuntime:
        """The same runtime, with another per-call timeout."""
        return SyncRuntime(self._runtime, self._loop, timeout)

    def close(self) -> None:
        """Close every provider. A second close does nothing."""
        self._loop.run(self._runtime.aclose(), timeout=self.timeout)

    def __enter__(self) -> SyncRuntime:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close. While the block is raising, a close that fails is logged by class name
        only and the block's exception propagates; otherwise the close failure is raised
        with neither cause nor context."""
        if exc is None:
            self.close()
            return
        failed: str | None = None
        try:
            self.close()
        except Exception as error:
            failed = type(error).__name__
        if failed is not None:
            logger.warning("closing the sync runtime failed: %s", failed)

    def __repr__(self) -> str:
        return f"SyncRuntime(providers={sorted(self._runtime.providers)})"


def start_sync(
    settings: ConduitSettings,
    *,
    task_classes: Iterable[TaskClass],
    factories: Iterable[ProviderFactory],
    secrets: SecretResolver,
    events: EventSink | None = None,
    catalogue: Catalogue | None = None,
    loop: LoopThread | None = None,
    timeout: float | None = DEFAULT_TIMEOUT_S,
) -> SyncRuntime:
    """``start`` for sync code: run on ``loop`` (the default loop unless given), with
    ``timeout`` for start-up and for every later call. Raises what ``start`` raises."""
    owner = loop or default_loop()
    runtime = owner.run(
        start(
            settings,
            task_classes=task_classes,
            factories=factories,
            secrets=secrets,
            events=events,
            catalogue=catalogue,
        ),
        timeout=timeout,
    )
    return SyncRuntime(runtime, owner, timeout)


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "LoopThread",
    "SyncFacade",
    "SyncProvider",
    "SyncRuntime",
    "SyncTaskBinding",
    "default_loop",
    "start_sync",
    "sync",
]
