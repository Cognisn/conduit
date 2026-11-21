"""The base every conformance suite builds on."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, ClassVar

#: Scenarios abandoned at their bound. Their loops are closed, so they never run again;
#: they are kept referenced on purpose, so asyncio never reports them as destroyed while
#: pending in whatever test happens to collect them. The leak is deliberate and small: a
#: hung backend's scenario per failing test.
_ABANDONED: set[asyncio.Future[Any]] = set()


class ConformanceSuite:
    """A suite of scenarios one backend interface must pass.

    Each test method runs one async scenario on a fresh event loop from
    ``loop_factory`` (the default policy's loop when ``None``), bounded by
    ``scenario_timeout_s``: a backend that hangs fails the scenario, saying so, rather
    than hanging the test run. When the bound is reached the scenario is cancelled; if
    it has still not finished ``scenario_cancel_grace_s`` later (a close that hangs as
    well, or a backend that swallows the cancellation), the suite stops waiting for it,
    abandons its event loop and fails the same way. The bound works through the event
    loop, so a backend that blocks the loop itself (``time.sleep``, a synchronous
    network call) is not bounded: run blocking work in a worker thread.
    """

    #: The interface the suite checks, for reports.
    interface: ClassVar[str] = ""
    #: The event loop each scenario runs on; set it for backends that need another loop.
    loop_factory: ClassVar[Callable[[], asyncio.AbstractEventLoop] | None] = None
    #: The longest one scenario may take, making and closing the backend included. Room
    #: for a database reached over a network; raise it for a slower backend.
    scenario_timeout_s: ClassVar[float] = 60.0
    #: How long a scenario cancelled at the bound may take to finish before the suite
    #: stops waiting for it.
    scenario_cancel_grace_s: ClassVar[float] = 5.0

    def run(self, scenario: Callable[[], Awaitable[None]]) -> None:
        suite = type(self)
        bound = suite.scenario_timeout_s
        expired = False
        abandoned = False

        async def main() -> None:
            nonlocal expired, abandoned
            task = asyncio.ensure_future(scenario())
            done, _ = await asyncio.wait({task}, timeout=bound)
            if task in done:
                await task  # the scenario's own outcome, its failures and timeouts included
                return
            expired = True
            task.cancel()
            done, _ = await asyncio.wait({task}, timeout=suite.scenario_cancel_grace_s)
            if task not in done:
                abandoned = True
                _ABANDONED.add(task)
            elif not task.cancelled():
                task.exception()  # retrieved, so the loop does not report it as unread

        runner = asyncio.Runner(loop_factory=suite.loop_factory)
        try:
            runner.run(main())
        finally:
            if abandoned:
                # Closing the runner would cancel the stuck scenario again and wait for it
                # without end; close only the loop, leaving the scenario behind.
                runner.get_loop().close()
                if suite.loop_factory is None:
                    # As Runner.close does: no closed loop is left as the current one.
                    asyncio.set_event_loop(None)
            else:
                runner.close()
        # Raised here, outside every handler, so it keeps no timeout as its context.
        if expired:
            raise AssertionError(
                f"the scenario did not finish within {bound} s: the backend hung "
                "(raise scenario_timeout_s if it is only slow)"
            )
