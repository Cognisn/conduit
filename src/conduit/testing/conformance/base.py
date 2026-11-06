"""The base every conformance suite builds on."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import ClassVar


class ConformanceSuite:
    """A suite of scenarios one backend interface must pass.

    Each test method runs one async scenario on a fresh event loop from
    ``loop_factory`` (the default policy's loop when ``None``), bounded by
    ``scenario_timeout_s``: a backend that hangs fails the scenario, saying so, rather
    than hanging the test run.
    """

    #: The interface the suite checks, for reports.
    interface: ClassVar[str] = ""
    #: The event loop each scenario runs on; set it for backends that need another loop.
    loop_factory: ClassVar[Callable[[], asyncio.AbstractEventLoop] | None] = None
    #: The longest one scenario may take, making and closing the backend included. Room
    #: for a database reached over a network; raise it for a slower backend.
    scenario_timeout_s: ClassVar[float] = 60.0

    def run(self, scenario: Callable[[], Awaitable[None]]) -> None:
        bound = type(self).scenario_timeout_s
        expired = False

        async def main() -> None:
            nonlocal expired
            scope = asyncio.timeout(bound)
            try:
                async with scope:
                    await scenario()
            except TimeoutError:
                if not scope.expired():
                    raise  # the scenario's own timeout, not the bound
                expired = True

        with asyncio.Runner(loop_factory=type(self).loop_factory) as runner:
            runner.run(main())
        # Raised here, outside every handler, so it keeps no timeout as its context.
        if expired:
            raise AssertionError(
                f"the scenario did not finish within {bound} s: the backend hung "
                "(raise scenario_timeout_s if it is only slow)"
            )
