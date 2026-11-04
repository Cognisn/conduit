"""The base every conformance suite builds on."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import ClassVar


class ConformanceSuite:
    """A suite of scenarios one backend interface must pass.

    Each test method runs one async scenario on a fresh event loop from
    ``loop_factory`` (the default policy's loop when ``None``).
    """

    #: The interface the suite checks, for reports.
    interface: ClassVar[str] = ""
    #: The event loop each scenario runs on; set it for backends that need another loop.
    loop_factory: ClassVar[Callable[[], asyncio.AbstractEventLoop] | None] = None

    def run(self, scenario: Callable[[], Awaitable[None]]) -> None:
        async def main() -> None:
            await scenario()

        with asyncio.Runner(loop_factory=type(self).loop_factory) as runner:
            runner.run(main())
