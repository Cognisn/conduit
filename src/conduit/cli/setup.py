"""What every command shares: the interface's context, provider factories and task classes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from conduit import CapabilityFloor, EnvironmentSecretResolver, ProviderFactory, TaskClass
from conduit.cli.config import CHAT, SUMMARY, CliConfig
from conduit.cli.demo import demo_script
from conduit.cli.output import Screen
from conduit.providers import builtin_factories
from conduit.providers.stub import StubFactory


class LineSource(Protocol):
    """Where the person's input comes from: standard input, or a test's."""

    def readline(self) -> str: ...

    def isatty(self) -> bool: ...


@dataclass
class Context:
    """Where a command reads and writes, and what it is built from.

    ``open_url`` opens the OAuth sign-in page (the browser by default). ``factories``
    are the provider adapters (``interface_factories()`` by default).
    """

    screen: Screen
    stdin: LineSource
    environ: Mapping[str, str]
    factories: Sequence[ProviderFactory]
    open_url: Callable[[str], Any] | None = None

    def secrets(self) -> EnvironmentSecretResolver:
        """``secret://NAME`` resolved from the environment variable ``NAME``."""
        return EnvironmentSecretResolver(environ=self.environ)


def interface_factories() -> list[ProviderFactory]:
    """Every adapter conduit ships, with the stub also offering the ``demo`` script."""
    others = [factory for factory in builtin_factories() if factory.kind != "stub"]
    return [StubFactory(scripts={"demo": demo_script}), *others]


#: The task classes the interface declares, as a product declares its own.
CHAT_CLASS = TaskClass(
    name=CHAT,
    floor=CapabilityFloor(tools=True),
    description="The conversation: a chat model with tool use.",
)
SUMMARY_CLASS = TaskClass(
    name=SUMMARY,
    description="Summaries for the summarising compaction strategy.",
)


def task_classes(config: CliConfig) -> list[TaskClass]:
    """``chat``, and ``summary`` when the configuration assigns it."""
    classes = [CHAT_CLASS]
    if SUMMARY in config.conduit.assignments:
        classes.append(SUMMARY_CLASS)
    return classes
