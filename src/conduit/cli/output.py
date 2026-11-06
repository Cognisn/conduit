"""What the terminal shows, and the rules for showing it safely.

- **Secrets are never printed.** An error is shown by its class name only, unless it is a
  ``ConduitError``, whose message names configuration (a provider, a field, a secret's
  reference) and never a value or provider text. Settings are shown by their kinds,
  regions and modes; a credential appears only as its ``secret://`` reference.
- **Text from outside is shown as text.** Model output, tool results and anything else
  the interface did not write goes through ``sanitise_untrusted`` (control, escape and
  invisible characters become visible markers, so it cannot drive the terminal) and is
  printed without rich markup or emoji codes, so brackets and ``:name:`` in it are shown
  as written. Every line goes through ``Screen``, table cells and titles included.
- **Events are shown as metadata.** Usage, cache reads and writes, compactions, skill
  loads and MCP calls are shown by their counts, names and outcomes, as conduit's events
  carry them.
"""

from __future__ import annotations

from collections.abc import Sequence

from rich.console import Console
from rich.table import Table
from rich.text import Text

from conduit import ConduitError, Event, ModelCallFinishedEvent, Usage
from conduit.agent import sanitise_untrusted
from conduit.compaction import CompactionEvent
from conduit.skills import (
    SkillLoadedEvent,
    SkillRefusedEvent,
    SkillUnloadedEvent,
    SkillUsedEvent,
)


def describe_error(error: BaseException) -> str:
    """An error as the terminal shows it: never a value it might carry."""
    if isinstance(error, ConduitError):
        return f"{type(error).__name__}: {sanitise_untrusted(str(error))}"
    return type(error).__name__


def usage_text(usage: Usage | None) -> str:
    """The four usage counts, named as conduit names them."""
    if usage is None:
        return "no usage reported"
    return (
        f"input {usage.input_tokens}, output {usage.output_tokens}, "
        f"cache read {usage.cache_read_tokens}, cache write {usage.cache_write_tokens}"
    )


class Screen:
    """The console, written to only through these methods."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self._open = False  # streamed text has left a line unfinished

    def end_line(self) -> None:
        """Finish a line streamed text left open, so the next line starts on its own."""
        if self._open:
            self.console.print()
            self._open = False

    def say(self, text: str, *, style: str | None = None) -> None:
        """A line the interface wrote itself (it may name configuration)."""
        self.end_line()
        self.console.print(
            sanitise_untrusted(text),
            style=style,
            markup=False,
            emoji=False,
            highlight=False,
            soft_wrap=True,
        )

    def stream(self, text: str) -> None:
        """A piece of streamed model output, left on an open line."""
        if text:
            self.console.print(
                sanitise_untrusted(text),
                end="",
                markup=False,
                emoji=False,
                highlight=False,
                soft_wrap=True,
            )
            self._open = True

    def error(self, text: str) -> None:
        self.say(text, style="red")

    def note(self, text: str) -> None:
        self.say(text, style="dim")

    def table(self, title: str, columns: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
        table = Table(title=Text(sanitise_untrusted(title)), title_justify="left", highlight=False)
        for column in columns:
            table.add_column(column, overflow="fold")
        for row in rows:
            table.add_row(*(Text(sanitise_untrusted(cell)) for cell in row))
        self.end_line()
        self.console.print(table)


class EventPrinter:
    """An event sink that shows usage, cache, compaction and skill events as they happen."""

    def __init__(self, screen: Screen, *, verbose: bool = False) -> None:
        self._screen = screen
        self.verbose = verbose
        self.calls = 0
        self.usage = Usage.zero()

    async def emit(self, event: Event) -> None:
        screen = self._screen
        if isinstance(event, ModelCallFinishedEvent):
            self.calls += 1
            if event.usage is not None:
                self.usage = self.usage + event.usage
            where = f"{event.provider} {event.model}"
            if event.outcome == "ok":
                screen.note(f"[usage] {where}: {usage_text(event.usage)}")
                if event.usage is not None and (
                    event.usage.cache_read_tokens or event.usage.cache_write_tokens
                ):
                    screen.note(
                        f"[cache] read {event.usage.cache_read_tokens}, "
                        f"written {event.usage.cache_write_tokens}"
                    )
            else:
                why = event.failure.value if event.failure else event.error_type or "failed"
                screen.note(f"[usage] {where}: failed ({why}); {usage_text(event.usage)}")
        elif isinstance(event, CompactionEvent):
            head = f"[compaction] {event.trigger} {event.strategy}: {event.outcome}"
            if event.outcome == "compacted":
                screen.say(
                    f"{head}; before {event.tokens_before} tokens in {event.messages_before} "
                    f"messages, after {event.tokens_after} tokens in {event.messages_after} "
                    f"messages"
                )
            else:
                screen.say(f"{head} ({event.reason}); {event.tokens_before} tokens")
        elif isinstance(event, SkillLoadedEvent):
            screen.note(
                f"[skill] loaded {event.skill} ({event.body_size} bytes, "
                f"{event.resource_count} resources); {event.loaded_count} loaded"
            )
        elif isinstance(event, SkillUsedEvent):
            what = f" resource {event.resource}" if event.resource else ""
            screen.note(f"[skill] used {event.skill}{what}")
        elif isinstance(event, SkillUnloadedEvent):
            screen.note(f"[skill] unloaded {event.skill}")
        elif isinstance(event, SkillRefusedEvent):
            screen.note(f"[skill] refused ({event.reason})")
        elif event.type == "mcp_call":
            # McpCallEvent lives behind the mcp extra; matched by its type, read by name.
            server = getattr(event, "server", "?")
            tool = getattr(event, "remote_tool", "?")
            outcome = getattr(event, "outcome", "?")
            code = getattr(event, "error_code", None)
            screen.note(f"[mcp] {server} {tool}: {outcome}" + (f" ({code})" if code else ""))
        elif self.verbose:
            screen.note(f"[event] {event.type}")
