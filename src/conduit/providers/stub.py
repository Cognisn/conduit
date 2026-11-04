"""The stub provider: deterministic and scriptable, for tests. Not for production.

The stub never reaches a network. A script decides each turn from the request and the
call's index: ``echo`` (the default) repeats the last user text, ``scripted`` plays
turns in order, and ``answer_tool_results`` is reactive, calling a tool first and then
answering from the tool results it is sent. Every turn streams like a real provider
(usage, text in chunks, tool calls started and finished, a final stop with usage).

Usage is computed deterministically and carries all four counts. A request with a
cache breakpoint writes its prefix to the stub's cache the first time and reads it on
every later call with the same prefix, so cache accounting can be tested end to end.

Constructing a stub needs ``not_for_production=True``, in code and in settings.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Self

from pydantic import PositiveInt, SecretStr, model_validator

from conduit.core.errors import ConfigurationError, ProviderError
from conduit.core.provider import BaseProvider, Provider
from conduit.core.settings import ProviderSettings
from conduit.core.streaming import (
    StreamEvent,
    StreamStopped,
    TextDelta,
    ToolCallFinished,
    ToolCallStarted,
    UsageReported,
)
from conduit.core.types import (
    CacheBreakpoint,
    ModelRequest,
    StopReason,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)

logger = logging.getLogger("conduit.providers.stub")


@dataclass(frozen=True, slots=True)
class StubToolCall:
    name: str
    input: dict[str, Any] = field(default_factory=dict)
    id: str | None = None

    def block(self, call_id: str) -> ToolUseBlock:
        return ToolUseBlock(id=self.id or call_id, name=self.name, input=self.input)


@dataclass(frozen=True, slots=True)
class StubTurn:
    """One scripted model turn. Usage and stop reason are derived unless given."""

    text: str = ""
    tool_calls: tuple[StubToolCall, ...] = ()
    stop_reason: StopReason | None = None
    usage: Usage | None = None


StubScript = Callable[[ModelRequest, int], StubTurn]


def echo(request: ModelRequest, index: int) -> StubTurn:
    """Repeat the text of the last user message."""
    for message in reversed(request.messages):
        if message.role == "user":
            return StubTurn(
                text="".join(b.text for b in message.content if isinstance(b, TextBlock))
            )
    return StubTurn()


def scripted(turns: Sequence[StubTurn | ProviderError]) -> StubScript:
    """Play turns (or raise provider errors) in order, one per call."""
    steps = tuple(turns)

    def script(request: ModelRequest, index: int) -> StubTurn:
        if index >= len(steps):
            raise AssertionError(f"stub script is exhausted after {len(steps)} calls")
        step = steps[index]
        if isinstance(step, ProviderError):
            raise step
        return step

    return script


def _latest_tool_results(request: ModelRequest) -> tuple[ToolResultBlock, ...]:
    if not request.messages or request.messages[-1].role != "user":
        return ()
    return tuple(b for b in request.messages[-1].content if isinstance(b, ToolResultBlock))


def answer_tool_results(
    first: StubTurn, answer: Callable[[tuple[ToolResultBlock, ...]], StubTurn]
) -> StubScript:
    """Reactive: ``first`` until the request carries tool results, then ``answer(results)``."""

    def script(request: ModelRequest, index: int) -> StubTurn:
        results = _latest_tool_results(request)
        return answer(results) if results else first

    return script


def _tokens(text: str) -> int:
    return (len(text) + 3) // 4


def _segments(request: ModelRequest) -> list[str | None]:
    """The request as text segments in prompt order; ``None`` marks a cache breakpoint."""
    segments: list[str | None] = []
    if request.tools:
        segments.append(json.dumps([t.model_dump(mode="json") for t in request.tools]))
        if request.cache_tools:
            segments.append(None)
    for block in request.system:
        segments.append(None if isinstance(block, CacheBreakpoint) else block.text)
    for message in request.messages:
        for content in message.content:
            if isinstance(content, CacheBreakpoint):
                segments.append(None)
            else:
                segments.append(f"{message.role}:{content.model_dump_json()}")
    return segments


class StubProvider(BaseProvider):
    kind = "stub"

    def __init__(
        self, script: StubScript = echo, *, not_for_production: bool, chunk_size: int = 4
    ) -> None:
        if not not_for_production:
            raise ConfigurationError(
                "the stub provider is not for production; pass not_for_production=True"
            )
        logger.warning("stub provider built: deterministic test answers, not for production")
        self._script = script
        self._chunk = max(1, chunk_size)
        self._cache: set[str] = set()
        self.requests: list[ModelRequest] = []

    def _usage(self, request: ModelRequest, turn: StubTurn, calls: list[ToolUseBlock]) -> Usage:
        segments = _segments(request)
        last_break = max((i for i, s in enumerate(segments) if s is None), default=-1)
        prefix = "".join(s for s in segments[: last_break + 1] if s is not None)
        rest = "".join(s for s in segments[last_break + 1 :] if s is not None)
        read = write = 0
        if prefix:
            key = hashlib.sha256(prefix.encode("utf-8")).hexdigest()
            if key in self._cache:
                read = _tokens(prefix)
            else:
                write = _tokens(prefix)
                self._cache.add(key)
        output = turn.text + "".join(json.dumps(c.input, sort_keys=True) for c in calls)
        return Usage(
            input_tokens=_tokens(rest),
            output_tokens=_tokens(output),
            cache_read_tokens=read,
            cache_write_tokens=write,
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        index = len(self.requests)
        self.requests.append(request)
        turn = self._script(request, index)
        calls = [c.block(f"toolu_stub_{index}_{i}") for i, c in enumerate(turn.tool_calls)]
        usage = turn.usage or self._usage(request, turn, calls)
        yield UsageReported(
            usage=usage.model_copy(update={"output_tokens": 0}),
        )
        for start in range(0, len(turn.text), self._chunk):
            yield TextDelta(text=turn.text[start : start + self._chunk])
        for call in calls:
            yield ToolCallStarted(id=call.id, name=call.name)
            yield ToolCallFinished(id=call.id, name=call.name, input=call.input)
        stop = turn.stop_reason or (StopReason.TOOL_USE if calls else StopReason.END_TURN)
        yield StreamStopped(stop_reason=stop, usage=usage)


class StubSettings(ProviderSettings):
    """Settings for a configured stub. ``not_for_production: true`` is required."""

    kind: Literal["stub"] = "stub"
    script: str = "echo"
    not_for_production: bool = False
    chunk_size: PositiveInt = 4

    @model_validator(mode="after")
    def _acknowledged(self) -> Self:
        if not self.not_for_production:
            raise ValueError("a stub provider needs not_for_production: true")
        return self


class StubFactory:
    """The stub's adapter for the provider registry. Scripts are looked up by name."""

    kind = "stub"
    settings_model = StubSettings

    def __init__(self, scripts: Mapping[str, StubScript] | None = None) -> None:
        self._scripts: dict[str, StubScript] = {"echo": echo, **(scripts or {})}

    def create_client(self, settings: ProviderSettings, secrets: Mapping[str, SecretStr]) -> Any:
        return None  # the stub has no client

    def create_provider(self, name: str, settings: ProviderSettings, client: Any) -> Provider:
        if not isinstance(settings, StubSettings):
            raise ConfigurationError("stub settings expected")
        try:
            script = self._scripts[settings.script]
        except KeyError:
            raise ConfigurationError(f"no stub script named {settings.script!r}") from None
        return StubProvider(
            script,
            not_for_production=settings.not_for_production,
            chunk_size=settings.chunk_size,
        )
