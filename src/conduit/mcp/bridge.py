"""MCP tools bridged into conduit's tool catalogue (C14, C19).

``McpToolBridge`` lists the tools of connected servers and gives each a
``McpTool``: a declared tool for ``ToolCatalogue.from_tools``, named
``<server>__<tool>`` and serving only the task classes the product granted the
server (narrowed per tool by ``tool_task_classes``). The bridge is also the
``ToolExecutor`` for those tools; ``executor(product)`` hands every other tool to
the product's own executor.

Naming. A server's tool name is accepted only if it holds nothing but letters,
digits, ``_``, ``.`` and ``-`` (at most 128 characters). It is lower-cased and every
run of other characters becomes one ``_``; server names never hold ``__``, so the
name splits one way. A name the catalogue cannot hold (over 64 characters) is left
out, and two tools of one server that reach the same name are both left out rather
than one shadowing the other.

Descriptions are untrusted. A tool's description is the server's, so it is marked
with its source, and it reaches the model in the tool catalogue: it is made safe by
the core's ``sanitise_untrusted``, then its whitespace collapsed to single spaces. The
descriptions and titles inside its input schema are made safe the same way. Every other
string in the schema (property names, ``enum``, ``const``, ``default``, ``examples``,
``$comment``, ``pattern`` and the rest) is kept as the server sent it, since changing
it would change what the tool accepts; a tool with such a string holding a hidden
character (by the core's rule, ``conduit.core.untrusted.is_hidden_character``: a
control, format or surrogate character, a default-ignorable character, or a line or
paragraph separator) or anything resembling a frame marker is left out instead, as a
tool whose name cannot be namespaced is.

Results are untrusted. A result goes back to the model as the core's untrusted block
(``conduit.core.untrusted.frame_untrusted``), ``<<UNTRUSTED source="mcp"
server="<server>" tool="<tool>" nonce="...">>``, so it is sanitised by the same rules
as every other external text (base64 runs included) and cut at ``max_result_chars``.

A result is framed here exactly once: every ``McpTool`` declares ``untrusted_framed``,
so the agent runner sends it as it is rather than framing it again. Give ``McpTool``
only to the bridge's executor (``McpToolBridge.executor``), which is what that
declaration promises.

Every call emits one ``McpCallEvent`` to the product's sink, however it ends,
with the arguments and result as hashes and sizes; their text only when the
capture policy allows tool content. A failing sink is logged by class name and
never fails the call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from pydantic import ValidationError

from conduit.core.errors import ConfigurationError, UnknownTool
from conduit.core.events import CapturePolicy, ContentKind, EventSink
from conduit.core.tools import ToolCallContext, ToolExecutor, ToolOutcome
from conduit.core.untrusted import (
    ESCAPED_DELIMITER,
    frame_untrusted,
    is_hidden_character,
    sanitise_untrusted,
)
from conduit.mcp.client import CallResult, McpConnection, RemoteTool
from conduit.mcp.errors import McpError
from conduit.mcp.events import McpCallEvent

__all__ = ["DEFAULT_MAX_RESULT_CHARS", "McpTool", "McpToolBridge", "namespaced_name"]

logger = logging.getLogger("conduit.mcp")

_REMOTE_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_CATALOGUE_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SEPARATOR = "__"
_DESCRIPTION_LIMIT = 1024
#: The schema keys whose text a model reads as prose.
_SCHEMA_PROSE = frozenset({"description", "title"})
#: The default cap on a result's length, in characters, as the agent runner's.
DEFAULT_MAX_RESULT_CHARS = 20_000
#: What frames a result with no cap: longer than any result a server can send.
_NO_CAP = sys.maxsize


def namespaced_name(server: str, tool: str) -> str | None:
    """``<server>__<tool>`` as the catalogue holds it, or ``None`` if it cannot be."""
    if not _REMOTE_NAME.fullmatch(tool):
        return None
    local = re.sub(r"[^a-z0-9]+", "_", tool.lower()).strip("_")
    if not local:
        return None
    name = f"{server}{_SEPARATOR}{local}"
    return name if _CATALOGUE_NAME.fullmatch(name) else None


def _raw_safe(text: str) -> bool:
    """Whether ``text`` can reach a model as it is: no character the core's rule hides
    (``is_hidden_character``) and nothing ``sanitise_untrusted`` would escape as a frame
    marker."""
    if any(is_hidden_character(char) for char in text):
        return False
    return sanitise_untrusted(text).count(ESCAPED_DELIMITER) == text.count(ESCAPED_DELIMITER)


class _Unsafe(Exception):
    """A schema string that cannot be shown to a model unchanged."""


def _safe_schema(value: Any) -> Any:
    """``value`` with every description and title made safe, at any depth; ``_Unsafe``
    for any other string, or any key, that is not safe as it is."""
    if isinstance(value, Mapping):
        safe: dict[Any, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and not _raw_safe(key):
                raise _Unsafe
            if key in _SCHEMA_PROSE and isinstance(item, str):
                safe[key] = sanitise_untrusted(item)
            else:
                safe[key] = _safe_schema(item)
        return safe
    if isinstance(value, list):
        return [_safe_schema(item) for item in value]
    if isinstance(value, str) and not _raw_safe(value):
        raise _Unsafe
    return value


def _schema(remote: RemoteTool) -> dict[str, Any] | None:
    """The tool's input schema as the catalogue holds it, or ``None`` when it holds a
    string that cannot reach a model unchanged."""
    schema = remote.input_schema
    if not (isinstance(schema, Mapping) and schema.get("type") == "object"):
        return {"type": "object"}
    unsafe = False
    safe: Any = None
    try:
        safe = _safe_schema(schema)
    except _Unsafe:
        unsafe = True
    return None if unsafe else dict(safe)


def _description(server: str, remote: RemoteTool) -> str:
    text = " ".join(sanitise_untrusted(remote.description).split())
    if not text:
        text = f"Tool {remote.name}."
    described = f"[MCP server {server}] {text}"
    return described[:_DESCRIPTION_LIMIT]


@dataclass(frozen=True, eq=False)
class McpTool:
    """One MCP server tool, declared for the catalogue.

    ``untrusted_framed`` tells the agent runner that the bridge frames its results as
    untrusted itself, so they are framed once (``conduit.agent.tool_results_preframed``).
    """

    #: The bridge frames this tool's results; the runner sends them as they are.
    untrusted_framed: ClassVar[bool] = True

    name: str
    description: str
    task_classes: frozenset[str]
    input_schema: dict[str, Any]
    server: str
    remote_name: str
    connection: McpConnection = field(repr=False)


class McpToolBridge:
    """The tools of one or more MCP connections, and their executor."""

    def __init__(
        self,
        connections: Iterable[McpConnection],
        *,
        events: EventSink | None = None,
        capture: CapturePolicy | None = None,
        max_result_chars: int | None = DEFAULT_MAX_RESULT_CHARS,
    ) -> None:
        if max_result_chars is not None and max_result_chars < 1:
            raise ConfigurationError("max_result_chars must be positive, or None for no cap")
        self._max_result_chars = max_result_chars
        self._connections: dict[str, McpConnection] = {}
        for connection in connections:
            if connection.name in self._connections:
                raise ConfigurationError(f"two MCP connections are named {connection.name!r}")
            self._connections[connection.name] = connection
        self._events = events
        self._capture = capture or CapturePolicy()

    def __repr__(self) -> str:
        return f"McpToolBridge({sorted(self._connections)!r})"

    async def tools(self) -> tuple[McpTool, ...]:
        """Every bridged tool of every connection. A server that fails raises ``McpError``."""
        found: list[McpTool] = []
        for connection in self._connections.values():
            found += self._bridge(connection, await connection.list_tools())
        return tuple(found)

    def _bridge(self, connection: McpConnection, remotes: Iterable[RemoteTool]) -> list[McpTool]:
        settings = connection.settings
        server = settings.name
        allowed = settings.allowed_tools
        candidates: dict[str, list[RemoteTool]] = {}
        for remote in remotes:
            if allowed is not None and remote.name not in allowed:
                continue
            name = namespaced_name(server, remote.name)
            if name is None:
                logger.warning("MCP server %s: a tool name cannot be namespaced; left out", server)
                continue
            candidates.setdefault(name, []).append(remote)
        bridged: list[McpTool] = []
        for name, remotes_for_name in candidates.items():
            if len(remotes_for_name) > 1:
                logger.warning(
                    "MCP server %s: %d tools share the name %s; all left out",
                    server,
                    len(remotes_for_name),
                    name,
                )
                continue
            (remote,) = remotes_for_name
            schema = _schema(remote)
            if schema is None:
                logger.warning(
                    "MCP server %s: a tool's input schema holds unsafe text; left out", server
                )
                continue
            granted = settings.tool_task_classes.get(remote.name, settings.task_classes)
            bridged.append(
                McpTool(
                    name=name,
                    description=_description(server, remote),
                    task_classes=frozenset(granted) & settings.task_classes,
                    input_schema=schema,
                    server=server,
                    remote_name=remote.name,
                    connection=connection,
                )
            )
        return bridged

    def executor(self, fallback: ToolExecutor | None = None) -> ToolExecutor:
        """An executor running MCP tools here and every other tool through ``fallback``."""

        async def run(
            tool: Any, arguments: dict[str, Any], context: ToolCallContext
        ) -> ToolOutcome:
            if isinstance(tool, McpTool) or fallback is None:
                return await self(tool, arguments, context)
            return await fallback(tool, arguments, context)

        return run

    async def __call__(
        self, tool: Any, arguments: dict[str, Any], context: ToolCallContext
    ) -> ToolOutcome:
        if (
            not isinstance(tool, McpTool)
            or self._connections.get(tool.server) is not tool.connection
        ):
            raise UnknownTool(
                f"not an MCP tool of this bridge, for task class {context.task_class}"
            )
        started = time.perf_counter()
        arguments_text = json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)
        if context.task_class not in tool.task_classes:
            await self._emit(tool, context, arguments_text, None, "forbidden", started)
            return ToolOutcome(
                ok=False,
                content=f"The tool {tool.name} is not available to this task.",
                error_code="forbidden",
            )
        result: CallResult | None = None
        failure: str | None = None
        try:
            result = await tool.connection.call_tool(tool.remote_name, arguments)
        except McpError as error:
            failure = error.failure.value
        except asyncio.CancelledError:
            await self._emit(tool, context, arguments_text, None, "cancelled", started)
            raise
        if result is None:
            await self._emit(tool, context, arguments_text, None, failure, started)
            return ToolOutcome(
                ok=False,
                content=f"The MCP server {tool.server} could not complete the call ({failure}).",
                error_code=failure,
            )
        error_code = "tool_error" if result.is_error else None
        await self._emit(tool, context, arguments_text, result.text, error_code, started)
        return ToolOutcome(
            ok=not result.is_error,
            content=frame_untrusted(
                result.text,
                source="mcp",
                attributes={"server": tool.server, "tool": tool.remote_name},
                max_length=self._max_result_chars or _NO_CAP,
            ).text,
            error_code=error_code,
        )

    async def _emit(
        self,
        tool: McpTool,
        context: ToolCallContext,
        arguments: str,
        result: str | None,
        error_code: str | None,
        started: float,
    ) -> None:
        if self._events is None:
            return
        event: McpCallEvent | None = None
        try:
            event = McpCallEvent(
                call_id=context.call_id,
                run_id=context.run_id,
                correlation_id=context.correlation_id,
                task_class=context.task_class,
                server=tool.server,
                tool=tool.name,
                remote_tool=tool.remote_name,
                outcome="ok" if error_code is None else "failed",
                error_code=error_code,
                arguments=self._capture.capture(ContentKind.TOOL, arguments),
                result=None if result is None else self._capture.capture(ContentKind.TOOL, result),
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        except ValidationError:
            logger.warning("MCP call on %s not recorded: an id is not an identifier", tool.server)
        if event is None:
            return
        try:
            await self._events.emit(event)
        except Exception as error:
            logger.warning(
                "event sink %s failed on mcp_call: %s",
                type(self._events).__name__,
                type(error).__name__,
            )
