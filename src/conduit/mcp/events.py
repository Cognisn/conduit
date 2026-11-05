"""The MCP call event (C19).

One per tool call to an MCP server, however it ends: the server and tool, the
outcome, the duration, and the arguments and result as hashes and sizes. Their text
is present only when the product's capture policy allows tool content
(``ToolContent``), and the event-content tripwire checks the class's shape.
"""

from __future__ import annotations

from typing import Literal

from pydantic import NonNegativeFloat

from conduit.core.events import CallOutcome, Event, ToolContent
from conduit.core.identifiers import Identifier

__all__ = ["McpCallEvent"]


class McpCallEvent(Event):
    type: Literal["mcp_call"] = "mcp_call"
    call_id: Identifier
    server: Identifier  # the configured server name
    tool: Identifier  # the namespaced name the model sees
    remote_tool: Identifier  # the server's own name for the tool
    outcome: CallOutcome
    #: ``tool_error`` (the tool reported an error), or the ``McpFailure`` value, or
    #: ``cancelled``.
    error_code: Identifier | None = None
    arguments: ToolContent
    result: ToolContent | None = None
    duration_ms: NonNegativeFloat
