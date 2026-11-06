"""MCP servers: connecting them (with the OAuth sign-in on a loopback callback) and the
``mcp`` command.

Each configured server is ``McpServerSettings``. An OAuth server is signed in to through
``LoopbackSignIn``: the browser opens at the authorisation page and the redirect comes
back to ``http://127.0.0.1:<callback_port>/callback``. Tokens are held in memory for the
session unless the person names a token directory (``--tokens``). A token, code or
client secret is never shown.

The mcp extra is needed only when servers are configured; without it the interface says
which extra to install and carries on without them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from conduit import EventSink, TokenStore
from conduit.cli.config import CHAT, CliConfig
from conduit.cli.output import describe_error
from conduit.cli.setup import Context
from conduit.core.settings import refusals
from conduit.stores.json import JsonTokenStore
from conduit.stores.memory import MemoryTokenStore

if TYPE_CHECKING:
    from conduit.mcp import McpConnection, McpTool

MISSING_EXTRA = (
    "MCP servers need the mcp extra: install cognisn-conduit[cli,mcp] "
    '(uvx --from "cognisn-conduit[cli,mcp]" conduit).'
)


def token_store(directory: str | None) -> TokenStore:
    """In memory unless the person asked for a directory."""
    return JsonTokenStore(directory) if directory is not None else MemoryTokenStore()


async def connect_servers(
    context: Context, config: CliConfig, tokens: TokenStore
) -> list[McpConnection] | None:
    """A connection to every configured server that connects; ``None`` without the extra.

    A server that is refused or fails to connect is reported and left out.
    """
    if not config.mcp.servers:
        return []
    screen = context.screen
    try:
        from conduit.mcp import LoopbackSignIn, McpConnection, McpServerSettings, OAuthAuth
    except ImportError:
        screen.error(MISSING_EXTRA)
        return None
    connections: list[McpConnection] = []
    for index, raw in enumerate(config.mcp.servers):
        found: tuple[list[str], list[str]] | None = None
        settings: Any = None
        try:
            settings = McpServerSettings.model_validate(raw)
        except ValidationError as error:
            found = refusals(error, raw)
        if found is not None or settings is None:
            fields = ", ".join((found or ([], []))[0])
            screen.error(f"MCP server {index + 1} was refused at {fields}")
            continue
        if CHAT not in settings.task_classes:
            screen.note(f"[mcp] {settings.name} does not grant the {CHAT} task class its tools")
        sign_in = LoopbackSignIn(config.mcp.callback_port, open_url=context.open_url)
        if isinstance(settings.auth, OAuthAuth):
            screen.say(
                f"[mcp] {settings.name} signs in with OAuth: if no tokens are held, the "
                f"browser opens and the sign-in returns to {sign_in.redirect_uri}"
            )
        connection = McpConnection(
            settings, secrets=context.secrets(), tokens=tokens, sign_in=sign_in
        )
        failure: str | None = None
        try:
            await connection.connect()
        except Exception as error:
            failure = describe_error(error)
        if failure is not None:
            screen.error(f"[mcp] {settings.name} did not connect: {failure}")
            continue
        screen.say(f"[mcp] connected to {settings.name} ({settings.transport.type})")
        connections.append(connection)
    return connections


async def close_servers(connections: Sequence[McpConnection]) -> None:
    for connection in connections:
        await connection.close()  # never raises: a failing close is logged by class


async def bridged_tools(
    connections: Sequence[McpConnection], events: EventSink
) -> tuple[Any, tuple[McpTool, ...]]:
    """The bridge over ``connections`` and its tools. A server that fails is an ``McpError``."""
    from conduit.mcp import McpToolBridge

    bridge = McpToolBridge(connections, events=events)
    return bridge, await bridge.tools()


async def mcp_command(context: Context, config: CliConfig, tokens_dir: str | None) -> int:
    """Connect every configured server, list its tools, and close."""
    screen = context.screen
    if not config.mcp.servers:
        screen.say("No MCP servers are configured (mcp.servers in the configuration file).")
        return 0
    connections = await connect_servers(context, config, token_store(tokens_dir))
    if connections is None:
        return 1
    failed = len(connections) < len(config.mcp.servers)
    try:
        from conduit.mcp import McpToolBridge

        rows = []
        for connection in connections:
            listing: tuple[McpTool, ...] = ()
            try:
                listing = await McpToolBridge([connection]).tools()
            except Exception as error:
                screen.error(f"[mcp] {connection.name}: tools not listed: {describe_error(error)}")
                failed = True
            for tool in listing:
                rows.append(
                    [
                        connection.name,
                        tool.name,
                        ", ".join(sorted(tool.task_classes)),
                        tool.description,
                    ]
                )
        screen.table("MCP tools", ["server", "tool", "task classes", "description"], rows)
    finally:
        await close_servers(connections)
    return 1 if failed else 0
