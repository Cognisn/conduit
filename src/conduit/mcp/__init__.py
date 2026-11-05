"""MCP client over the official SDK, behind the mcp extra."""

# The names below load on first use, so this package imports without the extra; using
# one without it raises ImportError naming the missing library (C14).
#
# - McpConnection: one MCP server over stdio, streamable HTTP or SSE, with connection
#   lifecycle, timeouts and clean shutdown.
# - McpServerSettings with its transports and authentication: static credentials (a
#   bearer token or an API key) as secret:// references, or OAuthAuth.
# - OAuthClient: the OAuth 2.1 client (discovery, dynamic registration, authorisation
#   code with PKCE, refresh rotation), holding tokens only in the product's TokenStore.
#   The interactive sign-in is a SignInHook; LoopbackSignIn is the terminal's.
# - McpToolBridge: MCP tools as catalogue tools named <server>__<tool>, granted per
#   task class, with results framed as untrusted content and an McpCallEvent per call.
#
#     from conduit.mcp import LoopbackSignIn, McpConnection, McpServerSettings, McpToolBridge
#
#     docs = McpConnection(settings, secrets=resolver, tokens=store, sign_in=LoopbackSignIn(8765))
#     await docs.connect()
#     bridge = McpToolBridge([docs], events=sink, capture=policy)
#     catalogue = ToolCatalogue.from_tools([*product_tools, *await bridge.tools()])
#     executor = bridge.executor(product_executor)

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from conduit.mcp.bridge import McpTool, McpToolBridge, frame_untrusted, namespaced_name
    from conduit.mcp.client import CallResult, McpConnection, RemoteTool, TransportOpener
    from conduit.mcp.errors import McpAuthError, McpError, McpFailure, OAuthFailure
    from conduit.mcp.events import McpCallEvent
    from conduit.mcp.oauth import OAuthClient
    from conduit.mcp.settings import (
        ApiKeyAuth,
        BearerTokenAuth,
        McpServerSettings,
        NoAuth,
        OAuthAuth,
        SseTransport,
        StdioTransport,
        StreamableHttpTransport,
    )
    from conduit.mcp.signin import LoopbackSignIn, SignInHook, SignInRequest, SignInResult

_HOMES = {
    "McpTool": "bridge",
    "McpToolBridge": "bridge",
    "frame_untrusted": "bridge",
    "namespaced_name": "bridge",
    "CallResult": "client",
    "McpConnection": "client",
    "RemoteTool": "client",
    "TransportOpener": "client",
    "McpAuthError": "errors",
    "McpError": "errors",
    "McpFailure": "errors",
    "OAuthFailure": "errors",
    "McpCallEvent": "events",
    "OAuthClient": "oauth",
    "ApiKeyAuth": "settings",
    "BearerTokenAuth": "settings",
    "McpServerSettings": "settings",
    "NoAuth": "settings",
    "OAuthAuth": "settings",
    "SseTransport": "settings",
    "StdioTransport": "settings",
    "StreamableHttpTransport": "settings",
    "LoopbackSignIn": "signin",
    "SignInHook": "signin",
    "SignInRequest": "signin",
    "SignInResult": "signin",
}

__all__ = [
    "ApiKeyAuth",
    "BearerTokenAuth",
    "CallResult",
    "LoopbackSignIn",
    "McpAuthError",
    "McpCallEvent",
    "McpConnection",
    "McpError",
    "McpFailure",
    "McpServerSettings",
    "McpTool",
    "McpToolBridge",
    "NoAuth",
    "OAuthAuth",
    "OAuthClient",
    "OAuthFailure",
    "RemoteTool",
    "SignInHook",
    "SignInRequest",
    "SignInResult",
    "SseTransport",
    "StdioTransport",
    "StreamableHttpTransport",
    "TransportOpener",
    "frame_untrusted",
    "namespaced_name",
]


def __getattr__(name: str) -> Any:
    home = _HOMES.get(name)
    if home is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{home}"), name)
