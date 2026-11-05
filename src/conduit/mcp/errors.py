"""Errors the MCP client raises.

Each carries the server's configured name and a fixed failure class, never the
server's own text, a URL, a token, a code, a client secret or a tool's arguments or
result. Every one is raised after the handler of the exception that caused it, so
``__cause__`` and ``__context__`` are both ``None`` (D0040).
"""

from __future__ import annotations

from enum import StrEnum

from conduit.core.errors import ConduitError

__all__ = ["McpAuthError", "McpError", "McpFailure", "OAuthFailure"]


class McpFailure(StrEnum):
    CONNECT = "connect"  # the transport or the session could not be opened
    TIMEOUT = "timeout"
    CLOSED = "closed"  # the connection is not open, or was lost
    REJECTED = "rejected"  # the server answered the request with an error
    PROTOCOL = "protocol"  # the server's answer broke the protocol
    AUTH = "auth"  # authentication failed; see ``McpAuthError.reason``


class OAuthFailure(StrEnum):
    INSECURE_ENDPOINT = "insecure_endpoint"  # an endpoint is not https (or loopback http)
    DISCOVERY = "discovery"  # metadata missing, unreadable or incomplete
    AUDIENCE_MISMATCH = "audience_mismatch"  # the metadata names another resource
    ISSUER_MISMATCH = "issuer_mismatch"  # the metadata or the callback names another issuer
    PKCE_UNSUPPORTED = "pkce_unsupported"  # the server does not offer S256
    REGISTRATION = "registration"  # dynamic client registration failed
    REDIRECT_MISMATCH = "redirect_mismatch"  # a redirect URI was refused or did not match
    STATE_MISMATCH = "state_mismatch"
    AUTHORISATION_DENIED = "authorisation_denied"  # the callback carried an error
    # The code could not be exchanged for tokens.
    TOKEN_EXCHANGE = "token_exchange"  # noqa: S105 - a failure class, not a secret
    REFRESH = "refresh"  # the refresh grant failed for a reason other than invalid_grant
    SIGN_IN_REQUIRED = "sign_in_required"  # no usable tokens and no interactive sign-in
    SIGN_IN_FAILED = "sign_in_failed"  # the sign-in hook failed or timed out
    STORE = "store"  # the token store could not read or write
    # The server refused a freshly issued token.
    TOKEN_REJECTED = "token_rejected"  # noqa: S105 - a failure class, not a secret


class McpError(ConduitError):
    """An MCP server could not be used. Names the server and the failure class only."""

    def __init__(self, server: str, failure: McpFailure, detail: str | None = None) -> None:
        suffix = f" ({detail})" if detail else ""
        super().__init__(f"MCP server {server!r}: {failure.value}{suffix}")
        self.server = server
        self.failure = failure


class McpAuthError(McpError):
    """Authentication with an MCP server failed. ``reason`` says at which step."""

    def __init__(self, server: str, reason: OAuthFailure) -> None:
        super().__init__(server, McpFailure.AUTH, reason.value)
        self.reason = reason
