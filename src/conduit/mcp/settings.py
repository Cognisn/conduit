"""Settings for one MCP server: transport, authentication, task-class grant and timeouts.

Typed and buildable in code or from a mapping (C15). Credentials are only ever
``secret://`` references (C14): a bearer token, an API key, a pre-registered OAuth
client secret and every credential a stdio server takes in its environment. A plain
environment variable whose name marks it as a credential, or whose value reads as an
HTTP authorisation value, is refused; such a value goes in ``secret_env``. Validation
errors name fields, never values.

HTTP transports need ``https``; plain ``http`` is accepted on a loopback host only
(``127.0.0.1``, ``::1`` or ``localhost``), for a server on the same machine.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from ipaddress import ip_address
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator

from conduit.core.secrets import SecretRef
from conduit.core.settings import FieldsRefused, SettingsModel

__all__ = [
    "SERVER_NAME_PATTERN",
    "ApiKeyAuth",
    "BearerTokenAuth",
    "McpAuth",
    "McpServerSettings",
    "McpTransport",
    "NoAuth",
    "OAuthAuth",
    "SseTransport",
    "StdioTransport",
    "StreamableHttpTransport",
    "check_endpoint_url",
    "is_loopback_host",
]

#: A server name: lower-case words joined by single underscores, at most 32 characters.
#: It never holds a double underscore, so ``<server>__<tool>`` splits one way only.
SERVER_NAME_PATTERN = r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$"
_SERVER_NAME_LIMIT = 32
_TASK_CLASS = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_HEADER = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]{1,128}$")
#: Headers the transports set themselves, which a product's API key may not replace.
_RESERVED_HEADERS = frozenset(
    {
        "accept",
        "connection",
        "content-length",
        "content-type",
        "host",
        "last-event-id",
        "mcp-protocol-version",
        "mcp-session-id",
        "transfer-encoding",
    }
)
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
#: Name segments that mark an environment variable as a credential.
_CREDENTIAL_ENV_SEGMENTS = frozenset(
    {
        "key",
        "keys",
        "apikey",
        "secret",
        "secrets",
        "token",
        "tokens",
        "password",
        "passwords",
        "passphrase",
        "passwd",
        "pwd",
        "credential",
        "credentials",
        "bearer",
        "jwt",
        "cookie",
        "authorization",
        "authorisation",
    }
)
_SEGMENT = re.compile(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])")
_AUTHORISATION_VALUE = re.compile(r"^\s*(?:bearer|basic|token)\s+\S{8,}\s*$", re.IGNORECASE)
_TIMEOUT_LIMIT = 3600.0


def is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def check_endpoint_url(url: str) -> str | None:
    """Why ``url`` cannot be an MCP or OAuth endpoint, or ``None`` if it can.

    The reason never quotes the URL.
    """
    try:
        parts = urlsplit(url)
        port = parts.port  # raises ValueError for a malformed port
    except ValueError:
        return "it is not a valid URL"
    del port
    if parts.scheme not in ("https", "http") or not parts.hostname:
        return "it must be an https URL with a host"
    if parts.username is not None or parts.password is not None:
        return "it must not carry user information"
    if parts.fragment or url.endswith("#"):
        return "it must not carry a fragment"
    if parts.scheme == "http" and not is_loopback_host(parts.hostname):
        return "plain http is accepted on a loopback host only"
    return None


def _credential_env_name(name: str) -> bool:
    segments = {s.lower() for s in _SEGMENT.findall(name)}
    segments.add(re.sub(r"[^a-z0-9]", "", name.lower()))
    return bool(segments & _CREDENTIAL_ENV_SEGMENTS)


class StdioTransport(SettingsModel):
    """A server run as a subprocess, spoken to over its standard input and output.

    The subprocess gets the SDK's small default environment (``PATH`` and the user's
    profile variables), then ``env``, then the resolved ``secret_env``.
    """

    type: Literal["stdio"] = "stdio"
    command: str = Field(min_length=1, max_length=4096)
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = Field(default_factory=dict)
    secret_env: Mapping[str, SecretRef] = Field(default_factory=dict)
    cwd: str | None = Field(default=None, min_length=1, max_length=4096)

    @field_validator("env")
    @classmethod
    def _plain_env(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        bad_names = [name for name in value if not _ENV_NAME.fullmatch(name)]
        if bad_names:
            raise ValueError("an environment variable name is not valid")
        refused = [
            name
            for name, item in value.items()
            if _credential_env_name(name) or _AUTHORISATION_VALUE.match(item)
        ]
        if refused:
            raise FieldsRefused("a credential must be a secret_env reference", sorted(refused))
        return dict(value)

    @field_validator("secret_env")
    @classmethod
    def _secret_env(cls, value: Mapping[str, SecretRef]) -> Mapping[str, SecretRef]:
        if any(not _ENV_NAME.fullmatch(name) for name in value):
            raise ValueError("an environment variable name is not valid")
        return dict(value)


class _HttpTransport(SettingsModel):
    url: str = Field(min_length=1, max_length=2048)

    @field_validator("url")
    @classmethod
    def _url(cls, value: str) -> str:
        reason = check_endpoint_url(value)
        if reason is not None:
            raise ValueError(f"the server URL was refused: {reason}")
        return value


class StreamableHttpTransport(_HttpTransport):
    type: Literal["streamable_http"] = "streamable_http"


class SseTransport(_HttpTransport):
    """The older HTTP with server-sent events transport."""

    type: Literal["sse"] = "sse"


McpTransport = Annotated[
    StdioTransport | StreamableHttpTransport | SseTransport, Field(discriminator="type")
]


class NoAuth(SettingsModel):
    type: Literal["none"] = "none"


class BearerTokenAuth(SettingsModel):
    """A static bearer token, sent as ``Authorization: Bearer <token>``."""

    type: Literal["bearer"] = "bearer"
    token: SecretRef


class ApiKeyAuth(SettingsModel):
    """A static API key, sent in its own header."""

    type: Literal["api_key"] = "api_key"
    key: SecretRef
    header: str = "X-API-Key"

    @field_validator("header")
    @classmethod
    def _header(cls, value: str) -> str:
        if not _HEADER.fullmatch(value) or value.lower() in _RESERVED_HEADERS:
            raise ValueError("the API key header must be a header name the transport does not set")
        return value


class OAuthAuth(SettingsModel):
    """The OAuth 2.1 client (C14).

    Without ``client_id`` the client registers itself dynamically with the
    authorisation server. With it, the client is pre-registered; ``client_secret``
    is then its secret, if it has one.
    """

    type: Literal["oauth"] = "oauth"
    scopes: tuple[str, ...] = ()
    client_id: str | None = Field(default=None, min_length=1, max_length=256)
    client_secret: SecretRef | None = None
    client_name: str = Field(default="conduit", min_length=1, max_length=128)
    #: Seconds before an access token's expiry at which it is refreshed.
    refresh_margin: float = Field(default=60.0, ge=0, le=_TIMEOUT_LIMIT)

    @field_validator("scopes")
    @classmethod
    def _scopes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not s or any(c.isspace() or c in '"\\' for c in s) for s in value):
            raise ValueError("a scope must be a non-empty token without spaces or quotes")
        return value

    @model_validator(mode="after")
    def _secret_needs_id(self) -> Self:
        if self.client_secret is not None and self.client_id is None:
            raise ValueError("a client secret needs a pre-registered client id")
        return self


McpAuth = Annotated[NoAuth | BearerTokenAuth | ApiKeyAuth | OAuthAuth, Field(discriminator="type")]


def _timeout(default: float) -> float:
    return Field(default=default, gt=0, le=_TIMEOUT_LIMIT)


class McpServerSettings(SettingsModel):
    """One MCP server the product connects to.

    ``task_classes`` is the product's grant: the server's tools serve these task
    classes and no others. ``allowed_tools`` restricts which of the server's tools
    are bridged at all (by the server's own tool name; ``None`` bridges every tool),
    and ``tool_task_classes`` narrows the grant for named tools. A tool's grant can
    only narrow the server's, never widen it.
    """

    name: str = Field(pattern=SERVER_NAME_PATTERN, max_length=_SERVER_NAME_LIMIT)
    transport: McpTransport
    auth: McpAuth = Field(default_factory=NoAuth)
    task_classes: frozenset[str] = Field(min_length=1)
    allowed_tools: frozenset[str] | None = None
    tool_task_classes: Mapping[str, frozenset[str]] = Field(default_factory=dict)
    connect_timeout: float = _timeout(30.0)
    call_timeout: float = _timeout(120.0)
    list_timeout: float = _timeout(30.0)
    close_timeout: float = _timeout(10.0)
    #: Read proxy, CA bundle and ``.netrc`` settings from the environment (httpx's
    #: ``trust_env``). Off by default, so the environment cannot add credentials or
    #: reroute requests behind the settings' back.
    use_environment_proxies: bool = False

    @field_validator("task_classes")
    @classmethod
    def _task_classes(cls, value: frozenset[str]) -> frozenset[str]:
        if not all(_TASK_CLASS.fullmatch(t) for t in value):
            raise ValueError("a task class must be lower-case snake_case")
        return value

    @model_validator(mode="after")
    def _check(self) -> Self:
        if isinstance(self.transport, StdioTransport) and not isinstance(self.auth, NoAuth):
            raise FieldsRefused("HTTP authentication needs an HTTP transport", ["auth"])
        widened = [
            tool
            for tool, granted in self.tool_task_classes.items()
            if not granted or not granted <= self.task_classes
        ]
        if widened:
            raise FieldsRefused(
                "a tool's task classes must be a non-empty subset of the server's",
                [f"tool_task_classes.{tool}" for tool in sorted(widened)],
            )
        return self

    @property
    def token_key(self) -> str:
        """The token store key for this server's OAuth tokens: ``mcp:<name>``."""
        return f"mcp:{self.name}"

    @property
    def registration_key(self) -> str:
        """The token store key for this server's dynamic client registration."""
        return f"mcp:{self.name}:client"
