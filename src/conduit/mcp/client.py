"""A connection to one MCP server over the official SDK: stdio, streamable HTTP or SSE.

The SDK's transports are anyio task groups, which must be entered and left in the
same task. Each connection therefore lives in one task of its own (the runner): it
opens the transport and the session, initialises the session, reports that it is
ready, waits until ``close`` asks it to stop, and closes both in that same task. A
transport that fails in the background ends the runner; calls then fail with
``McpFailure.CLOSED`` rather than waiting for ever.

Every failure is an ``McpError`` naming the server and a failure class, raised
after the handler of what caused it (D0040). Cancellation always propagates
unchanged, and ``close`` never raises: a failing close is logged by class name and
never replaces a failure already on its way out.

Credentials are resolved through the product's ``SecretResolver`` on every
``connect``, so a rotated secret is picked up on the next connection.

Environment and logging (D0037):

- An HTTP server is reached with an httpx2 client conduit builds (httpx2 is the
  SDK's HTTP library from MCP SDK 2.0), with ``trust_env`` off unless
  ``use_environment_proxies`` is set, so ``HTTP_PROXY``, ``SSL_CERT_FILE`` and
  ``.netrc`` cannot reroute requests or add credentials. httpx2 checks
  certificates against the operating system's trust store. conduit's httpx2
  client never follows a redirect. The SDK's streamable HTTP and SSE transports
  follow one themselves only when it stays on the server's origin; the OAuth
  requests follow none.
- A stdio server gets the SDK's small default environment (``PATH`` and the user's
  profile variables), not the parent's whole environment, plus the configured
  ``env`` and ``secret_env``.
- The SDK reads no environment variable on the client side. Its transport loggers
  (``mcp.client.*``) record message content, tool arguments and results included,
  at ``DEBUG``; a product that must not log content keeps the ``mcp`` logger above
  ``DEBUG``. The SDK's session logs under ``client`` (tool names, at ``WARNING``).
  conduit's own logger (``conduit.mcp``) never records content, URLs with their
  query, or credentials.
- The SDK makes an OpenTelemetry client span for every request (the method name
  and request id, no content) and sends the W3C trace context to the server in the
  request's ``_meta``. Neither does anything unless the product configures an
  OpenTelemetry tracer provider.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self

import httpx2
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError as SdkMcpError
from mcp.types import (
    CONNECTION_CLOSED,
    AudioContent,
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    PaginatedRequestParams,
    ResourceLink,
    TextContent,
    TextResourceContents,
)

from conduit.core.backends import TokenStore
from conduit.core.errors import ConfigurationError
from conduit.core.secrets import SecretResolver
from conduit.mcp.errors import McpAuthError, McpError, McpFailure, OAuthFailure
from conduit.mcp.settings import (
    ApiKeyAuth,
    BearerTokenAuth,
    McpServerSettings,
    OAuthAuth,
    SseTransport,
    StdioTransport,
)

if TYPE_CHECKING:
    from conduit.mcp.oauth import OAuthClient
    from conduit.mcp.signin import SignInHook

__all__ = ["CallResult", "McpConnection", "RemoteTool", "TransportOpener"]

logger = logging.getLogger("conduit.mcp")

#: Opens a transport: an async context manager yielding the read and write streams
#: first (the SDK's transports yield exactly those two). For in-process servers and
#: for tests; ``McpConnection`` builds the stdio, streamable HTTP and SSE openers itself.
TransportOpener = Callable[[], AbstractAsyncContextManager[tuple[Any, ...]]]

#: Pages of a tool listing read before the listing is refused as a protocol failure.
_PAGE_LIMIT = 100
#: How long an SSE or streamable HTTP response stream may stay quiet.
_STREAM_READ_TIMEOUT = 300.0


@dataclass(frozen=True, slots=True)
class RemoteTool:
    """A tool as the server describes it. Its description is the server's, untrusted."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CallResult:
    """A tool call's result, rendered as text. ``is_error`` when the tool reported one."""

    is_error: bool
    text: str


class _ConnectFailed(Exception):
    """The runner could not open the connection. Carries the failure class only."""

    def __init__(self, failure: McpFailure, reason: OAuthFailure | None = None) -> None:
        super().__init__(failure.value)
        self.failure = failure
        self.reason = reason


def _leaves(error: BaseException) -> list[BaseException]:
    if isinstance(error, BaseExceptionGroup):
        return [leaf for inner in error.exceptions for leaf in _leaves(inner)]
    return [error]


_REFUSED = (401, 403)


class _LastStatus:
    """The status of the last HTTP response a connection attempt received.

    The SDK's streamable HTTP transport (from SDK 2.0) answers a refused request with a
    JSON-RPC error rather than raising the HTTP status, so a refused credential is read
    here, from conduit's own HTTP client, instead.
    """

    __slots__ = ("code",)

    def __init__(self) -> None:
        self.code: int | None = None

    async def record(self, response: httpx2.Response) -> None:
        self.code = response.status_code


def _classify_connect(error: BaseException, status: _LastStatus | None = None) -> _ConnectFailed:
    leaves = _leaves(error)
    for leaf in leaves:
        if isinstance(leaf, McpAuthError):
            return _ConnectFailed(McpFailure.AUTH, leaf.reason)
    for leaf in leaves:
        if isinstance(leaf, httpx2.HTTPStatusError) and leaf.response.status_code in _REFUSED:
            return _ConnectFailed(McpFailure.AUTH, OAuthFailure.TOKEN_REJECTED)
    if status is not None and status.code in _REFUSED:
        return _ConnectFailed(McpFailure.AUTH, OAuthFailure.TOKEN_REJECTED)
    return _ConnectFailed(McpFailure.CONNECT)


def render_content(result: CallToolResult) -> str:
    """The text a tool result carries. Binary content is described, never passed on."""
    parts: list[str] = []
    for item in result.content:
        if isinstance(item, TextContent):
            parts.append(item.text)
        elif isinstance(item, ImageContent | AudioContent):
            kind = "image" if isinstance(item, ImageContent) else "audio"
            parts.append(f"[{kind} content omitted: {item.mime_type}, {len(item.data)} characters]")
        elif isinstance(item, EmbeddedResource):
            resource = item.resource
            if isinstance(resource, TextResourceContents):
                parts.append(resource.text)
            elif isinstance(resource, BlobResourceContents):
                parts.append(
                    f"[binary resource omitted: {resource.mime_type or 'unknown type'}, "
                    f"{len(resource.blob)} characters]"
                )
        elif isinstance(item, ResourceLink):
            parts.append(f"[resource link: {item.uri}]")
        else:  # pragma: no cover - every content type the SDK defines is handled above
            parts.append("[unsupported content omitted]")
    if not parts and result.structured_content is not None:
        parts.append(json.dumps(result.structured_content, sort_keys=True, ensure_ascii=False))
    return "\n".join(parts)


def _lister(cursor: str | None) -> Callable[[ClientSession], Any]:
    def make(session: ClientSession) -> Any:
        params = PaginatedRequestParams(cursor=cursor) if cursor else None
        return session.list_tools(params=params)

    return make


class McpConnection:
    """One MCP server: connect, list its tools, call them, close.

    ``tokens`` and ``sign_in`` serve an OAuth server (``OAuthAuth``): tokens are held
    only in the product's ``TokenStore``, and ``sign_in`` runs the interactive
    sign-in when there are none (without it, a server with no usable tokens is
    refused with ``OAuthFailure.SIGN_IN_REQUIRED``). ``transport`` replaces the
    configured transport (an in-process server); ``http_transport`` (an httpx2
    transport, the SDK's HTTP library) replaces httpx2's network transport for every
    HTTP request this connection makes (custom networking, or tests).
    """

    def __init__(
        self,
        settings: McpServerSettings,
        *,
        secrets: SecretResolver,
        tokens: TokenStore | None = None,
        sign_in: SignInHook | None = None,
        transport: TransportOpener | None = None,
        http_transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self._secrets = secrets
        self._transport = transport
        self._http_transport = http_transport
        self._oauth: OAuthClient | None = None
        if isinstance(settings.auth, OAuthAuth):
            if tokens is None:
                raise ConfigurationError(
                    f"MCP server {settings.name!r} uses OAuth and needs a token store"
                )
            from conduit.mcp.oauth import OAuthClient

            url = getattr(settings.transport, "url", "")
            self._oauth = OAuthClient(
                settings,
                resource=url,
                store=tokens,
                sign_in=sign_in,
                secrets=secrets,
                http_transport=http_transport,
            )
        self._session: ClientSession | None = None
        self._runner: asyncio.Task[None] | None = None
        self._stop: asyncio.Event | None = None
        self._lost: asyncio.Event | None = None
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return self.settings.name

    @property
    def connected(self) -> bool:
        runner = self._runner
        return self._session is not None and runner is not None and not runner.done()

    @property
    def oauth(self) -> OAuthClient | None:
        """The OAuth client of an OAuth server (to sign out, for example)."""
        return self._oauth

    def __repr__(self) -> str:
        return f"McpConnection({self.name!r}, connected={self.connected})"

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    # -- Opening ---------------------------------------------------------------------

    async def _headers(self) -> dict[str, str]:
        auth = self.settings.auth
        if isinstance(auth, BearerTokenAuth):
            token = await self._secrets.resolve(auth.token)
            return {"Authorization": f"Bearer {token.get_secret_value()}"}
        if isinstance(auth, ApiKeyAuth):
            key = await self._secrets.resolve(auth.key)
            return {auth.header: key.get_secret_value()}
        return {}

    def _http_client(
        self,
        headers: Mapping[str, str] | None,
        auth: httpx2.Auth | None,
        timeout: httpx2.Timeout,
        status: _LastStatus,
    ) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(
            headers=dict(headers or {}),
            auth=auth,
            timeout=timeout,
            follow_redirects=False,
            trust_env=self.settings.use_environment_proxies,
            transport=self._http_transport,
            event_hooks={"response": [status.record]},
        )

    async def _opener(self, status: _LastStatus) -> TransportOpener:
        """The configured transport, with its credentials resolved now."""
        transport = self.settings.transport
        if isinstance(transport, StdioTransport):
            env = dict(transport.env)
            for name, ref in transport.secret_env.items():
                env[name] = (await self._secrets.resolve(ref)).get_secret_value()
            params = StdioServerParameters(
                command=transport.command, args=list(transport.args), env=env, cwd=transport.cwd
            )
            return lambda: stdio_client(params)

        headers = await self._headers()
        auth: httpx2.Auth | None = None
        if self._oauth is not None:
            await self._oauth.access_token(interactive=True)
            auth = self._oauth.httpx_auth()
        timeout = httpx2.Timeout(self.settings.connect_timeout, read=_STREAM_READ_TIMEOUT)
        url = transport.url

        if isinstance(transport, SseTransport):

            def factory(
                headers: dict[str, str] | None = None,
                timeout: httpx2.Timeout | None = None,
                auth: httpx2.Auth | None = None,
            ) -> httpx2.AsyncClient:
                return self._http_client(
                    headers, auth, timeout or httpx2.Timeout(self.settings.connect_timeout), status
                )

            return lambda: sse_client(
                url,
                headers=headers,
                timeout=self.settings.connect_timeout,
                sse_read_timeout=_STREAM_READ_TIMEOUT,
                httpx_client_factory=factory,
                auth=auth,
            )

        @asynccontextmanager
        async def streamable() -> AsyncIterator[tuple[Any, ...]]:
            async with (
                self._http_client(headers, auth, timeout, status) as client,
                streamable_http_client(url, http_client=client) as streams,
            ):
                yield streams

        return streamable

    async def connect(self) -> None:
        """Open the connection and initialise the session. Does nothing if open."""
        async with self._lock:
            if self.connected:
                return
            await self._end()
            status = _LastStatus()
            opener = self._transport if self._transport is not None else await self._opener(status)
            loop = asyncio.get_running_loop()
            ready: asyncio.Future[None] = loop.create_future()
            self._stop = asyncio.Event()
            self._lost = asyncio.Event()
            self._runner = loop.create_task(
                self._run(opener, ready, self._stop, self._lost, status),
                name=f"conduit-mcp-{self.name}",
            )
            failed: _ConnectFailed | None = None
            try:
                await asyncio.wait_for(asyncio.shield(ready), timeout=self.settings.connect_timeout)
            except TimeoutError:
                failed = _ConnectFailed(McpFailure.TIMEOUT)
            except asyncio.CancelledError:
                ready.cancel()
                await self._end(cancel=True)
                raise
            except _ConnectFailed as error:
                failed = _ConnectFailed(error.failure, error.reason)
            if failed is not None:
                ready.cancel()
                await self._end(cancel=True)
                logger.warning("MCP server %s did not connect: %s", self.name, failed.failure.value)
                if failed.reason is not None:
                    raise McpAuthError(self.name, failed.reason)
                raise McpError(self.name, failed.failure)
            logger.info("connected to MCP server %s (%s)", self.name, self.settings.transport.type)

    async def _run(
        self,
        opener: TransportOpener,
        ready: asyncio.Future[None],
        stop: asyncio.Event,
        lost: asyncio.Event,
        status: _LastStatus,
    ) -> None:
        """Own one connection from opening to closing (see the module docstring)."""
        try:
            async with AsyncExitStack() as stack:
                streams = await stack.enter_async_context(opener())
                session = await stack.enter_async_context(ClientSession(streams[0], streams[1]))
                await session.initialize()
                self._session = session
                if not ready.done():
                    ready.set_result(None)
                await stop.wait()
        except asyncio.CancelledError:
            if not ready.done():
                ready.cancel()
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as error:  # anyio groups included; reported, never chained
            failed = _classify_connect(error, status)
            if not ready.done():
                ready.set_exception(failed)
            elif not stop.is_set():
                logger.warning("MCP server %s connection lost: %s", self.name, type(error).__name__)
        finally:
            self._session = None
            lost.set()

    # -- Closing ---------------------------------------------------------------------

    async def _end(self, *, cancel: bool = False) -> None:
        """Stop the runner, cancelling it if it does not stop within the close timeout.

        ``cancel`` cancels it at once: a runner that never became ready is not
        waiting to be asked to stop.
        """
        runner = self._runner
        self._runner = None
        self._session = None
        if runner is None:
            return
        if self._stop is not None:
            self._stop.set()
        if runner.done():
            return
        try:
            if cancel:
                runner.cancel()
            done, _ = await asyncio.wait({runner}, timeout=self.settings.close_timeout)
            if not done and not cancel:
                runner.cancel()
                done, _ = await asyncio.wait({runner}, timeout=self.settings.close_timeout)
            if not done:
                logger.warning("MCP server %s did not close in time", self.name)
        except asyncio.CancelledError:
            runner.cancel()
            raise

    async def close(self) -> None:
        """Close the connection. Never raises, except to pass on a cancellation."""
        async with self._lock:
            was_open = self._runner is not None
            try:
                await self._end()
            except asyncio.CancelledError:
                raise
            except Exception as error:  # a failing close never replaces a failure
                logger.warning("MCP server %s failed to close: %s", self.name, type(error).__name__)
            if was_open:
                logger.info("disconnected from MCP server %s", self.name)

    # -- Requests --------------------------------------------------------------------

    async def _gone(self, session: ClientSession) -> bool:
        """Whether the connection really ended, once the SDK has answered CONNECTION_CLOSED.

        That code (-32000) is also the first of JSON-RPC's server-error range, so a live
        server may answer with it. A ping, bounded by the close timeout, tells them
        apart: a session the SDK has shut (closing, or its server gone) refuses it at once
        with the same code, a live server answers it, and one that cannot answer in time
        is treated as gone.
        """
        try:
            await asyncio.wait_for(session.send_ping(), timeout=self.settings.close_timeout)
        except SdkMcpError as error:
            return error.code == CONNECTION_CLOSED
        except Exception:  # no answer in time, or a broken transport
            return True
        return False

    async def _request(self, make: Callable[[ClientSession], Any], timeout: float) -> Any:
        session = self._session
        lost = self._lost
        if session is None or lost is None or not self.connected:
            raise McpError(self.name, McpFailure.CLOSED)
        work: asyncio.Future[Any] = asyncio.ensure_future(make(session))
        watcher = asyncio.ensure_future(lost.wait())
        try:
            done, _ = await asyncio.wait(
                {work, watcher}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
        except asyncio.CancelledError:
            work.cancel()
            watcher.cancel()
            await asyncio.wait({work, watcher}, timeout=self.settings.close_timeout)
            raise
        watcher.cancel()
        failure: McpFailure | None = None
        if work not in done:
            work.cancel()
            await asyncio.wait({work}, timeout=self.settings.close_timeout)
            failure = McpFailure.CLOSED if lost.is_set() else McpFailure.TIMEOUT
        elif work.cancelled():
            failure = McpFailure.CLOSED
        else:
            try:
                return work.result()
            except SdkMcpError as error:
                # The SDK answers a request its closing connection abandoned with
                # CONNECTION_CLOSED (from SDK 2.0); a live server may send the same code.
                if lost.is_set():
                    failure = McpFailure.CLOSED
                elif error.code != CONNECTION_CLOSED:
                    failure = McpFailure.REJECTED
            except Exception:
                failure = McpFailure.CLOSED if lost.is_set() else McpFailure.PROTOCOL
        if failure is None:  # decided outside the handler, so nothing is chained
            gone = await self._gone(session)
            failure = McpFailure.CLOSED if gone else McpFailure.REJECTED
        raise McpError(self.name, failure)

    async def list_tools(self) -> tuple[RemoteTool, ...]:
        """Every tool the server offers, across every page of its listing."""
        found: list[RemoteTool] = []
        cursor: str | None = None
        for _ in range(_PAGE_LIMIT):
            page = await self._request(_lister(cursor), self.settings.list_timeout)
            for tool in page.tools:
                found.append(
                    RemoteTool(
                        name=tool.name,
                        description=tool.description or "",
                        input_schema=dict(tool.input_schema or {}),
                    )
                )
            cursor = page.next_cursor
            if not cursor:
                return tuple(found)
        raise McpError(self.name, McpFailure.PROTOCOL, "too many pages of tools")

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> CallResult:
        """Call one of the server's tools by its own name."""
        result = await self._request(
            lambda s: s.call_tool(name, dict(arguments)), self.settings.call_timeout
        )
        return CallResult(is_error=bool(result.is_error), text=render_content(result))
