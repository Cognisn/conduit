"""The interactive sign-in hook (C14), and a loopback implementation for a terminal.

The OAuth client builds the authorisation request and checks the answer; the hook
only takes the person to the authorisation server and brings back the address the
server redirected their browser to. A server product implements ``SignInHook``
itself: it sends the person to ``authorization_url`` through its own interface,
receives the redirect at ``redirect_uri`` (one of its own routes) and returns the
full URL it received. The client then checks that URL's address against
``redirect_uri``, its ``state`` and its issuer before it uses the code.

``LoopbackSignIn`` is the terminal interface's hook: it listens on a loopback port,
opens the browser at the authorisation URL, and waits for the one redirect.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit

from conduit.core.errors import ConfigurationError
from conduit.mcp.settings import is_loopback_host

__all__ = ["LoopbackSignIn", "SignInHook", "SignInRequest", "SignInResult"]

logger = logging.getLogger("conduit.mcp")

_REQUEST_LIMIT = 16 * 1024
_PAGE = (
    b"<!doctype html><html><head><meta charset='utf-8'><title>Signed in</title></head>"
    b"<body><p>Sign-in complete. You can close this window.</p></body></html>"
)


@dataclass(frozen=True, slots=True)
class SignInRequest:
    """What the hook needs: where to send the person, and where they come back to."""

    server: str  # the configured MCP server name
    authorization_url: str
    redirect_uri: str

    def __repr__(self) -> str:  # the URL carries the state; keep it out of logs
        return f"SignInRequest(server={self.server!r})"


@dataclass(frozen=True, slots=True)
class SignInResult:
    """The full URL the authorisation server redirected the browser to."""

    callback_url: str

    def __repr__(self) -> str:  # the URL carries the code
        return "SignInResult(<callback>)"


@runtime_checkable
class SignInHook(Protocol):
    """Takes a person through an interactive sign-in. Supplied by the product."""

    #: Where the authorisation server sends the browser back to. Registered with the
    #: server and checked against every callback. https, or http on a loopback host.
    redirect_uri: str

    async def sign_in(self, request: SignInRequest) -> SignInResult: ...


class LoopbackSignIn:
    """Opens the browser and waits for the redirect on a loopback port.

    ``port`` is fixed (an authorisation server matches registered redirect URIs
    exactly). ``open_url`` opens the address (``webbrowser.open`` by default, run in a
    worker thread); if it reports failure, the address is written to standard error
    for the person to open themselves. Only a request for ``path`` completes the
    sign-in; anything else is answered 404. The listener is closed when the sign-in
    ends, however it ends. A sign-in not completed within ``timeout`` seconds raises
    ``TimeoutError``.
    """

    def __init__(
        self,
        port: int,
        *,
        host: str = "127.0.0.1",
        path: str = "/callback",
        timeout: float = 300.0,
        open_url: Callable[[str], Any] | None = None,
    ) -> None:
        if not is_loopback_host(host):
            raise ConfigurationError("a loopback sign-in listens on a loopback address only")
        if not 0 < port < 65536:
            raise ConfigurationError("a loopback sign-in needs a port from 1 to 65535")
        if not path.startswith("/") or "?" in path or "#" in path:
            raise ConfigurationError("a loopback sign-in path starts with / and has no query")
        if timeout <= 0:
            raise ConfigurationError("a loopback sign-in needs a positive timeout")
        self._host = host
        self._port = port
        self._path = path
        self._timeout = timeout
        self._open_url = open_url or webbrowser.open
        shown = f"[{host}]" if ":" in host else host
        self.redirect_uri = f"http://{shown}:{port}{path}"

    def __repr__(self) -> str:
        return f"LoopbackSignIn({self.redirect_uri!r})"

    async def sign_in(self, request: SignInRequest) -> SignInResult:
        loop = asyncio.get_running_loop()
        received: asyncio.Future[str] = loop.create_future()

        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            try:
                target = await self._read_target(reader)
                matched = target is not None and urlsplit(target).path == self._path
                if matched and target is not None and not received.done():
                    await self._answer(writer, 200, _PAGE)
                    received.set_result(target)
                else:
                    await self._answer(writer, 404, b"Not found")
            except (OSError, asyncio.IncompleteReadError, asyncio.LimitOverrunError):
                pass
            finally:
                writer.close()
                with contextlib.suppress(OSError):
                    await writer.wait_closed()

        server = await asyncio.start_server(handle, self._host, self._port, limit=_REQUEST_LIMIT)
        try:
            opened = await asyncio.to_thread(self._open_url, request.authorization_url)
            if opened is False:
                sys.stderr.write(
                    f"Open this address in a browser to sign in to {request.server}:\n"
                    f"{request.authorization_url}\n"
                )
            target = await asyncio.wait_for(received, timeout=self._timeout)
        finally:
            server.close()
            await server.wait_closed()
        shown = f"[{self._host}]" if ":" in self._host else self._host
        return SignInResult(callback_url=f"http://{shown}:{self._port}{target}")

    @staticmethod
    async def _read_target(reader: asyncio.StreamReader) -> str | None:
        head = await reader.readuntil(b"\r\n\r\n")
        if len(head) > _REQUEST_LIMIT:
            return None
        line = head.split(b"\r\n", 1)[0].decode("latin-1")
        parts = line.split(" ")
        if len(parts) != 3 or parts[0] != "GET" or not parts[1].startswith("/"):
            return None
        return parts[1]

    @staticmethod
    async def _answer(writer: asyncio.StreamWriter, status: int, body: bytes) -> None:
        reason = "OK" if status == 200 else "Not Found"
        kind = "text/html; charset=utf-8" if status == 200 else "text/plain"
        writer.write(
            f"HTTP/1.1 {status} {reason}\r\nContent-Type: {kind}\r\n"
            f"Content-Length: {len(body)}\r\nCache-Control: no-store\r\n"
            "Connection: close\r\n\r\n".encode("latin-1")
            + body
        )
        await writer.drain()
