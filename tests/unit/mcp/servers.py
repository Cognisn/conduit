"""In-process MCP test servers built with the SDK, and the transports that reach them.

Everything here is synthetic. The memory transport runs the server in the same event
loop over anyio memory streams; ``LoopbackServer`` runs an ASGI app (the SDK's
streamable HTTP or SSE app, or the fake authorisation server) under uvicorn on
127.0.0.1, inside this worktree's slot unless ``CONDUIT_MCP_TEST_PORT_BASE`` says
otherwise.
"""

from __future__ import annotations

import base64
import contextlib
import os
import socket
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from typing import Any

import anyio
import uvicorn
from mcp.server.fastmcp import FastMCP, Image
from mcp.shared.memory import create_client_server_memory_streams
from mcp.types import EmbeddedResource, ResourceLink, TextResourceContents
from pydantic import AnyUrl

#: The first port of this worktree's slot (24); offsets within the block are fixed below.
PORT_BASE = int(os.environ.get("CONDUIT_MCP_TEST_PORT_BASE", "22400"))
OFFSETS = {"http": 10, "sse": 11, "auth": 12, "protected": 13, "callback": 20, "callback2": 21}

PIXEL = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAwS2OUAAAAABJRU5ErkJggg=="
)


def port(name: str) -> int:
    return PORT_BASE + OFFSETS[name]


def build_server(name: str = "example", **settings: Any) -> FastMCP:
    server = FastMCP(name, **settings)

    @server.tool(description="Echo the text back.")
    def echo(text: str) -> str:
        return text

    @server.tool(description="Add two integers.")
    def add(a: int, b: int) -> int:
        return a + b

    @server.tool(description="Always fails.")
    def fail() -> str:
        raise ValueError("synthetic tool failure")

    @server.tool(description="Sleep, then answer.")
    async def slow(seconds: float) -> str:
        await anyio.sleep(seconds)
        return "awake"

    @server.tool(name="Get-Item.v2", description="A tool whose name needs namespacing care.")
    def get_item(item: str) -> str:
        return f"item {item}"

    @server.tool(description="A one-pixel image.")
    def picture() -> Image:
        return Image(data=PIXEL, format="png")

    @server.tool(description="Mixed content.")
    def mixed() -> list[Any]:
        return [
            "first part",
            EmbeddedResource(
                type="resource",
                resource=TextResourceContents(
                    uri=AnyUrl("file:///example/notes.txt"), mimeType="text/plain", text="notes"
                ),
            ),
            ResourceLink(type="resource_link", uri=AnyUrl("https://example.com/doc"), name="doc"),
        ]

    return server


Streams = tuple[Any, Any]


def memory_transport(server: FastMCP) -> Callable[[], Any]:
    """A transport opener for ``McpConnection`` that runs ``server`` in-process."""

    @asynccontextmanager
    async def opener() -> AsyncIterator[Streams]:
        lowlevel = server._mcp_server
        async with (
            create_client_server_memory_streams() as (client, srv),
            anyio.create_task_group() as group,
        ):
            group.start_soon(
                lambda: lowlevel.run(srv[0], srv[1], lowlevel.create_initialization_options())
            )
            try:
                yield client
            finally:
                group.cancel_scope.cancel()

    return opener


def silent_transport() -> Callable[[], Any]:
    """A transport whose server never answers: initialisation waits forever."""

    @asynccontextmanager
    async def opener() -> AsyncIterator[Streams]:
        to_client, from_server = anyio.create_memory_object_stream[Any](1)
        to_server, from_client = anyio.create_memory_object_stream[Any](1)
        async with to_client, from_server, to_server, from_client:
            yield from_server, to_server

    return opener


def failing_transport(error: BaseException) -> Callable[[], Any]:
    @asynccontextmanager
    async def opener() -> AsyncIterator[Streams]:
        raise error
        yield  # pragma: no cover

    return opener


def stubborn_transport(server: FastMCP, hold: float) -> Callable[[], Any]:
    """A working transport whose exit takes ``hold`` seconds and ignores cancellation."""
    inner = memory_transport(server)

    @asynccontextmanager
    async def opener() -> AsyncIterator[Streams]:
        try:
            async with inner() as streams:
                yield streams
        finally:
            with anyio.CancelScope(shield=True):
                await anyio.sleep(hold)

    return opener


def _wait_for_port(host: str, number: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with contextlib.suppress(OSError), socket.create_connection((host, number), timeout=0.2):
            return
        time.sleep(0.05)
    raise RuntimeError(f"loopback test server on port {number} did not start")


class LoopbackServer:
    """Runs an ASGI app under uvicorn on 127.0.0.1 in a background thread."""

    def __init__(self, app: Any, number: int) -> None:
        self.port = number
        self._server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=number, log_level="warning", lifespan="on")
        )
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> LoopbackServer:
        self._thread.start()
        _wait_for_port("127.0.0.1", self.port)
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=15)
        if self._thread.is_alive():  # pragma: no cover - a hung server fails the test run
            self._server.force_exit = True
            self._thread.join(timeout=5)


@contextlib.contextmanager
def serving(app: Any, name: str) -> Iterator[LoopbackServer]:
    with LoopbackServer(app, port(name)) as server:
        yield server
