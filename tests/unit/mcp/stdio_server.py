"""A synthetic MCP server over stdio, run as a subprocess by the client tests.

It imports nothing from conduit or the test package, so it runs from its file path.
"""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

server = FastMCP("stdio-example")


@server.tool(description="Echo the text back.")
def echo(text: str) -> str:
    return text


@server.tool(description="Report whether an environment variable is set, never its value.")
def has_env(name: str) -> str:
    value = os.environ.get(name)
    return "unset" if value is None else f"set, {len(value)} characters"


if __name__ == "__main__":
    server.run("stdio")
