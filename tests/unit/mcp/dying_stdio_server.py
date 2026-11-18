"""A synthetic MCP server over stdio whose one tool ends the server process mid-call.

It imports nothing from conduit or the test package, so it runs from its file path.
"""

from __future__ import annotations

import os

from mcp.server.mcpserver import MCPServer

server = MCPServer("dying-example")


@server.tool(description="End the server process without answering.")
def die() -> str:
    os._exit(0)


if __name__ == "__main__":
    server.run("stdio")
