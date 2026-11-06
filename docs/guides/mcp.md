# MCP servers

The `mcp` extra adds a client for Model Context Protocol servers, built on the official MCP Python SDK. A product connects to the servers it configures, and their tools join its tool catalogue for the task classes it grants them to, so the agent runner calls them like any other tool: under the run's budgets and approvals, with results framed as untrusted content and an event for every call.

```sh
pip install "cognisn-conduit[mcp]"
```

## Configuring a server

`McpServerSettings` describes one server: a name, a transport, authentication, the task classes its tools serve, and timeouts. Like every conduit setting it is buildable from a mapping, and holds credentials only as `secret://` references.

| Transport | Settings |
|---|---|
| `stdio` | A subprocess spoken to over standard input and output: `command`, `args`, `cwd`, `env` (plain values) and `secret_env` (credentials, as references). |
| `streamable_http` | The streamable HTTP transport at `url`. |
| `sse` | The older HTTP with server-sent events transport at `url`. |

HTTP URLs must be https; plain http is accepted on a loopback host only. The HTTP client ignores proxy variables, certificate settings and `.netrc` in the environment unless `use_environment_proxies` is set.

| Authentication | Sends |
|---|---|
| `none` (default) | nothing |
| `bearer` | `Authorization: Bearer <token>`, from a secret reference |
| `api_key` | a key in its own header (`X-API-Key` by default), from a secret reference |
| `oauth` | an OAuth 2.1 access token, obtained and refreshed by conduit's client |

```python
import asyncio
import sys
from pathlib import Path

from conduit.mcp import McpServerSettings

tickets = McpServerSettings.model_validate(
    {
        "name": "tickets",
        "transport": {"type": "streamable_http", "url": "https://mcp.example.com/tickets"},
        "auth": {"type": "bearer", "token": "secret://mcp/tickets_token"},
        "task_classes": ["investigate", "triage"],
        "allowed_tools": ["search_tickets", "get_ticket", "close_ticket"],
        "tool_task_classes": {"close_ticket": ["triage"]},
        "call_timeout": 60,
    }
)
assert tickets.token_key == "mcp:tickets"
```

`task_classes` is the product's grant: the server's tools serve these task classes and no others. `allowed_tools` bridges only the named tools (by the server's own names), and `tool_task_classes` narrows the grant for a tool; it can never widen it. A stdio server's `env` refuses a variable whose name or value looks like a credential: put those in `secret_env`.

## Connecting and bridging tools

`McpConnection` is one server: `connect`, `list_tools`, `call_tool` and `close` (or `async with`). `McpToolBridge` turns the tools of one or more connections into catalogue tools named `<server>__<tool>`, and gives the runner an executor that sends their calls to the right server and every other call to the product's own executor.

This example runs a small MCP server over stdio, written with the SDK's `FastMCP`, and lets an agent call its tool:

```python
Path("ticket_server.py").write_text(
    "from mcp.server.fastmcp import FastMCP\n"
    "\n"
    "server = FastMCP('tickets')\n"
    "\n"
    "@server.tool(description='Count open tickets for a host.')\n"
    "def open_tickets(host: str) -> str:\n"
    "    return f'3 open tickets for {host}'\n"
    "\n"
    "if __name__ == '__main__':\n"
    "    server.run('stdio')\n",
    encoding="utf-8",
    newline="\n",
)

from conduit import (
    CapabilityFloor,
    ConduitSettings,
    Event,
    LiteralSecretResolver,
    Message,
    TaskClass,
    TaskTools,
    TextBlock,
    ToolCallContext,
    ToolCatalogue,
    ToolOutcome,
    ToolResultBlock,
    start,
)
from conduit.agent import AgentRunner, AgentTask, RunStatus
from conduit.mcp import McpConnection, McpToolBridge
from conduit.providers.stub import StubFactory, StubToolCall, StubTurn, answer_tool_results

local_tickets = McpServerSettings.model_validate(
    {
        "name": "tickets",
        "transport": {"type": "stdio", "command": sys.executable, "args": ["ticket_server.py"]},
        "task_classes": ["investigate"],
        "connect_timeout": 60,
    }
)


def answer_from(results: tuple[ToolResultBlock, ...]) -> StubTurn:
    # The stub repeats the tool result it was sent, as the server's answer reached it.
    (block,) = results[0].content
    assert isinstance(block, TextBlock)
    return StubTurn(text=block.text)


model = answer_tool_results(
    StubTurn(
        tool_calls=(StubToolCall(name="tickets__open_tickets", input={"host": "192.0.2.10"}),)
    ),
    answer_from,
)


class ListSink:
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def emit(self, event: Event) -> None:
        self.events.append(event)


async def product_tools(tool: object, arguments: dict, context: ToolCallContext) -> ToolOutcome:
    return ToolOutcome(ok=False, content="no such tool", error_code="forbidden")


async def ask_the_server() -> list[Event]:
    sink = ListSink()
    runtime = await start(
        ConduitSettings.model_validate(
            {
                "providers": {
                    "local": {"kind": "stub", "script": "model", "not_for_production": True}
                },
                "assignments": {"investigate": {"provider": "local", "model": "stub-echo"}},
            }
        ),
        task_classes=[TaskClass(name="investigate", floor=CapabilityFloor(tools=True))],
        factories=[StubFactory(scripts={"model": model})],
        secrets=LiteralSecretResolver({}),
    )
    try:
        async with McpConnection(local_tickets, secrets=LiteralSecretResolver({})) as connection:
            bridge = McpToolBridge([connection], events=sink)
            catalogue = ToolCatalogue.from_tools(await bridge.tools())
            assert catalogue.names() == ["tickets__open_tickets"]
            runner = AgentRunner(
                runtime,
                tools={
                    "investigate": TaskTools(
                        catalogue.registry_for("investigate"), bridge.executor(product_tools)
                    )
                },
            )
            result = await runner.run(
                AgentTask(
                    task_class="investigate",
                    messages=(Message.user("How many tickets are open for 192.0.2.10?"),),
                )
            )
            assert result.status is RunStatus.COMPLETED
            # The server's answer, framed as untrusted content on its way to the model.
            assert "3 open tickets for 192.0.2.10" in result.text
            assert result.text != "3 open tickets for 192.0.2.10"
    finally:
        await runtime.aclose()
    return sink.events


mcp_events = asyncio.run(ask_the_server())
(call,) = [e for e in mcp_events if e.type == "mcp_call"]
assert call.outcome == "ok"
```

Combine MCP tools with the product's own in one catalogue: `ToolCatalogue.from_tools([*product_tools, *await bridge.tools()])`. Each MCP tool's description is the server's, so it is untrusted; the bridge prefixes it with the server's name and caps its length. Every result is framed as untrusted content before the model sees it, and capped at `max_result_chars`. Each call emits an `McpCallEvent` (server, tool, outcome, duration, and the arguments and result as captured content under the capture policy).

From sync code, connect on the facade's loop (`loop.run(connection.connect())`) and close the same way, as [the sync facade](integration.md#the-sync-facade) explains.

## OAuth sign-in

With `auth: {"type": "oauth"}`, conduit's OAuth 2.1 client obtains tokens for the server: it discovers the authorisation server from the MCP server's metadata, registers itself dynamically (or uses a pre-registered `client_id`, with an optional `client_secret` reference), runs the authorisation code flow with PKCE, and rotates refresh tokens. Tokens and registrations are persisted only through the product's `TokenStore` (under `mcp:<server>` and `mcp:<server>:client`); conduit writes no token file of its own.

`issuer` pins the authorisation server: discovery that names any other is refused before a request goes to it. A client secret needs the pin, so it can only ever be sent to the server that issued it.

```python
from conduit.mcp import LoopbackSignIn
from conduit.stores import MemoryTokenStore

wiki = McpServerSettings.model_validate(
    {
        "name": "wiki",
        "transport": {"type": "streamable_http", "url": "https://mcp.example.org/wiki"},
        "auth": {
            "type": "oauth",
            "scopes": ["wiki:read"],
            "issuer": "https://auth.example.org",
        },
        "task_classes": ["investigate"],
    }
)

terminal = McpConnection(
    wiki,
    secrets=LiteralSecretResolver({}),
    tokens=MemoryTokenStore(),
    sign_in=LoopbackSignIn(8765),
)
assert terminal.oauth is not None
```

When there are no usable tokens, the client calls the product's **sign-in hook**. The hook takes the person to the authorisation server and brings back the address the server redirected their browser to; the client checks that address's `state`, issuer and redirect URI before it uses the code. Without a hook, a server with no usable tokens is refused (`OAuthFailure.SIGN_IN_REQUIRED`).

- **A terminal or desktop product** uses `LoopbackSignIn(port)`: it listens on a loopback port, opens the browser at the authorisation URL, and waits for the one redirect (`http://127.0.0.1:<port>/callback`). The port is fixed, because an authorisation server matches registered redirect URIs exactly.
- **A server product** implements `SignInHook` itself: it sends the person to `authorization_url` through its own interface, receives the redirect on one of its own routes, and returns the full URL it received.

```python
from conduit.mcp import SignInRequest, SignInResult


class WebSignIn:
    """Sends the signed-in user to the authorisation server through the product's UI."""

    redirect_uri = "https://app.example.com/mcp/callback"

    def __init__(self, notify_user, wait_for_callback) -> None:
        self._notify_user = notify_user  # shows the person a link, in the product's UI
        self._wait_for_callback = wait_for_callback  # resolves when the route is hit

    async def sign_in(self, request: SignInRequest) -> SignInResult:
        await self._notify_user(request.server, request.authorization_url)
        callback_url = await self._wait_for_callback(request.server)
        return SignInResult(callback_url=callback_url)


async def show_link(server: str, url: str) -> None:
    assert url.startswith("https://auth.example.org/")


async def callback_route(server: str) -> str:
    return "https://app.example.com/mcp/callback?code=example-code&state=example-state"


hook = WebSignIn(show_link, callback_route)
result = asyncio.run(
    hook.sign_in(
        SignInRequest(
            server="wiki",
            authorization_url="https://auth.example.org/authorize?state=example-state",
            redirect_uri=hook.redirect_uri,
        )
    )
)
assert result.callback_url.startswith(hook.redirect_uri)
```

A `SignInRequest` and `SignInResult` keep their URLs out of `repr`, since they carry the state and the code. For a multi-user server product, give each user's connection its own token store key space (a token store per user, or keys prefixed by user) so one user's tokens never serve another.
