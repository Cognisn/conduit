"""MCP servers in the terminal interface: a stdio server in a chat, and the OAuth sign-in.

The stdio server is the MCP tests' synthetic one, run as a subprocess. The OAuth server
and the protected MCP server are the MCP tests' fakes, served on 127.0.0.1 inside this
worktree's slot (``CONDUIT_MCP_TEST_PORT_BASE``) at offsets the MCP tests do not use.
The browser is simulated; every value the fake server issues is synthetic.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

pytest.importorskip("mcp")

from tests.unit.cli.helpers import STDIO_SERVER, cli, lines, write_config
from tests.unit.mcp.fake_auth import FakeAuthServer
from tests.unit.mcp.servers import PORT_BASE, LoopbackServer, build_server
from tests.unit.mcp.test_signin import protect

AUTH_PORT = PORT_BASE + 50
PROTECTED_PORT = PORT_BASE + 51
CALLBACK_PORT = PORT_BASE + 52


def stdio_server(**extra: Any) -> dict[str, Any]:
    return {
        "name": "docs",
        "transport": {"type": "stdio", "command": sys.executable, "args": [str(STDIO_SERVER)]},
        "task_classes": ["chat"],
        "connect_timeout": 60,
        **extra,
    }


class TestStdioServer:
    async def test_its_tools_are_listed(self, tmp_path: Path) -> None:
        path = write_config(tmp_path, {"mcp": {"servers": [stdio_server()]}})
        result = await cli("mcp", "--config", path)
        assert result.code == 0, result.output
        assert "[mcp] connected to docs (stdio)" in result.output
        assert "docs__echo" in result.output and "docs__has_env" in result.output

    async def test_the_model_calls_its_tool_in_a_chat(self, tmp_path: Path) -> None:
        path = write_config(tmp_path, {"mcp": {"servers": [stdio_server()]}})
        result = await cli(
            "chat", "--config", path, stdin=lines("/tools", 'call docs__echo {"text": "hi mcp"}')
        )
        assert result.code == 0, result.output
        assert "Tools: add, docs__echo, docs__has_env, word_count" in result.output
        assert "[mcp] docs echo: ok" in result.output
        assert "[tool] docs__echo: ok" in result.output
        assert "hi mcp" in result.output.split("Tool results:", 1)[1]

    async def test_no_mcp_leaves_the_servers_alone(self, tmp_path: Path) -> None:
        path = write_config(tmp_path, {"mcp": {"servers": [stdio_server()]}})
        result = await cli("chat", "--config", path, "--no-mcp", stdin=lines("/tools"))
        assert "Tools: add, word_count" in result.output
        assert "[mcp] connected" not in result.output

    async def test_a_refused_server_is_named_by_its_fields(self, tmp_path: Path) -> None:
        server = stdio_server()
        server["transport"]["env"] = {"EXAMPLE_TOKEN": "zz-plain-token-value-zz"}
        path = write_config(tmp_path, {"mcp": {"servers": [server]}})
        result = await cli("mcp", "--config", path)
        assert result.code == 1
        assert "MCP server 1 was refused at transport.stdio.env.EXAMPLE_TOKEN" in result.output
        assert "zz-plain-token-value-zz" not in result.output

    async def test_a_server_that_does_not_start_is_reported(self, tmp_path: Path) -> None:
        server = stdio_server(connect_timeout=20)
        server["transport"]["args"] = [str(tmp_path / "absent_server.py")]
        path = write_config(tmp_path, {"mcp": {"servers": [server]}})
        result = await cli("chat", "--config", path, stdin=lines("/tools"))
        assert "[mcp] docs did not connect: McpError" in result.output
        assert "Tools: add, word_count" in result.output

    async def test_without_the_extra_it_names_the_extra(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "conduit.mcp", None)
        path = write_config(tmp_path, {"mcp": {"servers": [stdio_server()]}})
        result = await cli("chat", "--config", path, stdin=lines("hello"))
        assert "MCP servers need the mcp extra" in result.output
        assert "hello" in result.output


def browser(url: str) -> bool:
    """The person approves: the browser follows every redirect to the callback."""
    with httpx.Client(follow_redirects=True, timeout=10, trust_env=False) as client:
        client.get(url)
    return True


@pytest.fixture(scope="module")
def fake() -> FakeAuthServer:
    return FakeAuthServer(
        resource=f"http://127.0.0.1:{PROTECTED_PORT}/mcp",
        issuer=f"http://127.0.0.1:{AUTH_PORT}",
    )


@pytest.fixture(scope="module")
def servers(fake: FakeAuthServer) -> Iterator[None]:
    app = build_server("protected", host="127.0.0.1").streamable_http_app()
    with LoopbackServer(fake.asgi(), AUTH_PORT), LoopbackServer(protect(app, fake), PROTECTED_PORT):
        yield


def oauth_config(fake: FakeAuthServer) -> dict[str, Any]:
    return {
        "mcp": {
            "callback_port": CALLBACK_PORT,
            "servers": [
                {
                    "name": "protected",
                    "transport": {"type": "streamable_http", "url": fake.resource},
                    "auth": {"type": "oauth", "scopes": ["read"]},
                    "task_classes": ["chat"],
                    "connect_timeout": 20,
                }
            ],
        }
    }


@pytest.mark.usefixtures("servers")
class TestOAuthSignIn:
    async def test_the_sign_in_returns_on_the_loopback_callback(
        self, fake: FakeAuthServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        opened: list[str] = []

        def watching(url: str) -> bool:
            opened.append(url)
            return browser(url)

        path = write_config(tmp_path, oauth_config(fake))
        result = await cli("mcp", "--config", path, open_url=watching)
        assert result.code == 0, result.output
        redirect = f"http://127.0.0.1:{CALLBACK_PORT}/callback"
        assert f"the sign-in returns to {redirect}" in result.output
        assert "[mcp] connected to protected (streamable_http)" in result.output
        assert "protected__echo" in result.output
        assert len(opened) == 1 and opened[0].startswith(fake.issuer)
        (authorise,) = [r for r in fake.authorise_requests if r["redirect_uri"] == redirect]
        assert authorise["resource"] == fake.resource
        issued = fake.issued()
        assert issued
        for value in issued:
            assert value not in result.output
        # Without --tokens the tokens were held in memory: nothing was written.
        assert [p for p in work.rglob("*")] == []

    async def test_tokens_are_kept_only_where_asked(
        self, fake: FakeAuthServer, tmp_path: Path
    ) -> None:
        kept = tmp_path / "tokens"
        path = write_config(tmp_path, oauth_config(fake))
        first = await cli("mcp", "--config", path, "--tokens", str(kept), open_url=browser)
        assert first.code == 0, first.output
        assert any(p.is_file() for p in kept.rglob("*"))

        def no_browser(url: str) -> bool:
            raise AssertionError("signed in again although tokens were kept")

        again = await cli("mcp", "--config", path, "--tokens", str(kept), open_url=no_browser)
        assert again.code == 0, again.output
        assert "protected__echo" in again.output

    async def test_a_chat_calls_the_signed_in_server(
        self, fake: FakeAuthServer, tmp_path: Path
    ) -> None:
        path = write_config(tmp_path, oauth_config(fake))
        result = await cli(
            "chat",
            "--config",
            path,
            open_url=browser,
            stdin=lines('call protected__add {"a": 20, "b": 22}'),
        )
        assert "[mcp] protected add: ok" in result.output
        assert "42" in result.output.split("Tool results:", 1)[1]
        for value in fake.issued():
            assert value not in result.output
