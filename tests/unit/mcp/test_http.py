"""Streamable HTTP and SSE servers on loopback, with static credentials by reference."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import pytest
from starlette.responses import PlainTextResponse

from conduit.core.secrets import LiteralSecretResolver
from conduit.mcp.client import McpConnection
from conduit.mcp.errors import McpError, McpFailure
from conduit.mcp.settings import McpServerSettings
from tests.unit.mcp.servers import LoopbackServer, build_server, port, serving

BEARER = "zz-synthetic-bearer-token-7731zz"
API_KEY = "zz-synthetic-api-key-8842zz"


def require_header(app: Any, header: str, value: str, seen: list[str]) -> Any:
    """``app`` behind a check of one request header; records what each request sent."""
    wanted = header.lower().encode("latin-1")

    async def guarded(scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            sent = dict(scope["headers"]).get(wanted, b"").decode("latin-1")
            seen.append(sent)
            if sent != value:
                await PlainTextResponse("unauthorised", status_code=401)(scope, receive, send)
                return
        await app(scope, receive, send)

    return guarded


@pytest.fixture(scope="module")
def bearer_seen() -> list[str]:
    return []


@pytest.fixture(scope="module")
def http_server(bearer_seen: list[str]) -> Iterator[LoopbackServer]:
    app = build_server("http-example", host="127.0.0.1").streamable_http_app()
    with serving(
        require_header(app, "Authorization", f"Bearer {BEARER}", bearer_seen), "http"
    ) as s:
        yield s


@pytest.fixture(scope="module")
def key_seen() -> list[str]:
    return []


@pytest.fixture(scope="module")
def sse_server(key_seen: list[str]) -> Iterator[LoopbackServer]:
    app = build_server("sse-example", host="127.0.0.1").sse_app()
    with serving(require_header(app, "X-Example-Key", API_KEY, key_seen), "sse") as s:
        yield s


def http_settings(**overrides: Any) -> McpServerSettings:
    data: dict[str, Any] = {
        "name": "remote",
        "transport": {"type": "streamable_http", "url": f"http://127.0.0.1:{port('http')}/mcp"},
        "auth": {"type": "bearer", "token": "secret://remote_token"},
        "task_classes": ["research"],
        "connect_timeout": 10,
        "close_timeout": 5,
    }
    data.update(overrides)
    return McpServerSettings.model_validate(data)


def sse_settings(**overrides: Any) -> McpServerSettings:
    data: dict[str, Any] = {
        "name": "events",
        "transport": {"type": "sse", "url": f"http://127.0.0.1:{port('sse')}/sse"},
        "auth": {"type": "api_key", "key": "secret://events_key", "header": "X-Example-Key"},
        "task_classes": ["research"],
        "connect_timeout": 10,
        "close_timeout": 5,
    }
    data.update(overrides)
    return McpServerSettings.model_validate(data)


SECRETS = LiteralSecretResolver(
    {"remote_token": BEARER, "events_key": API_KEY, "wrong": "zz-wrong-credential-0000zz"}
)


class TestStreamableHttp:
    async def test_a_bearer_token_by_reference(
        self, http_server: LoopbackServer, bearer_seen: list[str]
    ) -> None:
        async with McpConnection(http_settings(), secrets=SECRETS) as conn:
            assert "echo" in {t.name for t in await conn.list_tools()}
            assert (await conn.call_tool("echo", {"text": "over http"})).text == "over http"
        assert bearer_seen
        assert set(bearer_seen) == {f"Bearer {BEARER}"}

    async def test_a_wrong_token_is_an_auth_failure(self, http_server: LoopbackServer) -> None:
        settings = http_settings(auth={"type": "bearer", "token": "secret://wrong"})
        with pytest.raises(McpError) as caught:
            await McpConnection(settings, secrets=SECRETS).connect()
        assert caught.value.failure is McpFailure.AUTH
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None

    async def test_a_server_that_is_not_there_is_a_connect_failure(self) -> None:
        settings = http_settings(
            transport={
                "type": "streamable_http",
                "url": f"http://127.0.0.1:{port('callback2')}/mcp",
            },
            connect_timeout=5,
        )
        with pytest.raises(McpError) as caught:
            await McpConnection(settings, secrets=SECRETS).connect()
        assert caught.value.failure in (McpFailure.CONNECT, McpFailure.TIMEOUT)
        assert caught.value.__context__ is None

    async def test_environment_proxies_are_ignored_unless_asked_for(
        self, http_server: LoopbackServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A proxy no one listens on: honouring it would make every request fail.
        proxy = f"http://127.0.0.1:{port('callback2')}"
        for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
            monkeypatch.setenv(name, proxy)
        for name in ("NO_PROXY", "no_proxy"):
            monkeypatch.delenv(name, raising=False)
        async with McpConnection(http_settings(), secrets=SECRETS) as conn:
            assert (await conn.call_tool("echo", {"text": "direct"})).text == "direct"
        honouring = McpConnection(
            http_settings(use_environment_proxies=True, connect_timeout=5), secrets=SECRETS
        )
        with pytest.raises(McpError):
            await honouring.connect()


class TestSse:
    async def test_an_api_key_by_reference(
        self, sse_server: LoopbackServer, key_seen: list[str]
    ) -> None:
        async with McpConnection(sse_settings(), secrets=SECRETS) as conn:
            assert "add" in {t.name for t in await conn.list_tools()}
            assert (await conn.call_tool("add", {"a": 20, "b": 22})).text == "42"
        assert key_seen
        assert set(key_seen) == {API_KEY}


class TestCredentialsStayOut:
    async def test_no_credential_reaches_a_log_record(
        self,
        http_server: LoopbackServer,
        sse_server: LoopbackServer,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.DEBUG)
        async with McpConnection(http_settings(), secrets=SECRETS) as conn:
            await conn.call_tool("echo", {"text": "logged call"})
        async with McpConnection(sse_settings(), secrets=SECRETS) as conn:
            await conn.call_tool("echo", {"text": "logged call"})
        with pytest.raises(McpError):
            await McpConnection(
                http_settings(auth={"type": "bearer", "token": "secret://wrong"}), secrets=SECRETS
            ).connect()
        assert caplog.records, "the check would be vacuous with nothing logged"
        for record in caplog.records:
            text = record.getMessage() + repr(record.args)
            assert BEARER not in text
            assert API_KEY not in text
            assert "zz-wrong-credential-0000zz" not in text
