"""The MCP client: connection lifecycle, every transport, timeouts and clean shutdown."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from conduit.core.errors import SecretUnavailable
from conduit.core.secrets import LiteralSecretResolver
from conduit.mcp.client import McpConnection
from conduit.mcp.errors import McpError, McpFailure
from conduit.mcp.settings import McpServerSettings
from tests.unit.mcp.servers import (
    build_server,
    failing_transport,
    hanging_transport,
    memory_transport,
    rejecting_server,
    silent_transport,
    stubborn_transport,
)

STDIO_SERVER = Path(__file__).with_name("stdio_server.py")
SYNTHETIC_SECRET = "zz-stdio-env-secret-0000zz"


def settings(**overrides: Any) -> McpServerSettings:
    data: dict[str, Any] = {
        "name": "example",
        "transport": {"type": "stdio", "command": sys.executable, "args": [str(STDIO_SERVER)]},
        "task_classes": ["research"],
        "connect_timeout": 20,
    }
    data.update(overrides)
    return McpServerSettings.model_validate(data)


def connection(transport: Any = None, **overrides: Any) -> McpConnection:
    return McpConnection(
        settings(**overrides),
        secrets=LiteralSecretResolver({}),
        transport=transport if transport is not None else memory_transport(build_server()),
    )


def assert_clean(error: BaseException) -> None:
    assert error.__cause__ is None
    assert error.__context__ is None


class TestLifecycle:
    async def test_connect_list_call_close(self) -> None:
        conn = connection()
        assert not conn.connected
        await conn.connect()
        assert conn.connected
        tools = {tool.name: tool for tool in await conn.list_tools()}
        assert {"echo", "add", "fail", "slow", "Get-Item.v2"} <= set(tools)
        assert tools["add"].input_schema["type"] == "object"
        assert "a" in tools["add"].input_schema["properties"]
        result = await conn.call_tool("add", {"a": 2, "b": 3})
        assert not result.is_error
        assert result.text == "5"
        await conn.close()
        assert not conn.connected

    async def test_used_as_a_context_manager(self) -> None:
        async with connection() as conn:
            assert conn.connected
            assert (await conn.call_tool("echo", {"text": "hello"})).text == "hello"
        assert not conn.connected

    async def test_connect_is_idempotent_and_close_is_repeatable(self) -> None:
        conn = connection()
        await conn.connect()
        await conn.connect()
        await conn.close()
        await conn.close()

    async def test_a_call_after_close_is_refused_cleanly(self) -> None:
        conn = connection()
        await conn.connect()
        await conn.close()
        with pytest.raises(McpError) as caught:
            await conn.call_tool("echo", {"text": "x"})
        assert caught.value.failure is McpFailure.CLOSED
        assert_clean(caught.value)

    async def test_a_call_before_connect_is_refused(self) -> None:
        with pytest.raises(McpError) as caught:
            await connection().list_tools()
        assert caught.value.failure is McpFailure.CLOSED

    async def test_a_tool_error_is_a_result_not_an_exception(self) -> None:
        async with connection() as conn:
            result = await conn.call_tool("fail", {})
        assert result.is_error
        assert "synthetic tool failure" in result.text

    async def test_an_unknown_tool_is_answered_by_the_server(self) -> None:
        async with connection() as conn:
            result = await conn.call_tool("no_such_tool", {})
        assert result.is_error

    async def test_non_text_content_is_described(self) -> None:
        async with connection() as conn:
            picture = await conn.call_tool("picture", {})
            mixed = await conn.call_tool("mixed", {})
        assert "image/png" in picture.text
        assert "iVBOR" not in picture.text  # the data itself is not passed on
        assert "first part" in mixed.text
        assert "notes" in mixed.text
        assert "https://example.com/doc" in mixed.text

    async def test_the_reconnect_after_close_works(self) -> None:
        conn = connection()
        await conn.connect()
        await conn.close()
        await conn.connect()
        assert (await conn.call_tool("echo", {"text": "again"})).text == "again"
        await conn.close()


class TestFailures:
    async def test_a_connect_that_never_answers_times_out(self) -> None:
        conn = connection(silent_transport(), connect_timeout=0.3)
        started = time.monotonic()
        with pytest.raises(McpError) as caught:
            await conn.connect()
        assert time.monotonic() - started < 5
        assert caught.value.failure is McpFailure.TIMEOUT
        assert_clean(caught.value)
        assert not conn.connected

    async def test_a_transport_that_fails_is_a_clean_connect_failure(self) -> None:
        conn = connection(failing_transport(OSError(f"refused {SYNTHETIC_SECRET}")))
        with pytest.raises(McpError) as caught:
            await conn.connect()
        assert caught.value.failure is McpFailure.CONNECT
        assert_clean(caught.value)
        for text in (str(caught.value), repr(caught.value), repr(caught.value.args)):
            assert SYNTHETIC_SECRET not in text

    async def test_a_slow_call_times_out_and_the_connection_survives(self) -> None:
        async with connection(call_timeout=0.3) as conn:
            with pytest.raises(McpError) as caught:
                await conn.call_tool("slow", {"seconds": 5})
            assert caught.value.failure is McpFailure.TIMEOUT
            assert_clean(caught.value)
            assert (await conn.call_tool("echo", {"text": "still here"})).text == "still here"

    async def test_cancellation_propagates_unchanged(self) -> None:
        async with connection() as conn:
            task = asyncio.create_task(conn.call_tool("slow", {"seconds": 5}))
            await asyncio.sleep(0.2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert (await conn.call_tool("echo", {"text": "after"})).text == "after"

    async def test_a_cancelled_connect_leaves_nothing_running(self) -> None:
        conn = connection(silent_transport(), connect_timeout=30)
        task = asyncio.create_task(conn.connect())
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not conn.connected
        assert conn._runner is None

    async def test_close_while_a_call_is_running_fails_the_call_at_once(self) -> None:
        conn = connection(call_timeout=60)
        await conn.connect()
        task = asyncio.create_task(conn.call_tool("slow", {"seconds": 30}))
        await asyncio.sleep(0.2)
        started = time.monotonic()
        await conn.close()
        with pytest.raises(McpError) as caught:
            await task
        assert time.monotonic() - started < 5  # not left waiting for the call timeout
        assert caught.value.failure is McpFailure.CLOSED
        assert_clean(caught.value)

    async def test_close_gives_up_after_the_close_timeout(self) -> None:
        conn = connection(stubborn_transport(build_server(), hold=3), close_timeout=0.3)
        await conn.connect()
        started = time.monotonic()
        await conn.close()
        assert time.monotonic() - started < 2.5
        assert not conn.connected

    async def test_close_cancels_a_runner_that_does_not_stop(self) -> None:
        conn = connection(hanging_transport(build_server()), close_timeout=0.3)
        await conn.connect()
        runner = conn._runner
        assert runner is not None
        await conn.close()
        await asyncio.sleep(0)
        assert runner.done()

    async def test_a_server_error_is_a_clean_rejection(self) -> None:
        conn = connection(memory_transport(rejecting_server()))
        async with conn:
            with pytest.raises(McpError) as caught:
                await conn.list_tools()
        assert caught.value.failure is McpFailure.REJECTED
        assert_clean(caught.value)
        for text in (str(caught.value), repr(caught.value), repr(caught.value.args)):
            assert SYNTHETIC_SECRET not in text

    async def test_a_body_failure_is_not_replaced_by_close(self) -> None:
        conn = connection(stubborn_transport(build_server(), hold=1), close_timeout=0.2)
        with pytest.raises(LookupError, match="body"):
            async with conn:
                raise LookupError("body")

    async def test_an_unresolvable_secret_fails_before_anything_starts(self) -> None:
        conn = McpConnection(
            settings(
                transport={
                    "type": "stdio",
                    "command": sys.executable,
                    "args": [str(STDIO_SERVER)],
                    "secret_env": {"EXAMPLE_KEY": "secret://missing"},
                }
            ),
            secrets=LiteralSecretResolver({}),
        )
        with pytest.raises(SecretUnavailable):
            await conn.connect()
        assert conn._runner is None


class TestStdio:
    async def test_a_stdio_subprocess_server(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CONDUIT_NOT_INHERITED", "set in the parent")
        conn = McpConnection(
            settings(
                transport={
                    "type": "stdio",
                    "command": sys.executable,
                    "args": [str(STDIO_SERVER)],
                    "env": {"EXAMPLE_MODE": "test"},
                    "secret_env": {"EXAMPLE_KEY": "secret://stdio_key"},
                }
            ),
            secrets=LiteralSecretResolver({"stdio_key": SYNTHETIC_SECRET}),
        )
        async with conn:
            assert {t.name for t in await conn.list_tools()} == {"echo", "has_env"}
            assert (await conn.call_tool("echo", {"text": "over stdio"})).text == "over stdio"
            key = await conn.call_tool("has_env", {"name": "EXAMPLE_KEY"})
            mode = await conn.call_tool("has_env", {"name": "EXAMPLE_MODE"})
            absent = await conn.call_tool("has_env", {"name": "CONDUIT_NOT_INHERITED"})
        assert key.text == f"set, {len(SYNTHETIC_SECRET)} characters"
        assert mode.text == "set, 4 characters"
        assert absent.text == "unset"

    async def test_a_missing_command_is_a_clean_connect_failure(self, tmp_path: Path) -> None:
        missing = tmp_path / "no-such-server.exe"
        conn = McpConnection(
            settings(transport={"type": "stdio", "command": str(missing)}),
            secrets=LiteralSecretResolver({}),
        )
        with pytest.raises(McpError) as caught:
            await conn.connect()
        assert caught.value.failure is McpFailure.CONNECT
        assert_clean(caught.value)
        assert str(missing) not in str(caught.value)
