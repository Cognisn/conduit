"""What reaches the terminal, the log levels, Ctrl-C, and checks that call no model.

Hostile text is synthetic: escape sequences (CSI, OSC 8 hyperlinks), a C1 control and
a bidirectional override, fed through every way text reaches the screen: a table cell,
a table title, an interface line and streamed model output.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
import textwrap
import threading
import time
from collections.abc import AsyncIterator, Iterator, Mapping
from io import StringIO
from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import SecretStr

from conduit import BaseProvider, Catalogue, ModelRequest, Provider, ProviderSettings
from conduit.cli import app
from conduit.cli.app import main, run
from conduit.cli.output import Screen
from conduit.cli.setup import Context, interface_factories
from conduit.core.streaming import StreamEvent
from conduit.providers.stub import StubSettings
from tests.unit.cli.helpers import cli, console, lines, stub_conduit, write_config

ESC_CSI = "\x1b[2J"
OSC_8 = "\x1b]8;;http://example.com\x07link\x1b]8;;\x07"
C1_CSI = "\x9b31m"
BIDI = "‮"
FORBIDDEN = ("\x1b", "\x9b", "\x07", BIDI)


def assert_clean(output: str) -> None:
    for char in FORBIDDEN:
        assert char not in output, repr(char)


class TestHostileTextIsShownSafely:
    async def test_an_mcp_tool_description(self, tmp_path: Path) -> None:
        pytest.importorskip("mcp")
        description = f"Looks {OSC_8} fine {ESC_CSI} {C1_CSI} {BIDI}evil"
        server = tmp_path / "hostile_server.py"
        server.write_text(
            textwrap.dedent(
                f"""\
                from mcp.server.fastmcp import FastMCP

                server = FastMCP("hostile")


                @server.tool(description={description!r})
                def probe() -> str:
                    return "ok"


                if __name__ == "__main__":
                    server.run("stdio")
                """
            ),
            encoding="utf-8",
            newline="\n",
        )
        mcp = {
            "servers": [
                {
                    "name": "hostile",
                    "transport": {
                        "type": "stdio",
                        "command": sys.executable,
                        "args": [str(server)],
                    },
                    "task_classes": ["chat"],
                    "connect_timeout": 60,
                }
            ]
        }
        result = await cli("mcp", "--config", write_config(tmp_path, {"mcp": mcp}))
        assert result.code == 0, result.output
        assert "hostile__probe" in result.output and "fine" in result.output
        assert_clean(result.output)

    async def test_a_skill_description(self, tmp_path: Path) -> None:
        skill = tmp_path / "skills" / "hostile-skill"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            f"---\nname: hostile-skill\ndescription: Reads {BIDI}backwards for example.com\n"
            "---\nSay hello.\n",
            encoding="utf-8",
            newline="\n",
        )
        result = await cli("chat", "--skills", str(tmp_path / "skills"), stdin=lines("/skills"))
        assert "hostile-skill" in result.output and "backwards" in result.output
        assert_clean(result.output)

    async def test_a_catalogue_cell(self, tmp_path: Path) -> None:
        overlay = {
            "name": "hostile",
            "add": [
                {
                    "model_id": f"stub-{ESC_CSI}{OSC_8}{C1_CSI}{BIDI}x",
                    "provider": "stub",
                    "residency_scope": "local",
                    "context_window": 4096,
                    "output_limit": 1024,
                }
            ],
        }
        path = write_config(tmp_path, {"conduit": stub_conduit(catalogue=[overlay])})
        result = await cli("models", "--config", path, "--provider", "stub")
        assert "stub-echo" in result.output
        assert_clean(result.output)

    async def test_a_table_title(self, tmp_path: Path) -> None:
        conduit = stub_conduit(residency={"regions": [f"ap-southeast-2{ESC_CSI}{C1_CSI}{BIDI}"]})
        result = await cli("check", "--config", write_config(tmp_path, {"conduit": conduit}))
        assert "Assignments (residency:" in result.output
        assert_clean(result.output)

    async def test_an_interface_line(self) -> None:
        result = await cli("chat", stdin=lines(f"/{ESC_CSI}{OSC_8}{C1_CSI}{BIDI}"))
        assert "No command /" in result.output
        assert_clean(result.output)

    async def test_streamed_model_output(self) -> None:
        result = await cli("chat", stdin=lines(f"a{ESC_CSI}b{OSC_8}c{C1_CSI}d{BIDI}e"))
        assert_clean(result.output)

    async def test_emoji_codes_are_shown_as_written(self, tmp_path: Path) -> None:
        conduit = stub_conduit()
        conduit["providers"]["stub"]["chunk_size"] = 4096
        path = write_config(tmp_path, {"conduit": conduit})
        result = await cli("chat", "--config", path, stdin=lines("/:warning: :smile:", "x :+1: y"))
        assert "No command /:warning:" in result.output
        assert "x :+1: y" in result.output
        assert "⚠" not in result.output and "\U0001f604" not in result.output


class FreshRoot:
    """Takes the root logger back to a fresh process's state (no handler, WARNING), so
    ``main`` configures logging as it would there; puts every handler and level back."""

    NAMES = ("conduit", "botocore", "mcp", "httpx", "anthropic")

    def __init__(self) -> None:
        self.root = logging.getLogger()
        self.level = self.root.level
        self.levels = {name: logging.getLogger(name).level for name in self.NAMES}
        self.handlers: list[logging.Handler] = []

    def __call__(self) -> None:
        # Called in the test body: pytest adds its capture handler for the call phase.
        for handler in list(self.root.handlers):
            self.root.removeHandler(handler)
            self.handlers.append(handler)
        self.root.setLevel(logging.WARNING)

    def restore(self) -> None:
        for handler in list(self.root.handlers):
            self.root.removeHandler(handler)
        for handler in self.handlers:
            self.root.addHandler(handler)
        self.root.setLevel(self.level)
        for name, level in self.levels.items():
            logging.getLogger(name).setLevel(level)


@pytest.fixture
def fresh_root() -> Iterator[FreshRoot]:
    state = FreshRoot()
    try:
        yield state
    finally:
        state.restore()


class TestLogLevels:
    def test_debug_applies_to_conduit_alone(self, fresh_root: FreshRoot) -> None:
        fresh_root()
        assert main(["--log-level", "debug", "providers"], console=console(StringIO())) == 0
        assert logging.getLogger().handlers, "main configured no handler"
        assert logging.getLogger("conduit").getEffectiveLevel() == logging.DEBUG
        assert logging.getLogger("conduit.mcp").getEffectiveLevel() == logging.DEBUG
        for name in ("botocore", "botocore.auth", "mcp", "mcp.client.session", "httpx"):
            assert logging.getLogger(name).getEffectiveLevel() >= logging.WARNING, name
        assert logging.getLogger().getEffectiveLevel() >= logging.WARNING

    def test_without_the_option_conduit_stays_at_warning(self, fresh_root: FreshRoot) -> None:
        fresh_root()
        assert main(["providers"], console=console(StringIO())) == 0
        assert logging.getLogger("conduit").getEffectiveLevel() >= logging.WARNING


class BlockingStdin:
    """Input that waits, as a person at the prompt does, until released."""

    def __init__(self) -> None:
        self.release = threading.Event()

    def isatty(self) -> bool:
        return False

    def readline(self) -> str:
        self.release.wait(15)
        return ""


class TestInterruption:
    def test_ctrl_c_at_the_prompt_ends_the_interface_promptly(self) -> None:
        stdin = BlockingStdin()
        buffer = StringIO()
        context = Context(
            screen=Screen(console(buffer)),
            stdin=stdin,
            environ={},
            factories=interface_factories(),
        )

        async def interrupted() -> None:
            task = asyncio.create_task(run(["chat"], context))
            await asyncio.sleep(1.5)  # the chat is waiting at the prompt
            task.cancel()  # what asyncio.run does to its task on Ctrl-C
            with contextlib.suppress(asyncio.CancelledError):
                await task

        began = time.monotonic()
        try:
            asyncio.run(interrupted())
        finally:
            elapsed = time.monotonic() - began
            stdin.release.set()
        assert "conduit chat on stub stub-echo" in buffer.getvalue()
        assert elapsed < 6, f"shutdown waited {elapsed:.1f}s for the blocked read"

    def test_an_interrupt_exits_with_130_and_no_traceback(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        async def interrupt(argv: Any, context: Any) -> int:
            raise KeyboardInterrupt

        monkeypatch.setattr(app, "run", interrupt)
        buffer = StringIO()
        try:
            code = main(["chat"], console=console(buffer))
        except KeyboardInterrupt:
            pytest.fail("the interrupt escaped main")
        assert code == 130
        assert "Interrupted." in buffer.getvalue()
        captured = capsys.readouterr()
        assert "Traceback" not in buffer.getvalue() + captured.out + captured.err


class NoCalls(BaseProvider):
    """A stub-kind provider that records, and refuses, every model call."""

    kind = "stub"
    calls: ClassVar[list[ModelRequest]] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        NoCalls.calls.append(request)
        raise AssertionError("check called a model")
        yield  # pragma: no cover


class NoCallFactory:
    kind = "stub"
    settings_model = StubSettings

    def create_client(self, settings: ProviderSettings, secrets: Mapping[str, SecretStr]) -> Any:
        return None

    def create_provider(
        self,
        name: str,
        settings: ProviderSettings,
        client: Any,
        *,
        catalogue: Catalogue | None = None,
    ) -> Provider:
        return NoCalls()


class TestCheckCallsNoModel:
    async def test_check_builds_and_resolves_without_a_model_call(self) -> None:
        NoCalls.calls = []
        others = [f for f in interface_factories() if f.kind != "stub"]
        result = await cli("check", factories=[NoCallFactory(), *others])
        assert result.code == 0, result.output
        assert "ready: no credentials needed" in result.output
        assert NoCalls.calls == []
