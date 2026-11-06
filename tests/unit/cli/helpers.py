"""Running the terminal interface in-process, against a captured console.

Everything here is synthetic. The interface runs on the stub provider unless a test
gives it a configuration; no test reaches a live model or the network.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

from rich.console import Console

from conduit import ProviderFactory
from conduit.cli.app import run
from conduit.cli.output import Screen
from conduit.cli.setup import Context, interface_factories

REPO = Path(__file__).resolve().parents[3]
SKILLS = REPO / "tests" / "fixtures" / "skills"
STDIO_SERVER = REPO / "tests" / "unit" / "mcp" / "stdio_server.py"


@dataclass(frozen=True)
class Outcome:
    code: int
    output: str


def console(buffer: StringIO) -> Console:
    return Console(
        file=buffer, width=240, color_system=None, force_terminal=False, legacy_windows=False
    )


async def cli(
    *argv: str,
    stdin: str = "",
    environ: Mapping[str, str] | None = None,
    factories: Sequence[ProviderFactory] | None = None,
    open_url: Callable[[str], Any] | None = None,
) -> Outcome:
    """Run ``conduit <argv>`` with ``stdin`` as the person's input."""
    buffer = StringIO()
    context = Context(
        screen=Screen(console(buffer)),
        stdin=StringIO(stdin),
        environ=dict(environ or {}),
        factories=list(factories) if factories is not None else interface_factories(),
        open_url=open_url,
    )
    code = await run(list(argv), context)
    # Table cell borders read as spaces, so a row can be matched as words.
    return Outcome(code, buffer.getvalue().replace("\u2502", " "))


def write_config(directory: Path, data: Mapping[str, Any], name: str = "conduit.json") -> str:
    path = directory / name
    path.write_text(json.dumps(data), encoding="utf-8", newline="\n")
    return str(path)


def stub_conduit(**extra: Any) -> dict[str, Any]:
    """The stub demonstration's conduit settings, with changes."""
    return {
        "providers": {"stub": {"kind": "stub", "script": "demo", "not_for_production": True}},
        "assignments": {
            "chat": {"provider": "stub", "model": "stub-echo", "max_output_tokens": 1024}
        },
        **extra,
    }


def lines(*typed: str) -> str:
    """What the person types, one line each, then /quit."""
    return "".join(f"{line}\n" for line in (*typed, "/quit"))
