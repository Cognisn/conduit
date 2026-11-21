"""The terminal interface's configuration file: conduit's settings and the demonstration's.

A configuration file is JSON (``.json``) or TOML (``.toml``)::

    {
      "conduit": { "providers": {...}, "assignments": {"chat": {...}}, "residency": {...} },
      "system": "You are a helpful assistant.",
      "budget": {"max_model_calls": 8, "max_tool_calls": 8},
      "skills": "skills",
      "mcp": {"callback_port": 33418, "servers": [{"name": "docs", ...}]},
      "compaction": {"keep_turns": 2}
    }

``conduit`` is ``ConduitSettings`` as conduit reads it: credentials only as
``secret://NAME`` references, resolved from the environment variable ``NAME``. The
interface declares the task class ``chat`` (which needs tools), and ``summary`` when the
file assigns it (the summarising compaction strategy uses it). ``skills`` is a
directory, relative to the file. Without a file the interface runs on the stub provider.

A file that cannot be read or validated is refused naming the fields at fault, never a
value from the file.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from conduit import ConduitSettings, ConfigurationError, refusals
from conduit.agent import RunBudget

#: The task class every chat runs on.
CHAT = "chat"
#: The task class the summarising compaction strategy runs on, when assigned.
SUMMARY = "summary"

DEFAULT_SYSTEM = (
    "You are the conduit demonstration assistant. Use the tools you are given when they "
    "help, and keep answers short."
)

#: The stub's demonstration script (``conduit.cli.demo``) on the shipped stub model.
DEFAULT_CONDUIT: dict[str, Any] = {
    "providers": {"stub": {"kind": "stub", "script": "demo", "not_for_production": True}},
    "assignments": {CHAT: {"provider": "stub", "model": "stub-echo", "max_output_tokens": 1024}},
}

DEFAULT_CALLBACK_PORT = 33418


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class McpSection(_Section):
    """MCP servers to connect when a chat starts, and the OAuth sign-in's loopback port.

    Each server is ``McpServerSettings`` as ``conduit.mcp`` reads it; it is validated when
    the servers are connected, so a file without servers needs no mcp extra.
    """

    callback_port: int = Field(default=DEFAULT_CALLBACK_PORT, ge=1, le=65535)
    servers: tuple[dict[str, Any], ...] = ()


class CompactionSection(_Section):
    """The compaction strategies offered: by threshold, and on ``/compact``."""

    keep_turns: int = Field(default=2, ge=1, le=64)
    drop_tool_results_at: float = Field(default=0.6, gt=0, le=1)
    keep_last_turns_at: float = Field(default=0.85, gt=0, le=1)
    summarise_at: float = Field(default=0.75, gt=0, le=1)


class CliConfig(_Section):
    conduit: ConduitSettings = Field(
        default_factory=lambda: ConduitSettings.model_validate(DEFAULT_CONDUIT)
    )
    system: str = DEFAULT_SYSTEM
    budget: RunBudget = Field(
        default_factory=lambda: RunBudget(max_model_calls=8, max_tool_calls=8)
    )
    skills: str | None = None
    mcp: McpSection = Field(default_factory=McpSection)
    compaction: CompactionSection = Field(default_factory=CompactionSection)


def _read(path: Path) -> Any:
    """The file's data, or a ``ConfigurationError`` that quotes none of it."""
    problem = ""
    data: Any = None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        problem = f"cannot be read ({type(error).__name__})"
        text = ""
    if not problem:
        if path.suffix.lower() == ".toml":
            try:
                data = tomllib.loads(text)
            except tomllib.TOMLDecodeError:
                problem = "is not valid TOML"
        else:
            try:
                data = json.loads(text)
            except json.JSONDecodeError as error:
                problem = f"is not JSON (line {error.lineno}, column {error.colno})"
    if problem:
        raise ConfigurationError(f"the configuration file {problem}")
    if not isinstance(data, dict):
        raise ConfigurationError("the configuration file must hold an object (a table)")
    return data


def parse_config(data: dict[str, Any], *, base: Path | None = None) -> CliConfig:
    """A validated configuration. ``skills`` is resolved against ``base``."""
    found: tuple[list[str], list[str]] | None = None
    config: CliConfig | None = None
    try:
        config = CliConfig.model_validate(data)
    except ValidationError as error:
        found = refusals(error, data)
    if found is not None or config is None:
        fields, reasons = found or ([], [])
        why = f" ({'; '.join(reasons)})" if reasons else ""
        raise ConfigurationError(f"the configuration was refused at {', '.join(fields)}{why}")
    if config.skills is not None and base is not None:
        config = config.model_copy(update={"skills": str((base / config.skills).resolve())})
    return config


def load_config(path: str | Path | None) -> CliConfig:
    """The configuration in ``path``, or the stub demonstration's without one."""
    if path is None:
        return CliConfig()
    file = Path(path)
    return parse_config(_read(file), base=file.parent)
