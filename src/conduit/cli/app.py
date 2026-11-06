"""The ``conduit`` command: arguments, dispatch and the outermost error handling.

::

    conduit providers [--config FILE]           adapters, their extras, configured providers
    conduit models [--config FILE] [--provider NAME]
                                                the model catalogue, with any overlays
    conduit check [--config FILE]               build every provider, resolve every
                                                assignment (residency refusals included)
    conduit mcp [--config FILE] [--tokens DIR]  connect every MCP server and list its tools
    conduit chat [--config FILE] [options]      streaming chat with tools, skills, MCP,
                                                compaction, usage and cache events

Exit status: 0 when the command succeeded, 1 when something was refused or failed, 2 for
a usage error, 130 when interrupted.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import sys
import webbrowser
from collections.abc import Callable, Sequence
from typing import Any

from rich.console import Console

from conduit import ConduitError, __version__
from conduit.cli.chat import Chat, ChatOptions
from conduit.cli.config import CliConfig, load_config
from conduit.cli.output import Screen, describe_error
from conduit.cli.servers import mcp_command
from conduit.cli.setup import Context, interface_factories
from conduit.cli.status import check, models, providers

DESCRIPTION = (
    "conduit's terminal interface: providers and credential checks, streaming chat with "
    "tools under run budgets, skills, MCP servers, compaction, usage and cache events. "
    "Without --config it runs on the stub provider. Nothing is kept unless you ask."
)


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return value


def _positive_float(text: str) -> float:
    value = float(text)
    if not value > 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="conduit", description=DESCRIPTION)
    top.add_argument("--version", action="version", version=f"conduit {__version__}")
    top.add_argument(
        "--log-level",
        default="warning",
        choices=["debug", "info", "warning", "error"],
        help="conduit's own log level, on standard error (default: warning)",
    )
    commands = top.add_subparsers(dest="command", metavar="COMMAND")

    def command(name: str, text: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=text, description=text)
        sub.add_argument(
            "--config", metavar="FILE", help="a JSON or TOML configuration file (default: stub)"
        )
        return sub

    command("providers", "list provider adapters, their extras and the configured providers")
    listing = command("models", "list the model catalogue, with the configuration's overlays")
    listing.add_argument("--provider", metavar="NAME", help="only this catalogue provider")
    command("check", "build every provider and resolve every assignment, calling no model")
    mcp = command("mcp", "connect every configured MCP server (signing in if needed)")
    mcp.add_argument("--tokens", metavar="DIR", help="keep OAuth tokens in DIR (default: memory)")
    chat = command("chat", "chat with streaming, tools, skills, MCP and compaction")
    chat.add_argument("--skills", metavar="DIR", help="a directory of SKILL.md skills")
    chat.add_argument(
        "--transcripts", metavar="DIR", help="keep the conversation in DIR (default: not kept)"
    )
    chat.add_argument("--tokens", metavar="DIR", help="keep OAuth tokens in DIR (default: memory)")
    chat.add_argument("--no-mcp", action="store_true", help="do not connect MCP servers")
    chat.add_argument("--events", action="store_true", help="show every event")
    chat.add_argument("--max-model-calls", type=_positive_int, metavar="N")
    chat.add_argument("--max-tool-calls", type=_positive_int, metavar="N")
    chat.add_argument("--max-tokens", type=_positive_int, metavar="N")
    chat.add_argument("--max-wall-time", type=_positive_float, metavar="SECONDS")
    return top


async def dispatch(context: Context, arguments: argparse.Namespace) -> int:
    config: CliConfig = load_config(arguments.config)
    name = arguments.command
    if name == "providers":
        return providers(context, config)
    if name == "models":
        return models(context, config, arguments.provider)
    if name == "check":
        return await check(context, config)
    if name == "mcp":
        return await mcp_command(context, config, arguments.tokens)
    options = ChatOptions(
        skills=arguments.skills,
        transcripts=arguments.transcripts,
        tokens=arguments.tokens,
        connect_mcp=not arguments.no_mcp,
        events=arguments.events,
        max_model_calls=arguments.max_model_calls,
        max_tool_calls=arguments.max_tool_calls,
        max_tokens=arguments.max_tokens,
        max_wall_time_s=arguments.max_wall_time,
    )
    return await Chat(context, config, options).run()


async def run(argv: Sequence[str], context: Context) -> int:
    """Run one command. Every error ends here, shown without any value it carries."""
    top = parser()
    try:
        arguments = top.parse_args(list(argv))
    except SystemExit as stop:  # --help, --version and usage errors
        return stop.code if isinstance(stop.code, int) else 2
    if arguments.command is None:
        top.print_help()
        return 2
    failure: str | None = None
    try:
        return await dispatch(context, arguments)
    except ConduitError as error:
        failure = describe_error(error)
    except Exception as error:
        failure = describe_error(error)  # the class only: its message might carry a value
    context.screen.error(f"conduit {arguments.command}: {failure}")
    return 1


def main(
    argv: Sequence[str] | None = None,
    *,
    console: Console | None = None,
    open_url: Callable[[str], Any] | None = None,
) -> int:
    """The ``conduit`` entry point."""
    args = list(sys.argv[1:] if argv is None else argv)
    # Other libraries stay at WARNING whatever is asked: some (the MCP SDK, botocore) log
    # message content or signatures at DEBUG. --log-level sets conduit's own logger only.
    logging.basicConfig(
        level=logging.WARNING, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s"
    )
    if "--log-level" in args:
        with contextlib.suppress(SystemExit):  # a bad value is reported by run()
            known, _ = parser().parse_known_args(args)
            logging.getLogger("conduit").setLevel(known.log_level.upper())
    context = Context(
        screen=Screen(console or Console()),
        stdin=sys.stdin,
        environ=dict(os.environ),
        factories=interface_factories(),
        open_url=open_url or webbrowser.open,
    )
    try:
        return asyncio.run(run(args, context))
    except KeyboardInterrupt:
        context.screen.error("Interrupted.")
        return 130
