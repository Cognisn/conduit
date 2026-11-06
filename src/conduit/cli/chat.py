"""``chat``: streaming chat with tool use under the runner's budgets.

A chat is a product in miniature, built on conduit's public API:

- the runtime is started from the configuration (``conduit.start``), so every
  assignment is resolved, and refused, before the first message;
- each message is one ``AgentRunner`` run on the ``chat`` task class, under the run
  budget (model calls, tool calls, tokens, wall time); its text streams as it arrives;
- the tools are the interface's own (``add``, ``word_count``), ``use_skill`` when a
  skills directory is given, and the tools of every configured MCP server;
- a ``Compactor`` is the runner's compaction hook: by threshold, or forced with
  ``/compact``, and each compaction is shown with its size before and after;
- usage and cache reads and writes are shown for every model call, from the provider
  registry's events.

**No history is kept unless asked.** The conversation lives in memory for the session
and is gone when it ends. ``--transcripts DIR`` asks for it to be written to a JSON
transcript store there; ``--tokens DIR`` does the same for MCP OAuth tokens. Nothing
else is written anywhere.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any

from conduit import (
    ConduitError,
    Message,
    Runtime,
    TextBlock,
    ToolCatalogue,
    TranscriptStore,
    start,
)
from conduit.agent import AgentRunner, AgentTask, RunBudget, RunStatus
from conduit.agent.events import (
    BudgetExhaustedEvent,
    ConversationCompactedEvent,
    RunFinishedEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolCallStartedEvent,
)
from conduit.cli.config import CHAT, SUMMARY, CliConfig
from conduit.cli.demo import DEMO_TOOLS, demo_executor
from conduit.cli.output import EventPrinter, describe_error, usage_text
from conduit.cli.servers import bridged_tools, close_servers, connect_servers, token_store
from conduit.cli.setup import Context, task_classes
from conduit.compaction import (
    CompactionStrategy,
    Compactor,
    DropToolResults,
    KeepLastTurns,
    Summarise,
    estimate_tokens,
)
from conduit.core.events import CapturePolicy
from conduit.core.tools import TaskTools, ToolExecutor
from conduit.skills import FilesystemSkillSource, SkillError, SkillSession, with_use_skill
from conduit.stores.json import JsonTranscriptStore

HELP = """Commands:
  /help              this list
  /tools             the tools the model may call
  /skills            the skills listed and loaded
  /load NAME         load a skill now (the model can also call use_skill)
  /compact [NAME]    force a compaction before the next model call (strategy NAME,
                     or the most aggressive)
  /usage             usage so far, and the conversation's size
  /events            show every event, or stop
  /quit              leave (the conversation is not kept unless --transcripts was given)
To have the stub model call a tool, type: call <tool> <JSON arguments>
  for example: call add {"a": 2, "b": 3}"""


@dataclass(frozen=True, slots=True)
class ChatOptions:
    skills: str | None = None
    transcripts: str | None = None
    tokens: str | None = None
    connect_mcp: bool = True
    events: bool = False
    max_model_calls: int | None = None
    max_tool_calls: int | None = None
    max_tokens: int | None = None
    max_wall_time_s: float | None = None


def run_budget(config: CliConfig, options: ChatOptions) -> RunBudget:
    """The configuration's budget, with any limit given on the command line."""
    changes: dict[str, Any] = {}
    for field, value in (
        ("max_model_calls", options.max_model_calls),
        ("max_tool_calls", options.max_tool_calls),
        ("max_tokens", options.max_tokens),
        ("max_wall_time_s", options.max_wall_time_s),
    ):
        if value is not None:
            changes[field] = value
    data = {**config.budget.model_dump(), **changes}
    return RunBudget.model_validate(data)


def strategies(config: CliConfig, runtime: Runtime) -> list[CompactionStrategy]:
    section = config.compaction
    chosen: list[CompactionStrategy] = [
        DropToolResults(threshold=section.drop_tool_results_at, keep_turns=section.keep_turns),
        KeepLastTurns(threshold=section.keep_last_turns_at, keep_turns=section.keep_turns),
    ]
    if SUMMARY in runtime.assignments:
        chosen.append(
            Summarise(
                runtime,
                task_class=SUMMARY,
                threshold=section.summarise_at,
                keep_turns=section.keep_turns,
            )
        )
    return chosen


class Chat:
    """One chat session: what it holds lives in memory, and goes when it ends."""

    def __init__(self, context: Context, config: CliConfig, options: ChatOptions) -> None:
        self.context = context
        self.screen = context.screen
        self.config = config
        self.options = options
        self.budget = run_budget(config, options)
        self.printer = EventPrinter(context.screen, verbose=options.events)
        self.history: tuple[Message, ...] = ()
        self.transcripts: TranscriptStore | None = (
            JsonTranscriptStore(options.transcripts) if options.transcripts else None
        )
        # A session id only when a transcript is kept: it names the stored session.
        self.session_id = f"cli-{uuid.uuid4().hex}" if self.transcripts is not None else None
        self.skills: SkillSession | None = None
        self.tool_names: list[str] = []
        self.compactor: Compactor | None = None
        self.runner: AgentRunner | None = None
        self.model = ""

    async def run(self) -> int:
        screen = self.screen
        try:
            runtime = await start(
                self.config.conduit,
                task_classes=task_classes(self.config),
                factories=self.context.factories,
                secrets=self.context.secrets(),
                events=self.printer,
            )
        except Exception as error:
            screen.error(f"The chat cannot start: {describe_error(error)}")
            return 1
        connections: list[Any] = []
        try:
            if self.options.connect_mcp:
                tokens = token_store(self.options.tokens)
                connections = await connect_servers(self.context, self.config, tokens) or []
            return await self._session(runtime, connections)
        finally:
            await close_servers(connections)
            await runtime.aclose()

    async def _session(self, runtime: Runtime, connections: list[Any]) -> int:
        screen = self.screen
        tools: list[Any] = list(DEMO_TOOLS)
        executor: ToolExecutor = demo_executor
        directory = self.options.skills or self.config.skills
        if directory is not None:
            try:
                source = FilesystemSkillSource(directory)
            except ConduitError as error:
                screen.error(f"The skills directory was refused: {describe_error(error)}")
                return 1
            self.skills = SkillSession(
                source,
                events=self.printer,
                transcript=self.transcripts,
                session_id=self.session_id,
                task_class=CHAT,
            )
            tools.append(self.skills.tool(task_classes=[CHAT]))
        if connections:
            try:
                bridge, remote = await bridged_tools(connections, self.printer)
            except Exception as error:
                screen.error(f"[mcp] tools not listed: {describe_error(error)}")
            else:
                tools.extend(remote)
                executor = bridge.executor(demo_executor)
        catalogue = ToolCatalogue.from_tools(tools)
        registry = catalogue.registry_for(CHAT)
        self.tool_names = registry.names()
        self.compactor = Compactor(
            strategies(self.config, runtime), transcripts=self.transcripts, events=self.printer
        )
        self.runner = AgentRunner(
            runtime,
            tools={CHAT: TaskTools(registry, executor)},
            transcripts=self.transcripts,
            compaction=self.compactor,
            # Content goes to this terminal only: the runner is given no event sink.
            capture=CapturePolicy(prompts=True),
        )
        assignment = runtime.assignments.for_task(CHAT)
        self.model = f"{assignment.provider} {assignment.model}"
        screen.say(
            f"conduit chat on {self.model} ({assignment.entry.residency_scope.value}). "
            "Type /help for commands, /quit to leave."
        )
        if self.transcripts is None:
            screen.note("The conversation is kept in memory only, and is gone when you leave.")
        else:
            screen.note(f"The conversation is written to the transcript {self.session_id}.")
        return await self._loop()

    async def _read(self) -> str | None:
        stdin = self.context.stdin
        if stdin.isatty():
            self.screen.console.print("you> ", end="", markup=False, highlight=False)
        line = await asyncio.to_thread(stdin.readline)
        return None if line == "" else line.rstrip("\r\n")

    async def _loop(self) -> int:
        while True:
            line = await self._read()
            if line is None:
                return 0
            text = line.strip()
            if not text:
                continue
            if text.startswith("/"):
                if not await self._command(text):
                    return 0
                continue
            await self._turn(line)

    # Commands -------------------------------------------------------------------------

    async def _command(self, text: str) -> bool:
        """Run a slash command; ``False`` to leave."""
        screen = self.screen
        name, _, rest = text.partition(" ")
        argument = rest.strip() or None
        if name in ("/quit", "/exit"):
            return False
        if name == "/help":
            screen.say(HELP)
        elif name == "/tools":
            screen.say("Tools: " + (", ".join(self.tool_names) or "none"))
        elif name == "/skills":
            await self._show_skills()
        elif name == "/load":
            await self._load(argument)
        elif name == "/compact":
            self._compact(argument)
        elif name == "/usage":
            self._usage()
        elif name == "/events":
            self.printer.verbose = not self.printer.verbose
            screen.say(f"Every event is {'shown' if self.printer.verbose else 'not shown'}.")
        else:
            screen.error(f"No command {name}. Type /help for the list.")
        return True

    async def _show_skills(self) -> None:
        if self.skills is None:
            self.screen.say("No skills directory was given (--skills DIR).")
            return
        listed = await self.skills.summaries()
        loaded = {skill.name for skill in self.skills.loaded}
        rows = [
            [s.name, s.version or "-", "loaded" if s.name in loaded else "", s.description]
            for s in listed
        ]
        self.screen.table("Skills", ["name", "version", "state", "description"], rows)

    async def _load(self, name: str | None) -> None:
        if self.skills is None:
            self.screen.say("No skills directory was given (--skills DIR).")
            return
        if name is None:
            self.screen.error("Name the skill: /load NAME")
            return
        failure: str | None = None
        try:
            await self.skills.load(name)
        except SkillError as error:
            failure = describe_error(error)
        if failure is not None:
            self.screen.error(f"The skill was not loaded: {failure}")
        else:
            self.screen.say(f"Skill {name} is loaded: its instructions are in the system prompt.")

    def _size(self) -> str:
        tokens = estimate_tokens(messages=self.history)
        return f"{len(self.history)} messages, about {tokens} tokens"

    def _compact(self, strategy: str | None) -> None:
        if self.compactor is None:
            return
        failure: str | None = None
        try:
            self.compactor.force(self.session_id, strategy=strategy)
        except ConduitError as error:
            failure = describe_error(error)
        if failure is not None:
            names = ", ".join(s.name for s in self.compactor.strategies)
            self.screen.error(f"{failure}. Strategies: {names}")
            return
        self.screen.say(
            f"Compaction forced ({strategy or 'the most aggressive strategy'}). Before: "
            f"{self._size()}. It runs before the next model call, and shows its size "
            "before and after."
        )

    def _usage(self) -> None:
        printer = self.printer
        self.screen.say(
            f"Usage so far: {usage_text(printer.usage)} over {printer.calls} model calls. "
            f"Conversation: {self._size()}."
        )

    # A turn -----------------------------------------------------------------------------

    async def _turn(self, text: str) -> None:
        runner = self.runner
        if runner is None:
            return
        screen = self.screen
        skills = self.skills
        task = AgentTask(
            task_class=CHAT,
            messages=(Message.user(text),),
            history=self.history,
            system=(TextBlock(text=self.config.system),),
            budget=self.budget,
            session_id=self.session_id,
            cache_tools=True,
            system_hook=skills.system if skills is not None else None,
            tool_wrapper=with_use_skill if skills is not None else None,
        )
        try:
            run = runner.start(task)
        except ConduitError as error:
            screen.error(f"The message was refused: {describe_error(error)}")
            return
        failure: str | None = None
        try:
            async with aclosing(run.events()) as events:
                async for event in events:
                    if isinstance(event, TextDeltaEvent):
                        screen.stream(event.text.text or "")
                    else:
                        self._show(event)
        except Exception as error:
            failure = describe_error(error)
        screen.end_line()
        if failure is not None:
            screen.error(f"[run] stopped: {failure}")
            return
        result = run.result
        self.history = result.messages
        if result.status is not RunStatus.COMPLETED:
            why = result.failure.value if result.failure else "failed"
            detail = result.budget or result.provider_failure
            screen.error(f"[run] failed: {why}" + (f" ({detail.value})" if detail else ""))

    def _show(self, event: Any) -> None:
        screen = self.screen
        if isinstance(event, ToolCallStartedEvent):
            screen.note(f"[tool] {event.tool} called")
        elif isinstance(event, ToolCallEvent):
            code = f" ({event.error_code})" if event.error_code else ""
            screen.note(f"[tool] {event.tool}: {event.outcome}{code}")
        elif isinstance(event, BudgetExhaustedEvent):
            screen.error(
                f"[budget] the {event.budget.value} budget is exhausted: "
                f"{event.used:g} used of {event.limit:g}"
            )
        elif isinstance(event, ConversationCompactedEvent):
            screen.say(
                f"[compaction] the conversation went from {event.messages_before} to "
                f"{event.messages_after} messages"
            )
        elif isinstance(event, RunFinishedEvent):
            screen.note(
                f"[run] {event.outcome}: {usage_text(event.usage)}; "
                f"{event.model_calls} model calls, {event.tool_calls} tool calls"
            )
        elif self.printer.verbose:
            screen.note(f"[event] {event.type}")
