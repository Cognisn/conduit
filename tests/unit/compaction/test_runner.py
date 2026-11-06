"""The compactor as the agent runner's hook, end to end against the stub provider.

The stub's cache accounting writes the prompt up to its last cache breakpoint on the
first call and reads it on every later call with the same prefix (one prefix per call,
the last breakpoint's). That is enough to show the stable prefix (tools, system prompt
and loaded skills) is never disturbed by a compaction: cache reads continue on the
very next call. A real provider also reads the system prefix while it writes a new
summary breakpoint; the stub, keeping one prefix per call, reads the summary from the
call after.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from conduit.agent import AgentRunner, AgentTask, CompactionHook
from conduit.agent.events import ConversationCompactedEvent
from conduit.compaction import CompactionEvent, Compactor, KeepLastTurns
from conduit.core.tools import TaskTools, ToolCallContext, ToolCatalogue, ToolOutcome
from conduit.core.transcript import CompactionPoint, MessageEntry, Transcript
from conduit.core.types import CacheBreakpoint, Message, ModelRequest, TextBlock, Usage
from conduit.providers.stub import StubToolCall, StubTurn
from conduit.skills.disclosure import SkillSession
from conduit.skills.memory import MemorySkillSource
from conduit.stores.memory import MemoryTranscriptStore
from tests.unit.compaction.helpers import ScriptLog, Sink, filler, runtime
from tests.unit.skills.helpers import skill

BASE = (TextBlock(text="You help the example.com team with release checks."),)
CALLS = 10


def _script(calls: int = CALLS) -> ScriptLog:
    def play(request: ModelRequest, index: int) -> StubTurn:
        if index < calls:
            return StubTurn(tool_calls=(StubToolCall(name="lookup", input={"q": f"item {index}"}),))
        return StubTurn(text="All items are checked.")

    return ScriptLog(play)


class Lookup:
    name = "lookup"
    description = "Look up a synthetic release item."
    task_classes = frozenset({"chat"})
    requires_approval = False
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"q": {"type": "string"}},
    }


async def _execute(tool: Any, arguments: dict[str, Any], context: ToolCallContext) -> ToolOutcome:
    return ToolOutcome(ok=True, content=f"{arguments['q']}: " + filler(1_200, "row"))


def _tools() -> dict[str, TaskTools]:
    return {"chat": TaskTools(ToolCatalogue.from_tools([Lookup()]).registry_for("chat"), _execute)}


async def _skills() -> SkillSession:
    session = SkillSession(MemorySkillSource([skill("release-notes", "Write release notes.\n")]))
    await session.load("release-notes")
    return session


def _task(
    session: SkillSession, *, history: tuple[Message, ...] = (), text: str = "Check."
) -> AgentTask:
    return AgentTask(
        task_class="chat",
        messages=(Message.user(text),),
        history=history,
        system=BASE,
        system_hook=session.system,
        cache_tools=True,
        session_id="s-1",
    )


def _compacted(request: ModelRequest) -> bool:
    first = request.messages[0].content[0]
    return isinstance(first, TextBlock) and first.text.startswith("NOTICE")


def _usages(transcript: Transcript) -> list[Usage]:
    return [
        e.usage
        for e in transcript.entries
        if isinstance(e, MessageEntry) and e.message.role == "assistant" and e.usage is not None
    ]


def _compactor(store: MemoryTranscriptStore, sink: Sink, *, cache_summary: bool) -> Compactor:
    return Compactor(
        [KeepLastTurns(threshold=0.5, keep_turns=1, notice="NOTICE: earlier turns removed.")],
        transcripts=store,
        events=sink,
        cache_summary=cache_summary,
    )


def test_the_compactor_is_a_compaction_hook() -> None:
    assert isinstance(Compactor([KeepLastTurns()]), CompactionHook)


async def test_cache_reads_continue_on_the_call_after_a_compaction() -> None:
    log = _script()
    rt = await runtime(log)
    store, sink = MemoryTranscriptStore(), Sink()
    compactor = _compactor(store, sink, cache_summary=False)
    runner = AgentRunner(rt, tools=_tools(), transcripts=store, compaction=compactor)
    result = await runner.run(_task(await _skills()))
    assert result.text == "All items are checked."
    compacted = [i for i, r in enumerate(log.requests) if _compacted(r)]
    assert compacted, "the conversation never reached the threshold"
    first = compacted[0]
    usages = _usages(await store.load("s-1"))
    assert len(usages) == len(log.requests)
    assert usages[0].cache_write_tokens > 0 and usages[0].cache_read_tokens == 0
    prefix = usages[1].cache_read_tokens
    assert prefix > 0
    # Every call after the first reads the same prefix, the first compacted call included.
    assert all(u.cache_read_tokens == prefix for u in usages[1:])
    assert all(u.cache_write_tokens == 0 for u in usages[1:])
    assert usages[first].input_tokens < usages[first - 1].input_tokens


async def test_a_cached_summary_is_read_from_the_call_after() -> None:
    log = _script()
    rt = await runtime(log)
    store, sink = MemoryTranscriptStore(), Sink()
    runner = AgentRunner(
        rt,
        tools=_tools(),
        transcripts=store,
        compaction=_compactor(store, sink, cache_summary=True),
    )
    await runner.run(_task(await _skills()))
    compacted = [i for i, r in enumerate(log.requests) if _compacted(r)]
    assert compacted
    first = compacted[0]
    assert first + 1 < len(log.requests), "no call followed the compaction"
    usages = _usages(await store.load("s-1"))
    system_prefix = usages[1].cache_read_tokens
    # The compacted call caches the stable prefix with the summary after it...
    assert usages[first].cache_write_tokens > system_prefix
    # ...and the next call reads both.
    assert usages[first + 1].cache_read_tokens == usages[first].cache_write_tokens
    assert usages[first + 1].cache_read_tokens > system_prefix


async def test_system_tools_and_loaded_skills_stay_byte_identical() -> None:
    log = _script()
    rt = await runtime(log)
    store, sink = MemoryTranscriptStore(), Sink()
    runner = AgentRunner(
        rt,
        tools=_tools(),
        transcripts=store,
        compaction=_compactor(store, sink, cache_summary=True),
    )
    await runner.run(_task(await _skills()))
    assert any(_compacted(r) for r in log.requests)
    first = log.requests[0]
    assert any(isinstance(b, TextBlock) and "Write release notes." in b.text for b in first.system)
    assert isinstance(first.system[-1], CacheBreakpoint)
    for request in log.requests[1:]:
        assert [b.model_dump_json() for b in request.system] == [
            b.model_dump_json() for b in first.system
        ]
        assert [t.model_dump_json() for t in request.tools] == [
            t.model_dump_json() for t in first.tools
        ]
        assert request.cache_tools
    # The summary follows the prefix: it is a message, never part of the system blocks.
    assert all("NOTICE" not in b.model_dump_json() for r in log.requests for b in r.system)


async def test_the_transcript_reloads_to_the_conversation_the_run_ended_with() -> None:
    log = _script()
    rt = await runtime(log)
    store, sink = MemoryTranscriptStore(), Sink()
    compactor = _compactor(store, sink, cache_summary=True)
    runner = AgentRunner(rt, tools=_tools(), transcripts=store, compaction=compactor)
    result = await runner.run(_task(await _skills()))
    transcript = await store.load("s-1")
    points = [e for e in transcript.entries if isinstance(e, CompactionPoint)]
    assert points
    assert len(points) == len(sink.of(CompactionEvent))
    assert transcript.active_messages() == result.messages
    assert [m.model_dump_json() for m in transcript.active_messages()] == [
        m.model_dump_json() for m in result.messages
    ]
    for point, event in zip(points, sink.of(CompactionEvent), strict=True):
        assert point.tokens_before == event.tokens_before
        assert point.tokens_after == event.tokens_after
        assert point.tokens_after is not None and point.tokens_before is not None
        assert point.tokens_after < point.tokens_before


async def test_a_reloaded_session_carries_on_reading_the_cache() -> None:
    log = _script()
    rt = await runtime(log)
    store, sink = MemoryTranscriptStore(), Sink()
    runner = AgentRunner(
        rt,
        tools=_tools(),
        transcripts=store,
        compaction=_compactor(store, sink, cache_summary=True),
    )
    session = await _skills()
    await runner.run(_task(session))
    before = len(log.requests)
    history = (await store.load("s-1")).active_messages()
    await runner.run(_task(session, history=history, text="Anything else?"))
    usages = _usages(await store.load("s-1"))
    # The next run's first call sends the same prefix and summary: a cache read.
    assert usages[before].cache_read_tokens >= usages[before - 1].cache_read_tokens > 0


async def test_a_forced_compaction_through_the_runner() -> None:
    log = ScriptLog(lambda request, index: StubTurn(text=f"answer {index}"))
    rt = await runtime(log)
    store, sink = MemoryTranscriptStore(), Sink()
    compactor = Compactor(
        [KeepLastTurns(threshold=1.0, keep_turns=1, notice="NOTICE")],
        transcripts=store,
        events=sink,
    )
    run_sink = Sink()
    runner = AgentRunner(rt, transcripts=store, compaction=compactor, events=run_sink)
    session = await _skills()
    history: tuple[Message, ...] = ()
    for turn in range(3):
        result = await runner.run(_task(session, history=history, text=f"question {turn}"))
        history = result.messages
    assert sink.of(CompactionEvent) == []
    compactor.force("s-1")
    result = await runner.run(_task(session, history=history, text="question 3"))
    [event] = sink.of(CompactionEvent)
    assert (event.trigger, event.outcome) == ("forced", "compacted")
    assert _compacted(log.requests[-1])
    assert [e.messages_after for e in run_sink.of(ConversationCompactedEvent)] == [3]
    assert (await store.load("s-1")).active_messages() == result.messages


async def test_concurrent_runs_compact_only_the_forced_session() -> None:
    log = ScriptLog(lambda request, index: StubTurn(text="ok"))
    rt = await runtime(log)
    store, sink = MemoryTranscriptStore(), Sink()
    compactor = Compactor(
        [KeepLastTurns(threshold=1.0, keep_turns=1)], transcripts=store, events=sink
    )
    runner = AgentRunner(rt, transcripts=store, compaction=compactor)
    session = await _skills()

    history = tuple(
        m for i in range(3) for m in (Message.user(f"q{i}"), Message.assistant(f"a{i}"))
    )

    async def run(session_id: str) -> tuple[Message, ...]:
        await store.append(session_id, [MessageEntry(message=m) for m in history])
        task = AgentTask(
            task_class="chat",
            messages=(Message.user("next"),),
            history=history,
            system=BASE,
            system_hook=session.system,
            session_id=session_id,
        )
        await asyncio.sleep(0)
        return (await runner.run(task)).messages

    compactor.force("s-b")
    results = await asyncio.gather(*(run(f"s-{name}") for name in "abcd"))
    assert [len(r) for r in results] == [8, 4, 8, 8]
    assert [e.run_id is not None for e in sink.of(CompactionEvent)] == [True]
    for name, messages in zip("abcd", results, strict=True):
        assert (await store.load(f"s-{name}")).active_messages() == messages
