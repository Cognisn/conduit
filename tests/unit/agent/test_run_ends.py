"""A run ended part-way (abandoned, cancelled, failing) leaves no tool use unanswered.

The stored transcript must stay loadable by every provider, so every tool use the
conversation holds gets a result: the real one if the tool finished, a stopped one if not.
A store that fails while closing them never replaces the end the run is on.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

import pytest

from conduit.agent import AgentRunner, AgentTask
from conduit.agent.events import RunFinishedEvent, ToolCallStartedEvent
from conduit.core.events import ToolCallEvent
from conduit.core.tools import ToolCallContext, ToolOutcome
from conduit.core.transcript import MessageEntry, TranscriptEntry
from conduit.core.types import Message, ToolResultBlock, ToolUseBlock
from conduit.providers.stub import StubToolCall, StubTurn, scripted
from conduit.stores.memory import MemoryTranscriptStore
from tests.unit.agent.helpers import (
    Executor,
    SampleTool,
    ScriptLog,
    Sink,
    result_text,
    runtime,
    task_tools,
)

STOPPED = "The tool call was not run: the run stopped."


def _two_lookups() -> ScriptLog:
    return ScriptLog(
        scripted(
            [
                StubTurn(
                    tool_calls=(
                        StubToolCall(name="lookup", input={"q": "a"}),
                        StubToolCall(name="lookup", input={"q": "b"}),
                    )
                )
            ]
        )
    )


def _task() -> AgentTask:
    return AgentTask(task_class="chat", messages=(Message.user("go"),), session_id="s-1")


async def _stored(store: MemoryTranscriptStore) -> tuple[Message, ...]:
    return (await store.load("s-1")).active_messages()


def _assert_every_use_answered(messages: Sequence[Message]) -> list[ToolResultBlock]:
    uses = [b for b in messages[-2].content if isinstance(b, ToolUseBlock)]
    assert messages[-2].role == "assistant" and uses
    assert messages[-1].role == "user"
    results = [b for b in messages[-1].content if isinstance(b, ToolResultBlock)]
    assert [r.tool_use_id for r in results] == [u.id for u in uses]
    return results


class SlowSecond:
    """Answers the first call at once and hangs on the second."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.calls = 0

    async def __call__(
        self, tool: Any, arguments: dict[str, Any], context: ToolCallContext
    ) -> ToolOutcome:
        self.calls += 1
        if self.calls == 1:
            return ToolOutcome(ok=True, content="first done")
        self.started.set()
        await asyncio.sleep(30)
        raise AssertionError("not reached")


async def test_a_stream_abandoned_at_a_tool_call_answers_every_open_use() -> None:
    store = MemoryTranscriptStore()
    sink = Sink()
    rt = await runtime(_two_lookups())
    executor = Executor()
    tools = {"chat": task_tools([SampleTool("lookup")], executor)}
    run = AgentRunner(rt, tools=tools, transcripts=store, events=sink).start(_task())
    stream = run.events()
    async for event in stream:
        if isinstance(event, ToolCallStartedEvent):
            break
    await stream.aclose()
    assert executor.calls == []
    results = _assert_every_use_answered(await _stored(store))
    assert [r.is_error for r in results] == [True, True]
    assert [result_text(r) for r in results] == [STOPPED, STOPPED]
    assert [e.error_type for e in sink.of(RunFinishedEvent)] == ["abandoned"]


async def test_a_run_cancelled_in_a_tool_keeps_finished_results_and_stops_the_rest() -> None:
    store = MemoryTranscriptStore()
    rt = await runtime(_two_lookups())
    executor = SlowSecond()
    tools = {"chat": task_tools([SampleTool("lookup")], executor)}
    task = asyncio.create_task(AgentRunner(rt, tools=tools, transcripts=store).run(_task()))
    await asyncio.wait_for(executor.started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    results = _assert_every_use_answered(await _stored(store))
    assert [r.is_error for r in results] == [False, True]
    assert "first done" in result_text(results[0])
    assert result_text(results[1]) == STOPPED


async def test_a_stream_abandoned_after_a_finished_tool_keeps_its_result() -> None:
    store = MemoryTranscriptStore()
    rt = await runtime(_two_lookups())
    tools = {"chat": task_tools([SampleTool("lookup")], Executor(answers={"lookup": "got it"}))}
    run = AgentRunner(rt, tools=tools, transcripts=store).start(_task())
    stream = run.events()
    async for event in stream:
        if isinstance(event, ToolCallEvent):  # the first tool has finished
            break
    await stream.aclose()
    results = _assert_every_use_answered(await _stored(store))
    assert [r.is_error for r in results] == [False, True]
    assert "got it" in result_text(results[0])


async def test_an_abandoned_run_with_no_open_tool_use_adds_nothing() -> None:
    store = MemoryTranscriptStore()
    rt = await runtime(ScriptLog(scripted([StubTurn(text="a long answer to abandon")])))
    run = AgentRunner(rt, transcripts=store).start(_task())
    stream = run.events()
    await anext(stream)
    await anext(stream)
    await stream.aclose()
    roles = [
        e.message.role for e in (await store.load("s-1")).entries if isinstance(e, MessageEntry)
    ]
    assert roles[-1] in ("user", "assistant")
    assert not any(isinstance(b, ToolResultBlock) for m in await _stored(store) for b in m.content)


class _FailsOnResults(MemoryTranscriptStore):
    """Stores everything but a message of tool results, where it fails."""

    async def append(
        self, session_id: str, entries: Sequence[TranscriptEntry]
    ) -> tuple[TranscriptEntry, ...]:
        for entry in entries:
            if isinstance(entry, MessageEntry) and any(
                isinstance(b, ToolResultBlock) for b in entry.message.content
            ):
                raise OSError("zz store detail 5512 zz")
        return await super().append(session_id, entries)


async def test_a_failing_store_never_replaces_the_cancellation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rt = await runtime(_two_lookups())
    executor = SlowSecond()
    tools = {"chat": task_tools([SampleTool("lookup")], executor)}
    task = asyncio.create_task(
        AgentRunner(rt, tools=tools, transcripts=_FailsOnResults()).run(_task())
    )
    await asyncio.wait_for(executor.started.wait(), 5)
    with caplog.at_level(logging.WARNING, logger="conduit.agent"):
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
    assert caught.value.__context__ is None or isinstance(
        caught.value.__context__, asyncio.CancelledError
    )
    assert "OSError" in caplog.text
    assert "zz store detail" not in caplog.text


async def test_a_failing_store_never_replaces_an_abandonment() -> None:
    sink = Sink()
    rt = await runtime(_two_lookups())
    tools = {"chat": task_tools([SampleTool("lookup")], Executor())}
    run = AgentRunner(rt, tools=tools, transcripts=_FailsOnResults(), events=sink).start(_task())
    stream = run.events()
    async for event in stream:
        if isinstance(event, ToolCallStartedEvent):
            break
    await stream.aclose()  # raises nothing
    assert [e.error_type for e in sink.of(RunFinishedEvent)] == ["abandoned"]


async def test_a_run_ending_on_an_unexpected_error_answers_every_open_use() -> None:
    def broken(tool: Any) -> bool:
        raise KeyError("zz")

    store = MemoryTranscriptStore()
    rt = await runtime(_two_lookups())
    tools = {"chat": task_tools([SampleTool("lookup")], Executor())}
    runner = AgentRunner(rt, tools=tools, transcripts=store, requires_approval=broken)
    with pytest.raises(KeyError):
        await runner.run(_task())
    results = _assert_every_use_answered(await _stored(store))
    assert [result_text(r) for r in results] == [STOPPED, STOPPED]
