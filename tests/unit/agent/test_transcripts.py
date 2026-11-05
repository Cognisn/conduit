"""Run transcripts: every message a run adds is appended through the supplied store."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from conduit.agent import AgentRunner, AgentTask, RunStatus
from conduit.core.transcript import MessageEntry, TranscriptEntry
from conduit.core.types import Message, ProviderStateBlock, ToolResultBlock
from conduit.providers.stub import StubToolCall, StubTurn, answer_tool_results
from conduit.stores.memory import MemoryTranscriptStore
from tests.unit.agent.helpers import Executor, SampleTool, ScriptLog, runtime, task_tools


def _script() -> ScriptLog:
    state = ProviderStateBlock(provider_kind="stub", state_type="thinking", payload=b"sig")
    return ScriptLog(
        answer_tool_results(
            StubTurn(tool_calls=(StubToolCall(name="lookup"),), state=(state,)),
            lambda results: StubTurn(text="done"),
        )
    )


def _task(session_id: str | None = "session-1", **fields: object) -> AgentTask:
    return AgentTask(
        task_class="chat",
        messages=(Message.user("look it up"),),
        session_id=session_id,
        **fields,  # type: ignore[arg-type]
    )


async def test_the_input_and_every_added_message_are_appended_in_order() -> None:
    store = MemoryTranscriptStore()
    rt = await runtime(_script())
    tools = {"chat": task_tools([SampleTool("lookup")], Executor())}
    result = await AgentRunner(rt, tools=tools, transcripts=store).run(_task())
    transcript = await store.load("session-1")
    entries = [e for e in transcript.entries if isinstance(e, MessageEntry)]
    assert [e.message for e in entries] == list(result.messages)
    assert [e.message.role for e in entries] == ["user", "assistant", "user", "assistant"]
    assert [e.seq for e in entries] == [1, 2, 3, 4]
    # Assistant entries name the model and carry the call's usage.
    assert entries[1].model == "stub:stub-echo"
    assert entries[1].usage is not None and entries[1].usage.output_tokens >= 0
    assert entries[0].model is None and entries[0].usage is None
    # Provider state survives in the stored conversation, unchanged.
    assert entries[1].message.provider_state("stub")[0].payload == b"sig"
    assert transcript.active_messages() == result.messages


async def test_history_is_not_appended_again() -> None:
    store = MemoryTranscriptStore()
    rt = await runtime(_script())
    tools = {"chat": task_tools([SampleTool("lookup")], Executor())}
    history = (Message.user("before"), Message.assistant("earlier"))
    result = await AgentRunner(rt, tools=tools, transcripts=store).run(_task(history=history))
    transcript = await store.load("session-1")
    assert len(transcript.entries) == len(result.messages) - 2
    assert transcript.entries[0].message == Message.user("look it up")  # type: ignore[union-attr]


async def test_without_a_session_id_nothing_is_appended() -> None:
    store = MemoryTranscriptStore()
    rt = await runtime(_script())
    tools = {"chat": task_tools([SampleTool("lookup")], Executor())}
    result = await AgentRunner(rt, tools=tools, transcripts=store).run(_task(None))
    assert result.status is RunStatus.COMPLETED
    assert await store.list_sessions() == []


async def test_a_failed_run_still_leaves_a_well_formed_transcript() -> None:
    store = MemoryTranscriptStore()
    rt = await runtime(_script())
    executor = Executor(fail_with=RuntimeError("boom"))
    tools = {"chat": task_tools([SampleTool("lookup")], executor)}
    await AgentRunner(rt, tools=tools, transcripts=store).run(_task())
    transcript = await store.load("session-1")
    last = transcript.entries[-1]
    assert isinstance(last, MessageEntry)
    block = last.message.content[0]
    assert isinstance(block, ToolResultBlock) and block.is_error


class _Failing(MemoryTranscriptStore):
    async def append(
        self, session_id: str, entries: Sequence[TranscriptEntry]
    ) -> tuple[TranscriptEntry, ...]:
        raise OSError("disk full")


async def test_a_store_that_fails_fails_the_run_with_its_own_error() -> None:
    rt = await runtime(_script())
    with pytest.raises(OSError, match="disk full"):
        await AgentRunner(rt, transcripts=_Failing()).run(_task())
