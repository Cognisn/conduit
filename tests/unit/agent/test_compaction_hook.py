"""The compaction hook: called before every model call, it may replace the conversation."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from conduit.agent import AgentRunner, AgentTask, CompactionRefused, CompactionRequest
from conduit.agent.events import ConversationCompactedEvent, RunFinishedEvent
from conduit.core.types import Message
from conduit.providers.stub import StubToolCall, StubTurn, answer_tool_results
from tests.unit.agent.helpers import Executor, SampleTool, ScriptLog, Sink, runtime, task_tools


def _script() -> ScriptLog:
    return ScriptLog(
        answer_tool_results(
            StubTurn(tool_calls=(StubToolCall(name="lookup"),)),
            lambda results: StubTurn(text="done"),
        )
    )


class Hook:
    def __init__(self, replace_on: int | None = None) -> None:
        self.requests: list[CompactionRequest] = []
        self.replace_on = replace_on

    async def before_model_call(self, request: CompactionRequest) -> Sequence[Message] | None:
        self.requests.append(request)
        if request.model_calls == self.replace_on:
            return (Message.user("summary of everything so far"),)
        return None


def _task() -> AgentTask:
    return AgentTask(task_class="chat", messages=(Message.user("go"),), session_id="s-1")


async def test_the_hook_sees_every_model_call_before_it_is_made() -> None:
    hook = Hook()
    rt = await runtime(_script())
    tools = {"chat": task_tools([SampleTool("lookup")], Executor())}
    result = await AgentRunner(rt, tools=tools, compaction=hook).run(_task())
    assert [r.model_calls for r in hook.requests] == [0, 1]
    first, second = hook.requests
    assert first.task_class == "chat"
    assert first.run_id == result.run_id
    assert first.session_id == "s-1"
    assert first.assignment.model == "stub-echo"
    assert first.assignment.entry.context_window == 1_000_000
    assert first.messages == (Message.user("go"),)
    assert first.last_usage is None
    assert len(second.messages) == 3
    assert second.last_usage is not None
    assert [t.name for t in second.tools] == ["lookup"]


async def test_a_replacement_conversation_is_used_for_the_call() -> None:
    hook = Hook(replace_on=1)
    log = _script()
    sink = Sink()
    rt = await runtime(log)
    tools = {"chat": task_tools([SampleTool("lookup")], Executor())}
    run = AgentRunner(rt, tools=tools, compaction=hook, events=sink).start(_task())
    events = [e async for e in run]
    assert log.requests[1].messages == (Message.user("summary of everything so far"),)
    compacted = [e for e in events if isinstance(e, ConversationCompactedEvent)]
    assert [(e.messages_before, e.messages_after) for e in compacted] == [(3, 1)]
    assert sink.of(ConversationCompactedEvent) == compacted
    assert run.result.messages[0] == Message.user("summary of everything so far")


@pytest.mark.parametrize("replacement", [(), [object()], "text"])
async def test_an_unusable_replacement_is_refused(replacement: object) -> None:
    class Bad:
        async def before_model_call(self, request: CompactionRequest) -> object:
            return replacement

    sink = Sink()
    rt = await runtime(_script())
    with pytest.raises(CompactionRefused):
        await AgentRunner(rt, compaction=Bad(), events=sink).run(_task())  # type: ignore[arg-type]
    finished = sink.of(RunFinishedEvent)
    assert [(e.outcome, e.error_type) for e in finished] == [("failed", "CompactionRefused")]


async def test_a_replacement_ending_with_the_assistant_is_refused() -> None:
    class Bad:
        async def before_model_call(self, request: CompactionRequest) -> Sequence[Message]:
            return (Message.user("a"), Message.assistant("b"))

    rt = await runtime(_script())
    with pytest.raises(CompactionRefused):
        await AgentRunner(rt, compaction=Bad()).run(_task())
