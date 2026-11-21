"""Run events: typed, metadata only, with content only as the capture policy allows.

A marker is planted everywhere content can be (the user message, the system prompt, tool
arguments, tool results, model output and opaque provider state). With capture off it
never reaches the stream or the sink; with capture on it does, so the check is not vacuous.
"""

from __future__ import annotations

import base64

from conduit.agent import AgentRunner, AgentTask
from conduit.agent import events as agent_events
from conduit.agent.events import (
    RunFinishedEvent,
    RunStartedEvent,
    TextDeltaEvent,
    ToolCallStartedEvent,
)
from conduit.core.events import CapturePolicy, ContentKind, Event, ToolCallEvent
from conduit.core.types import Message, ProviderStateBlock, TextBlock
from conduit.providers.stub import StubToolCall, StubTurn, answer_tool_results
from tests.tripwires.test_event_content import _event_classes, _model_problems
from tests.unit.agent.helpers import Executor, SampleTool, ScriptLog, Sink, runtime, task_tools

MARKER = "zz synthetic content marker 4417 zz"
STATE = b"zz synthetic provider state 5521 zz"
STATE_SEEN = base64.b64encode(STATE).decode("ascii")


def _leaks(event: Event) -> bool:
    text = event.model_dump_json() + repr(event)
    return MARKER in text or STATE_SEEN in text or STATE.decode("ascii") in text


async def _events(
    capture: CapturePolicy, *, output_filter: object = None
) -> tuple[list[Event], Sink]:
    state = ProviderStateBlock(provider_kind="stub", state_type="thinking", payload=STATE)
    script = answer_tool_results(
        StubTurn(
            text=f"Thinking: {MARKER}",
            tool_calls=(StubToolCall(name="lookup", input={"q": MARKER}),),
            state=(state,),
        ),
        lambda results: StubTurn(text=f"Answer: {MARKER}"),
    )
    sink = Sink()
    rt = await runtime(ScriptLog(script))
    tools = {"chat": task_tools([SampleTool("lookup")], Executor(answers={"lookup": MARKER}))}
    runner = AgentRunner(
        rt,
        tools=tools,
        events=sink,
        capture=capture,
        output_filter=output_filter,  # type: ignore[arg-type]
    )
    task = AgentTask(
        task_class="chat",
        system=(TextBlock(text=f"System: {MARKER}"),),
        messages=(Message.user(f"User: {MARKER}"),),
    )
    streamed: list[Event] = [e async for e in runner.start(task)]
    return streamed, sink


def test_every_agent_event_class_is_seen_by_the_tripwire_and_holds_only_metadata() -> None:
    classes = _event_classes()
    ours = {
        cls
        for cls in vars(agent_events).values()
        if isinstance(cls, type)
        and issubclass(cls, Event)
        and cls is not Event
        and cls.__module__ == agent_events.__name__
    }
    assert len(ours) >= 9
    assert ours <= classes
    problems = [p for cls in ours for p in _model_problems(cls, cls.__name__)]
    assert not problems, problems


async def test_no_content_reaches_the_stream_or_the_sink_with_capture_off() -> None:
    streamed, sink = await _events(CapturePolicy())
    assert streamed and sink.events
    leaks = [e.type for e in [*streamed, *sink.events] if _leaks(e)]
    assert not leaks, leaks


async def test_content_reaches_both_when_the_product_opts_in() -> None:
    streamed, sink = await _events(CapturePolicy(prompts=True, tool_content=True))
    text = "".join(e.text.text or "" for e in streamed if isinstance(e, TextDeltaEvent))
    assert MARKER in text
    started = next(e for e in streamed if isinstance(e, ToolCallStartedEvent))
    assert started.arguments.text is not None and MARKER in started.arguments.text
    finished = next(e for e in streamed if isinstance(e, ToolCallEvent))
    assert finished.result is not None and finished.result.text == MARKER
    run_finished = streamed[-1]
    assert isinstance(run_finished, RunFinishedEvent)
    assert run_finished.output is not None and run_finished.output.text == f"Answer: {MARKER}"
    assert any(_leaks(e) for e in sink.events)


async def test_tool_content_stays_out_when_only_prompts_are_captured() -> None:
    streamed, sink = await _events(CapturePolicy(prompts=True))
    for event in [*streamed, *sink.events]:
        if isinstance(event, ToolCallStartedEvent | ToolCallEvent):
            assert not _leaks(event), event.type
    assert any(isinstance(e, TextDeltaEvent) and e.text.text for e in streamed)


async def test_prompt_content_stays_out_when_only_tool_content_is_captured() -> None:
    streamed, _ = await _events(CapturePolicy(tool_content=True))
    for event in streamed:
        if isinstance(event, TextDeltaEvent | RunFinishedEvent | RunStartedEvent):
            assert not _leaks(event), event.type


async def test_text_deltas_are_never_sent_to_the_sink() -> None:
    streamed, sink = await _events(CapturePolicy())
    assert any(isinstance(e, TextDeltaEvent) for e in streamed)
    assert not sink.of(TextDeltaEvent)
    assert [type(e) for e in streamed if not isinstance(e, TextDeltaEvent)] == [
        type(e) for e in sink.events if e.run_id == streamed[0].run_id
    ]


async def test_with_an_output_filter_deltas_withhold_their_text() -> None:
    async def mask(text: str) -> str:
        return text.replace(MARKER, "[masked]")

    streamed, _ = await _events(CapturePolicy(prompts=True, tool_content=True), output_filter=mask)
    deltas = [e for e in streamed if isinstance(e, TextDeltaEvent)]
    assert deltas
    assert all(d.text.text is None for d in deltas)
    assert all(d.text.size > 0 for d in deltas)
    finished = streamed[-1]
    assert isinstance(finished, RunFinishedEvent)
    assert finished.filtered
    assert finished.output is not None and finished.output.text == "Answer: [masked]"


async def test_every_run_event_carries_the_run_and_task_class() -> None:
    streamed, _ = await _events(CapturePolicy())
    run_id = streamed[0].run_id
    assert run_id is not None
    for event in streamed:
        assert event.run_id == run_id
        assert event.task_class == "chat"
        assert event.correlation_id == run_id  # defaults to the run id


async def test_the_finished_event_carries_the_final_usage_and_counts() -> None:
    streamed, _ = await _events(CapturePolicy())
    finished = streamed[-1]
    assert isinstance(finished, RunFinishedEvent)
    assert finished.outcome == "completed"
    assert finished.model_calls == 2
    assert finished.tool_calls == 1
    assert finished.usage.output_tokens > 0
    assert finished.output is not None
    assert finished.output.kind is ContentKind.PROMPT
    assert finished.output.text is None


async def test_the_started_event_describes_the_run_without_content() -> None:
    streamed, _ = await _events(CapturePolicy())
    started = streamed[0]
    assert isinstance(started, RunStartedEvent)
    assert (started.provider, started.kind, started.model) == ("primary", "stub", "stub-echo")
    assert started.message_count == 1
    assert started.tool_count == 1
