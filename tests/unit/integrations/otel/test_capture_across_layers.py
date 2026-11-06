"""The content-capture policy across every layer, through every sink and exporter.

One run uses all of it: a system prompt and user message, model output, a product tool
behind an approval, an MCP tool (an in-memory MCP server), a skill loaded with
``use_skill``, and a compaction that replaces the conversation. A marker is planted in
every piece of content. All of it goes through one ``FanOutSink`` under the deployment's
policy, to a memory sink and the OpenTelemetry sink with in-memory exporters.

- Capture off: the marker reaches no sink and no exporter.
- Prompts only: no tool content (tool arguments and results, MCP arguments and results)
  reaches a sink; prompt content does.
- Both on: every kind of content reaches the memory sink (so the check is not vacuous),
  and still none reaches the OpenTelemetry exporters, which export metadata only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pytest

pytest.importorskip("mcp")  # the MCP layer is the mcp extra's

from conduit.agent import AgentRunner, AgentTask, CompactionRequest
from conduit.agent.events import (
    ConversationCompactedEvent,
    RunFinishedEvent,
    ToolCallStartedEvent,
)
from conduit.core.events import (
    CapturePolicy,
    Event,
    EventSink,
    FanOutSink,
    MemoryEventSink,
    ModelCallFinishedEvent,
    ModelCallStartedEvent,
    ToolCallEvent,
)
from conduit.core.runtime import start
from conduit.core.secrets import LiteralSecretResolver
from conduit.core.tools import TaskTools, ToolCatalogue
from conduit.core.types import Message, ModelRequest, TextBlock
from conduit.integrations.otel import OtelEventSink
from conduit.mcp.bridge import McpToolBridge
from conduit.mcp.events import McpCallEvent
from conduit.providers.stub import StubFactory, StubToolCall, StubTurn
from conduit.skills.disclosure import SkillSession
from conduit.skills.events import SkillLoadedEvent
from conduit.skills.memory import MemorySkillSource
from conduit.skills.tool import with_use_skill
from tests.unit.agent.helpers import CHAT, REVIEW, Approvals, Executor, SampleTool, settings
from tests.unit.integrations.otel.conftest import Telemetry
from tests.unit.integrations.otel.test_sink import every_exported_text
from tests.unit.mcp.test_bridge import connection
from tests.unit.skills.helpers import skill

MARKER = "zz synthetic cross-layer marker 7302 zz"


def _script(request: ModelRequest, index: int) -> StubTurn:
    if index == 0:
        return StubTurn(
            text=f"Thinking: {MARKER}",
            tool_calls=(
                StubToolCall(name="lookup", input={"q": MARKER}),
                StubToolCall(name="docs__echo", input={"text": MARKER}),
                StubToolCall(name="use_skill", input={"name": "triage"}),
            ),
        )
    return StubTurn(text=f"Answer: {MARKER}")


class Compactor:
    async def before_model_call(self, request: CompactionRequest) -> Sequence[Message] | None:
        if request.model_calls == 1:
            return (Message.user(f"Summary so far: {MARKER}"),)
        return None


@dataclass
class Observed:
    events: list[Event]
    exported: str
    span_count: int


async def _observe(
    capture: CapturePolicy, telemetry: Telemetry, also: Sequence[EventSink] = ()
) -> Observed:
    memory = MemoryEventSink()
    otel = OtelEventSink(
        tracer_provider=telemetry.tracer_provider, meter_provider=telemetry.meter_provider
    )
    bus = FanOutSink([memory, otel, *also], policy=capture)
    # The deployment's policy also governs the registry's model-call events.
    rt = await start(
        settings().model_copy(update={"capture": capture}),
        task_classes=[CHAT, REVIEW],
        factories=[StubFactory(scripts={"primary": _script, "specialist": _script})],
        secrets=LiteralSecretResolver({}),
        events=bus,
    )
    async with connection() as conn:
        bridge = McpToolBridge([conn], events=bus, capture=capture)
        session = SkillSession(
            MemorySkillSource([skill("triage", f"Steps: {MARKER}\n")]), events=bus
        )
        catalogue = ToolCatalogue.from_tools(
            [
                SampleTool("lookup", requires_approval=True),
                *[t for t in await bridge.tools() if t.name == "docs__echo"],
                session.tool(task_classes={"chat"}),
            ]
        )
        executor = bridge.executor(Executor(answers={"lookup": f"Found: {MARKER}"}))
        runner = AgentRunner(
            rt,
            tools={"chat": TaskTools(catalogue.registry_for("chat"), executor)},
            approvals=Approvals(approve=True),
            events=bus,
            capture=capture,
            compaction=Compactor(),
        )
        task = AgentTask(
            task_class="chat",
            system=(TextBlock(text=f"System: {MARKER}"),),
            messages=(Message.user(f"User: {MARKER}"),),
            system_hook=session.system,
            tool_wrapper=with_use_skill,
        )
        async for _ in runner.start(task):
            pass
    otel.close()
    return Observed(memory.events, every_exported_text(telemetry), len(telemetry.spans()))


def _one[E: Event](events: list[Event], kind: type[E], **match: object) -> E:
    found = [
        e
        for e in events
        if isinstance(e, kind) and all(getattr(e, k) == v for k, v in match.items())
    ]
    assert len(found) == 1, (kind.__name__, [type(e).__name__ for e in events])
    return found[0]


def _every_layer_is_present(events: list[Event]) -> None:
    types = {e.type for e in events}
    assert {
        "run_started",
        "model_call_started",
        "model_call_finished",
        "tool_call_started",
        "approval_requested",
        "approval_decided",
        "mcp_call",
        "skill_loaded",
        "tool_call",
        "conversation_compacted",
        "run_finished",
    } <= types, sorted(types)


async def test_with_capture_off_no_content_reaches_any_sink_or_exporter(
    telemetry: Telemetry,
) -> None:
    observed = await _observe(CapturePolicy(), telemetry)
    _every_layer_is_present(observed.events)
    leaks = [e.type for e in observed.events if MARKER in e.model_dump_json() + repr(e)]
    assert not leaks, leaks
    assert observed.span_count >= 9
    assert MARKER not in observed.exported


async def test_with_prompts_only_tool_content_stays_out(telemetry: Telemetry) -> None:
    observed = await _observe(CapturePolicy(prompts=True), telemetry)
    events = observed.events
    _every_layer_is_present(events)
    tool_bearing = [
        e for e in events if isinstance(e, ToolCallStartedEvent | ToolCallEvent | McpCallEvent)
    ]
    assert len(tool_bearing) >= 7  # three started, three finished, one MCP call
    leaks = [e.type for e in tool_bearing if MARKER in e.model_dump_json()]
    assert not leaks, leaks
    first = next(e for e in events if isinstance(e, ModelCallStartedEvent))
    assert first.request.text is not None and f"System: {MARKER}" in first.request.text
    assert '"q":"' + MARKER not in first.request.text
    finished = _one(events, RunFinishedEvent)
    assert finished.output is not None and finished.output.text == f"Answer: {MARKER}"
    assert MARKER not in observed.exported


async def test_with_both_on_every_kind_of_content_arrives_but_never_in_telemetry(
    telemetry: Telemetry,
) -> None:
    observed = await _observe(CapturePolicy(prompts=True, tool_content=True), telemetry)
    events = observed.events
    _every_layer_is_present(events)
    started = [e for e in events if isinstance(e, ModelCallStartedEvent)]
    assert all(e.request.text and MARKER in e.request.text for e in started)
    finished_calls = [e for e in events if isinstance(e, ModelCallFinishedEvent)]
    assert any(e.response and e.response.text and MARKER in e.response.text for e in finished_calls)
    lookup = _one(events, ToolCallEvent, tool="lookup")
    assert lookup.arguments.text is not None and MARKER in lookup.arguments.text
    assert lookup.result is not None and lookup.result.text is not None
    assert MARKER in lookup.result.text
    lookup_started = _one(events, ToolCallStartedEvent, tool="lookup")
    assert lookup_started.arguments.text is not None and MARKER in lookup_started.arguments.text
    mcp = _one(events, McpCallEvent)
    assert mcp.arguments.text is not None and MARKER in mcp.arguments.text
    assert mcp.result is not None and mcp.result.text is not None
    assert MARKER in mcp.result.text
    run = _one(events, RunFinishedEvent)
    assert run.output is not None and run.output.text == f"Answer: {MARKER}"
    # Events that never carry content still never do.
    for kind in (SkillLoadedEvent, ConversationCompactedEvent):
        assert MARKER not in _one(events, kind).model_dump_json()
    # The OpenTelemetry exporters hold metadata only, whatever the policy allowed.
    assert observed.span_count >= 9
    assert MARKER not in observed.exported
