"""The runner's sink and capture policy default to the runtime's (Q0027), and its
untrusted framing is the core's."""

from __future__ import annotations

from collections.abc import Mapping

import conduit.agent
import conduit.agent.untrusted
import conduit.core.untrusted
from conduit.agent import AgentRunner, AgentTask
from conduit.agent.events import RunFinishedEvent, RunStartedEvent
from conduit.core.events import CapturePolicy
from conduit.core.runtime import Runtime, TaskBinding, start
from conduit.core.secrets import LiteralSecretResolver
from conduit.core.tasks import ResolvedAssignment
from conduit.core.types import Message
from conduit.providers.stub import StubFactory, StubTurn
from tests.unit.agent.helpers import CHAT, REVIEW, ScriptLog, Sink, settings


def _answer() -> ScriptLog:
    return ScriptLog(lambda request, index: StubTurn(text="the answer"))


async def _runtime(sink: Sink, capture: CapturePolicy | None = None) -> Runtime:
    configured = settings()
    if capture is not None:
        configured = configured.model_copy(update={"capture": capture})
    script = _answer()
    return await start(
        configured,
        task_classes=[CHAT, REVIEW],
        factories=[StubFactory(scripts={"primary": script, "specialist": script})],
        secrets=LiteralSecretResolver({}),
        events=sink,
    )


def _task() -> AgentTask:
    return AgentTask(task_class="chat", messages=(Message.user("go"),))


async def test_without_a_sink_run_events_go_to_the_runtime_s() -> None:
    sink = Sink()
    rt = await _runtime(sink)
    await AgentRunner(rt).run(_task())
    assert len(sink.of(RunStartedEvent)) == 1
    assert len(sink.of(RunFinishedEvent)) == 1


async def test_an_explicit_sink_overrides_the_runtime_s() -> None:
    runtime_sink, own = Sink(), Sink()
    rt = await _runtime(runtime_sink)
    await AgentRunner(rt, events=own).run(_task())
    assert len(own.of(RunFinishedEvent)) == 1
    assert runtime_sink.of(RunFinishedEvent) == []


async def test_without_a_policy_the_runtime_s_capture_applies() -> None:
    sink = Sink()
    rt = await _runtime(sink, CapturePolicy(prompts=True))
    await AgentRunner(rt, events=sink).run(_task())
    [finished] = sink.of(RunFinishedEvent)
    assert finished.output is not None and finished.output.text == "the answer"


async def test_an_explicit_policy_overrides_the_runtime_s() -> None:
    sink = Sink()
    rt = await _runtime(sink, CapturePolicy(prompts=True))
    await AgentRunner(rt, events=sink, capture=CapturePolicy()).run(_task())
    [finished] = sink.of(RunFinishedEvent)
    assert finished.output is not None and finished.output.text is None


async def test_the_runtime_s_policy_still_holds_events_it_delivers() -> None:
    # A looser runner policy cannot take content past the runtime's own sink.
    sink = Sink()
    rt = await _runtime(sink)
    await AgentRunner(rt, capture=CapturePolicy(prompts=True)).run(_task())
    [finished] = sink.of(RunFinishedEvent)
    assert finished.output is not None and finished.output.text is None


async def test_bindings_without_a_runtime_s_sink_still_run() -> None:
    class Bare:
        def __init__(self, inner: Runtime) -> None:
            self._inner = inner

        @property
        def assignments(self) -> Mapping[str, ResolvedAssignment]:
            return self._inner.assignments

        def for_task(self, task_class: str) -> TaskBinding:
            return self._inner.for_task(task_class)

    sink = Sink()
    rt = await _runtime(sink)
    result = await AgentRunner(Bare(rt)).run(_task())
    assert result.text == "the answer"
    assert sink.of(RunFinishedEvent) == []


def test_the_runner_s_framing_is_the_core_s() -> None:
    for name in (
        "FRAMING_NOTE",
        "UntrustedBlock",
        "UntrustedText",
        "frame_untrusted",
        "sanitise_untrusted",
        "untrusted_text_block",
    ):
        assert getattr(conduit.agent, name) is getattr(conduit.core.untrusted, name), name
        assert getattr(conduit.agent.untrusted, name) is getattr(conduit.core.untrusted, name)
