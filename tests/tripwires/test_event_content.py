"""Tripwire: no event carries content unless the content-capture policy allows it.

Two halves, never skipped or weakened:

- Structure: every event class anywhere in conduit holds only metadata. Strings must
  be ``Identifier`` (no whitespace, bounded) or a literal; content may appear only as
  ``CapturedContent``, whose text the policy controls. No ``dict``, ``Any``, ``bytes``
  or plain ``str`` field.
- Behaviour: a marker planted in every place content can be (system prompt, messages,
  tool results, tool arguments, model output, opaque provider state sent and received)
  never reaches a sink with capture off, and does reach it with capture on (so the
  check is not vacuous). Tool content an emitter mislabels as prompt content stays out
  when only prompts are captured. Provider state is prompt content: it is captured with
  prompts, and never under the tool-content setting alone.
"""

from __future__ import annotations

import base64
import enum
import importlib
import pkgutil
import types
from datetime import datetime
from typing import Annotated, Any, Literal, Union, get_args, get_origin

import pytest
from pydantic import BaseModel
from pydantic.fields import FieldInfo

import conduit
from conduit.core.events import (
    CapturedContent,
    CapturePolicy,
    ContentKind,
    Event,
    FanOutSink,
    MemoryEventSink,
    ModelCallFinishedEvent,
    ModelCallStartedEvent,
    ToolCallEvent,
)
from conduit.core.identifiers import IDENTIFIER_MARK
from conduit.core.registry import ProviderRegistry
from conduit.core.secrets import LiteralSecretResolver
from conduit.core.streaming import ProviderStateReported
from conduit.core.types import (
    Message,
    ModelRequest,
    ProviderStateBlock,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)
from conduit.providers.stub import StubFactory, StubToolCall, StubTurn, scripted
from tests.tripwires.importscan import OPTIONAL_LIBRARIES

MARKER = "zz synthetic content marker 4417 zz"
#: Provider state is opaque bytes, carried as base64 wherever it is serialised.
STATE_MARKER = b"zz synthetic provider state 5521 zz"
STATE_SEEN = base64.b64encode(STATE_MARKER).decode("ascii")


def _leaks(text: str) -> bool:
    return MARKER in text or STATE_SEEN in text or STATE_MARKER.decode("ascii") in text


def _state() -> ProviderStateBlock:
    return ProviderStateBlock(provider_kind="stub", state_type="thinking", payload=STATE_MARKER)


_SCALARS = (int, float, bool, datetime, type(None))
_OPTIONAL_ROOTS = {name.split(".")[0] for name in OPTIONAL_LIBRARIES}


def _import_everything() -> list[str]:
    imported = []
    for info in pkgutil.walk_packages(conduit.__path__, prefix="conduit."):
        try:
            importlib.import_module(info.name)
        except ImportError as error:
            # Only a missing optional extra may excuse a module; anything else fails.
            if (error.name or "").split(".")[0] not in _OPTIONAL_ROOTS:
                raise
            continue
        imported.append(info.name)
    return imported


def _event_classes() -> set[type[Event]]:
    _import_everything()
    found: set[type[Event]] = set()
    pending: list[type[Event]] = [Event]
    while pending:
        cls = pending.pop()
        for sub in cls.__subclasses__():
            if sub not in found:
                found.add(sub)
                pending.append(sub)
    return found | {Event}


def _problems(annotation: Any, metadata: list[Any], where: str) -> list[str]:
    origin = get_origin(annotation)
    if origin is Annotated:
        inner, *extra = get_args(annotation)
        return _problems(inner, [*metadata, *extra], where)
    if origin in (Union, types.UnionType):
        return [p for arg in get_args(annotation) for p in _problems(arg, metadata, where)]
    if origin is Literal:
        return []
    if origin in (tuple, frozenset, list):
        return [
            p
            for arg in get_args(annotation)
            if arg is not Ellipsis
            for p in _problems(arg, [], where)
        ]
    if annotation is str:
        return [] if IDENTIFIER_MARK in metadata else [f"{where}: plain str"]
    if annotation is CapturedContent:
        return []
    if annotation in _SCALARS:
        return []
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return []
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _model_problems(annotation, where)
    return [f"{where}: {annotation!r} is not a metadata type"]


def _model_problems(model: type[BaseModel], where: str) -> list[str]:
    problems: list[str] = []
    for name, info in model.model_fields.items():
        assert isinstance(info, FieldInfo)
        problems += _problems(info.annotation, list(info.metadata), f"{where}.{name}")
    return problems


class TestStructure:
    def test_the_scan_sees_the_core_events(self) -> None:
        classes = _event_classes()
        assert {ModelCallStartedEvent, ModelCallFinishedEvent, ToolCallEvent} <= classes

    def test_every_event_field_is_metadata_or_captured_content(self) -> None:
        problems = [p for cls in _event_classes() for p in _model_problems(cls, cls.__name__)]
        assert not problems, "\n".join(problems)

    @pytest.mark.parametrize(
        "annotation",
        [str, dict[str, Any], Any, bytes, list[str], ProviderStateBlock, ProviderStateReported],
    )
    def test_the_check_refuses_content_bearing_types(self, annotation: Any) -> None:
        assert _problems(annotation, [], "synthetic")


def _request() -> ModelRequest:
    return ModelRequest(
        model="stub-echo",
        system=(TextBlock(text=f"System: {MARKER}"),),
        messages=(
            Message.user(f"User: {MARKER}"),
            Message(
                role="assistant",
                content=(
                    _state(),
                    ToolUseBlock(id="toolu_1", name="lookup", input={"q": MARKER}),
                ),
            ),
            Message(
                role="user",
                content=(
                    ToolResultBlock(tool_use_id="toolu_1", content=(TextBlock(text=MARKER),)),
                ),
            ),
        ),
        tools=(ToolSpec(name="lookup", description=MARKER, input_schema={"type": "object"}),),
        max_output_tokens=256,
    )


async def _events(capture: CapturePolicy) -> list[Event]:
    sink = MemoryEventSink()
    script = scripted(
        [
            StubTurn(
                text=f"Answer: {MARKER}",
                tool_calls=(StubToolCall(name="lookup", input={"q": MARKER}),),
                state=(_state(),),
            )
        ]
    )
    registry = ProviderRegistry(
        secrets=LiteralSecretResolver({}),
        events=sink,
        capture=capture,
        factories=[StubFactory(scripts={"marker": script})],
    )
    provider = await registry.build(
        "test", {"kind": "stub", "script": "marker", "not_for_production": True}
    )
    await provider.complete(_request())
    # An upper layer's tool event, emitted with a looser policy than the deployment's.
    loose = CapturePolicy(prompts=True, tool_content=True)
    bus = FanOutSink([sink], policy=capture)
    await bus.emit(
        ToolCallEvent(
            call_id="toolu_1",
            tool="lookup",
            outcome="ok",
            arguments=loose.capture(ContentKind.TOOL, MARKER),
            result=loose.capture(ContentKind.TOOL, MARKER),
            duration_ms=1.0,
        )
    )
    # The same tool content, mislabelled by its emitter as prompt content.
    await bus.emit(
        ToolCallEvent(
            call_id="toolu_2",
            tool="lookup",
            outcome="ok",
            arguments=loose.capture(ContentKind.PROMPT, MARKER),
            result=loose.capture(ContentKind.PROMPT, MARKER),
            duration_ms=1.0,
        )
    )
    return sink.events


class TestBehaviour:
    async def test_no_content_reaches_a_sink_with_capture_off(self) -> None:
        events = await _events(CapturePolicy())
        assert len(events) == 4
        leaks = [e.type for e in events if _leaks(e.model_dump_json()) or _leaks(repr(e))]
        assert not leaks, leaks

    async def test_tool_content_stays_out_when_only_prompts_are_captured(self) -> None:
        events = await _events(CapturePolicy(prompts=True))
        for tool in events[-2:]:  # labelled as tool content, then mislabelled as prompt
            assert isinstance(tool, ToolCallEvent)
            assert MARKER not in tool.model_dump_json()
        started = events[0]
        assert isinstance(started, ModelCallStartedEvent)
        assert started.request.text is not None
        assert '"q":"' + MARKER not in started.request.text  # tool arguments omitted
        assert MARKER in started.request.text  # the prompt itself is captured
        assert STATE_SEEN in started.request.text  # provider state is prompt content

    async def test_provider_state_stays_out_when_only_tool_content_is_captured(self) -> None:
        events = await _events(CapturePolicy(tool_content=True))
        started = events[0]
        assert isinstance(started, ModelCallStartedEvent)
        assert STATE_SEEN not in started.model_dump_json()
        assert all(not _leaks(e.model_dump_json()) for e in events[:2])

    async def test_content_reaches_a_sink_when_the_product_opts_in(self) -> None:
        events = await _events(CapturePolicy(prompts=True, tool_content=True))
        assert all(MARKER in e.model_dump_json() for e in events)
