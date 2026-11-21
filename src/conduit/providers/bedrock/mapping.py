"""Mapping between conduit's types and the Bedrock Converse wire format.

Requests: system blocks, messages (text, images, tool use and tool results) and tool
specifications become Converse parameters. A ``CacheBreakpoint`` becomes a
``cachePoint`` only when the catalogue says the model supports prompt caching, and at
most four are sent (Bedrock's limit for Claude models): earlier breakpoints are
dropped first, since a cache read still finds the longest prefix ending at a later
one. Bedrock processes cache points in the order tools, system, messages. Amazon Nova
takes cache points in system and messages only, so a Nova request never gets one after
its tools.

Tool use and streaming: a few models (``NO_STREAMING_TOOL_USE``, from the Bedrock user
guide's client-side tool use page) refuse tools on ConverseStream, so the provider sends
a request with tools for them through Converse instead (``streams_tool_use``). A model
is matched by its own id or by an inference profile id naming it.

Responses: ConverseStream events, or a whole Converse response, become conduit stream
events ending with one ``StreamStopped`` whose usage comes from the final metadata.
Bedrock's ``inputTokens`` counts uncached input only, so the four counts map one to
one: ``inputTokens``, ``outputTokens``, ``cacheReadInputTokens``,
``cacheWriteInputTokens``.

Reasoning: a reasoning block (``reasoningText`` with its text and signature, or
``redactedContent``), streamed in pieces or whole in a Converse response, is reported
when it is whole, in its place, as one ``ProviderStateReported``: a ``bedrock`` state
block of type ``reasoning_content`` whose payload is the block's ``reasoningContent``
value as JSON (redacted bytes as base64). It is never streamed as text. Sending back,
each ``bedrock`` state block becomes that ``reasoningContent`` block again, in its place
in the assistant message, as Claude on Bedrock needs for tool use with reasoning; other
kinds' state is left out. Bedrock state that is not a whole reasoning block is refused
as ``REJECTED`` before any call, with nothing of the payload in the error.

Amazon Nova writes its reasoning into the reply text as ``<thinking>...</thinking>``
(the Nova user guide's "Invoking a tool": chain-of-thought is the default for tool
calling, "contained in <thinking> tags", with "multiple tool calls and thinking blocks in
a response"). For a Nova model only (``splits_thinking``, matched on the base model id),
each span in a text block becomes one ``ProviderStateReported``, in its place, with the
same payload shape as a reasoning block (``reasoningText``, no signature) but its own
state type, ``thinking_text``, and the answer text carries neither the tags nor what
they hold. A span still open when its block ends (a reply cut off by its token limit) is
reasoning, not answer text. Whitespace after a closing tag is dropped (one space is kept
where answer text before the span did not end in whitespace), as is the span's own leading and
trailing whitespace, and an empty span reports nothing. Streamed text holds back only a
suffix that could begin ``<thinking>``, and releases it as text if the block ends first.
Other models' text is never parsed.

``thinking_text`` state is Nova's own text format, with no signature, so it is never sent
back to any model, as a reasoning block or as text. Native ``reasoning_content`` state
keeps the rule above for every model. A message left with no content by that
is left out, and the messages either side of it, now of the same role, are merged.
"""

from __future__ import annotations

import base64
import contextlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.streaming import (
    ProviderStateReported,
    StreamEvent,
    StreamStopped,
    TextDelta,
    ToolCallFinished,
    ToolCallStarted,
)
from conduit.core.types import (
    CacheBreakpoint,
    ImageBlock,
    Message,
    ModelRequest,
    ProviderStateBlock,
    StopReason,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)

CACHE_POINT: dict[str, Any] = {"cachePoint": {"type": "default"}}
MAX_CACHE_POINTS = 4
#: Inference profile prefixes: a profile id is one of these, a dot, and the model id.
PROFILE_PREFIXES = frozenset({"us", "eu", "apac", "au", "jp", "ca", "in", "us-gov", "global"})
#: Models that take tools on Converse but refuse them on ConverseStream ("This model
#: doesn't support tool use in streaming mode."), as the Bedrock user guide's
#: client-side tool use page lists them; a request with tools goes through Converse.
NO_STREAMING_TOOL_USE = frozenset(
    {
        "meta.llama3-1-8b-instruct-v1:0",
        "meta.llama3-1-70b-instruct-v1:0",
        "meta.llama3-3-70b-instruct-v1:0",
        "meta.llama4-scout-17b-instruct-v1:0",
        "meta.llama4-maverick-17b-instruct-v1:0",
        "mistral.mistral-large-2402-v1:0",
        "mistral.mistral-large-2407-v1:0",
        "mistral.pixtral-large-2502-v1:0",
        "writer.palmyra-x4-v1:0",
        "writer.palmyra-x5-v1:0",
    }
)
#: Model families whose cache points go in system and messages only (Amazon Nova's
#: model cards: "Fields that accept prompt cache checkpoints: system and messages").
NO_TOOL_CACHE_POINT = ("amazon.nova-",)
#: Model ids whose replies carry reasoning as ``<thinking>`` spans in the text.
THINKING_TAG_MODELS = ("amazon.nova-",)
THINKING_OPEN = "<thinking>"
THINKING_CLOSE = "</thinking>"
#: The provider kind and state type of a reasoning block's state.
KIND = "bedrock"
REASONING_STATE = "reasoning_content"
#: The state type of a ``<thinking>`` span taken from reply text: never sent back.
THINKING_STATE = "thinking_text"
_REASONING_TEXT_FIELDS = frozenset({"text", "signature"})


def _reasoning_text(value: Any) -> dict[str, str] | None:
    """A ``reasoningText`` value (text, and a signature if any), or ``None``."""
    if not isinstance(value, Mapping) or "text" not in value:
        return None
    if not set(value) <= _REASONING_TEXT_FIELDS:
        return None
    if not all(isinstance(item, str) for item in value.values()):
        return None
    return dict(value)


def reasoning_state(content: Any, state_type: str = REASONING_STATE) -> ProviderStateBlock:
    """The state block for a Converse ``reasoningContent`` value, or PROTOCOL."""
    payload: dict[str, Any] | None = None
    if isinstance(content, Mapping) and len(content) == 1:
        if "reasoningText" in content:
            text = _reasoning_text(content["reasoningText"])
            if text is not None:
                payload = {"reasoningText": text}
        elif isinstance(content.get("redactedContent"), bytes):
            encoded = base64.b64encode(content["redactedContent"]).decode("ascii")
            payload = {"redactedContent": encoded}
    if payload is None:
        raise _protocol()
    return ProviderStateBlock(
        provider_kind=KIND,
        state_type=state_type,
        payload=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
    )


def _reasoning_content(
    state: ProviderStateBlock, state_type: str = REASONING_STATE
) -> dict[str, Any]:
    """The ``reasoningContent`` block a ``bedrock`` state block holds, or REJECTED."""
    data: Any = None
    # UnicodeDecodeError and JSONDecodeError; RecursionError for deep nesting.
    with contextlib.suppress(ValueError, RecursionError):
        data = json.loads(state.payload.decode("utf-8"))
    wire: dict[str, Any] | None = None
    if state.state_type == state_type and isinstance(data, dict) and len(data) == 1:
        if "reasoningText" in data:
            text = _reasoning_text(data["reasoningText"])
            if text is not None:
                wire = {"reasoningText": text}
        elif isinstance(data.get("redactedContent"), str):
            with contextlib.suppress(ValueError):  # binascii.Error
                wire = {"redactedContent": base64.b64decode(data["redactedContent"], validate=True)}
    # Raised here, after the handlers: their errors quote the payload.
    if wire is None:
        raise ProviderError(ProviderFailure.REJECTED)
    return {"reasoningContent": wire}


_IMAGE_FORMATS = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/gif": "gif",
    "image/webp": "webp",
}

_STOP_REASONS = {
    "end_turn": StopReason.END_TURN,
    "tool_use": StopReason.TOOL_USE,
    "max_tokens": StopReason.MAX_TOKENS,
    "model_context_window_exceeded": StopReason.MAX_TOKENS,
    "stop_sequence": StopReason.STOP_SEQUENCE,
    "guardrail_intervened": StopReason.REFUSAL,
    "content_filtered": StopReason.REFUSAL,
    "refusal": StopReason.REFUSAL,
}


def _protocol() -> ProviderError:
    return ProviderError(ProviderFailure.PROTOCOL)


def base_model_id(model: str) -> str:
    """The model id an inference profile id names, or ``model`` itself."""
    prefix, dot, rest = model.partition(".")
    return rest if dot and prefix in PROFILE_PREFIXES else model


def streams_tool_use(model: str) -> bool:
    """Whether ConverseStream accepts tools for ``model`` (Bedrock's tool use page)."""
    return base_model_id(model) not in NO_STREAMING_TOOL_USE


def splits_thinking(model: str) -> bool:
    """Whether ``model`` writes its reasoning into the reply as ``<thinking>`` spans."""
    return base_model_id(model).startswith(THINKING_TAG_MODELS)


def caches_tools(model: str) -> bool:
    """Whether ``model`` accepts a cache point after its tools."""
    return not base_model_id(model).startswith(NO_TOOL_CACHE_POINT)


class _CachePoints:
    """Hands out cache points in processing order, keeping only the last four."""

    def __init__(self, request: ModelRequest, *, enabled: bool) -> None:
        #: Whether the tools take a cache point: asked for, and accepted by the model.
        self.tools = request.cache_tools and bool(request.tools) and caches_tools(request.model)
        total = 0
        if enabled:
            total += int(self.tools)
            total += sum(isinstance(b, CacheBreakpoint) for b in request.system)
            total += sum(
                isinstance(b, CacheBreakpoint) for m in request.messages for b in m.content
            )
        self._skip = max(0, total - MAX_CACHE_POINTS)
        self._enabled = enabled

    def take(self) -> list[dict[str, Any]]:
        if not self._enabled:
            return []
        if self._skip:
            self._skip -= 1
            return []
        return [CACHE_POINT]


def _image(block: ImageBlock) -> dict[str, Any]:
    return {"image": {"format": _IMAGE_FORMATS[block.media_type], "source": {"bytes": block.data}}}


def _content(block: Any, points: _CachePoints) -> list[dict[str, Any]]:
    if isinstance(block, TextBlock):
        return [{"text": block.text}]
    if isinstance(block, ImageBlock):
        return [_image(block)]
    if isinstance(block, CacheBreakpoint):
        return points.take()
    if isinstance(block, ProviderStateBlock):
        # Its own reasoning goes back in place; another kind's state is left out, and so
        # is tag text (Nova's own format, unsigned), after the same check of its payload.
        if block.provider_kind != KIND:
            return []
        if block.state_type == THINKING_STATE:
            _reasoning_content(block, THINKING_STATE)
            return []
        return [_reasoning_content(block)]
    if isinstance(block, ToolUseBlock):
        return [{"toolUse": {"toolUseId": block.id, "name": block.name, "input": block.input}}]
    if isinstance(block, ToolResultBlock):
        return [
            {
                "toolResult": {
                    "toolUseId": block.tool_use_id,
                    "content": [
                        {"text": b.text} if isinstance(b, TextBlock) else _image(b)
                        for b in block.content
                    ],
                    "status": "error" if block.is_error else "success",
                }
            }
        ]
    raise TypeError(f"unsupported content block {type(block).__name__}")


def _message(message: Message, points: _CachePoints) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for block in message.content:
        content.extend(_content(block, points))
    return {"role": message.role, "content": content}


def _messages(messages: Sequence[Message], points: _CachePoints) -> list[dict[str, Any]]:
    """The Converse messages, leaving out a turn that held only tag text.

    Bedrock refuses an empty message and needs the roles to alternate, so the user
    messages either side of a left-out turn become one. Messages the caller sent in the
    same role are passed on as they are. A turn left with only cache points is left out in
    the same way.
    """
    out: list[dict[str, Any]] = []
    left_out = False
    for message in messages:
        mapped = _message(message, points)
        if message.content and all("cachePoint" in b for b in mapped["content"]):
            # Nothing but cache points is not valid content either: leave the turn out
            # and carry its points to the end of the message before it, which keeps the
            # cached prefix, unless a point is already there.
            if out and not any("cachePoint" in b for b in out[-1]["content"][-1:]):
                out[-1]["content"].extend(mapped["content"][:1])
            left_out = True
            continue
        if left_out and out and out[-1]["role"] == mapped["role"]:
            out[-1]["content"].extend(mapped["content"])
        else:
            out.append(mapped)
        left_out = False
    return out


def build_converse_request(request: ModelRequest, *, prompt_caching: bool) -> dict[str, Any]:
    """Converse (and ConverseStream) parameters for ``request``."""
    if request.output_schema is not None:
        # Structured output is not mapped to Converse yet; refuse rather than ignore it.
        raise ProviderError(ProviderFailure.REJECTED)
    points = _CachePoints(request, enabled=prompt_caching)
    params: dict[str, Any] = {"modelId": request.model}
    # Processing order is tools, system, messages, so cache points are taken in that order.
    tools: list[dict[str, Any]] = [
        {
            "toolSpec": {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": {"json": tool.input_schema},
            }
        }
        for tool in request.tools
    ]
    if points.tools:
        tools.extend(points.take())
    system: list[dict[str, Any]] = []
    for block in request.system:
        system.extend(_content(block, points))
    if system:
        params["system"] = system
    params["messages"] = _messages(request.messages, points)
    if tools:
        params["toolConfig"] = {"tools": tools}
    inference: dict[str, Any] = {"maxTokens": request.max_output_tokens}
    if request.temperature is not None:
        inference["temperature"] = request.temperature
    if request.stop_sequences:
        inference["stopSequences"] = list(request.stop_sequences)
    params["inferenceConfig"] = inference
    return params


def _count(raw: Mapping[str, Any], key: str, *, required: bool) -> int:
    if key not in raw:
        if required:
            raise _protocol()
        return 0
    value = raw[key]
    if type(value) is not int or value < 0:
        raise _protocol()
    return value


def usage_from_bedrock(raw: Mapping[str, Any]) -> Usage:
    """conduit's four usage counts from a Converse ``usage`` block."""
    return Usage(
        input_tokens=_count(raw, "inputTokens", required=True),
        output_tokens=_count(raw, "outputTokens", required=True),
        cache_read_tokens=_count(raw, "cacheReadInputTokens", required=False),
        cache_write_tokens=_count(raw, "cacheWriteInputTokens", required=False),
    )


def stop_reason_from_bedrock(raw: str) -> StopReason:
    return _STOP_REASONS.get(raw, StopReason.OTHER)


def _tool_input(text: str) -> dict[str, Any]:
    if not text:
        return {}
    value: Any = None
    with contextlib.suppress(json.JSONDecodeError):  # its message quotes the input
        value = json.loads(text)
    if not isinstance(value, dict):
        raise _protocol()
    return value


def _partial_tag(text: str, tag: str) -> int:
    """The length of the longest proper prefix of ``tag`` that ``text`` ends with."""
    for size in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:size]):
            return size
    return 0


class _ThinkingSplitter:
    """Splits one text block's pieces into answer text and ``<thinking>`` spans.

    ``feed`` and ``end`` return ``(is_thinking, text)`` parts in order. Outside a span it
    holds back only a suffix that could begin ``<thinking>``; inside one it keeps the span
    until the closing tag arrives.
    """

    def __init__(self) -> None:
        self._held = ""  # a possible tag start, kept for the next piece
        self._thinking: list[str] | None = None  # the open span's text, if one is open
        self._trim = False  # drop whitespace that follows a closing tag
        self._gap = False  # whitespace was dropped after a closing tag
        self._last = ""  # the last character of answer text given out

    def feed(self, piece: str) -> list[tuple[bool, str]]:
        parts: list[tuple[bool, str]] = []
        buffer = self._held + piece
        self._held = ""
        while buffer:
            if self._thinking is None:
                if self._trim:
                    stripped = buffer.lstrip()
                    self._gap = self._gap or len(stripped) < len(buffer)
                    buffer = stripped
                    if not buffer:
                        break
                start = buffer.find(THINKING_OPEN)
                if start < 0:
                    keep = _partial_tag(buffer, THINKING_OPEN)
                    release = buffer[: len(buffer) - keep]
                    self._held = buffer[len(buffer) - keep :]
                    if release:
                        parts.append((False, self._answer(release)))
                    break
                if start:
                    parts.append((False, self._answer(buffer[:start])))
                self._thinking = []
                buffer = buffer[start + len(THINKING_OPEN) :]
            else:
                end = buffer.find(THINKING_CLOSE)
                if end < 0:
                    keep = _partial_tag(buffer, THINKING_CLOSE)
                    self._thinking.append(buffer[: len(buffer) - keep])
                    self._held = buffer[len(buffer) - keep :]
                    break
                self._thinking.append(buffer[:end])
                parts.extend(self._close())
                buffer = buffer[end + len(THINKING_CLOSE) :]
        return parts

    def _answer(self, text: str) -> str:
        """``text`` as answer text, with the one space a dropped gap leaves between words."""
        if self._trim and self._gap and self._last and not self._last.isspace():
            text = " " + text
        self._trim = self._gap = False
        self._last = text[-1]
        return text

    def _close(self) -> list[tuple[bool, str]]:
        text = "".join(self._thinking or []).strip()
        self._thinking = None
        self._trim = True
        return [(True, text)] if text else []

    def end(self) -> list[tuple[bool, str]]:
        """What is left when the block ends: an open span is reasoning, a held tag start text."""
        parts: list[tuple[bool, str]] = []
        if self._thinking is not None:
            self._thinking.append(self._held)
            parts.extend(self._close())
        elif self._held:
            parts.append((False, self._answer(self._held)))
        self._held = ""
        self._trim = self._gap = False
        return parts


def _joined(parts: list[tuple[bool, str]]) -> list[tuple[bool, str]]:
    """``parts`` with neighbouring answer text joined, for a block that arrives whole."""
    joined: list[tuple[bool, str]] = []
    for is_thinking, text in parts:
        if joined and not is_thinking and not joined[-1][0]:
            joined[-1] = (False, joined[-1][1] + text)
        else:
            joined.append((is_thinking, text))
    return joined


def _thinking_events(parts: list[tuple[bool, str]]) -> Iterator[StreamEvent]:
    for is_thinking, text in parts:
        if is_thinking:
            state = reasoning_state({"reasoningText": {"text": text}}, THINKING_STATE)
            yield ProviderStateReported(state=state)
        else:
            yield TextDelta(text=text)


@dataclass
class _OpenCall:
    id: str
    name: str
    input: list[str] = field(default_factory=list)


@dataclass
class _OpenReasoning:
    """A reasoning block streamed in pieces: text and signature, or redacted bytes."""

    text: list[str] = field(default_factory=list)
    signature: list[str] = field(default_factory=list)
    redacted: bytes | None = None

    def add(self, piece: Any) -> None:
        if not isinstance(piece, Mapping) or len(piece) != 1:
            raise _protocol()
        ((name, value),) = piece.items()
        if name == "redactedContent" and isinstance(value, bytes):
            if self.text or self.signature or self.redacted is not None:
                raise _protocol()
            self.redacted = value
        elif name in _REASONING_TEXT_FIELDS and isinstance(value, str):
            if self.redacted is not None:
                raise _protocol()
            (self.text if name == "text" else self.signature).append(value)
        else:
            raise _protocol()

    def whole(self) -> dict[str, Any]:
        """The block's ``reasoningContent`` value, as a Converse response gives it."""
        if self.redacted is not None:
            return {"redactedContent": self.redacted}
        text = {"text": "".join(self.text)}
        if self.signature:
            text["signature"] = "".join(self.signature)
        return {"reasoningText": text}


class StreamMapper:
    """Turns ConverseStream events into conduit stream events, one event at a time."""

    def __init__(self, model: str | None = None) -> None:
        #: Whether a text block's ``<thinking>`` spans are split out (Nova models only).
        self._splits = model is not None and splits_thinking(model)
        self._splitters: dict[int, _ThinkingSplitter] = {}
        self._calls: dict[int, _OpenCall] = {}
        self._reasoning: dict[int, _OpenReasoning] = {}
        self._stop: StopReason | None = None
        self._usage: Usage | None = None

    def feed(self, raw: Mapping[str, Any]) -> Iterator[StreamEvent]:
        # ValueError covers pydantic's ValidationError, whose text quotes the input.
        broken = False
        try:
            yield from self._feed(raw)
        except (
            KeyError,
            TypeError,
            AttributeError,
            ValueError,
        ):
            broken = True
        if broken:
            raise _protocol()

    def _feed(self, raw: Mapping[str, Any]) -> Iterator[StreamEvent]:
        if "contentBlockStart" in raw:
            body = raw["contentBlockStart"]
            use = body.get("start", {}).get("toolUse")
            if use is not None:
                index = body["contentBlockIndex"]
                if index in self._calls:
                    raise _protocol()
                call = _OpenCall(id=use["toolUseId"], name=use["name"])
                self._calls[index] = call
                yield ToolCallStarted(id=call.id, name=call.name)
        elif "contentBlockDelta" in raw:
            body = raw["contentBlockDelta"]
            delta = body["delta"]
            if "text" in delta:
                if delta["text"] and self._splits:
                    splitter = self._splitters.setdefault(
                        body["contentBlockIndex"], _ThinkingSplitter()
                    )
                    yield from _thinking_events(splitter.feed(delta["text"]))
                elif delta["text"]:
                    yield TextDelta(text=delta["text"])
            elif "toolUse" in delta:
                open_call = self._calls.get(body["contentBlockIndex"])
                if open_call is None:
                    raise _protocol()
                open_call.input.append(delta["toolUse"].get("input", ""))
            elif "reasoningContent" in delta:
                index = body["contentBlockIndex"]
                if index in self._calls:
                    raise _protocol()
                self._reasoning.setdefault(index, _OpenReasoning()).add(delta["reasoningContent"])
            # Other deltas are not surfaced.
        elif "contentBlockStop" in raw:
            index = raw["contentBlockStop"]["contentBlockIndex"]
            ended = self._splitters.pop(index, None)
            if ended is not None:
                yield from _thinking_events(ended.end())
            reasoning = self._reasoning.pop(index, None)
            if reasoning is not None:
                yield ProviderStateReported(state=reasoning_state(reasoning.whole()))
            finished = self._calls.pop(index, None)
            if finished is not None:
                yield ToolCallFinished(
                    id=finished.id, name=finished.name, input=_tool_input("".join(finished.input))
                )
        elif "messageStop" in raw:
            self._stop = stop_reason_from_bedrock(raw["messageStop"]["stopReason"])
        elif "metadata" in raw:
            if self._usage is not None:
                raise _protocol()  # one metadata event per stream; never replace usage
            self._usage = usage_from_bedrock(raw["metadata"]["usage"])

    def finish(self) -> Iterator[StreamEvent]:
        """The final ``StreamStopped``, once the stream has ended."""
        if self._stop is None or self._usage is None or self._calls or self._reasoning:
            raise _protocol()
        for index in sorted(self._splitters):  # a block that never saw its stop event
            yield from _thinking_events(self._splitters[index].end())
        self._splitters.clear()
        yield StreamStopped(stop_reason=self._stop, usage=self._usage)


def events_from_converse_response(
    response: Mapping[str, Any], *, model: str | None = None
) -> list[StreamEvent]:
    """The stream events a whole Converse response describes."""
    splits = model is not None and splits_thinking(model)
    try:
        content: Sequence[Mapping[str, Any]] = response["output"]["message"]["content"]
        events: list[StreamEvent] = []
        for block in content:
            if "text" in block:
                if block["text"] and splits:
                    splitter = _ThinkingSplitter()
                    parts = splitter.feed(block["text"]) + splitter.end()
                    events.extend(_thinking_events(_joined(parts)))
                elif block["text"]:
                    events.append(TextDelta(text=block["text"]))
            elif "toolUse" in block:
                use = block["toolUse"]
                if not isinstance(use["input"], dict):
                    raise _protocol()
                events.append(ToolCallStarted(id=use["toolUseId"], name=use["name"]))
                events.append(
                    ToolCallFinished(id=use["toolUseId"], name=use["name"], input=use["input"])
                )
            elif "reasoningContent" in block:
                state = reasoning_state(block["reasoningContent"])
                events.append(ProviderStateReported(state=state))
        events.append(
            StreamStopped(
                stop_reason=stop_reason_from_bedrock(response["stopReason"]),
                usage=usage_from_bedrock(response["usage"]),
            )
        )
    except (
        KeyError,
        TypeError,
        AttributeError,
        ValueError,
    ):
        broken = True
    else:
        broken = False
    if broken:
        raise _protocol()
    return events
