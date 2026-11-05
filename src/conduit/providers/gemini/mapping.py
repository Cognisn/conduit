"""Mapping between conduit's types and google-genai's generate_content parameters.

Requests become ``generate_content_stream`` keyword arguments (``model``,
``contents``, ``config``) as plain dicts, which the SDK validates itself; responses
arrive as chunks, which the provider turns into their JSON form (snake_case names,
bytes as base64) before ``StreamMapper`` reads them.

- **Cache breakpoints are ignored.** The Gemini API caches repeated prefixes
  implicitly and reports the tokens read from that cache; explicit caches
  (``cachedContents``) are separately created resources that conduit does not manage.
  Nothing is ever reported as written to a cache, so ``cache_write_tokens`` is zero.
- **Usage.** ``prompt_token_count`` includes ``cached_content_token_count``, so input
  is the prompt minus the cached part (plus ``tool_use_prompt_token_count``), and
  ``cache_read_tokens`` is the cached part. Output is ``candidates_token_count`` plus
  ``thoughts_token_count``: thinking is billed as output. Each chunk repeats the usage
  so far; the last chunk's is the call's usage, never a sum, and a stream whose last
  chunk carries no usage is a protocol failure. A negative count is one too.
- **Function calls** arrive whole in one chunk. A call id from the API is kept and
  sent back with the call and its response; where the API gives none, a conduit id
  marked ``GENERATED_PREFIX`` is made and never sent.
- **Thought signatures.** Gemini 3 models return a ``thought_signature`` with function
  calls and expect it back with the call on the next request. conduit's core types
  cannot carry it yet (Q0009), so the mapper reports each signature by call id and
  the provider remembers it. A call whose signature was never seen (history from
  another provider, a restarted process) carries Google's documented
  ``skip_thought_signature_validator`` value on the first call of its turn, a last
  resort that costs answer quality but keeps the conversation going.
- Thought parts (``thought: true``) are not streamed.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import ValidationError

from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.identifiers import new_id
from conduit.core.streaming import (
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
    StopReason,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)

#: Marks a call id conduit made because the API gave none; such ids are never sent.
GENERATED_PREFIX = "conduit_gemini_"
#: Google's documented value for a function call whose real signature is unavailable.
SKIP_SIGNATURE = b"skip_thought_signature_validator"

_PARAGRAPH = "\n\n"
_REFUSALS = frozenset(
    {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY"}
)
_STOP_REASONS = {"STOP": StopReason.END_TURN, "MAX_TOKENS": StopReason.MAX_TOKENS}


def _protocol() -> ProviderError:
    return ProviderError(ProviderFailure.PROTOCOL)


def _texts(blocks: tuple[Any, ...]) -> str:
    return _PARAGRAPH.join(b.text for b in blocks if isinstance(b, TextBlock))


def _image(block: ImageBlock) -> dict[str, Any]:
    return {"inline_data": {"mime_type": block.media_type, "data": block.data}}


def _sent_id(call_id: str) -> dict[str, str]:
    return {} if call_id.startswith(GENERATED_PREFIX) else {"id": call_id}


def _tool_names(request: ModelRequest) -> dict[str, str]:
    return {
        block.id: block.name
        for message in request.messages
        for block in message.content
        if isinstance(block, ToolUseBlock)
    }


def _parts(
    message: Message, names: Mapping[str, str], signatures: Mapping[str, bytes]
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    first_call = True
    for block in message.content:
        if isinstance(block, TextBlock):
            parts.append({"text": block.text})
        elif isinstance(block, ImageBlock):
            parts.append(_image(block))
        elif isinstance(block, ToolUseBlock):
            part: dict[str, Any] = {
                "function_call": {**_sent_id(block.id), "name": block.name, "args": block.input}
            }
            signature = signatures.get(block.id)
            if signature is None and first_call:
                signature = SKIP_SIGNATURE
            if signature is not None:
                part["thought_signature"] = signature
            parts.append(part)
            first_call = False
        elif isinstance(block, ToolResultBlock):
            try:
                name = names[block.tool_use_id]
            except KeyError:
                # A result for a call the history does not hold: refused before any call.
                raise ProviderError(ProviderFailure.REJECTED) from None
            text = _texts(block.content)
            parts.append(
                {
                    "function_response": {
                        **_sent_id(block.tool_use_id),
                        "name": name,
                        "response": {"error": text} if block.is_error else {"output": text},
                    }
                }
            )
            parts += [_image(b) for b in block.content if isinstance(b, ImageBlock)]
        # CacheBreakpoint: ignored (module docstring).
    return parts


def build_request(request: ModelRequest, *, signatures: Mapping[str, bytes]) -> dict[str, Any]:
    """``generate_content_stream`` keyword arguments for ``request``."""
    names = _tool_names(request)
    contents: list[dict[str, Any]] = []
    for message in request.messages:
        parts = _parts(message, names, signatures)
        if parts:
            role = "model" if message.role == "assistant" else "user"
            contents.append({"role": role, "parts": parts})
    config: dict[str, Any] = {"max_output_tokens": request.max_output_tokens}
    if request.temperature is not None:
        config["temperature"] = request.temperature
    if request.stop_sequences:
        config["stop_sequences"] = list(request.stop_sequences)
    system = _texts(tuple(b for b in request.system if not isinstance(b, CacheBreakpoint)))
    if system:
        config["system_instruction"] = system
    if request.tools:
        config["tools"] = [
            {
                "function_declarations": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters_json_schema": tool.input_schema,
                    }
                    for tool in request.tools
                ]
            }
        ]
    if request.output_schema is not None:
        config["response_mime_type"] = "application/json"
        config["response_json_schema"] = request.output_schema
    return {"model": request.model, "contents": contents, "config": config}


def _count(usage: Mapping[str, Any], key: str) -> int:
    value = usage.get(key) or 0
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise _protocol()
    return value


def _usage(usage: Mapping[str, Any]) -> Usage:
    prompt = _count(usage, "prompt_token_count")
    cached = _count(usage, "cached_content_token_count")
    if cached > prompt:
        raise _protocol()
    return Usage(
        input_tokens=prompt - cached + _count(usage, "tool_use_prompt_token_count"),
        output_tokens=_count(usage, "candidates_token_count")
        + _count(usage, "thoughts_token_count"),
        cache_read_tokens=cached,
        cache_write_tokens=0,
    )


def _signature(value: Any) -> bytes:
    if not isinstance(value, str):
        raise _protocol()
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise _protocol() from None


class StreamMapper:
    """Turns stream chunks (JSON form) into conduit stream events."""

    def __init__(self, *, new_call_id: Callable[[], str] | None = None) -> None:
        self._new_call_id = new_call_id or (lambda: new_id(GENERATED_PREFIX.rstrip("_")))
        self._called = False
        self._finish: str | None = None
        self._blocked = False
        self._usage: Mapping[str, Any] | None = None
        #: Thought signatures seen in this stream, by call id.
        self.signatures: dict[str, bytes] = {}

    def _call(self, part: Mapping[str, Any]) -> list[StreamEvent]:
        call = part["function_call"]
        if not isinstance(call, Mapping):
            raise _protocol()
        name, args, given = call.get("name"), call.get("args") or {}, call.get("id")
        if not isinstance(name, str):
            raise _protocol()
        call_id = given if isinstance(given, str) and given else self._new_call_id()
        if "thought_signature" in part:
            self.signatures[call_id] = _signature(part["thought_signature"])
        try:
            events: list[StreamEvent] = [
                ToolCallStarted(id=call_id, name=name),
                ToolCallFinished(id=call_id, name=name, input=args),
            ]
        except ValidationError:
            raise _protocol() from None
        self._called = True
        return events

    def feed(self, chunk: Mapping[str, Any]) -> list[StreamEvent]:
        """The events one chunk carries."""
        events: list[StreamEvent] = []
        candidates = chunk.get("candidates") or []
        if not isinstance(candidates, list):
            raise _protocol()
        if candidates:
            if self._finish is not None:
                raise _protocol()  # content after the finish
            candidate = candidates[0]
            if not isinstance(candidate, Mapping):
                raise _protocol()
            content = candidate.get("content") or {}
            for part in content.get("parts") or []:
                if not isinstance(part, Mapping):
                    raise _protocol()
                if part.get("thought"):
                    continue
                if "function_call" in part:
                    events += self._call(part)
                elif isinstance(part.get("text"), str) and part["text"]:
                    events.append(TextDelta(text=part["text"]))
            finish = candidate.get("finish_reason")
            if finish is not None:
                self._finish = str(finish)
        feedback = chunk.get("prompt_feedback") or {}
        if isinstance(feedback, Mapping) and feedback.get("block_reason"):
            self._blocked = True
        # The call's usage is that of the last chunk fed, never an earlier one: a
        # stream whose final chunk carries none is broken (finish() refuses it).
        usage = chunk.get("usage_metadata")
        if usage is not None and not isinstance(usage, Mapping):
            raise _protocol()
        self._usage = usage
        return events

    def finish(self) -> list[StreamEvent]:
        """The final ``StreamStopped``. A stream without a finish or usage is broken."""
        if self._usage is None or (self._finish is None and not self._blocked):
            raise _protocol()
        if self._blocked or self._finish in _REFUSALS:
            reason = StopReason.REFUSAL
        else:
            reason = _STOP_REASONS.get(self._finish or "", StopReason.OTHER)
            if reason is StopReason.END_TURN and self._called:
                reason = StopReason.TOOL_USE
        return [StreamStopped(stop_reason=reason, usage=_usage(self._usage))]
