"""Provider-neutral messages, content blocks, tool specifications, usage and responses.

Every provider adapter maps these to and from its own wire format, so upper layers
never depend on one provider's SDK. Every model is immutable and serialisable to JSON,
so a transcript store can hold them as they are.

Usage keeps four counts apart, all required: ``input_tokens`` (uncached input only),
``output_tokens``, ``cache_read_tokens`` and ``cache_write_tokens``. An adapter must
state every one explicitly, and a mapping carrying any other key is refused, so a
provider's own name for a count (Bedrock's ``cacheReadInputTokens``, for example) can
never be read as zero by mistake.

``ProviderStateBlock`` carries state a provider needs back unchanged on a later
request (a model's signed reasoning, for example): the provider kind that made it,
its type and an opaque payload. conduit never reads, renders or edits the payload.
Each adapter is expected to send back its own kind's blocks and ignore every other
kind's, so a conversation can move between providers; the providers follow-up brings
every adapter into line. Only an assistant message may carry one.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt, model_validator

from conduit.core.settings import KIND_PATTERN

TOOL_NAME_PATTERN = r"^[A-Za-z][A-Za-z0-9_-]{0,63}$"
#: Request fields that identify a call rather than describe it; not part of the digest.
CALL_CONTEXT_FIELDS = frozenset({"run_id", "correlation_id", "task_class"})
#: A provider-state type: a short name without spaces (``thinking``, ``thought_signature``).
STATE_TYPE_PATTERN = r"^[A-Za-z][A-Za-z0-9_.:-]{0,63}$"


class _Frozen(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )


class TextBlock(_Frozen):
    type: Literal["text"] = "text"
    text: str


class ImageBlock(_Frozen):
    type: Literal["image"] = "image"
    media_type: Literal["image/png", "image/jpeg", "image/gif", "image/webp"]
    data: bytes


class CacheBreakpoint(_Frozen):
    """Marks the end of a stable prefix that may be cached (C19).

    Adapters translate it to their provider's cache marker, or ignore it where the
    provider or model does not support prompt caching.
    """

    type: Literal["cache_breakpoint"] = "cache_breakpoint"


class ToolUseBlock(_Frozen):
    type: Literal["tool_use"] = "tool_use"
    id: str = Field(min_length=1, max_length=256)
    name: str = Field(pattern=TOOL_NAME_PATTERN)
    input: dict[str, Any]


class ToolResultBlock(_Frozen):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str = Field(min_length=1, max_length=256)
    content: tuple[TextBlock | ImageBlock, ...]
    is_error: bool = False


class ProviderStateBlock(_Frozen):
    """Opaque state a provider needs back unchanged: never text, never read by conduit.

    ``provider_kind`` is the adapter kind that made it (``anthropic``, ``gemini``);
    only that adapter is expected to send it back, and every other adapter to ignore
    it. The payload
    is left out of ``repr`` and ``str``, so it is never rendered.
    """

    type: Literal["provider_state"] = "provider_state"
    provider_kind: str = Field(pattern=KIND_PATTERN)
    state_type: str = Field(pattern=STATE_TYPE_PATTERN)
    payload: bytes = Field(repr=False)


ContentBlock = Annotated[
    TextBlock | ImageBlock | CacheBreakpoint | ToolUseBlock | ToolResultBlock | ProviderStateBlock,
    Field(discriminator="type"),
]
SystemBlock = Annotated[TextBlock | CacheBreakpoint, Field(discriminator="type")]
ResponseBlock = Annotated[
    TextBlock | ToolUseBlock | ProviderStateBlock, Field(discriminator="type")
]


def _state_of(blocks: tuple[Any, ...], kind: str | None) -> tuple[ProviderStateBlock, ...]:
    return tuple(
        b
        for b in blocks
        if isinstance(b, ProviderStateBlock) and (kind is None or b.provider_kind == kind)
    )


Role = Literal["user", "assistant"]


class Message(_Frozen):
    role: Role
    content: tuple[ContentBlock, ...]

    @model_validator(mode="after")
    def _state_only_from_the_assistant(self) -> Message:
        if self.role != "assistant" and _state_of(self.content, None):
            raise ValueError("only an assistant message may carry provider state")
        return self

    def provider_state(self, kind: str) -> tuple[ProviderStateBlock, ...]:
        """The provider-state blocks ``kind`` made, in order; every other kind's are left out."""
        return _state_of(self.content, kind)

    @classmethod
    def user(cls, text: str) -> Message:
        return cls(role="user", content=(TextBlock(text=text),))

    @classmethod
    def assistant(cls, text: str) -> Message:
        return cls(role="assistant", content=(TextBlock(text=text),))


class ToolSpec(_Frozen):
    """A tool as the model sees it: name, description and JSON Schema for its input."""

    name: str = Field(pattern=TOOL_NAME_PATTERN)
    description: str
    input_schema: dict[str, Any]


class StopReason(StrEnum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    REFUSAL = "refusal"
    OTHER = "other"


class Usage(_Frozen):
    """Token counts for one model call, the four kinds kept apart and all required."""

    input_tokens: NonNegativeInt  # uncached input tokens only
    output_tokens: NonNegativeInt
    cache_read_tokens: NonNegativeInt
    cache_write_tokens: NonNegativeInt

    @classmethod
    def zero(cls) -> Usage:
        return cls(input_tokens=0, output_tokens=0, cache_read_tokens=0, cache_write_tokens=0)

    @property
    def total_input_tokens(self) -> int:
        """Every input token: uncached, read from cache and written to cache."""
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


class ModelRequest(_Frozen):
    """One model call, provider-neutral.

    ``cache_tools`` places a cache breakpoint after the tool definitions. The call
    context (``run_id``, ``correlation_id``, ``task_class``) identifies the call in
    events and is not part of ``digest()``.
    """

    model: str = Field(min_length=1)
    system: tuple[SystemBlock, ...] = ()
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...] = ()
    cache_tools: bool = False
    max_output_tokens: PositiveInt
    temperature: float | None = Field(default=None, ge=0, le=2)
    stop_sequences: tuple[str, ...] = ()
    output_schema: dict[str, Any] | None = None  # structured output, where supported
    timeout_s: float = Field(default=120, gt=0, le=3_600)
    run_id: str | None = None
    correlation_id: str | None = None
    task_class: str | None = None

    def digest(self) -> str:
        """sha256 over the request's content and parameters, excluding the call context."""
        payload = self.model_dump_json(exclude=set(CALL_CONTEXT_FIELDS))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ModelResponse(_Frozen):
    """The normalised response: content, stop reason, tool use and usage."""

    content: tuple[ResponseBlock, ...]
    stop_reason: StopReason
    usage: Usage
    model: str

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    @property
    def tool_uses(self) -> tuple[ToolUseBlock, ...]:
        return tuple(b for b in self.content if isinstance(b, ToolUseBlock))

    @property
    def provider_state(self) -> tuple[ProviderStateBlock, ...]:
        """Every provider-state block, in order. Send them back in the assistant message."""
        return _state_of(self.content, None)


def usage_from_mapping(values: Mapping[str, int]) -> Usage:
    """A ``Usage`` from conduit's own four names; any other key is refused."""
    return Usage.model_validate(dict(values))
