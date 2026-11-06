"""The live checks: start conduit for one target, stream, use a tool, cache, embed.

Every call goes through ``conduit.start`` (or ``start_embeddings``) and the provider
registry, exactly as a product's would: settings validated, secrets resolved through
``EnvironmentSecretResolver``, assignments resolved against the catalogue and the
residency policy, and every call metered.

Assertions are on shapes and numbers only. Provider text, tool input the model wrote,
keys, signatures and response bodies never reach an assertion message or the report:
each check reduces what it needs to counts, booleans and enum values first, and asserts
on those. A ``ProviderError`` carries only its failure class, which may be reported.
"""

from __future__ import annotations

import asyncio
import json
import math
import secrets
import struct
import zlib
from collections.abc import AsyncIterator, Iterable, Mapping
from contextlib import asynccontextmanager
from typing import Any

from conduit.core.catalogue import CatalogueOverlay
from conduit.core.errors import ProviderError
from conduit.core.registry import ProviderFactory
from conduit.core.residency import ResidencyPolicy
from conduit.core.runtime import ConduitSettings, TaskBinding, start
from conduit.core.settings import AssignmentSettings
from conduit.core.streaming import (
    StreamEvent,
    StreamStopped,
    TextDelta,
    ToolCallFinished,
    ToolCallStarted,
    collect,
)
from conduit.core.tasks import CapabilityFloor, TaskClass
from conduit.core.types import (
    CacheBreakpoint,
    ImageBlock,
    Message,
    ModelRequest,
    ModelResponse,
    StopReason,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    Usage,
)
from conduit.embeddings import EmbeddingBinding, builtin_embedding_factories, start_embeddings
from conduit.providers import builtin_factories
from tests.live.report import REPORT, Verdict
from tests.live.targets import Target, resolver

#: The task class every live chat check runs under.
LIVE_TASK = "live"
LIVE_EMBEDDING_TASK = "live-embedding"
#: Room for reasoning models to think before they answer.
DEFAULT_OUTPUT_BUDGET = 4096


@asynccontextmanager
async def started(
    target: Target,
    *,
    residency: ResidencyPolicy | None = None,
    overlays: Iterable[CatalogueOverlay] = (),
    max_output_tokens: int | None = DEFAULT_OUTPUT_BUDGET,
    environ: Mapping[str, str] | None = None,
    factories: Iterable[ProviderFactory] | None = None,
) -> AsyncIterator[TaskBinding]:
    """A running conduit for ``target``'s model, closed on the way out.

    ``factories`` replaces the shipped adapters (``test_harness.py`` gives a scripted
    stub, to prove the checks offline)."""
    settings = ConduitSettings.model_validate(
        {
            "providers": {"live": target.provider},
            "assignments": {
                LIVE_TASK: AssignmentSettings(
                    provider="live", model=target.model, max_output_tokens=max_output_tokens
                )
            },
            "residency": residency or ResidencyPolicy(),
            "catalogue": (*target.overlays, *overlays),
        }
    )
    runtime = await start(
        settings,
        task_classes=[TaskClass(name=LIVE_TASK)],
        factories=builtin_factories() if factories is None else factories,
        secrets=resolver(environ),
    )
    try:
        yield runtime.for_task(LIVE_TASK)
    finally:
        await runtime.aclose()


@asynccontextmanager
async def started_embeddings(
    target: Target,
    *,
    dimensions: int | None = None,
    environ: Mapping[str, str] | None = None,
) -> AsyncIterator[EmbeddingBinding]:
    """A running conduit for ``target``'s embedding model, closed on the way out."""
    settings = ConduitSettings.model_validate(
        {
            "providers": {"live": target.provider},
            "assignments": {
                LIVE_EMBEDDING_TASK: AssignmentSettings(provider="live", model=target.model)
            },
            "catalogue": target.overlays,
        }
    )
    runtime = await start_embeddings(
        settings,
        task_classes=[TaskClass(name=LIVE_EMBEDDING_TASK, floor=CapabilityFloor(kind="embedding"))],
        factories=builtin_embedding_factories(),
        secrets=resolver(environ),
        dimensions={LIVE_EMBEDDING_TASK: dimensions} if dimensions is not None else None,
    )
    try:
        yield runtime.for_task(LIVE_EMBEDDING_TASK)
    finally:
        await runtime.aclose()


# Shapes ---------------------------------------------------------------------------------


def usage_detail(usage: Usage) -> str:
    """The four counts, for the report."""
    return (
        f"input {usage.input_tokens}, output {usage.output_tokens}, "
        f"cache read {usage.cache_read_tokens}, cache write {usage.cache_write_tokens}"
    )


def assert_usage(usage: Usage) -> None:
    """All four counts are stated (the type requires it), with some input and output."""
    counts = (
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_read_tokens,
        usage.cache_write_tokens,
    )
    negative = sum(1 for count in counts if count < 0)
    total_input = usage.total_input_tokens
    output = usage.output_tokens
    assert negative == 0, "a usage count is negative"
    assert total_input > 0, "the call reported no input tokens"
    assert output > 0, "the call reported no output tokens"


async def stream_shape(binding: TaskBinding, request: ModelRequest) -> tuple[int, StreamStopped]:
    """The number of text deltas and the final event, checking the stream's shape."""
    deltas = 0
    stops = 0
    after_stop = 0
    final: StreamStopped | None = None
    async for event in binding.provider.stream(request):
        if final is not None:
            after_stop += 1
        if isinstance(event, TextDelta):
            deltas += 1
        elif isinstance(event, StreamStopped):
            stops += 1
            final = event
    assert stops == 1, "the stream did not end with exactly one StreamStopped"
    assert after_stop == 0, "events followed StreamStopped"
    assert final is not None
    return deltas, final


def ask(text: str) -> tuple[Message, ...]:
    return (Message.user(text),)


# Chat -----------------------------------------------------------------------------------


async def check_chat(binding: TaskBinding, subject: str) -> Usage:
    """A streaming chat: text deltas, one final event, end of turn, all four counts."""
    request = binding.request(messages=ask("Reply with the single word: ready."))
    deltas, final = await stream_shape(binding, request)
    stop = final.stop_reason
    assert deltas > 0, "the stream carried no text"
    assert stop is StopReason.END_TURN, f"stop reason {stop.value}, expected end_turn"
    assert_usage(final.usage)
    REPORT.record("chat", subject, "streaming chat", Verdict.MATCH, usage_detail(final.usage))
    return final.usage


LOOKUP_TOOL = ToolSpec(
    name="lookup_reference",
    description=(
        "Looks up the value stored under a reference code. It is the only source of "
        "these values: always call it to answer a question about a reference code."
    ),
    input_schema={
        "type": "object",
        "properties": {"code": {"type": "string", "description": "The reference code."}},
        "required": ["code"],
    },
)


async def check_tools(binding: TaskBinding, subject: str) -> int:
    """A tool loop: a streamed tool call, then the answer after the tool's result.

    The first response's content (provider state included: Anthropic thinking, Gemini
    thought signatures, Bedrock reasoning) goes back unchanged as the assistant
    message. Returns the number of provider-state blocks the first response carried.
    """
    question = ask(
        "What value is stored under reference code R-1234? Use the lookup_reference tool."
    )
    first_request = binding.request(messages=question, tools=(LOOKUP_TOOL,))
    counted = {"started": 0, "finished": 0}

    async def counting() -> AsyncIterator[StreamEvent]:
        async for event in binding.provider.stream(first_request):
            if isinstance(event, ToolCallStarted):
                counted["started"] += 1
            elif isinstance(event, ToolCallFinished):
                counted["finished"] += 1
            yield event

    first = await collect(counting(), model=first_request.model)
    started_calls = counted["started"]
    finished_calls = counted["finished"]
    assert started_calls >= 1, "no ToolCallStarted event was streamed"
    assert finished_calls == started_calls, "a started tool call was not finished"
    stop = first.stop_reason
    calls = first.tool_uses
    call_count = len(calls)
    assert stop is StopReason.TOOL_USE, f"stop reason {stop.value}, expected tool_use"
    assert call_count >= 1, "the response carried no tool use"
    named = sum(1 for call in calls if call.name == LOOKUP_TOOL.name)
    with_code = sum(1 for call in calls if isinstance(call.input.get("code"), str))
    assert named == call_count, "a tool call named another tool"
    assert with_code == call_count, "a tool call had no string code"
    state_blocks = len(first.provider_state)

    results = tuple(
        ToolResultBlock(tool_use_id=call.id, content=(TextBlock(text="The stored value is 42."),))
        for call in calls
    )
    follow_up = binding.request(
        messages=(
            *question,
            Message(role="assistant", content=first.content),
            Message(role="user", content=results),
        ),
        tools=(LOOKUP_TOOL,),
    )
    second = await binding.provider.complete(follow_up)
    second_stop = second.stop_reason
    answered = bool(second.text.strip())
    assert second_stop is StopReason.END_TURN, f"stop reason {second_stop.value} after the result"
    assert answered, "no answer followed the tool result"
    assert_usage(second.usage)
    REPORT.record(
        "tools",
        subject,
        "tool loop",
        Verdict.MATCH,
        f"{call_count} call(s), {state_blocks} provider-state block(s) sent back",
    )
    return state_blocks


# Caching --------------------------------------------------------------------------------


def cache_prefix(nonce: str, lines: int = 420) -> str:
    """A long, stable, synthetic system prompt (about 8,000 tokens), unique to a run.

    The nonce makes the first call a cache write rather than a read of an earlier run's
    prefix. Long enough for every catalogued model's minimum cacheable prefix.
    """
    body = "\n".join(
        f"Reference entry {n:04d} of ledger {nonce}: alpha bravo charlie delta echo "
        "foxtrot golf hotel india juliet kilo lima."
        for n in range(lines)
    )
    return f"You answer questions about a synthetic ledger.\n{body}"


async def _cached_call(binding: TaskBinding, prefix: str) -> Usage:
    request = binding.request(
        system=(TextBlock(text=prefix), CacheBreakpoint()),
        messages=ask("Reply with the single word: ok."),
    )
    _, final = await stream_shape(binding, request)
    assert_usage(final.usage)
    return final.usage


async def check_caching(
    binding: TaskBinding, subject: str, *, explicit: bool, strict: bool, attempts: int = 3
) -> tuple[Usage, Usage]:
    """Two calls on one cached prefix: a write (where the provider reports writes), then
    a read.

    ``explicit`` providers (cache breakpoints: Anthropic, Bedrock, bedrock-mantle) must
    report a write on the first call. The second call must read from the cache when
    ``strict``; otherwise a missing read is recorded as unobserved. Implicit caches are
    best effort, so the second call is tried up to ``attempts`` times.
    """
    prefix = cache_prefix(secrets.token_hex(8))
    first = await _cached_call(binding, prefix)
    if explicit:
        wrote = first.cache_write_tokens
        assert wrote > 0, "the first call on a new prefix reported no cache write"
    second = await _cached_call(binding, prefix)
    tries = 1
    while second.cache_read_tokens == 0 and not explicit and tries < attempts:
        await asyncio.sleep(2)
        second = await _cached_call(binding, prefix)
        tries += 1
    read = second.cache_read_tokens
    detail = f"first: {usage_detail(first)}; second: {usage_detail(second)}; tries {tries}"
    if read > 0:
        REPORT.record("caching", subject, "cached second call", Verdict.MATCH, detail)
    elif strict:
        REPORT.record("caching", subject, "cached second call", Verdict.MISMATCH, detail)
        raise AssertionError("the second call on the same prefix read nothing from the cache")
    else:
        REPORT.record("caching", subject, "cached second call", Verdict.UNOBSERVED, detail)
    return first, second


# Further capabilities (catalogue verification) ----------------------------------------


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))


#: A 1 by 1 opaque white PNG, built here so its checksums are right.
TINY_PNG = (
    b"\x89PNG\r\n\x1a\n"
    + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    + _png_chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
    + _png_chunk(b"IEND", b"")
)

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


async def try_complete(
    binding: TaskBinding, request: ModelRequest
) -> tuple[ModelResponse | None, str]:
    """The response, or ``None`` and the failure class (never provider text)."""
    try:
        return await binding.provider.complete(request), ""
    except ProviderError as error:
        return None, f"ProviderError({error.failure.value})"


async def check_output_limit(binding: TaskBinding, limit: int) -> tuple[bool, str]:
    """Whether the provider accepts a request at the catalogued output limit."""
    request = binding.request(
        messages=ask("Reply with the single word: ok."), max_output_tokens=limit
    )
    response, failure = await try_complete(binding, request)
    return response is not None, failure


async def check_structured_output(binding: TaskBinding) -> tuple[bool, str]:
    """Whether a request with an output schema is accepted and answered with its JSON."""
    request = binding.request(
        messages=ask("Give the answer 'ok' in the requested JSON form."),
        output_schema=ANSWER_SCHEMA,
    )
    response, failure = await try_complete(binding, request)
    if response is None:
        return False, failure
    parsed: Any = None
    try:
        parsed = json.loads(response.text)
    except ValueError:
        parsed = None
    shaped = isinstance(parsed, dict) and isinstance(parsed.get("answer"), str)
    return shaped, "" if shaped else "the answer was not the schema's JSON"


async def check_vision(binding: TaskBinding) -> tuple[bool, str]:
    """Whether a request carrying an image is accepted."""
    request = binding.request(
        messages=(
            Message(
                role="user",
                content=(
                    ImageBlock(media_type="image/png", data=TINY_PNG),
                    TextBlock(text="Reply with the single word: seen."),
                ),
            ),
        )
    )
    response, failure = await try_complete(binding, request)
    return response is not None, failure


# Embeddings -----------------------------------------------------------------------------

EMBEDDING_TEXTS = (
    "The ledger opens with entry one.",
    "A second, unrelated sentence about weather on example.org.",
    "Entry three closes the ledger.",
)


async def check_embeddings(binding: EmbeddingBinding, subject: str, expected: int) -> None:
    """One finite vector per text, each of ``expected`` length, and the usage shape."""
    response = await binding.embed(EMBEDDING_TEXTS)
    count = len(response.vectors)
    lengths = sorted({len(vector) for vector in response.vectors})
    finite = all(math.isfinite(x) for vector in response.vectors for x in vector)
    assert count == len(EMBEDDING_TEXTS), f"{count} vectors for {len(EMBEDDING_TEXTS)} texts"
    assert lengths == [expected], f"vector lengths {lengths}, expected {expected}"
    assert finite, "a vector held a non-finite number"
    usage = response.usage
    if usage is None:
        detail = f"{expected} dimensions; usage not reported"
    else:
        others = usage.output_tokens + usage.cache_read_tokens + usage.cache_write_tokens
        assert others == 0, "embedding usage carried more than input tokens"
        detail = f"{expected} dimensions; input tokens {usage.input_tokens}"
    REPORT.record("embeddings", subject, f"embed at {expected}", Verdict.MATCH, detail)
