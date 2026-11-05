"""The mantle provider: the anthropic adapter's mapping, gated by the mantle catalogue."""

from __future__ import annotations

import json
from typing import Any

import pytest

from conduit.core.catalogue import Catalogue, CatalogueOverlay, default_catalogue
from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.streaming import ProviderStateReported, StreamStopped
from conduit.core.types import (
    CacheBreakpoint,
    Message,
    ModelRequest,
    ProviderStateBlock,
    TextBlock,
    ToolSpec,
    Usage,
)
from conduit.providers.anthropic.request import KIND as ANTHROPIC_STATE
from conduit.providers.bedrock_mantle import BedrockMantleProvider

from ..anthropic import fakes
from ..anthropic.fakes import (
    message_end,
    message_start,
    scripted_client,
    text_block,
    thinking_block,
    usage,
)

SONNET_5 = "anthropic.claude-sonnet-5"
HAIKU_4_5 = "anthropic.claude-haiku-4-5"
TOOL = ToolSpec(name="lookup", description="Look up.", input_schema={"type": "object"})
SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}


def _simple(*body: dict[str, Any]) -> list[Any]:
    return [
        message_start(usage(input_tokens=10, output_tokens=1)),
        *body,
        *message_end("end_turn", usage(10, 5)),
    ]


def _provider(client: Any, catalogue: Catalogue | None = None) -> BedrockMantleProvider:
    return BedrockMantleProvider(
        client, name="mantle_melbourne", catalogue=catalogue or default_catalogue()
    )


def _cached_request(model: str, **fields: Any) -> ModelRequest:
    return ModelRequest(
        model=model,
        system=(TextBlock(text="system"), CacheBreakpoint()),
        messages=(Message(role="user", content=(TextBlock(text="hi"), CacheBreakpoint())),),
        tools=(TOOL,),
        cache_tools=True,
        max_output_tokens=64,
        **fields,
    )


def _with_capability(model: str, **capabilities: bool) -> Catalogue:
    entry = default_catalogue().require("bedrock_mantle", model)
    changed = entry.model_copy(
        update={"capabilities": entry.capabilities.model_copy(update=capabilities)}
    )
    return default_catalogue().apply(CatalogueOverlay(name="test", override=(changed,)))


def _cache_controls(params: dict[str, Any]) -> int:
    return json.dumps(params).count("cache_control")


async def _drain(provider: BedrockMantleProvider, request: ModelRequest) -> list[Any]:
    return [event async for event in provider.stream(request)]


def test_kind_and_repr() -> None:
    provider = _provider(scripted_client())
    assert provider.kind == "bedrock_mantle"
    assert provider.name == "mantle_melbourne"
    assert repr(provider) == "BedrockMantleProvider(name='mantle_melbourne')"


class TestPromptCaching:
    async def test_a_caching_model_sends_its_breakpoints(self) -> None:
        client = scripted_client(_simple(*text_block(0, "ok")))
        await _drain(_provider(client), _cached_request(SONNET_5))
        assert _cache_controls(client.messages.calls[0]) == 3  # tools, system, message

    async def test_a_model_the_catalogue_says_does_not_cache_sends_none(self) -> None:
        client = scripted_client(_simple(*text_block(0, "ok")))
        await _drain(_provider(client), _cached_request(HAIKU_4_5))
        params = client.messages.calls[0]
        assert _cache_controls(params) == 0
        # Everything else is still sent.
        assert params["system"] == [{"type": "text", "text": "system"}]
        assert params["tools"][0]["name"] == "lookup"
        assert params["messages"][0]["content"] == [{"type": "text", "text": "hi"}]

    async def test_the_answer_comes_from_the_catalogue_it_was_given(self) -> None:
        both = [_simple(*text_block(0, "ok")), _simple(*text_block(0, "ok"))]
        on = scripted_client(*both)
        await _drain(
            _provider(on, _with_capability(HAIKU_4_5, prompt_caching=True)),
            _cached_request(HAIKU_4_5),
        )
        assert _cache_controls(on.messages.calls[0]) == 3
        off = scripted_client(*both)
        await _drain(
            _provider(off, _with_capability(SONNET_5, prompt_caching=False)),
            _cached_request(SONNET_5),
        )
        assert _cache_controls(off.messages.calls[0]) == 0

    async def test_an_undeclared_model_sends_no_breakpoints(self) -> None:
        client = scripted_client(_simple(*text_block(0, "ok")))
        await _drain(_provider(client), _cached_request("anthropic.claude-undeclared"))
        assert _cache_controls(client.messages.calls[0]) == 0

    def test_prompt_caching_is_read_under_the_mantle_provider_name(self) -> None:
        provider = _provider(scripted_client())
        assert provider.prompt_caching(SONNET_5) is True
        assert provider.prompt_caching(HAIKU_4_5) is False
        # A bare id the bedrock provider declares but mantle does not.
        assert provider.prompt_caching("anthropic.claude-sonnet-4-6") is False


class TestStructuredOutput:
    def test_read_from_the_mantle_entries(self) -> None:
        provider = _provider(scripted_client())
        assert provider.structured_output(SONNET_5) is True
        assert provider.structured_output("anthropic.claude-undeclared") is False
        off = _provider(scripted_client(), _with_capability(SONNET_5, structured_output=False))
        assert off.structured_output(SONNET_5) is False

    async def test_a_schema_is_sent_where_supported(self) -> None:
        client = scripted_client(_simple(*text_block(0, '{"a": "b"}')))
        await _drain(_provider(client), _cached_request(SONNET_5, output_schema=SCHEMA))
        assert client.messages.calls[0]["output_config"]["format"]["schema"] == SCHEMA

    async def test_a_schema_is_refused_where_not_supported(self) -> None:
        client = scripted_client(_simple(*text_block(0, "ok")))
        catalogue = _with_capability(SONNET_5, structured_output=False)
        with pytest.raises(ProviderError) as caught:
            await _drain(
                _provider(client, catalogue), _cached_request(SONNET_5, output_schema=SCHEMA)
            )
        assert caught.value.failure is ProviderFailure.REJECTED
        assert client.messages.calls == []


class TestProviderState:
    async def test_thinking_is_reported_as_anthropic_state(self) -> None:
        client = scripted_client(_simple(*thinking_block(0), *text_block(1, "ok")))
        events = await _drain(_provider(client), _cached_request(SONNET_5))
        states = [e.state for e in events if isinstance(e, ProviderStateReported)]
        assert [s.provider_kind for s in states] == [ANTHROPIC_STATE]

    async def test_anthropic_state_goes_back_and_other_kinds_are_left_out(self) -> None:
        thinking = {"type": "thinking", "thinking": "", "signature": "c2ln"}
        mine = ProviderStateBlock(
            provider_kind=ANTHROPIC_STATE,
            state_type="thinking",
            payload=json.dumps(thinking).encode(),
        )
        foreign = ProviderStateBlock(provider_kind="gemini", state_type="x", payload=b"\x00")
        request = ModelRequest(
            model=SONNET_5,
            messages=(
                Message.user("q"),
                Message(role="assistant", content=(mine, foreign, TextBlock(text="a"))),
                Message.user("again"),
            ),
            max_output_tokens=64,
        )
        client = scripted_client(_simple(*text_block(0, "ok")))
        await _drain(_provider(client), request)
        sent = client.messages.calls[0]["messages"][1]["content"]
        assert sent == [thinking, {"type": "text", "text": "a"}]


class TestStreams:
    async def test_all_four_usage_counts_come_from_the_final_events(self) -> None:
        events = [
            message_start(usage(input_tokens=7, output_tokens=1, cache_read=300, cache_write=40)),
            *text_block(0, "ok"),
            *message_end("end_turn", usage(7, 9, cache_read=300, cache_write=40)),
        ]
        out = await _drain(_provider(scripted_client(events)), _cached_request(SONNET_5))
        stops = [e for e in out if isinstance(e, StreamStopped)]
        assert len(stops) == 1
        assert stops[0].usage == Usage(
            input_tokens=7, output_tokens=9, cache_read_tokens=300, cache_write_tokens=40
        )

    async def test_an_abandoned_stream_is_closed_at_once(self) -> None:
        client = scripted_client(_simple(*text_block(0, "a", "b", "c")))
        stream = _provider(client).stream(_cached_request(SONNET_5))
        await anext(stream)
        await stream.aclose()  # type: ignore[attr-defined]
        assert client.messages.streams[0].closed

    async def test_aclose_closes_the_client(self) -> None:
        client = scripted_client()
        await _provider(client).aclose()
        assert client.closed


class TestErrors:
    @pytest.mark.parametrize("when", ["create", "stream"])
    async def test_sdk_errors_become_failures_without_provider_text(self, when: str) -> None:
        error = fakes.RateLimitError("throttled: SECRET-MARKER", status_code=429)
        client = (
            scripted_client(error)
            if when == "create"
            else scripted_client([message_start(usage(5)), error])
        )
        with pytest.raises(ProviderError) as caught:
            await _drain(_provider(client), _cached_request(SONNET_5))
        assert caught.value.failure is ProviderFailure.UNAVAILABLE
        fakes.assert_clean(caught.value, "SECRET-MARKER")

    async def test_a_key_refusal_from_the_middleware_stays_rejected(self) -> None:
        # The bearer middleware raises ProviderError(REJECTED) inside messages.create.
        client = scripted_client(ProviderError(ProviderFailure.REJECTED))
        with pytest.raises(ProviderError) as caught:
            await _drain(_provider(client), _cached_request(SONNET_5))
        assert caught.value.failure is ProviderFailure.REJECTED
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None
