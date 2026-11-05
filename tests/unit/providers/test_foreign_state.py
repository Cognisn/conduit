"""Every adapter ignores the provider state other kinds made (Q0009).

A conversation can move between providers: an assistant message may carry state blocks
from the adapter that answered it, and any other adapter must build exactly the request
it would build without them, raising nothing and sending none of their payload.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from typing import Any

import pytest

from conduit.core.types import (
    ContentBlock,
    Message,
    ModelRequest,
    ProviderStateBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from conduit.providers.anthropic.request import build_params
from conduit.providers.bedrock.mapping import build_converse_request
from conduit.providers.gemini.mapping import build_request as build_gemini
from conduit.providers.ollama.mapping import build_chat_body
from conduit.providers.openai_compat.mapping import build_request as build_openai

MARKER = b"zz-foreign-state-zz"
KINDS = ("anthropic", "bedrock", "openai_compat", "gemini", "ollama", "stub", "example_gateway")

BUILDERS: dict[str, Callable[[ModelRequest], Any]] = {
    "anthropic": lambda r: build_params(r, structured_output=False),
    "bedrock": lambda r: build_converse_request(r, prompt_caching=False),
    "openai_compat": lambda r: build_openai(r, structured_output=False),
    "gemini": lambda r: build_gemini(r),
    "ollama": lambda r: build_chat_body(r, num_ctx=None),
}


def _state(kind: str) -> ProviderStateBlock:
    return ProviderStateBlock(provider_kind=kind, state_type="thinking", payload=MARKER)


def _conversation(answer: tuple[ContentBlock, ...]) -> ModelRequest:
    return ModelRequest(
        model="example-model",
        messages=(
            Message.user("Which network is host-a.example.com on?"),
            Message(role="assistant", content=answer),
            Message(
                role="user",
                content=(
                    ToolResultBlock(
                        tool_use_id="call_01", content=(TextBlock(text="host-a is 192.0.2.10"),)
                    ),
                ),
            ),
            Message(role="assistant", content=(TextBlock(text="It is on 192.0.2.0/24."),)),
            Message.user("Thanks."),
        ),
        max_output_tokens=64,
    )


ANSWER: tuple[ContentBlock, ...] = (
    TextBlock(text="Checking."),
    ToolUseBlock(id="call_01", name="lookup_host", input={"host": "host-a.example.com"}),
)


@pytest.mark.parametrize("adapter", sorted(BUILDERS))
def test_other_kinds_state_changes_nothing(adapter: str) -> None:
    build = BUILDERS[adapter]
    foreign = [_state(kind) for kind in KINDS if kind != adapter]
    with_state = _conversation((foreign[0], *ANSWER[:1], *foreign[1:], *ANSWER[1:]))
    assert build(with_state) == build(_conversation(ANSWER))
    sent = repr(build(with_state))
    assert MARKER.decode() not in sent
    assert base64.b64encode(MARKER).decode() not in sent


@pytest.mark.parametrize("adapter", sorted(BUILDERS))
def test_a_message_of_only_other_kinds_state_sends_none_of_it(adapter: str) -> None:
    foreign = tuple(_state(kind) for kind in KINDS if kind != adapter)
    request = ModelRequest(
        model="example-model",
        messages=(
            Message.user("hello"),
            Message(role="assistant", content=(*foreign, TextBlock(text="Hi."))),
            Message.user("again"),
        ),
        max_output_tokens=64,
    )
    sent = repr(BUILDERS[adapter](request))
    assert MARKER.decode() not in sent
    assert base64.b64encode(MARKER).decode() not in sent
