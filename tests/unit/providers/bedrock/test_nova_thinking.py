"""Amazon Nova writes its reasoning into the reply as <thinking> spans; conduit moves it.

Amazon's Nova user guide ("Invoking a tool") says the default behaviour for tool calling
is chain-of-thought, "contained in <thinking> tags", with "multiple tool calls and
thinking blocks in a response". For Nova models only, a span becomes one reasoning state
block (the path Claude's reasoning takes) and the answer text no longer carries it.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

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
    Message,
    ModelRequest,
    ProviderStateBlock,
    StopReason,
    TextBlock,
    Usage,
)
from conduit.providers.bedrock.mapping import (
    StreamMapper,
    build_converse_request,
    events_from_converse_response,
)

NOVA = "amazon.nova-lite-v1:0"
USAGE = Usage(input_tokens=11, output_tokens=7, cache_read_tokens=0, cache_write_tokens=0)
METADATA = {"metadata": {"usage": {"inputTokens": 11, "outputTokens": 7}, "metrics": {}}}


def _state(state_type: str, text: str) -> ProviderStateReported:
    payload = json.dumps({"reasoningText": {"text": text}}, separators=(",", ":"))
    return ProviderStateReported(
        state=ProviderStateBlock(
            provider_kind="bedrock", state_type=state_type, payload=payload.encode()
        )
    )


def thought(text: str) -> ProviderStateReported:
    """The state event for one <thinking> span: Claude's reasoning shape, marked as tag text."""
    return _state("thinking_text", text)


def native(text: str) -> ProviderStateReported:
    """The state event for a native ``reasoningContent`` block."""
    return _state("reasoning_content", text)


def delta(text: str, index: int = 0) -> dict[str, Any]:
    return {"contentBlockDelta": {"delta": {"text": text}, "contentBlockIndex": index}}


def stop(index: int = 0) -> dict[str, Any]:
    return {"contentBlockStop": {"contentBlockIndex": index}}


def streamed(pieces: list[str], *, model: str | None = NOVA) -> list[StreamEvent]:
    """One text block streamed in ``pieces``, then the stream's ending."""
    raw = [delta(p) for p in pieces] + [
        stop(),
        {"messageStop": {"stopReason": "end_turn"}},
        METADATA,
    ]
    mapper = StreamMapper(model)
    events: list[StreamEvent] = []
    for item in raw:
        events.extend(mapper.feed(item))
    events.extend(mapper.finish())
    return events


def whole(blocks: list[dict[str, Any]], *, model: str | None = NOVA) -> list[StreamEvent]:
    return events_from_converse_response(
        {
            "output": {"message": {"content": blocks}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 11, "outputTokens": 7},
        },
        model=model,
    )


STOPPED = StreamStopped(stop_reason=StopReason.END_TURN, usage=USAGE)
REPLY = (
    "<thinking>The user is asking for my name. I can respond with a simple statement "
    "of my name.</thinking> My name is Example. How can I assist you today?"
)
THOUGHT = "The user is asking for my name. I can respond with a simple statement of my name."


def one_chunk(text: str, *, model: str | None = NOVA) -> list[StreamEvent]:
    return streamed([text], model=model)


class TestWholeReply:
    def test_a_span_becomes_reasoning_and_leaves_the_answer(self) -> None:
        events = whole([{"text": REPLY}])
        assert events == [
            thought(THOUGHT),
            TextDelta(text="My name is Example. How can I assist you today?"),
            STOPPED,
        ]

    def test_no_thinking_is_plain_text(self) -> None:
        assert whole([{"text": "Just an answer."}]) == [TextDelta(text="Just an answer."), STOPPED]

    def test_several_spans_around_tool_use_all_become_reasoning(self) -> None:
        events = whole(
            [
                {"text": "<thinking>First I look the host up.</thinking>"},
                {
                    "toolUse": {
                        "toolUseId": "tu-1",
                        "name": "lookup_host",
                        "input": {"host": "host-a.example.com"},
                    }
                },
                {"text": "<thinking>Now a second tool.</thinking>"},
                {"toolUse": {"toolUseId": "tu-2", "name": "lookup_host", "input": {}}},
            ]
        )
        assert [type(e).__name__ for e in events] == [
            "ProviderStateReported",
            "ToolCallStarted",
            "ToolCallFinished",
            "ProviderStateReported",
            "ToolCallStarted",
            "ToolCallFinished",
            "StreamStopped",
        ]
        assert events[0] == thought("First I look the host up.")
        assert events[3] == thought("Now a second tool.")

    def test_two_spans_in_one_block_and_text_between_them(self) -> None:
        events = whole(
            [{"text": "<thinking>a</thinking>\n\nMiddle.\n<thinking>b</thinking>\nEnd."}]
        )
        assert events == [
            thought("a"),
            TextDelta(text="Middle.\n"),
            thought("b"),
            TextDelta(text="End."),
            STOPPED,
        ]

    def test_whitespace_around_a_span_is_trimmed_not_the_answer(self) -> None:
        events = whole([{"text": "<thinking>\n  padded \n</thinking>\n\n  Indented answer.  "}])
        assert events == [thought("padded"), TextDelta(text="Indented answer.  "), STOPPED]

    def test_an_empty_span_reports_nothing(self) -> None:
        assert whole([{"text": "<thinking> </thinking>Hello."}]) == [
            TextDelta(text="Hello."),
            STOPPED,
        ]

    def test_an_unclosed_span_is_reasoning_not_text(self) -> None:
        # Decision: a reply cut off inside a span (max tokens, a stop sequence) holds
        # thinking, so what follows the opening tag is reasoning and never answer text.
        events = whole([{"text": "Start. <thinking>Cut off mid thou"}])
        assert events == [TextDelta(text="Start. "), thought("Cut off mid thou"), STOPPED]

    def test_a_partial_tag_at_the_very_end_is_text(self) -> None:
        assert whole([{"text": "Ends with <thin"}]) == [
            TextDelta(text="Ends with <thin"),
            STOPPED,
        ]

    def test_the_tag_in_other_case_is_not_a_span(self) -> None:
        text = "<Thinking>kept</Thinking>"
        assert whole([{"text": text}]) == [TextDelta(text=text), STOPPED]

    def test_a_reasoning_block_in_the_response_still_passes_through(self) -> None:
        block = {"reasoningText": {"text": "Native."}}
        events = whole([{"reasoningContent": block}, {"text": "Hi."}])
        assert events == [native("Native."), TextDelta(text="Hi."), STOPPED]

    def test_a_word_before_a_span_keeps_one_space_from_the_text_after_it(self) -> None:
        events = whole([{"text": "The answer is<thinking>x</thinking> 42"}])
        assert events == [
            TextDelta(text="The answer is"),
            thought("x"),
            TextDelta(text=" 42"),
            STOPPED,
        ]

    def test_no_space_is_added_where_the_text_before_ended_in_whitespace(self) -> None:
        events = whole([{"text": "Line one.\n<thinking>x</thinking>\nLine two."}])
        assert events == [
            TextDelta(text="Line one.\n"),
            thought("x"),
            TextDelta(text="Line two."),
            STOPPED,
        ]

    def test_no_space_is_added_at_the_start_of_a_block(self) -> None:
        events = whole([{"text": "<thinking>x</thinking> 42"}])
        assert events == [thought("x"), TextDelta(text="42"), STOPPED]

    @pytest.mark.parametrize(
        "model",
        [
            "amazon.nova-micro-v1:0",
            "amazon.nova-lite-v1:0",
            "amazon.nova-pro-v1:0",
            "amazon.nova-premier-v1:0",
            "us.amazon.nova-pro-v1:0",
            "apac.amazon.nova-lite-v1:0",
            "eu.amazon.nova-micro-v1:0",
            "global.amazon.nova-2-lite-v1:0",
        ],
    )
    def test_every_nova_id_and_profile_is_parsed(self, model: str) -> None:
        assert whole([{"text": REPLY}], model=model)[0] == thought(THOUGHT)


class TestNonNovaIsUntouched:
    @pytest.mark.parametrize(
        "model",
        [
            None,
            "anthropic.claude-sonnet-4-6",
            "eu.anthropic.claude-sonnet-4-6",
            "openai.gpt-oss-120b-1:0",
            "mistral.mistral-large-2402-v1:0",
            "writer.palmyra-x5-v1:0",
            "notamazon.nova-lite-v1:0",
            "amazon.titan-text-premier-v1:0",
        ],
    )
    def test_whole_reply_text_is_not_parsed(self, model: str | None) -> None:
        assert whole([{"text": REPLY}], model=model) == [TextDelta(text=REPLY), STOPPED]

    @pytest.mark.parametrize("model", [None, "anthropic.claude-sonnet-4-6", "qwen.qwen3-235b"])
    def test_a_stream_is_not_parsed_and_not_held_back(self, model: str | None) -> None:
        events = streamed(["<thin", "king>a</thinking> b"], model=model)
        assert events == [
            TextDelta(text="<thin"),
            TextDelta(text="king>a</thinking> b"),
            STOPPED,
        ]

    def test_a_code_block_mentioning_the_tag_is_untouched(self) -> None:
        text = "Use the tag like this:\n```\n<thinking>plan</thinking>\n```\nand the word thinking."
        assert whole([{"text": text}], model="anthropic.claude-sonnet-4-6") == [
            TextDelta(text=text),
            STOPPED,
        ]


class TestStreamedReply:
    def test_a_span_in_one_chunk(self) -> None:
        assert one_chunk(REPLY) == [
            thought(THOUGHT),
            TextDelta(text="My name is Example. How can I assist you today?"),
            STOPPED,
        ]

    def test_tags_split_across_chunks(self) -> None:
        events = streamed(["<thin", "king>The user is", " asking.</thin", "king> Hello", " there."])
        assert events == [
            thought("The user is asking."),
            TextDelta(text="Hello"),
            TextDelta(text=" there."),
            STOPPED,
        ]

    def test_the_split_is_equal_at_every_cut_point(self) -> None:
        text = "Before. <thinking>One.</thinking> Mid. <thinking>Two.</thinking>\n\nEnd."
        expected = whole([{"text": text}])
        for cut in range(1, len(text)):
            events = streamed([text[:cut], text[cut:]])
            merged = _merged(events)
            assert merged == _merged(expected), cut

    def test_every_one_character_chunking_matches_the_whole_reply(self) -> None:
        text = "<thinking>Plan <b>it</b>.</thinking>\n Answer < 3 and <thin."
        assert _merged(streamed(list(text))) == _merged(whole([{"text": text}]))

    def test_ordinary_text_is_not_delayed(self) -> None:
        mapper = StreamMapper(NOVA)
        out = list(mapper.feed(delta("Plain words here")))
        assert out == [TextDelta(text="Plain words here")]

    def test_only_a_possible_tag_start_is_held_back(self) -> None:
        mapper = StreamMapper(NOVA)
        assert list(mapper.feed(delta("Hi <thin"))) == [TextDelta(text="Hi ")]
        # "<thinx" is not a tag start, so everything held is released with it.
        assert list(mapper.feed(delta("x"))) == [TextDelta(text="<thinx")]

    def test_a_lone_angle_bracket_is_held_one_chunk(self) -> None:
        mapper = StreamMapper(NOVA)
        assert list(mapper.feed(delta("a <"))) == [TextDelta(text="a ")]
        assert list(mapper.feed(delta(" b"))) == [TextDelta(text="< b")]

    def test_a_held_partial_tag_at_the_end_is_released_as_text(self) -> None:
        assert streamed(["Ends with <thin"]) == [
            TextDelta(text="Ends with "),
            TextDelta(text="<thin"),
            STOPPED,
        ]

    def test_an_unclosed_span_at_the_end_is_reasoning(self) -> None:
        assert streamed(["Start. <thinking>Cut ", "off"]) == [
            TextDelta(text="Start. "),
            thought("Cut off"),
            STOPPED,
        ]

    def test_a_partial_closing_tag_at_the_end_of_an_unclosed_span_is_reasoning(self) -> None:
        assert streamed(["<thinking>Cut off</thin"]) == [thought("Cut off</thin"), STOPPED]

    def test_whitespace_between_a_span_and_the_answer_is_dropped_across_chunks(self) -> None:
        events = streamed(["<thinking>a</thinking>", "\n", "\n  ", "Answer."])
        assert events == [thought("a"), TextDelta(text="Answer."), STOPPED]

    def test_spans_either_side_of_a_tool_call(self) -> None:
        raw = [
            delta("<thinking>Look it ", 0),
            delta("up.</thinking>", 0),
            stop(0),
            {
                "contentBlockStart": {
                    "start": {"toolUse": {"toolUseId": "tu-1", "name": "lookup_host"}},
                    "contentBlockIndex": 1,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"toolUse": {"input": '{"host": "a"}'}},
                    "contentBlockIndex": 1,
                }
            },
            stop(1),
            delta("<thinking>Another</thinking>", 2),
            stop(2),
            {
                "contentBlockStart": {
                    "start": {"toolUse": {"toolUseId": "tu-2", "name": "lookup_host"}},
                    "contentBlockIndex": 3,
                }
            },
            stop(3),
            {"messageStop": {"stopReason": "tool_use"}},
            METADATA,
        ]
        mapper = StreamMapper(NOVA)
        events: list[StreamEvent] = []
        for item in raw:
            events.extend(mapper.feed(item))
        events.extend(mapper.finish())
        assert events == [
            thought("Look it up."),
            ToolCallStarted(id="tu-1", name="lookup_host"),
            ToolCallFinished(id="tu-1", name="lookup_host", input={"host": "a"}),
            thought("Another"),
            ToolCallStarted(id="tu-2", name="lookup_host"),
            ToolCallFinished(id="tu-2", name="lookup_host", input={}),
            StreamStopped(stop_reason=StopReason.TOOL_USE, usage=USAGE),
        ]

    def test_the_separating_space_survives_chunking(self) -> None:
        text = "The answer is<thinking>x</thinking>\n 42, then<thinking>y</thinking> done."
        assert _merged(streamed(list(text))) == _merged(whole([{"text": text}]))
        events = streamed(["is<thinking>x</thinking>", " ", "42"])
        assert events == [TextDelta(text="is"), thought("x"), TextDelta(text=" 42"), STOPPED]

    def test_a_span_straight_after_a_span_keeps_the_pending_space(self) -> None:
        events = streamed(["is<thinking>x</thinking> <thinking>y</thinking> 42"])
        assert events == [
            TextDelta(text="is"),
            thought("x"),
            thought("y"),
            TextDelta(text=" 42"),
            STOPPED,
        ]

    def test_a_block_left_open_is_flushed_at_the_end_of_the_stream(self) -> None:
        mapper = StreamMapper(NOVA)
        events: list[StreamEvent] = []
        for item in [
            delta("Hi <thin"),
            {"messageStop": {"stopReason": "end_turn"}},
            METADATA,
        ]:
            events.extend(mapper.feed(item))
        events.extend(mapper.finish())
        assert events == [TextDelta(text="Hi "), TextDelta(text="<thin"), STOPPED]

    def test_blocks_at_different_indexes_do_not_share_a_partial_tag(self) -> None:
        mapper = StreamMapper(NOVA)
        events = list(mapper.feed(delta("<thin", 0)))
        events += list(mapper.feed(delta("king>x</thinking>y", 1)))
        events += list(mapper.feed(stop(1)))
        events += list(mapper.feed(stop(0)))
        # Block 1 never saw an opening tag, so its text is plain; block 0 keeps its own
        # held "<thin" until its own stop.
        assert events == [TextDelta(text="king>x</thinking>y"), TextDelta(text="<thin")]


def _merged(events: list[StreamEvent]) -> list[Any]:
    """Events with adjacent text joined, so chunking does not matter."""
    out: list[Any] = []
    for event in events:
        if isinstance(event, TextDelta) and out and isinstance(out[-1], TextDelta):
            out[-1] = TextDelta(text=out[-1].text + event.text)
        else:
            out.append(event)
    return out


ALL_MODELS = [
    "amazon.nova-lite-v1:0",
    "us.amazon.nova-pro-v1:0",
    "apac.amazon.nova-micro-v1:0",
    "anthropic.claude-sonnet-4-6",
    "openai.gpt-oss-120b-1:0",
]


def _params(model: str, *messages: Message) -> dict[str, Any]:
    return build_converse_request(
        ModelRequest(model=model, messages=messages, max_output_tokens=256),
        prompt_caching=False,
    )


class TestSendingReasoningBack:
    """Tag-derived reasoning is never sent back to any model; native reasoning keeps its rule."""

    def _sent(self, model: str, state: ProviderStateBlock) -> list[Any]:
        params = _params(
            model,
            Message.user("Hello"),
            Message(role="assistant", content=(state, TextBlock(text="Hi."))),
            Message.user("Again"),
        )
        sent: list[Any] = params["messages"][1]["content"]
        return sent

    @pytest.mark.parametrize("model", ALL_MODELS)
    def test_tag_derived_reasoning_is_left_out_for_every_model(self, model: str) -> None:
        sent = self._sent(model, thought("Private thought zz-thought-zz.").state)
        assert sent == [{"text": "Hi."}]
        assert "zz-thought-zz" not in repr(sent)

    @pytest.mark.parametrize("model", ALL_MODELS)
    def test_native_reasoning_keeps_the_existing_rule_for_every_model(self, model: str) -> None:
        sent = self._sent(model, native("Native thought.").state)
        assert sent[0] == {"reasoningContent": {"reasoningText": {"text": "Native thought."}}}
        assert sent[1] == {"text": "Hi."}

    def test_a_malformed_tag_state_is_still_refused_without_its_payload(self) -> None:
        state = ProviderStateBlock(
            provider_kind="bedrock", state_type="thinking_text", payload=b"zz-state-marker-zz"
        )
        with pytest.raises(ProviderError) as raised:
            self._sent("amazon.nova-lite-v1:0", state)
        assert raised.value.failure is ProviderFailure.REJECTED
        assert "zz-state-marker-zz" not in repr(raised.value)
        assert raised.value.__cause__ is None and raised.value.__context__ is None


class TestReasoningOnlyTurns:
    """A turn that held only <thinking> text is left out; its neighbours are merged."""

    ONLY = Message(role="assistant", content=(thought("zz-cut-off-zz").state,))

    @pytest.mark.parametrize("model", ["amazon.nova-lite-v1:0", "anthropic.claude-sonnet-4-6"])
    def test_the_empty_assistant_turn_is_left_out_and_users_are_merged(self, model: str) -> None:
        params = _params(model, Message.user("Hello"), self.ONLY, Message.user("Please carry on."))
        assert params["messages"] == [
            {"role": "user", "content": [{"text": "Hello"}, {"text": "Please carry on."}]}
        ]
        assert "zz-cut-off-zz" not in repr(params)

    def test_roles_still_alternate_around_other_turns(self) -> None:
        params = _params(
            "amazon.nova-lite-v1:0",
            Message.user("One"),
            Message(role="assistant", content=(TextBlock(text="Two."),)),
            Message.user("Three"),
            self.ONLY,
            Message.user("Four"),
            Message(role="assistant", content=(TextBlock(text="Five."),)),
        )
        assert [m["role"] for m in params["messages"]] == ["user", "assistant", "user", "assistant"]
        assert params["messages"][2]["content"] == [{"text": "Three"}, {"text": "Four"}]

    def test_a_trailing_empty_turn_is_left_out(self) -> None:
        params = _params("amazon.nova-lite-v1:0", Message.user("Hello"), self.ONLY)
        assert params["messages"] == [{"role": "user", "content": [{"text": "Hello"}]}]

    def test_a_turn_with_other_content_is_kept(self) -> None:
        both = Message(
            role="assistant", content=(thought("zz-cut-off-zz").state, TextBlock(text="Hi."))
        )
        params = _params("amazon.nova-lite-v1:0", Message.user("Hello"), both, Message.user("Go"))
        assert [m["role"] for m in params["messages"]] == ["user", "assistant", "user"]

    def test_same_role_neighbours_the_caller_sent_are_not_merged(self) -> None:
        params = _params("amazon.nova-lite-v1:0", Message.user("A"), Message.user("B"))
        assert len(params["messages"]) == 2
