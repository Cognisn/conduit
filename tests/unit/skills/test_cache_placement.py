"""Loaded skills sit in the stable request prefix and are cached with the system and tools.

A small driver plays a conversation against the stub provider, whose cache accounting
writes the prefix up to the last cache breakpoint on its first call and reads it on
every later call with the same prefix. The driver rebuilds the system prompt from the
skill session before each call, as an agent runner does.
"""

from __future__ import annotations

from typing import Any

from conduit.core.tools import TaskTools, ToolCallContext, ToolCatalogue, ToolOutcome
from conduit.core.types import (
    CacheBreakpoint,
    Message,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolResultBlock,
    Usage,
)
from conduit.providers.stub import StubProvider, StubToolCall, StubTurn, scripted
from conduit.skills.disclosure import SkillSession
from conduit.skills.filesystem import FilesystemSkillSource
from conduit.skills.tool import with_use_skill
from tests.unit.skills.helpers import FIXTURES

BASE = (TextBlock(text="You help the example.com team with their release work."),)


def _tokens(text: str) -> int:
    return (len(text) + 3) // 4  # the stub's own count


async def _refuse(tool: Any, arguments: dict[str, Any], context: ToolCallContext) -> ToolOutcome:
    raise AssertionError("only use_skill is called here")


class Driver:
    """Just enough of a runner: call, run tool calls, feed results back, repeat."""

    def __init__(self, session: SkillSession, turns: list[StubTurn]) -> None:
        self.session = session
        self.stub = StubProvider(scripted(turns), not_for_production=True)
        tool = session.tool(task_classes={"chat"})
        self.tools = TaskTools(
            ToolCatalogue.from_tools([tool]).registry_for("chat"), with_use_skill(_refuse)
        )
        self.messages: list[Message] = []
        self.requests: list[ModelRequest] = []

    async def call(self) -> ModelResponse:
        request = ModelRequest(
            model="stub-model",
            system=await self.session.system(BASE),
            messages=tuple(self.messages),
            tools=self.tools.specs(),
            max_output_tokens=512,
        )
        self.requests.append(request)
        response = await self.stub.complete(request)
        self.messages.append(Message(role="assistant", content=response.content))
        results = []
        for use in response.tool_uses:
            outcome = await self.tools.call(
                self.tools.resolve(use.name),
                use.input,
                ToolCallContext(task_class="chat", run_id="run_1", call_id=use.id),
            )
            results.append(
                ToolResultBlock(
                    tool_use_id=use.id,
                    content=(TextBlock(text=outcome.content),),
                    is_error=not outcome.ok,
                )
            )
        if results:
            self.messages.append(Message(role="user", content=tuple(results)))
        return response

    def say(self, text: str) -> None:
        self.messages.append(Message.user(text))


def _prefix_text(request: ModelRequest) -> str:
    """The text the stub caches: tool definitions, then the system up to its breakpoint."""
    import json

    tools = json.dumps([t.model_dump(mode="json") for t in request.tools])
    return tools + "".join(b.text for b in request.system if isinstance(b, TextBlock))


def _load(name: str) -> StubTurn:
    return StubTurn(tool_calls=(StubToolCall(name="use_skill", input={"name": name}),))


async def test_a_loaded_skill_is_cached_with_the_system_prompt_and_tools() -> None:
    session = SkillSession(FilesystemSkillSource(FIXTURES))
    driver = Driver(
        session,
        [_load("release-notes"), StubTurn(text="Here is a draft."), StubTurn(text="Revised.")],
    )
    driver.say("Draft the notes for this week.")
    first = (await driver.call()).usage
    second = (await driver.call()).usage
    driver.say("Shorter, please.")
    third = (await driver.call()).usage
    body = session.loaded[0].body

    # Call 1: the listing, system and tools are written to the cache.
    assert first.cache_read_tokens == 0
    assert first.cache_write_tokens == _tokens(_prefix_text(driver.requests[0]))
    # Call 2: the skill is now in the prefix, so the longer prefix is written once.
    request = driver.requests[1]
    assert isinstance(request.system[-1], CacheBreakpoint)
    assert body in "".join(b.text for b in request.system if isinstance(b, TextBlock))
    assert second.cache_read_tokens == 0
    assert second.cache_write_tokens == _tokens(_prefix_text(request))
    assert second.cache_write_tokens > first.cache_write_tokens + _tokens(body) - 2
    # Call 3: the whole prefix, the skill included, is read from the cache.
    assert driver.requests[2].system == request.system
    assert third.cache_read_tokens == second.cache_write_tokens
    assert third.cache_write_tokens == 0


async def test_the_body_is_never_sent_in_the_messages() -> None:
    session = SkillSession(FilesystemSkillSource(FIXTURES))
    driver = Driver(session, [_load("release-notes"), StubTurn(text="Done.")])
    driver.say("Draft the notes.")
    await driver.call()
    await driver.call()
    body = session.loaded[0].body
    for message in driver.messages:
        for block in message.content:
            if isinstance(block, ToolResultBlock):
                assert all(body not in c.text for c in block.content if isinstance(c, TextBlock))
            assert not (isinstance(block, TextBlock) and body in block.text)


async def test_a_resource_is_sent_once_as_a_tool_result_and_leaves_the_prefix_cached() -> None:
    session = SkillSession(FilesystemSkillSource(FIXTURES))
    driver = Driver(
        session,
        [
            _load("release-notes"),
            StubTurn(
                tool_calls=(
                    StubToolCall(
                        name="use_skill",
                        input={"name": "release-notes", "resource": "references/style.md"},
                    ),
                )
            ),
            StubTurn(text="Done."),
        ],
    )
    driver.say("Draft the notes.")
    await driver.call()
    second = (await driver.call()).usage
    third = (await driver.call()).usage
    assert third.cache_read_tokens == second.cache_write_tokens
    assert third.cache_write_tokens == 0
    style = session.loaded[0].resources["references/style.md"]
    results = [
        c.text
        for m in driver.messages
        for b in m.content
        if isinstance(b, ToolResultBlock)
        for c in b.content
        if isinstance(c, TextBlock)
    ]
    assert results.count(style) == 1
    assert all(style not in b.text for b in driver.requests[2].system if isinstance(b, TextBlock))


async def test_a_second_skill_rewrites_the_prefix_once_and_unloading_returns_to_it() -> None:
    session = SkillSession(FilesystemSkillSource(FIXTURES))
    driver = Driver(
        session,
        [
            _load("release-notes"),
            StubTurn(text="a"),
            _load("status-report"),
            StubTurn(text="b"),
            StubTurn(text="c"),
            StubTurn(text="d"),
        ],
    )
    driver.say("Start.")
    usages: list[Usage] = [(await driver.call()).usage for _ in range(2)]
    driver.say("Now the status report.")
    usages += [(await driver.call()).usage for _ in range(2)]
    driver.say("Again.")
    usages.append((await driver.call()).usage)
    assert [u.cache_write_tokens > 0 for u in usages] == [True, True, False, True, False]
    assert usages[4].cache_read_tokens == usages[3].cache_write_tokens

    await session.unload("status-report")
    driver.say("Back to the notes only.")
    after = (await driver.call()).usage
    # The one-skill prefix is still cached, so unloading costs no cache write.
    assert after.cache_write_tokens == 0
    assert after.cache_read_tokens == usages[1].cache_write_tokens
