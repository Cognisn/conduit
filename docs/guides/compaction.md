# Compaction

A long conversation eventually outgrows the model's context window, and well before that it costs more on every call. The compaction layer replaces the older part of a conversation with something shorter (a notice, the older turns without their tool results, or a summary written by a model) and keeps the most recent turns as they were. It is part of the core install.

## The compactor

`Compactor` is the agent runner's compaction hook: pass it as `AgentRunner(compaction=...)`. Before every model call it estimates the prompt's size, compares it with the assigned model's context window, and tries the most aggressive strategy whose threshold is reached. Below every threshold nothing changes.

Three strategies ship with conduit. Each keeps the last `keep_turns` turns and replaces what comes before:

| Strategy | Replaces the older turns with | Default threshold |
|---|---|---|
| `DropToolResults` | their words as text, with every tool result left out | 0.6 of the window |
| `Summarise` | a summary written by the model assigned to a task class the product declares | 0.8 |
| `KeepLastTurns` | a short notice that earlier turns were dropped | 0.9 |

A threshold is a share of the context window. With all three, a conversation is first trimmed of its tool results, then summarised, and as a last resort cut.

`compactor.force(session_id)` makes the next model call of that session compact whatever its size (with a named strategy, or the most aggressive one), which is how a product offers a "compact now" command. A forced compaction waits until one succeeds or finds nothing to replace.

```python
import asyncio

from conduit import (
    CapabilityFloor,
    ConduitSettings,
    Event,
    LiteralSecretResolver,
    Message,
    MessageEntry,
    TaskClass,
    start,
)
from conduit.agent import AgentRunner, AgentTask, RunStatus
from conduit.compaction import Compactor, DropToolResults, KeepLastTurns, Summarise
from conduit.providers.stub import StubFactory, StubTurn
from conduit.stores import MemoryTranscriptStore


class ListSink:
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def emit(self, event: Event) -> None:
        self.events.append(event)


def summariser(request: object, index: int) -> StubTurn:
    return StubTurn(text="The user asked about mail delays to example.org; they are resolved.")


settings = ConduitSettings.model_validate(
    {
        "providers": {
            "chat": {"kind": "stub", "not_for_production": True},
            "summaries": {"kind": "stub", "script": "summariser", "not_for_production": True},
        },
        "assignments": {
            "assistant": {"provider": "chat", "model": "stub-echo"},
            "conversation_summary": {"provider": "summaries", "model": "stub-echo"},
        },
    }
)
TASK_CLASSES = [
    TaskClass(name="assistant", floor=CapabilityFloor(tools=True)),
    TaskClass(name="conversation_summary", floor=CapabilityFloor(tools=True)),
]

history = [
    Message.user(f"Question {n} about mail to example.org?")
    if n % 2 == 0
    else Message.assistant(f"Answer {n}: delivery to example.org is delayed.")
    for n in range(12)
]


async def compacted_run() -> tuple[list[Event], int]:
    runtime = await start(
        settings,
        task_classes=TASK_CLASSES,
        factories=[StubFactory(scripts={"summariser": summariser})],
        secrets=LiteralSecretResolver({}),
    )
    store = MemoryTranscriptStore()
    sink = ListSink()
    try:
        # The product stored the conversation so far; the run continues it.
        await store.append("sess_mail", [MessageEntry(message=m) for m in history])
        compactor = Compactor(
            [
                DropToolResults(keep_turns=2),
                # The default summary floor: a model with tools (see "The summary call").
                Summarise(runtime, task_class="conversation_summary", keep_turns=2),
                KeepLastTurns(keep_turns=2),
            ],
            transcripts=store,
            events=sink,
            capture=settings.capture,
        )
        runner = AgentRunner(runtime, transcripts=store, compaction=compactor)

        compactor.force("sess_mail", strategy="summarise")
        transcript = await store.load("sess_mail")
        result = await runner.run(
            AgentTask(
                task_class="assistant",
                session_id="sess_mail",
                history=transcript.active_messages(),
                messages=(Message.user("Is it fixed now?"),),
            )
        )
        assert result.status is RunStatus.COMPLETED
        # The summary is a model call of the run, counted with the run's own.
        assert result.model_calls == 2

        reloaded = await store.load("sess_mail")
        return sink.events, len(reloaded.active_messages())
    finally:
        await runtime.aclose()


events, active = asyncio.run(compacted_run())
(compaction,) = [e for e in events if e.type == "compaction"]
assert (compaction.strategy, compaction.trigger, compaction.outcome) == (
    "summarise",
    "forced",
    "compacted",
)
assert compaction.summary is not None and compaction.summary.text is None  # hash and size only
assert active < len(history)  # a reload gives the compacted conversation
```

## Keeping the cached prefix stable

The hook returns messages only. The system prompt, the tool definitions and the loaded skills (which the skills layer places in the system blocks, before their cache breakpoint) are never touched, so they stay byte-identical and the provider keeps reading them from its cache. The summary becomes the first message after that prefix, and by default it ends with a cache breakpoint of its own (`cache_summary=True`), so from the call after a compaction the summary is read from the cache too.

Some providers limit how many breakpoints one request may carry: Anthropic allows four. With tools, the system prompt and loaded skills already using three or more, the summary's breakpoint can push one out. A product near that limit passes `Compactor(..., cache_summary=False)`.

## When a compaction is applied

- **While the prompt fits** (the prompt and the assignment's output budget are within the window), a compaction must bring the prompt under its threshold. If the fixed prefix, the kept tail and the strategy's allowance for a summary already reach it, nothing is tried (`skipped`, `prefix_too_large`), and a result still at or over the threshold is not applied (`skipped`, `not_below_threshold`). A conversation is therefore never compacted again on every call, and no summary is paid for that could not help.
- **Once the prompt no longer fits**, a compaction is needed, so it is applied whenever its result fits, even above the threshold. A result that cannot fit fails (`result_too_large`).
- **A forced compaction** skips these checks and is applied whenever there is something to replace.

A failure (`CompactionFailed`, with a fixed reason) leaves the conversation as it was. While the prompt still fits, the call goes ahead uncompacted and the next call tries again; once it no longer fits, the hook raises, since the call could not succeed, and the run ends with the failure `compaction_failed`. Every attempt, skip or failure emits a `CompactionEvent` with the strategy, trigger, outcome, reason and token estimates before and after.

## The summary call

`Summarise` sends the older turns to the model assigned to the task class the product names, as messages much as they were: text trimmed to `max_message_chars`, images named rather than sent, cache markers and provider state left out, and the summary instructions added to the last user turn. **Tool calls and tool results keep their own blocks**: each tool use is sent whole, so its result still answers it, and each result is trimmed to `tool_result_chars` (500 characters by default). The request also carries a definition for every tool those turns used (the run's own definition, or a minimal one for a tool the run no longer offers), because a model is sent tool calls only alongside their tools.

That is why `Summarise`'s default floor is `CapabilityFloor(tools=True)`: the summary model must accept tool definitions, calls and results. Pass a lower `floor` only when the conversations it summarises never hold tool calls (a chat with no tools), so no summary request ever carries one. The product declares the summary task class, with its floor and assignment, like any other: a smaller, cheaper model often serves well.

Tool arguments and results in the summary request are tool content, and the summary call's events are held to the capture policy like every model call's: the request is always recorded by hash and size, its text only when `prompts` is captured, and even then without the tool arguments and results unless `tool_content` is captured too.

## Summaries and the run's budget

A summary is a model call of the run: `Summarise` counts it against `max_model_calls`, holds its output to the tokens the budget has left, and charges its usage to `max_tokens`. A summary the budget refuses is never made, and the run ends as `budget_exhausted`, as for any over-budget call; so does a summary that spends the token budget exactly. Allow for summaries when you set `max_model_calls`. `RunResult.model_calls` counts every call counted, including the run's own call that was counted but never made because a compaction failed or was refused. The summary call is also metered by the provider registry like every model call, and its events carry the run's id.

A product's own hook charges its calls through `request.charge` (`begin`, then `record`), valid only during that `before_model_call`. A hook that raises ends the run with the failure `compaction_failed`; the error is logged by class name and not kept, so nothing it held reaches the result. Cancellation propagates.

```python
from collections.abc import Sequence

from conduit.agent import CompactionRequest, RunBudget, RunFailed, RunFailure


class BrokenHook:
    """A product's own compaction hook whose summary service is down."""

    async def before_model_call(self, request: CompactionRequest) -> Sequence[Message] | None:
        raise ConnectionError("summary service unreachable at 192.0.2.40")


async def failed_and_refused_runs() -> None:
    runtime = await start(
        settings,
        task_classes=TASK_CLASSES,
        factories=[StubFactory(scripts={"summariser": summariser})],
        secrets=LiteralSecretResolver({}),
    )
    question = (Message.user("Is it fixed now?"),)
    try:
        broken = await AgentRunner(runtime, compaction=BrokenHook()).run(
            AgentTask(task_class="assistant", messages=question)
        )
        assert broken.status is RunStatus.FAILED
        assert broken.failure is RunFailure.COMPACTION_FAILED
        try:
            broken.raise_for_failure()
        except RunFailed as failed:
            assert "192.0.2.40" not in str(failed) and failed.__cause__ is None

        # One model call allowed: the run's own call, so a forced summary is refused.
        compactor = Compactor([Summarise(runtime, task_class="conversation_summary", keep_turns=2)])
        compactor.force("sess_budget")
        refused = await AgentRunner(runtime, compaction=compactor).run(
            AgentTask(
                task_class="assistant",
                session_id="sess_budget",
                history=tuple(history),
                messages=question,
                budget=RunBudget(max_model_calls=1),
            )
        )
        assert refused.failure is RunFailure.BUDGET_EXHAUSTED
    finally:
        await runtime.aclose()


asyncio.run(failed_and_refused_runs())
```

## The token estimate

The compactor estimates about four characters a token, and takes the larger of that and the previous call's reported usage. The estimate undercounts some text (Chinese, Japanese and Korean, for example, take more tokens a character), so a prompt it judges to fit may still be refused by the provider for its length. Where that matters, pass `estimate=` with the model's own count: a function from the prompt's system blocks, tools and messages to a token count.

## Transcripts

With a transcript store and a session, the compactor checks that the stored transcript's active conversation is exactly the conversation it was shown (the runner appends every message it adds, so they agree when both share a store), then appends a `CompactionPoint` replacing through the last message before the kept tail, with the token estimates before and after. A reload gives the compacted conversation (`Transcript.active_messages()`), summary first, byte for byte as it was sent.
