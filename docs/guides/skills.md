# Skills

A skill is a folder holding a `SKILL.md` file (instructions for the model, in Markdown, with a short front matter) and, optionally, resource files the instructions refer to. conduit's skills layer lists the skills a conversation may use, loads one when the model asks for it with the `use_skill` tool, and keeps the loaded skills in the cached prefix of every later request. This is progressive disclosure: the model sees each skill's one-line description until it needs the skill, so a product can offer many skills without paying for their instructions on every call.

The layer is part of the core install and needs no extra.

## The SKILL.md format

```markdown
---
name: incident-summary
description: Summarise an incident for the example.com status page, in three short paragraphs.
version: 1.0.0
---
# Incident summary

1. Say what was affected and for how long.
2. Say what was done, without naming people.
3. Use the template in templates/summary.md.
```

- `name` (required): lower-case words joined by hyphens, at most 64 characters, the same as the folder's name.
- `description` (required): one line, up to 1,024 characters. It is all the model sees before loading the skill, so it should say when to use it.
- `version` (optional): a short label without spaces.
- `license`, `compatibility`, `allowed-tools` and `metadata` (optional) are recorded. conduit does not enforce `allowed-tools`: which tools a task may call is the product's policy, set by its tool catalogue.

Any other key is refused, as is any front matter beyond simple `key: value` lines and lists. A refusal names a fixed reason and the line, never the text it found.

Resources are files inside the skill's folder, read by relative POSIX path (`templates/summary.md`). Paths that climb out of the folder (`..`, absolute or drive paths, backslashes, a link pointing outside) are refused.

## Sources

A `SkillSource` lists skills and loads them. `FilesystemSkillSource(directory)` reads a directory of skill folders, confined to it, with size limits per skill and resource (`SourceLimits`). `MemorySkillSource` holds skills built in code. A product can supply its own (a database, an object store); see [Writing backends](backends.md#the-skill-source).

```python
import asyncio
from pathlib import Path

from conduit.skills import FilesystemSkillSource

skill = Path("skills/incident-summary")
(skill / "templates").mkdir(parents=True)
(skill / "SKILL.md").write_text(
    "---\n"
    "name: incident-summary\n"
    "description: Summarise an incident for the example.com status page.\n"
    "version: 1.0.0\n"
    "---\n"
    "# Incident summary\n\n"
    "Say what was affected, for how long, and what was done. Use templates/summary.md.\n",
    encoding="utf-8",
    newline="\n",
)
(skill / "templates" / "summary.md").write_text(
    "## <service>: <one line>\n\n<three short paragraphs>\n", encoding="utf-8", newline="\n"
)

source = FilesystemSkillSource("skills")
summaries = asyncio.run(source.list_skills())
assert [s.name for s in summaries] == ["incident-summary"]
```

## A skill session

A `SkillSession` serves one conversation. `session.system(base)` gives the request's system blocks: the product's own `base` blocks, then the listing of available skills, then each loaded skill in the order it was loaded, then one cache breakpoint. `session.tool(task_classes=...)` gives the `use_skill` tool to add to the product's tool catalogue, and `with_use_skill(executor)` wraps the product's tool executor so `use_skill` calls go to the session.

With the agent runner, pass `session.system` as the task's `system_hook` and `with_use_skill` as its `tool_wrapper`; both are per run, because a skill session belongs to one conversation:

```python
from conduit import (
    CapabilityFloor,
    ConduitSettings,
    LiteralSecretResolver,
    Message,
    ModelRequest,
    TaskClass,
    TextBlock,
    ToolCatalogue,
    start,
)
from conduit.agent import AgentRunner, AgentTask, RunStatus
from conduit.core.tools import TaskTools, ToolCallContext, ToolOutcome
from conduit.providers.stub import StubFactory, StubToolCall, StubTurn, answer_tool_results
from conduit.skills import SkillSession, with_use_skill

sent: list[ModelRequest] = []
model = answer_tool_results(
    StubTurn(tool_calls=(StubToolCall(name="use_skill", input={"name": "incident-summary"}),)),
    lambda results: StubTurn(text="## Mail: delayed delivery\n\nMail to example.org was slow."),
)


def recording(request: ModelRequest, index: int) -> StubTurn:
    sent.append(request)
    return model(request, index)


async def no_other_tools(tool: object, arguments: dict, context: ToolCallContext) -> ToolOutcome:
    return ToolOutcome(ok=False, content="no such tool", error_code="forbidden")


async def write_summary() -> str:
    runtime = await start(
        ConduitSettings.model_validate(
            {
                "providers": {
                    "local": {"kind": "stub", "script": "model", "not_for_production": True}
                },
                "assignments": {"status": {"provider": "local", "model": "stub-echo"}},
            }
        ),
        task_classes=[TaskClass(name="status", floor=CapabilityFloor(tools=True))],
        factories=[StubFactory(scripts={"model": recording})],
        secrets=LiteralSecretResolver({}),
    )
    try:
        session = SkillSession(source, task_class="status")
        catalogue = ToolCatalogue.from_tools([session.tool(task_classes=["status"])])
        runner = AgentRunner(
            runtime,
            tools={"status": TaskTools(catalogue.registry_for("status"), no_other_tools)},
        )
        result = await runner.run(
            AgentTask(
                task_class="status",
                messages=(Message.user("Write up this morning's mail delay."),),
                system=(TextBlock(text="You write status updates for example.com."),),
                system_hook=session.system,
                tool_wrapper=with_use_skill,
            )
        )
        assert result.status is RunStatus.COMPLETED
        assert [loaded.name for loaded in session.loaded] == ["incident-summary"]
        return result.text
    finally:
        await runtime.aclose()


asyncio.run(write_summary())

first, second = (
    "".join(b.text for b in request.system if isinstance(b, TextBlock)) for request in sent
)
assert "incident-summary" in first and "Say what was affected" not in first  # listed only
assert "Say what was affected" in second  # loaded into the system prompt
```

## Caching and the stable prefix

Loaded skills sit in the system blocks, before the session's cache breakpoint, with the product's system prompt and the tool definitions. Loading a skill rewrites the cached prefix once; every later call reads it from the cache. A skill's body is never sent in the messages: the `use_skill` result only says the skill is loaded and lists its resources. A resource the model reads goes once into the messages as a tool result, which leaves the prefix unchanged.

Rendering is deterministic, so the same skills loaded in the same order give byte-identical blocks, across sessions and after a restore. The listing is a snapshot taken on first use and unchanged until `session.refresh()`, so a skill added to the source mid-conversation does not disturb the cache.

## Limits, markers and events

`DisclosureLimits` bounds what a session holds: how many skills may be loaded at once and how many bytes they take in the prefix, how many skills are listed and how large the listing is, and how large a resource may be. `allowed=` restricts a session to named skills (the product's per-task or per-user policy). A request past a limit, for an unlisted skill or for a resource of a skill not yet loaded is an error result the model sees, with a fixed code (`skill_unknown`, `too_many_loaded`, `path_traversal` and so on) and never the model's own arguments.

With a transcript store and a session id, each load, use and unload is a `SkillUse` marker in the transcript, and `session.restore(transcript)` reloads a resumed conversation's skills in their original order without writing new markers. Each is also a skill event (`skill_loaded`, `skill_used`, `skill_unloaded`, `skill_refused`) carrying names, versions and sizes only.
