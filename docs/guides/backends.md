# Writing backends

conduit stores nothing of its own accord. Everything that persists or leaves the process goes through an interface the product supplies (C11): the secret resolver, the event sink, the transcript store, the MCP token store, the skill source and the approval handler. Every interface is an async `typing.Protocol`, so a backend is any class with the right methods; nothing needs to inherit from conduit.

conduit ships reference backends (in memory, local JSON files, and SQL behind the `sql` extra), and a conformance suite for each storage interface. A product that writes its own backend runs the suite against it under pytest.

| Interface | Protocol | Reference backends | Conformance suite |
|---|---|---|---|
| Secret resolver | `conduit.SecretResolver` | `EnvironmentSecretResolver`, `LiteralSecretResolver`; `KonfigSecretResolver` (konfig extra) | none (see below) |
| Event sink | `conduit.EventSink` | `FanOutSink`; `OtelEventSink` (otel), `WardenAuditSink` (warden) | none (see below) |
| Transcript store | `conduit.TranscriptStore` | `MemoryTranscriptStore`, `JsonTranscriptStore`, `SqlTranscriptStore` (sql) | `TranscriptStoreSuite` |
| Token store | `conduit.TokenStore` | `MemoryTokenStore`, `JsonTokenStore`, `SqlTokenStore` (sql) | `TokenStoreSuite` |
| Skill source | `conduit.SkillSource` | `FilesystemSkillSource`, `MemorySkillSource` | `SkillSourceSuite` |
| Approval handler | `conduit.ApprovalHandler` | none: approvals are the product's interface | none (see below) |

Three rules apply to every backend:

- **Async, and safe under concurrency.** conduit calls backends from concurrent tasks on one event loop. A backend that touches blocking I/O runs it in a worker thread (`asyncio.to_thread`) or uses an async driver.
- **No content in errors or logs.** A backend's errors and log records may name a session, a key or a reference, never a stored value, a token or a credential.
- **Never chain a driver's exception into one of your own** when the driver's message might carry a value: record what you need inside the `except` block and raise after it, so `__cause__` and `__context__` are both `None`. conduit's own errors follow this rule.

## The secret resolver

`resolve(ref)` turns a `secret://<name>` reference into a `pydantic.SecretStr`. The provider registry calls it when a provider is built (never per call), and the MCP client calls it when it connects. It raises `SecretUnavailable` (a `LookupError`) for a reference it cannot resolve, naming the reference only.

```python
import asyncio
from collections.abc import Mapping

from pydantic import SecretStr

from conduit import SecretRef
from conduit.core.errors import SecretUnavailable


class VaultSecretResolver:
    """Resolves references from the product's secret store (here, a mapping)."""

    def __init__(self, vault: Mapping[str, str]) -> None:
        self._vault = vault

    async def resolve(self, ref: SecretRef) -> SecretStr:
        value = self._vault.get(ref.name)
        if value is None:
            raise SecretUnavailable(f"{ref} is not in the vault")
        return SecretStr(value)

    def __repr__(self) -> str:
        return "VaultSecretResolver(<vault>)"  # never the values


resolver = VaultSecretResolver({"example/anthropic": "sk-example-not-real"})
key = asyncio.run(resolver.resolve(SecretRef.parse("secret://example/anthropic")))
assert key.get_secret_value() == "sk-example-not-real"
```

A reference name may hold letters, digits and `_ . / : -`, so it can name a path in a vault (`secret://prod/ai/anthropic`) or an ARN-like key. The konfig extra's `KonfigSecretResolver` resolves references through konfig's secrets store, whichever backend konfig has selected: its encrypted file, the operating system's keyring, an AWS Secrets Manager secret per name, or a designated Secrets Manager bundle. konfig's backends are synchronous, so each read runs in a worker thread.

There is no conformance suite for resolvers in this release; the example's behaviour (a value for a known reference, `SecretUnavailable` naming only the reference otherwise, no value in `repr`) is what conduit relies on.

## The event sink

`emit(event)` receives each event. Events are immutable pydantic models with a `type`, an `event_id`, `occurred_at`, and `run_id`, `correlation_id` and `task_class` where they apply. Serialise one with `event.model_dump(mode="json")`.

```python
import json

from conduit import Event, FanOutSink


class JsonLinesSink:
    """Appends each event to a JSON lines file, off the event loop."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = asyncio.Lock()

    async def emit(self, event: Event) -> None:
        line = json.dumps(event.model_dump(mode="json"), sort_keys=True)
        async with self._lock:
            await asyncio.to_thread(self._write, line)

    def _write(self, line: str) -> None:
        with open(self._path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
```

A sink should be quick: conduit awaits it on the call's path. A sink that ships events to a remote service queues them and sends from a background task. Wrap the product's sinks in a `FanOutSink` to deliver to several at once under one capture policy; a sink that raises is logged by class name and the others still receive the event.

Events carry content only inside `CapturedContent` fields (a hash, a size, and the text only when the capture policy allows it); every other string on an event is an identifier. A sink that must never hold content (an append-only audit trail) reads only the metadata, as the warden and otel sinks do.

There is no conformance suite for sinks in this release.

## The transcript store

A transcript is a session's ordered entries: `MessageEntry` (a message, with the model and usage where known), `CompactionPoint` (the compaction layer's summary, replacing every entry up to `replaces_through`) and `SkillUse` (a skill loaded, used or unloaded). The store:

- assigns each appended entry the session's next sequence number (1, 2, 3, ...), replacing any the caller set, and returns the entries as stored;
- keeps appends to one session atomic with respect to each other (a batch is never interleaved with another);
- refuses, with `ValueError` and storing nothing, an append that would leave the session unloadable (a compaction point replacing itself or a later entry);
- loads a session it never wrote as an empty transcript, lists sessions, and deletes them.

Entries are pydantic models with a JSON form, so any backend that stores text can hold them. Here is a store over a dictionary, as a product might write one over its own database:

```python
from collections.abc import Sequence

from pydantic import TypeAdapter, ValidationError

from conduit import Transcript
from conduit.core.transcript import TranscriptEntry

ENTRY: TypeAdapter[TranscriptEntry] = TypeAdapter(TranscriptEntry)


class DictTranscriptStore:
    def __init__(self) -> None:
        self._rows: dict[str, list[str]] = {}
        self._lock = asyncio.Lock()

    async def append(
        self, session_id: str, entries: Sequence[TranscriptEntry]
    ) -> tuple[TranscriptEntry, ...]:
        async with self._lock:
            rows = self._rows.get(session_id, [])
            stored = tuple(
                entry.model_copy(update={"seq": len(rows) + offset})
                for offset, entry in enumerate(entries, start=1)
            )
            loadable = True
            try:
                existing = tuple(ENTRY.validate_json(row) for row in rows)
                Transcript(session_id=session_id, entries=existing + stored)
            except ValidationError:
                loadable = False
            if not loadable:  # raised after the handler: nothing chained
                raise ValueError(f"the append to {session_id} would leave it unloadable")
            self._rows[session_id] = rows + [ENTRY.dump_json(e).decode() for e in stored]
            return stored

    async def load(self, session_id: str) -> Transcript:
        rows = self._rows.get(session_id, [])
        return Transcript(
            session_id=session_id, entries=tuple(ENTRY.validate_json(row) for row in rows)
        )

    async def list_sessions(self) -> Sequence[str]:
        return sorted(self._rows)

    async def delete(self, session_id: str) -> None:
        self._rows.pop(session_id, None)
```

`Transcript.active_messages()` gives the conversation as the model should see it: the latest compaction summary, then every later message. `loaded_skills()` gives the skills a resumed session had loaded.

## Running a conformance suite

Each suite is a class in `conduit.testing.conformance`. Subclass it in a test module and say how to make the backend; pytest collects the scenarios as tests:

```python
from conduit.testing.conformance import TranscriptStoreSuite


class TestDictTranscriptStore(TranscriptStoreSuite):
    async def make_store(self) -> DictTranscriptStore:
        return DictTranscriptStore()
```

The suite runs each scenario on its own event loop, so it needs no async pytest plugin. Override:

- `make_store` (or `make_source` for skills): a new, empty backend for one scenario;
- `reopen_store`: the same backend reached afresh (a new connection or instance), for durable stores, so the suite checks that data survives a reopen;
- `close_store`: release the backend after a scenario;
- `loop_factory`: the event loop to use (for example `asyncio.SelectorEventLoop` for psycopg's async driver on Windows).

Register the plugin to mark every suite test `conformance`, so `pytest -m conformance` runs exactly the suites:

```toml
# pyproject.toml
[tool.pytest.ini_options]
addopts = "-p conduit.testing.conformance"
```

Outside pytest, call a suite's test methods directly; each raises `AssertionError` on a failure:

```python
def run_suite(suite: object) -> list[str]:
    passed = []
    for name in sorted(dir(suite)):
        if name.startswith("test_"):
            getattr(suite, name)()
            passed.append(name)
    return passed


assert "test_concurrent_appends_to_one_session_stay_whole" in run_suite(TestDictTranscriptStore())
```

Running the suite against your backend is how you show it conforms; there is no other contract.

## The token store

The MCP client's OAuth tokens (and its dynamic client registrations) are held only in the product's token store, under keys such as `mcp:<server>` and `mcp:<server>:client`. `get` returns a `TokenSet` or `None`, `put` overwrites, `delete` removes. Store the mapping `TokenSet.to_storage()` returns (it reveals the token values for storage) and rebuild with `TokenSet.from_storage`. Tokens are credentials: encrypt them at rest, and never log them.

```python
from conduit.core.backends import TokenSet
from conduit.testing.conformance import TokenStoreSuite


class DictTokenStore:
    def __init__(self) -> None:
        self._rows: dict[str, str] = {}

    async def get(self, key: str) -> TokenSet | None:
        row = self._rows.get(key)
        return None if row is None else TokenSet.from_storage(json.loads(row))

    async def put(self, key: str, tokens: TokenSet) -> None:
        self._rows[key] = json.dumps(tokens.to_storage())

    async def delete(self, key: str) -> None:
        self._rows.pop(key, None)


class TestDictTokenStore(TokenStoreSuite):
    async def make_store(self) -> DictTokenStore:
        return DictTokenStore()


assert "test_a_deleted_key_is_gone" in run_suite(TestDictTokenStore())
```

## The skill source

`list_skills()` returns each skill's `SkillSummary` (name, description and version); `load_skill(name)` returns the `Skill` (its summary, its `SKILL.md` body, and its resources by relative path) and raises `LookupError` for one it does not hold. Validation of names, versions and resource paths is the skills layer's, so it applies to every source alike.

```python
from conduit.core.backends import Skill, SkillSummary
from conduit.testing.conformance import SkillSourceSuite


class DictSkillSource:
    def __init__(self, skills: Sequence[Skill]) -> None:
        self._skills = {skill.summary.name: skill for skill in skills}

    async def list_skills(self) -> Sequence[SkillSummary]:
        return [self._skills[name].summary for name in sorted(self._skills)]

    async def load_skill(self, name: str) -> Skill:
        skill = self._skills.get(name)
        if skill is None:
            raise LookupError(f"no skill named {name}")
        return skill


class TestDictSkillSource(SkillSourceSuite):
    async def make_source(self, skills: Sequence[Skill]) -> DictSkillSource:
        return DictSkillSource(skills)


assert "test_an_absent_skill_is_a_lookup_error" in run_suite(TestDictSkillSource())
```

## The approval handler

The agent runner sends a tool call the product marks as needing approval (a tool with a true `requires_approval` attribute, by default) to the handler first. `decide(request)` returns an `ApprovalDecision`; a refusal, a missing handler or a handler that raises becomes a tool result the model sees, and the tool never runs. How a person is asked (a terminal prompt, a queue in a web interface, a chat message) is the product's.

```python
from conduit.core.backends import ApprovalDecision, ApprovalRequest


class ChangeWindowApprovals:
    """Approves changes only to hosts on the product's allow list."""

    def __init__(self, allowed_hosts: set[str]) -> None:
        self._allowed = allowed_hosts

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        host = str(request.arguments.get("host", ""))
        if host in self._allowed:
            return ApprovalDecision(approved=True, decided_by="change-window@example.com")
        return ApprovalDecision(approved=False, reason="outside the change window")


approvals = ChangeWindowApprovals({"192.0.2.20"})
decision = asyncio.run(
    approvals.decide(
        ApprovalRequest(
            request_id="apr_1", tool="restart_service", arguments={"host": "192.0.2.21"}
        )
    )
)
assert not decision.approved
```

Pass it as `AgentRunner(approvals=...)`. A request's `arguments` are the model's, so treat them as untrusted when you show them to a person. There is no conformance suite for approval handlers in this release.

## The SQL backends

The `sql` extra's `SqlTranscriptStore` and `SqlTokenStore` keep their tables in their own `conduit` schema (Postgres) with their own Alembic chain and version table, so they live beside a product's own migrations without touching them. Migrate at start-up. This example uses SQLite in the working directory; for Postgres the URL is `postgresql+asyncpg://<user>@<host>/<database>`, with the password supplied the way the product supplies any database credential:

```python
from sqlalchemy.ext.asyncio import create_async_engine

from conduit import Message
from conduit.core.transcript import MessageEntry
from conduit.stores.sql import SqlTranscriptStore, current_revision, head_revision, migrate


async def sql_round_trip() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///conduit-example.db")
    try:
        await migrate(engine)
        assert await current_revision(engine) == head_revision()
        transcripts = SqlTranscriptStore(engine)
        await transcripts.append("sess_example", [MessageEntry(message=Message.user("hello"))])
        loaded = await transcripts.load("sess_example")
        assert [entry.seq for entry in loaded.entries] == [1]
    finally:
        await engine.dispose()


asyncio.run(sql_round_trip())
```

The drivers are asyncpg (Postgres) and aiosqlite (SQLite). `downgrade`, `current_revision`, `head_revision` and `revisions` are there for a product's own migration tooling; every migration is reversible.
