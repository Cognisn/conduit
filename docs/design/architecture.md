# conduit architecture

This document records conduit's standing design decisions, its package layout as built, the direction its dependencies may point, how start-up, the provider registry, residency and content capture work, its providers and its extras. It is versioned with the code; the guides in `docs/guides/` show how a product uses it. The plan of record, with the reasoning behind each decision, is held in the project's knowledge base as "2026-11-04 plan conduit 0.1.0 development" and "2026-11-04 decision conduit library scope and design rulings (C1 to C19)". Where this document and those records disagree, the records win and this document is corrected.

## 1. Purpose

conduit gives every Cognisn product one AI integration layer: model providers, a model catalogue with residency policy, streaming, prompt caching, normalised usage, an agent runner with budgets and hand-off, compaction, skills, an MCP client, embeddings, product-supplied storage, and audit and monitoring events. It supplies mechanisms; each product keeps its own policy.

## 2. Decisions

| # | Area | Decision |
|---|---|---|
| C1 | Scope | A full AI integration library: providers, catalogue, usage, residency, agent runner, compaction, skills and an MCP client. conduit supplies mechanisms; each product keeps its policy (prompt layers, guardrail rules, credit ledgers, approvals and conversation management stay in the product). |
| C2 | Packaging | One package with layered modules. The core is always installed; every other layer and every provider SDK is an optional extra. Dependencies point one way (upper layers depend on the core, never the reverse) and each layer is usable alone. Enforced by import tripwires. |
| C3 | Concurrency | Async-first throughout. A supported sync facade, generated from the async API, runs calls on a loop conduit owns and refuses to run on that loop's own thread. Blocking SDK calls (boto3) run in a worker thread. |
| C4 | Origin | Seeded from VOR's AI layer, generalised: VOR-specific pieces are replaced by conduit interfaces. SPARK's LLM layer is the second source. Termograph is a reference for lessons only: **no Termograph code is copied**, and Termograph's routing classifier stays in Termograph, above conduit. |
| C5 | Licence and distribution | Public on PyPI as `cognisn-conduit` under MIT. |
| C6 | Providers (first release) | Amazon Bedrock (AWS credential chain, credentials held as secrets through the secret resolver, and Bedrock API keys: short-term keys refreshed automatically, long-term keys only when a setting allows them); Anthropic (direct, and Claude in Microsoft Foundry through its Anthropic endpoint, with an API key or Microsoft Entra ID); one OpenAI-compatible adapter serving OpenAI, X.AI and Microsoft Foundry (API key and Microsoft Entra ID); Gemini; Ollama; and the stub. |
| C7 | ChatGPT | "ChatGPT" means the OpenAI API through the OpenAI provider. Signing in with a consumer ChatGPT subscription is deferred until OpenAI publishes a supported route for third-party products. |
| C8 | Name | conduit (`cognisn-conduit`, repository `Cognisn/conduit`). |
| C9 | Adoption | No product adopts conduit before 0.1.0 is released. |
| C10 | Demonstration interface | A terminal interface in the optional `cli` extra, run with `uvx --from "cognisn-conduit[cli]" conduit`: providers and credential checks (including Bedrock authentication modes and residency refusals), streaming chat with tool use under runner budgets, skills, an MCP server, forced compaction, and normalised usage and cache events. Public API only; keeps no history unless asked. |
| C11 | Persistence | Storage-agnostic. Products supply backends through interfaces: transcript store, event sink, skill source, secret resolver, approval handler, MCP token store. Every interface is async and ships with a conformance test suite the product runs against its own backend. |
| C12 | Reference backends | In-memory and local JSON file backends, plus an optional `sql` extra (SQLAlchemy; Postgres and SQLite) with its own `conduit` Postgres schema, its own Alembic chain and version table, and reversible migrations. |
| C13 | Embeddings | An embeddings interface with remote providers sharing the chat providers' authentication, residency policy and usage events, plus an optional local extra running sentence-transformers with CPU-only torch (no NVIDIA packages installed or audited in CI; install lines documented per installer). |
| C14 | MCP client authentication | Static credentials (API key or bearer token through the secret resolver) and an OAuth 2.1 client: server metadata discovery, dynamic client registration, authorisation code with PKCE, refresh token rotation. Tokens are held through a product-supplied token store; the interactive sign-in is a hook. Built on the official MCP Python SDK. |
| C15 | Configuration | konfig is an optional `konfig` extra. conduit's settings are typed models buildable in code or from a mapping; the extra adds a konfig loader and the default secret resolver. conduit logs through the standard `logging` module under the `conduit` logger. |
| C16 | Release shape | One 0.1.0 containing everything, built in waves with milestone stamps (a1 to a4), b1 feature complete, rc1 with the live provider suite. |
| C17 | Conventions | `release/v<version>` branches and `<kind>/<version>-<aspect>` workstreams; Python 3.12 or later; hatchling; Ruff, mypy strict, pytest; licence audit (no strong copyleft; weak copyleft only as named, reviewed exceptions); mutation checks on every guard. |
| C18 | Model catalogue | A versioned default catalogue shipped as package data (model id, provider, regions, residency scope, capability flags: tools, vision, prompt caching, structured output, context window, output limit, embedding dimensions). Products overlay it to add, override or withdraw entries; an unknown model is refused unless declared. Task classes and capability floors are a mechanism: products declare their own classes and conduit enforces floors at start-up. No learned routing in conduit. |
| C19 | Caching, streaming, audit and monitoring | Provider-neutral cache breakpoints on the stable request prefix (system, tools, loaded skills, compaction summary), translated by each adapter or ignored where unsupported; separate cache-read and cache-write tokens on every usage event; compaction keeps the cached prefix stable. Every provider streams; runs are an async stream of typed events, with usage taken from the final stream event. Structured events for every run, model call, tool call, MCP call, approval, skill load, compaction and hand-off go to the event sink, with optional `warden` and `otel` extras. Events carry identifiers, hashes, sizes and outcomes by default; prompt and tool content is recorded only when the product opts in. |
| C20 | Machine | Every development session of 0.1.0 runs on one designated workstation. Seed and second-source code is read from read-only reference clones. |

## 3. Package layout

The layout as built. Each layer's package docstring describes it in more detail.

```text
src/conduit/
  __init__.py      the top-level public API (the core's contracts, start and start_sync)
  _version.txt     the version, written only by scripts/stamp_version.py
  core/
    types.py       messages, content blocks (text, image, tool use, tool result, provider
                   state, cache breakpoint), tool specs, requests, responses, usage
    streaming.py   stream events (usage, text delta, tool call started and finished,
                   provider state, stop) and collect
    provider.py    the provider protocol and BaseProvider; the embedding request, response
                   and provider protocol
    registry.py    the provider registry, MeteredProvider and MeteredEmbedder, the model
                   call and embedding call events
    runtime.py     ConduitSettings, start, Runtime and TaskBinding
    settings.py    ProviderSettings and AssignmentSettings; credential and echo rules
    catalogue.py   the catalogue, its entries and overlays; data/catalogue.json
    tasks.py       task classes, capability floors and the resolution of assignments
    residency.py   residency scopes and policy
    secrets.py     secret references, the resolver protocol and two resolvers
    events.py      the event base, the sink protocol, FanOutSink and the capture policy
    tools.py       the tool protocol, catalogue, per-task registries and TaskTools
    transcript.py  transcript entries, Transcript and the transcript store protocol
    backends.py    token store, skill source and approval handler protocols
    facade.py      the sync facade: LoopThread, sync, start_sync and the sync views
    errors.py, identifiers.py
  providers/       stub (no extra), bedrock, bedrock_mantle, anthropic, openai_compat
                   (OpenAI, X.AI and Microsoft Foundry, with entra.py shared by the
                   Foundry endpoints), gemini, ollama; builtin_factories()
  embeddings/      start_embeddings and EmbeddingRuntime; bedrock, openai_compat,
                   gemini, ollama and stub embedders; local/ (sentence-transformers,
                   behind local-embeddings); builtin_embedding_factories()
  agent/           runner (AgentRunner, AgentTask, AgentRun), budgets, contracts
                   (output contracts, one retry), untrusted (framing), hooks (output
                   filter, approvals mark, compaction, system and tool hooks), events,
                   results
  compaction/      compactor (the runner's compaction hook, forced compaction),
                   strategies (DropToolResults, KeepLastTurns, Summarise), conversation
                   (estimates, cut points, rendering), events, errors
  skills/          format (SKILL.md and resource paths), filesystem and memory sources,
                   disclosure (SkillSession), tool (use_skill), events, errors
  mcp/             settings (transports and authentication), client (McpConnection),
                   oauth (OAuthClient), signin (SignInHook, LoopbackSignIn), bridge
                   (McpToolBridge), events, errors; behind the mcp extra
  stores/          memory and json (transcript and token stores); sql/ (models,
                   stores, migration and the Alembic chain), behind the sql extra
  integrations/    konfig/ (load_settings, KonfigSecretResolver), warden/
                   (WardenAuditSink), otel/ (OtelEventSink), each behind its extra
  testing/         replay (recorded responses and scrub checks) and conformance/ (the
                   transcript store, token store and skill source suites, a pytest
                   plugin)
  cli/             the terminal interface, behind the cli extra
```

Hatchling reads `_version.txt` for the distribution metadata, and `conduit.__version__` reads it at import. The public API is whatever is importable from `conduit`, or from a layer's package (`conduit.agent`, `conduit.mcp`, `conduit.providers.bedrock`), without a leading underscore.

## 4. Dependency direction

```text
                                       cli
                                        |
   agent  compaction  skills  mcp  embeddings  stores  integrations  testing  providers
     \__________________________________|________________________________________/
                                        |
                                       core
```

- `conduit.core` imports nothing from any other conduit package, nor the top-level `conduit` package. Its only runtime dependency is pydantic.
- Upper layers import the core. Each is usable alone; none imports another upper layer, except that the embeddings adapters reuse their chat adapter's settings and client (`conduit.embeddings.bedrock` uses `conduit.providers.bedrock`), the bedrock-mantle adapter reuses the Bedrock presign step and the Anthropic request mapping, and the Anthropic adapter's Foundry endpoint reuses the OpenAI-compatible adapter's Entra ID support. The compaction layer reaches the runner only through the core's hook shapes, and the skills layer reaches it through `AgentTask.system_hook` and `tool_wrapper`.
- Each optional third-party library (a provider SDK, or a layer's own dependency) is imported only inside the layer of the extra that installs it, and only when a client is built or the layer is used. Without the extra installed, nothing else in conduit fails to import.
- Nothing outside `conduit.cli` imports `conduit.cli`; the terminal interface uses the public API only.
- No module constructs a provider client outside the provider registry, only the registry wraps a provider in metering, and no event carries content unless the content-capture policy allows it.

`tests/tripwires/` enforces these rules by scanning the source tree statically, so an import inside a function, a `TYPE_CHECKING` block or an `importlib.import_module` call with a literal name is caught whether or not the extra is installed. The tripwires are never skipped or weakened.

## 5. Start-up, the registry and metering

`start(settings, task_classes=, factories=, secrets=, events=)` is the one way providers are built for a product:

1. The settings' catalogue overlays are applied to the default catalogue (withdraw, override, add; adding an existing entry, or overriding or withdrawing an absent one, is refused).
2. Each configured provider's settings are re-validated against its adapter's own model. A credential field must hold a `secret://` reference; a plain-text credential, at any depth and under any credential-like name, is refused, and the refusal names the field, never the value.
3. Every assignment is resolved: the task class must be declared, the model declared in the catalogue for the provider (`UndeclaredModel` otherwise), offered in the provider's region, at or above the task class's floor, within the residency policy, and its output budget within the model's output limit. The result is immutable.
4. Only then are the providers built, each for the assignments resolved to it: the registry resolves the provider's secret references through the product's resolver, calls the adapter's `create_client` (nothing else in conduit may), and wraps the adapter in a `MeteredProvider`.

Every refusal is a `ConfigurationError` raised before any client exists. Nothing about a running task changes part-way through, so no floor or residency check can fail mid-task.

`MeteredProvider` is the single funnel every model call passes through. The adapter it wraps is private. Before any client call it refuses a model the catalogue does not declare, a request above the model's output limit, and anything outside the provider's resolved assignments (`ModelNotAssigned`): a request for a task class must name that class's model. A provider built without assignments (`ProviderRegistry.build` used directly) is held to the registry's residency policy on every call instead. It emits a `model_call_started` event and exactly one `model_call_finished` event for every call, however it ends (`ok`; or `failed` with the error class, `cancelled` or `abandoned`), carrying the final usage with all four counts. `MeteredEmbedder` does the same for embedding calls, and checks every response's vectors.

`start_embeddings` resolves embedding task classes (floors of kind `embedding`) from the same settings through the same registry. The sync facade's `start_sync` runs `start` on a loop conduit owns.

## 6. Residency

Each catalogue entry declares its residency scope: `local` (the deployment's own hosts), `in_region` (processed in the region the provider is called in, one of the entry's regions), `cross_region` (processed in any of the entry's regions, such as a Bedrock geographic inference profile) or `global` (anywhere the provider chooses). A `ResidencyPolicy` allows a set of scopes and optionally a set of regions:

- a scope outside the allowed set is refused;
- with regions set, a `global` entry is always refused, a non-local entry needs the provider's region set and allowed, and every region a `cross_region` entry may use must be allowed;
- `local` entries need no region.

The policy is checked when assignments are resolved at start-up, again when a provider is built for its assignments, and on every call of a provider built without assignments. Residency covers where a model provider processes a request; where transcripts and events are stored is decided by the product's backends.

## 7. Events and content capture

Every layer emits typed events through the product's event sink: model and embedding calls (the registry), runs, tool calls, approvals, hand-offs, budgets and output contracts (the runner), skills, MCP calls and compaction. An event's string fields are identifiers, which cannot hold prose; content (prompts, model output, tool arguments and results, summaries) appears only inside a `CapturedContent` field, which always carries the content's SHA-256 hash and size and carries its text only when the `CapturePolicy` allows that kind (`prompts`, `tool_content`; neither by default). A field declared as tool content is held to the tool-content setting whatever its emitter labelled it.

`FanOutSink` delivers each event to several sinks concurrently and enforces its own capture policy on the way through, so a sink never receives content its policy does not allow; a failing sink is logged by class name and never stops the others. The warden and otel sinks record metadata only, whatever the policy allows. The event-content tripwire checks every event class in conduit.

Errors follow the same rule: a conduit error never keeps the exception it was raised for (`__cause__` and `__context__` are `None`), and its message names fields, models and classes, never a value, a credential or provider text.

## 8. Providers

| Kind | Extra | Service | Authentication |
|---|---|---|---|
| `bedrock` | `bedrock` | Amazon Bedrock runtime, Converse and ConverseStream | AWS credential chain, access keys as secret references, short-term Bedrock API keys (generated and refreshed by conduit), long-term API keys only when `allow_long_term_api_keys` is set |
| `bedrock_mantle` | `bedrock-mantle` | Claude through Bedrock's mantle endpoint (Anthropic Messages API, single-Region) | Short-term or long-term Bedrock API keys, as for `bedrock` |
| `anthropic` | `anthropic` | Anthropic direct; Claude in Microsoft Foundry | API key; for Foundry an API key or Microsoft Entra ID |
| `openai_compat` | `openai` | OpenAI, X.AI, Microsoft Foundry's OpenAI v1 endpoint | API key; for Foundry an API key or Microsoft Entra ID |
| `gemini` | `gemini` | The Gemini API | API key |
| `ollama` | `ollama` | An Ollama server, loopback by default | none |
| `stub` | none | Deterministic, scriptable test provider | none; needs `not_for_production` |

Each adapter passes its SDK every setting the SDK could otherwise take from the environment (keys, endpoints, regions), refuses environment variables that would override conduit's settings or make the SDK log request content where it can, and documents the rest (D0037). Embedding adapters exist for `bedrock`, `openai_compat`, `gemini`, `ollama`, `stub` and, behind `local-embeddings`, `sentence_transformers`.

## 9. Extras

| Extra | Layer | Holds |
|---|---|---|
| `bedrock` | `conduit.providers.bedrock`, `conduit.embeddings.bedrock` | boto3 and botocore |
| `bedrock-mantle` | `conduit.providers.bedrock_mantle` | the Anthropic SDK's mantle client, and botocore (from boto3) to presign short-term keys |
| `anthropic` | `conduit.providers.anthropic` | the Anthropic SDK, and azure-identity for Microsoft Entra ID on Claude in Microsoft Foundry |
| `openai` | `conduit.providers.openai_compat`, `conduit.embeddings.openai_compat` | the OpenAI SDK and azure-identity for Microsoft Entra ID |
| `gemini` | `conduit.providers.gemini`, `conduit.embeddings.gemini` | the Google Gen AI SDK |
| `ollama` | `conduit.providers.ollama`, `conduit.embeddings.ollama` | httpx |
| `all-providers` | | every provider library above |
| `local-embeddings` | `conduit.embeddings.local` | sentence-transformers with CPU-only torch (install lines per installer in the providers guide) |
| `mcp` | `conduit.mcp` | the official MCP Python SDK |
| `sql` | `conduit.stores.sql` | SQLAlchemy, Alembic, asyncpg and aiosqlite |
| `konfig` | `conduit.integrations.konfig` | cognisn-konfig |
| `warden` | `conduit.integrations.warden` | cognisn-warden (a pre-release), whose audit recorder needs psycopg (LGPL-3.0, a named licence exception) |
| `otel` | `conduit.integrations.otel` | the OpenTelemetry API (the product supplies the SDK and exporters) |
| `cli` | `conduit.cli` | the terminal interface's dependencies |

The agent, compaction and skills layers, the stub provider, and the memory and JSON stores need no extra. Every runtime extra's dependency tree is held to the licence policy (`scripts/licence_audit.py --all-extras`); weak copyleft is accepted only as a named, reviewed exception in that script.
