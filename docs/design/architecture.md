# conduit architecture

This document records conduit's standing design decisions, its package layout, the direction its dependencies may point, and its extras. It is versioned with the code. The plan of record, with the reasoning behind each decision, is held in the project's knowledge base as "2026-11-04 plan conduit 0.1.0 development" and "2026-11-04 decision conduit library scope and design rulings (C1 to C19)". Where this document and those records disagree, the records win and this document is corrected.

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

```
src/conduit/
  core/            message, content block, tool and stream event types; normalised
                   response; usage events (input, output, cache read, cache write);
                   provider protocol; settings models; catalogue loader, overlays and
                   default data; task classes and floors; residency policy; errors;
                   secret resolver, event sink, approval handler, transcript store and
                   token store protocols; event fan-out and content-capture policy;
                   cache breakpoint model; sync facade
  providers/       bedrock, anthropic, openai_compat (openai, xai, microsoft), gemini,
                   ollama, stub (each SDK behind its own extra)
  embeddings/      interface, remote providers; local/ behind the local-embeddings extra
  agent/           runner, budgets, hand-off, output contracts with one retry,
                   untrusted-content framing, output filter hook, tool registry use
  compaction/      strategies, cache-stable summaries, transcript integration
  skills/          SKILL.md format and validation, skill source protocol, filesystem
                   source, use_skill progressive disclosure
  mcp/             client over the official SDK (stdio, streamable HTTP, SSE), static
                   credentials, OAuth 2.1 client, MCP tools bridged into the registry
  stores/          memory and JSON file backends; sql/ behind the sql extra (schema
                   "conduit", Alembic chain)
  integrations/    konfig/ (loader, secret resolver), warden/ (audit adapter),
                   otel/ (spans and metrics), each behind its extra
  testing/         stub and recorded-response replay, conformance suites as a pytest
                   plugin, evaluation harness
  cli/             terminal interface behind the cli extra
```

The version lives only in `src/conduit/_version.txt`. Hatchling reads it for the distribution metadata, `conduit.__version__` reads it at import, and only `scripts/stamp_version.py` writes it.

## 4. Dependency direction

```
            cli
             |
   agent  compaction  skills  mcp  embeddings  stores  integrations  testing  providers
             \___________________________|___________________________/
                                         |
                                        core
```

- `conduit.core` imports nothing from any other conduit package, nor the top-level `conduit` package. It takes no runtime dependency beyond those agreed in the plan.
- Upper layers may import the core. A layer that needs another upper layer says so in its workstream's kick-off, and the tripwires are extended to match.
- Each optional third-party library (a provider SDK, or a layer's own dependency) is imported only inside the layer of the extra that installs it. Without the extra installed, nothing else in conduit fails to import.
- Nothing outside `conduit.cli` imports `conduit.cli`; the terminal interface uses the public API only.
- No module constructs a provider client outside the provider registry, and no event carries content unless the content-capture policy allows it (tripwires added by the core workstream).

`tests/tripwires/` enforces these rules by scanning the source tree statically, so an import inside a function, a `TYPE_CHECKING` block or an `importlib.import_module` call with a literal name is caught whether or not the extra is installed. The tripwires are never skipped or weakened.

## 5. Extras

| Extra | Layer | Holds |
|---|---|---|
| `bedrock` | `conduit.providers.bedrock` | boto3 and botocore |
| `anthropic` | `conduit.providers.anthropic` | the Anthropic SDK, and azure-identity for Microsoft Entra ID on Claude in Microsoft Foundry |
| `openai` | `conduit.providers.openai_compat` | the OpenAI SDK and Microsoft Entra ID support |
| `gemini` | `conduit.providers.gemini` | the Google Gen AI SDK |
| `ollama` | `conduit.providers.ollama` | the Ollama client |
| `all-providers` | | every provider extra above |
| `local-embeddings` | `conduit.embeddings.local` | sentence-transformers with CPU-only torch |
| `mcp` | `conduit.mcp` | the official MCP Python SDK |
| `sql` | `conduit.stores.sql` | SQLAlchemy and Alembic |
| `konfig` | `conduit.integrations.konfig` | cognisn-konfig |
| `warden` | `conduit.integrations.warden` | cognisn-warden |
| `otel` | `conduit.integrations.otel` | the OpenTelemetry API (the product supplies the SDK and exporters) |
| `cli` | `conduit.cli` | the terminal interface's dependencies |

Every extra is declared empty in `pyproject.toml` and filled by the workstream that builds its layer. Every runtime extra's dependency tree is held to the licence policy (`scripts/licence_audit.py --all-extras`).
