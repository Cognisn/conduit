# conduit

One AI integration layer for Cognisn products.

conduit gives a Python product model providers, a model catalogue with residency policy, streaming, prompt caching, normalised usage, an agent runner with budgets and hand-off, compaction, skills, an MCP client, embeddings, product-supplied storage, and audit and monitoring events. It supplies the mechanisms; each product keeps its own policy (its prompts, guardrails, approvals and billing).

## Status

**Pre-release. Not yet published.** conduit 0.1.0 is under development on the `release/v0.1.0` branch and nothing is on PyPI yet. Every layer described here is built and tested; the public API may still change before 0.1.0 is released.

## Design in brief

- **Async first.** Every interface is async, and every provider streams. A supported sync facade runs calls on an event loop conduit owns.
- **Layered, with optional extras.** The core is always installed and depends on pydantic alone. Every provider SDK and every layer's own dependencies are optional extras, and dependencies point one way: upper layers depend on the core, never the reverse.
- **Checked at start-up.** The product declares its task classes with capability floors, assigns a provider and model to each, and states its residency policy. An undeclared model, a model below a floor, an assignment that would breach residency, or a credential written in plain text is refused at start-up, before any client exists, never part-way through a task.
- **One funnel for every model call.** Every call passes through the provider registry, which holds it to its assignment and emits exactly one finished event with its usage, however the call ends.
- **Usage you can bill.** Every usage figure carries input, output, cache-read and cache-write tokens separately.
- **Content stays out of events by default.** Events carry identifiers, hashes, sizes and outcomes; prompt and tool content is recorded only when the product opts in.
- **Storage-agnostic.** Products supply their own transcript store, event sink, skill source, secret resolver, approval handler and MCP token store, with a conformance test suite for each store and source. In-memory, local JSON file and SQL reference backends are provided.

## Requirements

Python 3.12 or later.

## Installation

Once 0.1.0 is published, the PyPI package is `cognisn-conduit` and the import name is `conduit`:

```sh
pip install "cognisn-conduit[bedrock]"
uv add "cognisn-conduit[anthropic,mcp]"
```

The core install and the stub provider need no extra. [Setting up providers](https://github.com/Cognisn/conduit/blob/main/docs/guides/providers.md) has install lines for each installer, including CPU-only torch for local embeddings.

### Extras

| Extra | Adds |
|---|---|
| `bedrock` | Amazon Bedrock provider (Converse; AWS credential chain, access keys as secrets, and Bedrock API keys) and Titan embeddings |
| `bedrock-mantle` | Single-Region Claude through Amazon Bedrock's mantle endpoint, with Bedrock API keys |
| `anthropic` | Anthropic provider (direct, and Claude in Microsoft Foundry with an API key or Microsoft Entra ID) |
| `openai` | OpenAI-compatible provider (OpenAI, X.AI and Microsoft Foundry with an API key or Microsoft Entra ID) and embeddings |
| `gemini` | Gemini API provider and embeddings |
| `ollama` | Ollama provider and embeddings |
| `all-providers` | Every provider above but `bedrock-mantle` (its two libraries are already in `all-providers`) |
| `local-embeddings` | Local embeddings with sentence-transformers on CPU-only torch |
| `mcp` | MCP client over the official SDK (stdio, streamable HTTP and SSE), with static credentials and an OAuth 2.1 client |
| `sql` | SQL transcript and token stores (SQLAlchemy; Postgres and SQLite) with their own schema and migrations |
| `konfig` | konfig settings loader and secret resolver |
| `warden` | Records conduit events in warden's audit trail (metadata only). 37 runtime packages with the core, including psycopg (LGPL-3.0, a reviewed licence exception) |
| `otel` | OpenTelemetry spans and metrics (the OpenTelemetry API; the product supplies the SDK and exporters) |
| `cli` | The `conduit` terminal interface demonstrating the public API (`uvx --from "cognisn-conduit[cli]" conduit`); installs rich and the `mcp` extra |

The agent runner, compaction, skills and the in-memory and JSON stores are part of the core install.

## A first call

This runs as it stands, against the stub provider:

```python
import asyncio

from conduit import ConduitSettings, LiteralSecretResolver, Message, TaskClass, start
from conduit.providers import builtin_factories

settings = ConduitSettings.model_validate(
    {
        "providers": {"local": {"kind": "stub", "not_for_production": True}},
        "assignments": {"triage": {"provider": "local", "model": "stub-echo"}},
    }
)


async def main() -> str:
    runtime = await start(
        settings,
        task_classes=[TaskClass(name="triage")],
        factories=builtin_factories(),
        secrets=LiteralSecretResolver({}),
    )
    try:
        binding = runtime.for_task("triage")
        response = await binding.provider.complete(
            binding.request(messages=[Message.user("Hello from example.com")])
        )
        return response.text
    finally:
        await runtime.aclose()


assert asyncio.run(main()) == "Hello from example.com"
```

To call a real model, configure its provider (for example `{"kind": "bedrock", "region": "eu-west-2"}`), assign one of its models, and use an `EnvironmentSecretResolver` or your own resolver for its credentials.

Import the core's names (types, settings, events, errors, tools, transcript entries and backend models) from `conduit` itself, and each layer's names from its package: `conduit.agent`, `conduit.stores`, `conduit.mcp`, `conduit.providers.bedrock` and so on. Importing `conduit` loads the core alone.

## Guides

- [Integrating conduit](https://github.com/Cognisn/conduit/blob/main/docs/guides/integration.md): settings, start-up, task classes, catalogue overlays, residency, events and capture, the agent runner and the sync facade.
- [Setting up providers](https://github.com/Cognisn/conduit/blob/main/docs/guides/providers.md) and [embeddings](https://github.com/Cognisn/conduit/blob/main/docs/guides/embeddings.md).
- [Writing backends](https://github.com/Cognisn/conduit/blob/main/docs/guides/backends.md), their conformance suites, and evaluating agent tasks.
- [Skills](https://github.com/Cognisn/conduit/blob/main/docs/guides/skills.md), [MCP servers](https://github.com/Cognisn/conduit/blob/main/docs/guides/mcp.md), [compaction](https://github.com/Cognisn/conduit/blob/main/docs/guides/compaction.md), [audit and telemetry](https://github.com/Cognisn/conduit/blob/main/docs/guides/observability.md) and [the terminal interface](https://github.com/Cognisn/conduit/blob/main/docs/guides/terminal.md).
- [Adoption notes](https://github.com/Cognisn/conduit/blob/main/docs/guides/adoption.md) for products moving onto conduit.

The full design is in [docs/design/architecture.md](https://github.com/Cognisn/conduit/blob/main/docs/design/architecture.md).

## Development

```sh
pip install -e ".[dev,all-providers,bedrock-mantle,cli,konfig,local-embeddings,mcp,otel,sql,warden]"
ruff format --check .
ruff check .
mypy --strict
pytest
python scripts/licence_audit.py
python scripts/licence_audit.py --all-extras
```

The default test run never calls a live model or service. Tests marked `integration` need live providers and are skipped by default. Every Python example in this README and in the guides is executed by `tests/docs/test_guides.py`.

## Licence

MIT. Copyright (c) 2026 Cognisn Pty Ltd. See [LICENSE](https://github.com/Cognisn/conduit/blob/main/LICENSE).
