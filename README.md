# conduit

One AI integration layer for Cognisn products.

conduit gives a Python product model providers, a model catalogue with residency policy, streaming, prompt caching, normalised usage, an agent runner with budgets and hand-off, compaction, skills, an MCP client, embeddings, product-supplied storage, and audit and monitoring events. It supplies the mechanisms; each product keeps its own policy (its prompts, guardrails, approvals and billing).

## Status

**Pre-alpha. Not yet published.** conduit 0.1.0 is under development and nothing is on PyPI yet. The repository currently holds the package layout, the tripwires that enforce its dependency direction, and CI; the layers are filled in waves on the `release/v0.1.0` branch. The public API will change without notice until 0.1.0 is released.

## Design in brief

- **Async first.** Every interface is async, and every provider streams. A supported sync facade runs calls on a loop conduit owns.
- **Layered, with optional extras.** The core is always installed and depends on nothing else in conduit. Every other layer and every provider SDK is an optional extra, and dependencies point one way: upper layers depend on the core, never the reverse.
- **Storage-agnostic.** Products supply their own transcript store, event sink, skill source, secret resolver, approval handler and MCP token store through interfaces, each with a conformance test suite. In-memory, local JSON file and SQL reference backends are provided.
- **Catalogue and residency.** A versioned default model catalogue that products overlay. An undeclared model is refused, and an assignment that would breach the deployment's residency policy is refused at start-up.
- **Usage you can bill.** Every usage event carries input, output, cache-read and cache-write tokens separately.
- **Content stays out of events by default.** Events carry identifiers, hashes, sizes and outcomes; prompt and tool content is recorded only when the product opts in.

The full design is in [docs/design/architecture.md](https://github.com/Cognisn/conduit/blob/main/docs/design/architecture.md).

## Requirements

Python 3.12 or later.

## Installation

Once 0.1.0 is published, the PyPI package is `cognisn-conduit` and the import name is `conduit`:

```
pip install "cognisn-conduit[bedrock]"
```

### Extras

| Extra | Adds |
|---|---|
| `bedrock` | Amazon Bedrock provider (AWS credential chain and Bedrock API keys) |
| `anthropic` | Anthropic provider (direct, and Claude in Microsoft Foundry) |
| `openai` | OpenAI-compatible provider (OpenAI, X.AI and Microsoft Foundry) |
| `gemini` | Google Gemini provider |
| `ollama` | Ollama provider |
| `all-providers` | Every provider above |
| `local-embeddings` | Local embeddings with sentence-transformers (CPU-only torch) |
| `mcp` | MCP client over the official SDK, with static credentials and an OAuth 2.1 client |
| `sql` | SQL storage backends (SQLAlchemy; Postgres and SQLite) with their own schema and migrations |
| `konfig` | konfig settings loader and secret resolver |
| `warden` | Records conduit events into warden's audit trail |
| `otel` | OpenTelemetry spans and metrics |
| `cli` | A terminal interface demonstrating the public API (`uvx --from "cognisn-conduit[cli]" conduit`) |

The core install and the stub provider need no extra.

## Development

```
pip install -e ".[dev]"
ruff format --check .
ruff check .
mypy --strict
pytest
python scripts/licence_audit.py
```

## Licence

MIT. Copyright (c) 2026 Cognisn Pty Ltd. See [LICENSE](https://github.com/Cognisn/conduit/blob/main/LICENSE).
