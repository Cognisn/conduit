# conduit guides

| Guide | Covers |
|---|---|
| [Integrating conduit](integration.md) | Settings, start-up, task classes and floors, catalogue overlays, residency, calling a model, prompt caching, events and the capture policy, the agent runner, the sync facade, errors and logging |
| [Setting up providers](providers.md) | Install lines for each installer (including CPU-only torch), Amazon Bedrock and its authentication modes and residency, the bedrock-mantle endpoint, Anthropic direct and Microsoft Foundry, OpenAI, X.AI and Foundry with Microsoft Entra ID, Gemini, Ollama, the stub |
| [Embeddings](embeddings.md) | Embedding task classes, remote providers and local embeddings |
| [Writing backends](backends.md) | Each product-supplied interface, and running its conformance suite |
| [Skills](skills.md) | `SKILL.md`, skill sources, progressive disclosure with `use_skill`, and the cached prefix |
| [MCP servers](mcp.md) | Configuring and connecting MCP servers, bridging their tools, and OAuth sign-in |
| [Compaction](compaction.md) | Strategies, thresholds, forced compaction, the stable cached prefix and transcripts |
| [Audit and telemetry](observability.md) | The warden and otel extras |
| [The terminal interface](terminal.md) | The `cli` extra's demonstration interface |
| [Adoption notes](adoption.md) | Moving VOR, SPARK and Termograph onto conduit |

The design itself is in [docs/design/architecture.md](../design/architecture.md).

Every Python example in these guides runs as written: `tests/docs/test_guides.py` executes them against the stub provider, local stores and in-process servers, with no network and no credentials.
