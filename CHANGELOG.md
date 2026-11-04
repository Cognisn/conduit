# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0a1] - 2026-11-04

Wave 1: the core contracts every later layer builds on.

### Added

- Core types (`conduit.core.types`, `conduit.core.streaming`):
  - messages and content blocks (text, image, tool use, tool result, cache breakpoint), tool specifications and the normalised response;
  - typed stream events, ending with a stop event that carries the call's final usage;
  - usage with input, output, cache-read and cache-write tokens kept separate and each required.
- The provider protocol (`conduit.core.provider`), async and streaming first, with a non-streaming call defined as the collected stream.
- The provider registry (`conduit.core.registry`), the only place provider clients are made. It validates settings, resolves secret references, refuses undeclared models and records every model call as an event with its usage.
- Typed settings (`conduit.core.settings`, `conduit.core.runtime`), buildable in code or from a mapping, with credentials only as `secret://` references, and environment and literal secret resolvers.
- The model catalogue (`conduit.core.catalogue`):
  - a versioned default for Anthropic, Amazon Bedrock (from the Bedrock model cards), OpenAI, X.AI, Gemini, Ollama and the stub;
  - product overlays that add, override or withdraw entries;
  - refusal of undeclared models.
- Product-declared task classes with capability floors, and residency policy, both enforced when assignments are resolved at start-up. A cross-region inference profile under an in-region policy is refused.
- Events (`conduit.core.events`), with fan-out to several sinks and a content-capture policy: identifiers, hashes, sizes and outcomes by default; prompt and tool content only when the product opts in.
- The transcript model (compaction points and skill-use markers) and protocols for the transcript store, token store, skill source and approval handler; in-memory transcript and token stores (`conduit.stores.memory`).
- The tool protocol, tool catalogue and per-task tool registries (`conduit.core.tools`).
- The sync facade (`conduit.core.facade`), running calls on a loop thread conduit owns, with timeouts, and refusing to run on that loop's own thread.
- A deterministic, scriptable stub provider (`conduit.providers.stub`), including reactive scripts that answer tool results and cache-token accounting. It is marked not for production.
- Recorded-response replay with a scrubbing check (`conduit.testing.replay`).
- Conformance suites as a pytest plugin (`conduit.testing.conformance`), starting with the transcript store suite.
- Tripwires: no provider client constructed outside the registry; no event carries content the capture policy does not allow.
- The core's main types, the registry, start-up and the sync facade are re-exported from the top-level `conduit` package.

### Changed

- The core install depends on pydantic (MIT), its only runtime dependency.

## [0.1.0a0] - 2026-11-04

Opening stamp of the 0.1.0 release branch: the scaffold, with no library behaviour yet.

### Added

- Repository scaffold: README, MIT licence, project working agreement (`CLAUDE.md`), changelog, `.gitattributes` and `.gitignore`.
- `pyproject.toml` (hatchling, Python 3.12 or later) with the version read from `src/conduit/_version.txt`, no runtime dependencies, and every planned extra declared empty: `bedrock`, `anthropic`, `openai`, `gemini`, `ollama`, `all-providers`, `local-embeddings`, `mcp`, `sql`, `konfig`, `warden`, `otel` and `cli`.
- The `conduit` package and its empty layers: `core`, `providers`, `embeddings` (with `local`), `agent`, `compaction`, `skills`, `mcp`, `stores` (with `sql`), `integrations` (`konfig`, `warden`, `otel`), `testing` and `cli`.
- Import-direction tripwires in `tests/tripwires/`: `conduit.core` imports no other conduit package, each optional third-party library is imported only inside the layer of its extra, and nothing outside `conduit.cli` imports it.
- CI: Ruff, `mypy --strict`, pytest on Linux, macOS and Windows for Python 3.12 and 3.13, a distribution build, and the licence audit.
- `scripts/stamp_version.py`, which stamps a version only on its own `release/vX.Y.Z` branch, and `scripts/licence_audit.py`, the licence policy (no strong copyleft; weak copyleft only as a named exception) for the core and for each extra.
- `dev/worktree.json`, the worktree recipe for the workstation setup script.
- Design documents: `docs/design/architecture.md` and `docs/design/stack.md`.
