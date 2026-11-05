# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0a7] - 2026-11-04

Wave 2, sixth merge: the Gemini and Ollama adapters (`feat/0.1.0-provider-gemini-ollama`).

### Added

- The Gemini adapter (`conduit.providers.gemini`, `gemini` extra with google-genai):
  - an API key as a secret reference, with the key, backend and base URL pinned against the environment;
  - streaming text and function calls, with Gemini thought signatures remembered per call and returned on the next turn;
  - usage from the final chunk, with cached prompt tokens reported as cache reads;
  - cache breakpoints ignored, because Gemini caching is managed separately.
- The Ollama adapter (`conduit.providers.ollama`, `ollama` extra with httpx):
  - an httpx client with a loopback default, ignoring the environment's proxies and never following redirects;
  - `/api/chat` NDJSON streaming with stream lines bounded at 8 MiB; text, tool calls, and usage with cached prompt tokens as cache reads where Ollama reports them;
  - a call whose response finished is not failed by an error closing the body.
- Both adapters map every failure to a provider failure that carries no part of the failing exception, and a failing close never replaces a failure.
- Catalogue: four current Gemini models.
- The import tripwire allows a library several owning layers: httpx is owned by the Ollama adapter and the MCP layer, asyncpg and aiosqlite by the SQL stores, and google.genai is pinned to the Gemini adapter.

### Changed

- `builtin_factories()` includes the Gemini and Ollama factories.
- Catalogue: `nomic-embed-text`'s context window is 2048, and the catalogue's version is 2026.11.5.

### Removed

- Catalogue: Gemini 2.5 Pro and Gemini 2.5 Flash, which Google has deprecated.


## [0.1.0a6] - 2026-11-04

Wave 2, fifth merge: the OpenAI-compatible adapter (`feat/0.1.0-provider-openai-compat`).

### Added

- The OpenAI-compatible adapter (`conduit.providers.openai_compat`, `openai` extra with openai and azure-identity), for OpenAI, X.AI and Microsoft Foundry:
  - Foundry takes an API key or Microsoft Entra ID.
  - Streaming Chat Completions with usage on the final chunk; tool calls assembled from argument deltas; images and tool results.
  - Structured output where the catalogue allows it.
  - Cached prompt tokens reported as cache reads.
  - Errors, including malformed stream chunks and stream close failures, mapped to provider failures with no provider text and no chained exception.
  - Cache breakpoints are ignored, because caching on these services is automatic.
- Catalogue: grok-4.3 on X.AI; GPT-6 Astra and GPT-6.1 Sol on OpenAI.
- Licence audit: certifi (MPL-2.0), the CA bundle httpx and requests require, admitted as a named exception, used unmodified and installed separately (Q0016).

### Changed

- `builtin_factories()` includes the OpenAI-compatible factory.
- The default catalogue's version is 2026.11.4.

### Removed

- Catalogue: the retired grok-4 entry (X.AI now serves grok-4.3 in its place).


## [0.1.0a5] - 2026-11-04

Wave 2, fourth merge: the Anthropic adapter (`feat/0.1.0-provider-anthropic`), and the Bedrock error-chain follow-up.

### Added

- The Anthropic adapter (`conduit.providers.anthropic`, `anthropic` extra) for the direct API and Claude in Microsoft Foundry:
  - streaming text, and tool use assembled from input JSON deltas, reported once a call is known to be whole;
  - all four usage counts, taken from the final message events;
  - cache control from breakpoints, within the four-breakpoint limit;
  - images, tool results, stop reasons and structured output;
  - failures mapped to provider failures that carry no part of the failing exception.
  - Keys are secret references. The client takes its key, endpoint and headers only from conduit's settings, and refuses SDK environment variables that would override them or log request content.

### Changed

- Default catalogue: Claude Sonnet 4.6's output limit is 128K on the Anthropic API, Claude Fable 5.1 is added, and the catalogue's version is 2026.11.3.
- `builtin_factories()` includes the Anthropic factory.
- Bedrock provider and configuration errors carry no part of the failing exception: no cause and no context. A stream event or Converse response whose fields have the wrong type is a protocol failure.

## [0.1.0a4] - 2026-11-04

Wave 2, third merge: the Amazon Bedrock adapter (`feat/0.1.0-provider-bedrock`), and the stores follow-up (Q0018).

### Added

- The Amazon Bedrock adapter (`conduit.providers.bedrock`, `bedrock` extra with boto3):
  - Converse and ConverseStream in worker threads, reading at most a bounded number of stream events ahead.
  - The AWS credential chain with an optional profile, access keys as secret references, short-term Bedrock API keys generated and refreshed before expiry, and long-term keys refused unless explicitly allowed, however the settings were made.
  - Cache points where the catalogue says the model caches; usage with all four counts; tool use; errors mapped to provider failures with no provider text.
- Catalogue: in-region Claude Sonnet 4.6 (eu-west-2) and Claude Sonnet 5 (eu-west-2, ap-northeast-2, ap-southeast-1) on Bedrock, and the Sonnet 5 geographic and global profiles.

### Changed

- `builtin_factories()` includes the Bedrock factory.
- The default catalogue's version is 2026.11.2.
- The SQL stores refuse a stored row they cannot decode with `StoreError`, refuse a session id outside 1 to 256 characters with an error that does not repeat it, and name the driver's error class in `StoreError`. A JSON store write whose directory cannot be synced after the replace is no longer reported as failed (Q0018).

## [0.1.0a3] - 2026-11-04

Wave 2, second merge: JSON and SQL stores (`feat/0.1.0-stores`).

### Added

- JSON file transcript and token stores (`conduit.stores.json`): atomic LF writes, the directory synced on POSIX, per-record locks across tasks, event loops and threads, and refusal of path-like ids and keys.
- The `sql` extra (`conduit.stores.sql`): Postgres and SQLite transcript and token stores, with conduit's own `conduit` schema on Postgres, its own reversible Alembic chain and version table, and `migrate` for products to call.
- `StoreError` (`conduit.stores.errors`): store failures name the store, the operation and the error class, and never carry ids, keys, content or token values.
- `TokenStoreSuite` in `conduit.testing.conformance`; the token and transcript suites pass on the memory, JSON, SQLite and Postgres stores.
- A Postgres CI job, and a Postgres service and Compose template in the worktree recipe.

### Changed

- CI's types and test jobs install every runtime extra the package declares.

## [0.1.0a2] - 2026-11-04

Wave 2, first merge: core hardening (`fix/0.1.0-core-hardening`).

### Added

- `ToolContent` and `CaptureAs` (`conduit.core.events`), for declaring tool-content fields on events.
- `ModelNotAssigned` (`conduit.core.errors`), raised when a call names a model or task class outside the provider's resolved assignments.
- `ModelNotAssigned`, `ToolContent` and `CaptureAs` are re-exported from the top-level `conduit` package.
- The client-construction tripwire covers more SDK construction forms (aiobotocore, `service_name=`, session `create_client`, `getattr` construction) and requires that only the registry wraps a provider in metering.

### Changed

- A provider built at start-up now serves only the task-class assignments resolved for it. A call naming another model, or another task class's model, is refused with `ModelNotAssigned` before any client call. A provider built from the registry with an assignment has each assigned model checked against the registry's residency policy, and one built without is held to that policy (new `residency` argument) on every call.
- Every model call is refused before any client call if its `max_output_tokens` exceeds the model's output limit (`ConfigurationError`), whether or not it was built with `TaskBinding.request`.
- `MeteredProvider` no longer exposes the provider adapter it wraps, so every call goes through admission and metering.
- Every model call now emits exactly one finished event however it ends:
  - cancelled calls and abandoned streams are recorded as failed (`cancelled` or `abandoned`) with the last usage reported;
  - a stream its consumer closes after a complete stop is recorded as ok;
  - the provider's stream is closed before the event.
- The sync facade applies its timeouts on the loop, so a timed-out call has finished its cleanup before `TimeoutError` is raised. A `TimeoutError` raised by the call itself passes through unchanged. Closing the facade's loop cancels pending calls and finalises suspended async generators, and every call it could not finish fails with `FacadeError`, so no caller is left waiting. Closing is refused from the loop's own thread.
- The in-memory transcript store refuses an append that would leave the session unloadable, such as a compaction point replacing later entries. The transcript store conformance suite requires this of every store.
- Tool arguments and results in events are always treated as tool content under the capture policy, whatever kind the emitter labelled them.
- `TaskBinding.request` validates the request and refuses an explicit `max_output_tokens` above the model's output limit.
- The recorded-response scrub check also refuses IPv6 addresses outside 2001:db8::/32 and any 12-digit number other than the documentation account id.

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
