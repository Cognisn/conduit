# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0b1] - 2026-11-06

Wave 6, third and last merge: the live provider suite (`test/0.1.0-live-providers`). Feature complete.

### Added

- A live provider suite (`tests/live`), marked `integration` and deselected by default. It covers:
  - streaming chat, tool loops with provider state, usage with cached second calls, and embeddings;
  - Bedrock (every authentication mode, and a residency refusal of a cross-region profile under an in-region policy), bedrock-mantle, Anthropic and Claude in Foundry, OpenAI, X.AI, Foundry (API key and Entra ID), Gemini, Ollama and local embeddings;
  - a verification pass over the default catalogue that reports each observable mismatch.
- Credentials for the live suite come only from `CONDUIT_LIVE_` variables through `EnvironmentSecretResolver`, and never appear in failure output. The default suite proves that every live test skips with its variables named, and proves every live check against the stub.

### Changed

- Catalogue 2026.11.7: the bedrock chat entries no longer declare structured output, because the adapter does not map an output schema to Converse yet. A capability floor that needs structured output now refuses them at start-up. bedrock-mantle is unchanged.

## [0.1.0a20] - 2026-11-06

Wave 6, second merge: guides and documentation (`docs/0.1.0-guides`).

### Added

- Guides in `docs/guides/`:
  - integrating conduit;
  - setting up each provider, with install lines for each installer and CPU-only torch;
  - embeddings;
  - writing backends, with their conformance suites;
  - skills;
  - MCP with OAuth sign-in;
  - compaction;
  - the warden and otel extras;
  - the terminal interface.
- Adoption notes for VOR, SPARK and Termograph.
- A test that runs every Python example in the README, the architecture document and the guides, and refuses any fence an example could escape the run through.

### Changed

- The README and the architecture document describe the library as built, including the bedrock-mantle and cli extras.

## [0.1.0a19] - 2026-11-05

Wave 6, first merge: the terminal interface (`feat/0.1.0-cli`).

### Added

- The terminal interface: the `conduit` command in the `cli` extra, run with `uvx --from "cognisn-conduit[cli]" conduit`.
  - `providers` lists the adapters, their extras and the configured providers with their authentication modes; `models` lists the catalogue with overlays.
  - `check` builds every configured provider (Bedrock authentication modes included) and resolves every assignment, showing each residency or capability refusal, without calling a model.
  - `mcp` connects the configured MCP servers, signing in with OAuth on a loopback callback.
  - `chat` streams a conversation with tool use under the run budget, skills, MCP tools, forced compaction with its size before and after, and per-call usage with cache reads and writes.
  - It runs on the stub provider without a configuration, uses conduit's public API only, keeps no conversation or token unless `--transcripts` or `--tokens` names a directory, and never prints a secret. Text from models, tools, servers and skills is shown as written, without letting it drive the terminal, and `--log-level` raises only conduit's own loggers.
- The code graph in `graphify-out/`.

### Changed

- The `cli` extra installs rich and the `mcp` extra.
- The import tripwire names `conduit.cli` as rich's only importer.

## [0.1.0a18] - 2026-11-05

Wave 5, third and last merge: audit and telemetry (`feat/0.1.0-audit-telemetry`). Milestone M4 (compaction and audit).

### Added

- The otel extra (`conduit.integrations.otel`): `OtelEventSink` turns conduit events into OpenTelemetry spans and metrics.
  - Spans cover runs, model and embedding calls, tool calls, MCP calls, approvals, skill loads, compactions and hand-offs.
  - Metrics cover all four token counts, cache hits, latency, errors, budget use, hand-offs and compactions, with attribute keys from a fixed low-cardinality set.
  - Attributes are metadata only: identifiers, counts, durations and the hash and size of content, never content.
  - It uses the product's tracer and meter providers, or the global ones. Telemetry failures are logged by class, never raised, and spans are always ended.
- The warden extra (`conduit.integrations.warden`): `WardenAuditSink` records conduit events in warden's hash-chained audit trail as registered `conduit.<event type>` actions, which `register_conduit_actions` registers.
  - Records carry metadata only, never content.
  - It needs cognisn-warden (a pre-release) and its PostgreSQL driver, psycopg.
- The event-content tripwire checks every event class in every layer under each capture policy. Capture-policy tests cover the runner, tools, MCP, skills and compaction through every sink and exporter.

### Changed

- The licence audit names psycopg (LGPL-3.0, required by cognisn-warden for the warden extra) as an agreed exception, matched by exact package name only.
- The architecture document says the otel extra holds the OpenTelemetry API only; the product supplies the SDK and exporters.

## [0.1.0a17] - 2026-11-05

Wave 5, second merge: compaction (`feat/0.1.0-compaction`).

### Added

- Compaction (`conduit.compaction`), the agent runner's compaction hook:
  - strategies drop old tool results, keep the last turns, or summarise through a product-declared task class, whose capability floor is checked at start-up;
  - each strategy is triggered at a share of the assigned model's context window. While the prompt fits, a compaction is applied only when it brings the prompt under its threshold. Once the prompt no longer fits, a compaction is applied whenever its result fits, and fails with a fixed reason when nothing can;
  - a compaction can be forced per session, and runs once even under concurrent calls.
- Cache-stable compaction: the system prompt, tools and loaded skills are never touched, and the summary follows the cached prefix with its own cache breakpoint, so cache reads continue after a compaction.
- Summary requests keep tool calls and results in tool blocks, so the capture policy's tool-content setting applies to them, and they carry only the tools those turns used.
- Compaction points written through the transcript store, with tokens before and after, reloading to the compacted conversation.
- Compaction events carrying strategy, trigger, outcome, tokens and entries replaced, with the summary only as captured prompt content.
- Compaction failures carry fixed reasons only, and leave a conversation that still fits unchanged.

### Changed

- CI's postgres job also runs the compaction store tests against a real Postgres.

## [0.1.0a16] - 2026-11-05

Wave 5, first merge: Bedrock key logging (`fix/0.1.0-bedrock-key-logging`).

### Changed

- Short-term Bedrock API keys for the bedrock and bedrock-mantle providers now come from one presign step, `conduit.providers.bedrock.auth.short_term_keys`. It presigns with botocore's SigV4 query signer, caps the key's lifetime at the credentials' expiry, refreshes within a margin, and generates the first key at start-up.
- The bedrock provider keeps its API keys out of botocore's DEBUG records, in both key modes. botocore.auth `Signature:` records are dropped, and botocore.endpoint records are turned into fixed text when they are logged. Both filters are installed once per process, even when clients are created from several threads. The bedrock auth module documents what a product gives up by removing them, and that botocore's own credential-refresh logging cannot be filtered, so products keep botocore above DEBUG in production.
- `SignatureRecordFilter`, `SIGNER_LOGGER` and `keep_signatures_out_of_logs` have moved from `conduit.providers.bedrock_mantle.auth` to `conduit.providers.bedrock.auth` (the last is now `keep_keys_out_of_logs`).
- bedrock-mantle's refusal when no AWS credentials are found now names `NoCredentialsError`, as the bedrock provider's does.
- The repository working agreement names the shared `devwork` Kvasir library for new project knowledge.

## [0.1.0a15] - 2026-11-05

Wave 4, sixth and last merge: the bedrock-mantle adapter (`feat/0.1.0-provider-bedrock-mantle`). Milestone M3 (agent runner, skills, MCP client and embeddings).

### Added

- The bedrock-mantle adapter (`conduit.providers.bedrock_mantle`, `bedrock-mantle` extra with anthropic and boto3). It serves single-Region Claude through the Anthropic Messages API on Amazon Bedrock's bedrock-mantle endpoint.
  - Short-term Bedrock API keys are generated from AWS credentials and refreshed before they, or their credentials, expire.
  - The key is sent as a bearer token set at send time on every attempt, and is kept out of every log record.
  - Long-term keys are refused unless explicitly allowed. That refusal, and the Region and endpoint checks, hold however the settings were made.
  - Requests, streaming, usage and errors are the anthropic adapter's.
  - Prompt caching is used only where the catalogue says the model caches.
- Catalogue 2026.11.6: in-region bedrock-mantle entries for Claude Opus 5.5, Sonnet 5.5, Haiku 5.5, Fable 5.1, Opus 5, Sonnet 5, Opus 4.8, Opus 4.7 and Haiku 4.5, six of them in ap-southeast-4 (Melbourne).

### Changed

- `builtin_factories()` includes the bedrock-mantle factory.
- The client-construction tripwire covers the bedrock-mantle client.

## [0.1.0a14] - 2026-11-05

Wave 4, fifth merge: embeddings (`feat/0.1.0-embeddings`).

### Added

- Embeddings:
  - an embedding interface in the core, and a registry funnel that checks the model, the dimensions and every vector. The funnel refuses non-numeric, boolean, out-of-range and non-finite vector values as a protocol failure;
  - embedding call events carrying input-token usage under the capture policy;
  - `start_embeddings`, which resolves embedding task classes at start-up, so a chat model cannot serve an embedding task class or the reverse, and refuses dimensions below one;
  - remote embedding providers for Bedrock Titan, OpenAI-compatible services, Gemini and Ollama, sharing the chat adapters' settings and authentication;
  - the `local-embeddings` extra, which runs sentence-transformers on CPU-only torch;
  - a stub embedder for tests.

### Changed

- CI installs torch (`>=2.15,<3`) from the PyTorch CPU index wherever it installs every extra, and refuses any NVIDIA package. tqdm is a named licence exception.

## [0.1.0a13] - 2026-11-05

Wave 4, fourth merge: the MCP client (`feat/0.1.0-mcp-client`).

### Added

- The `mcp` extra (`conduit.mcp`, MCP Python SDK 1.x), which holds:
  - `McpConnection`, for stdio, streamable HTTP and SSE servers, with connection lifecycle, timeouts and clean shutdown;
  - static bearer token and API key credentials as `secret://` references.
- An OAuth 2.1 client for MCP servers (`OAuthClient`):
  - protected-resource and authorisation-server metadata discovery, with audience and issuer checks, and an optional pinned issuer (required with a client secret);
  - dynamic client registration;
  - authorisation code with PKCE (S256), state and redirect URI checks, and the RFC 9207 issuer parameter;
  - refresh token rotation, with one refresh shared by every client of a token store in the process.

  Tokens and the client registration are held only in the product's token store, under `mcp:<server>` and `mcp:<server>:client`. They are bound to the issuer that granted them and are never sent to another one.
- The `SignInHook` protocol for the interactive sign-in, and `LoopbackSignIn` for terminals: a browser and a loopback callback. It completes only on a callback carrying the request's state, and bounds every wait.
- `McpToolBridge`: MCP tools join the tool catalogue as `<server>__<tool>`, restricted to the task classes the product grants. Results are framed as untrusted content with a fresh nonce, invisible characters made visible and a size cap.
- `McpCallEvent` (`mcp_call`), one per MCP tool call, with server, tool, outcome, duration, and hashes and sizes of the arguments and result. Their text appears only under the capture policy's tool-content setting.

## [0.1.0a12] - 2026-11-05

Wave 4, third merge: SKILL.md skills (`feat/0.1.0-skills`).

### Added

- SKILL.md skills (`conduit.skills`):
  - the SKILL.md format, read with the standard library: name, description and optional version, licence, compatibility, allowed tools and metadata. Validation refuses anything outside the format, including control and line-separator characters in single-line fields, with a fixed reason and line, never the file's text;
  - a filesystem skill source over a product-given directory, with every resource path confined to its skill's folder. Traversal, absolute paths, trailing-dot aliases, and links or junctions outside it are refused. Each file is checked again on its open handle, and every read is bounded by the source's limits on file sizes, counts and depth. There is also an in-memory source;
  - progressive disclosure: skill summaries listed for the model, and a `use_skill` tool for a tool catalogue that loads a skill's instructions on demand and its resources on further request. Limits on the loaded skills and the listing count the request prefix as it is rendered;
  - loaded skills placed in the stable request prefix before a cache breakpoint, so they are cached with the system prompt and tools;
  - skill-use transcript markers (loaded, used, unloaded), and skill events carrying names, versions and sizes only.
- A skill source conformance suite (`conduit.testing.conformance.SkillSourceSuite`).

## [0.1.0a11] - 2026-11-05

Wave 4, second merge: the agent runner (`feat/0.1.0-agent-runner`).

### Added

- The agent runner (`conduit.agent`): drives a task on its task class's resolved assignment through model calls and tool calls, as an async stream of typed run events, to a typed result. It can be used from sync code through the facade.
- Run budgets:
  - model calls, tool calls, tokens (the product chooses which of the four counts are charged) and wall time, each stopping the run cleanly with a budget event and a typed failure;
  - budgets hold across hand-offs, and the wall-time budget also bounds the hooks;
  - when output tokens are charged, no call asks for more output than the budget has left.
- Hand-off to another task class the task allows, only through the runtime's resolved assignments, by a `hand_off` tool offered to the model, with hand-off events.
- Output contracts: a pydantic model the final answer must satisfy, retried once with the validation errors (fields and types only), then a typed failure.
- Untrusted-content framing for tool results and other external text, with escaped delimiters, visible invisible characters and a fresh nonce per block. A tool can declare that its executor frames its own results (`preframed`).
- Approvals: tools a product marks go through the approval handler first, and a refusal is a tool result.
- An output filter hook, applied to every assistant message.
- Run events carrying identifiers, sizes, hashes and outcomes, with content only under the capture policy.
- Run transcripts appended through a supplied transcript store, keeping provider state. A run abandoned, cancelled or failing part-way still answers every tool use it left open, so the stored conversation stays loadable.
- Hooks: compaction before each model call, and a per-run system hook and tool-executor wrapper, which the skills layer uses.

## [0.1.0a10] - 2026-11-05

Wave 4, first merge: the providers follow-up (`fix/0.1.0-providers-follow-up`).

### Added

- Anthropic, Gemini and Bedrock round-trip their own provider state: Anthropic thinking and redacted thinking blocks, Gemini thought signatures and Bedrock reasoning content are reported as provider-state blocks in stream order and sent back unchanged in their assistant messages, as tool loops with reasoning need. Malformed own state, including state nested past the decoder's limit, is refused before any call. Every adapter ignores other providers' state, so a conversation can move between providers.
- Claude in Microsoft Foundry takes Microsoft Entra ID as well as an API key (`entra_id`, the same settings as the OpenAI-compatible adapter's Foundry service); the anthropic extra adds azure-identity.

### Changed

- Every adapter takes its capability checks from the catalogue the registry passes to `create_provider`, so product overlays apply; the Bedrock, Ollama and OpenAI-compatible factories no longer take a catalogue, and registering a shipped factory no longer warns. Gemini and Ollama refuse an output schema for a model the catalogue does not mark for structured output.
- Gemini keeps no memory of thought signatures; the `signature_memory` settings are removed.
- X.AI usage counts reasoning tokens as output when `total_tokens` shows they are outside `completion_tokens`.
- Bedrock Converse calls run on their own thread, so a timed-out call holds no executor thread. A ConverseStream worker whose consumer stops taking events gives up after the request's timeout and closes the stream at once.
- Closing a Foundry client that uses Entra ID logs a failing credential close by class name, and that failure never replaces the SDK client's own.

## [0.1.0a9] - 2026-11-05

Wave 3: the core follow-up (`fix/0.1.0-core-follow-up`).

### Added

- `ProviderStateBlock`: opaque provider state (provider kind, type and payload), never rendered as text, carried in assistant messages and responses. The `ProviderStateReported` stream event carries it, and `collect()` keeps it in stream order. The stub can emit state and keeps the stub state sent back to it, and the transcript store suite stores and loads it.
- `start_sync`, returning a `SyncRuntime` whose task bindings (`SyncTaskBinding`) and providers (`SyncProvider`) are blocking views, so sync code never holds an async object.
- `SettingsRefused`: provider settings that fail validation name the provider and the failing fields, never pydantic's text or an input value.

### Changed

- `create_provider` receives the registry's applied catalogue (product overlays included) as the keyword `catalogue`. A factory without it still works for this release, with a `DeprecationWarning` attributed to the code that registered it.
- The provider registry re-validates every settings instance it is given, including ones made with `model_construct` or changed with `model_copy(update=...)`.
- The generic `ProviderSettings` refuses a plaintext credential in an unknown field at any depth: under a credential-like name, or anything held by a key, secret, token, password or credential field, or as an HTTP authorisation value. Only the field is named. Numbers, booleans and null are never treated as credentials.
- An unknown provider kind is named in the error only when it is a well-formed kind.
- `MeteredProvider.aclose` logs an adapter close that raises, by class names only, and never raises it. Leaving a `SyncRuntime` block that is already raising logs a failing close the same way and lets the block's error propagate.
- Errors raised by the core, the stub and the replay helpers keep neither cause nor context. Catalogue and recording load errors name failing locations.
- The Anthropic and Bedrock adapters leave provider state out of their requests until they round-trip their own.

## [0.1.0a8] - 2026-11-04

Wave 2, seventh and last merge: the konfig integration (`feat/0.1.0-konfig`), and the Wave 2 close. Milestone M2 (providers and stores).

### Added

- The konfig integration (`conduit.integrations.konfig`, `konfig` extra with cognisn-konfig):
  - `load_settings` builds `ConduitSettings` (providers, assignments, residency, capture and catalogue overlays) from a konfig configuration section, keeping credentials as `secret://` references, and refuses a missing section;
  - `KonfigSecretResolver` resolves `secret://` references through konfig's stores (encrypted file, keyring or AWS Secrets Manager) in a worker thread, refusing empty, non-text and placeholder values;
  - errors name the field or reference, never a value, and carry no part of the failing exception.
- The `all-providers` extra installs every provider adapter's dependencies: Bedrock, Anthropic, OpenAI-compatible, Gemini and Ollama.


## [0.1.0a7] - 2026-11-04

Wave 2, sixth merge: the Gemini and Ollama adapters (`feat/0.1.0-provider-gemini-ollama`).

### Added

- The Gemini adapter (`conduit.providers.gemini`, `gemini` extra with google-genai):
  - an API key as a secret reference, with the key, backend and base URL pinned against the environment;
  - streaming text and function calls, with Gemini thought signatures remembered per call and returned on the next turn;
  - usage from the final chunk, with cached prompt tokens reported as cache reads;
  - cache breakpoints ignored, as documented in the adapter.
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
