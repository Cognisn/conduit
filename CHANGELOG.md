# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Repository scaffold: README, MIT licence, project working agreement (`CLAUDE.md`), changelog, `.gitattributes` and `.gitignore`.
- `pyproject.toml` (hatchling, Python 3.12 or later) with the version read from `src/conduit/_version.txt`, no runtime dependencies, and every planned extra declared empty: `bedrock`, `anthropic`, `openai`, `gemini`, `ollama`, `all-providers`, `local-embeddings`, `mcp`, `sql`, `konfig`, `warden`, `otel` and `cli`.
- The `conduit` package and its empty layers: `core`, `providers`, `embeddings` (with `local`), `agent`, `compaction`, `skills`, `mcp`, `stores` (with `sql`), `integrations` (`konfig`, `warden`, `otel`), `testing` and `cli`.
- Import-direction tripwires in `tests/tripwires/`: `conduit.core` imports no other conduit package, each optional third-party library is imported only inside the layer of its extra, and nothing outside `conduit.cli` imports it.
- CI: Ruff, `mypy --strict`, pytest on Linux, macOS and Windows for Python 3.12 and 3.13, a distribution build, and the licence audit.
- `scripts/stamp_version.py`, which stamps a version only on its own `release/vX.Y.Z` branch, and `scripts/licence_audit.py`, the licence policy (no strong copyleft; weak copyleft only as a named exception) for the core and for each extra.
- `dev/worktree.json`, the worktree recipe for the workstation setup script.
- Design documents: `docs/design/architecture.md` and `docs/design/stack.md`.
