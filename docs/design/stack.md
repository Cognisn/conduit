# conduit stack and shared modules

## Stack

| Concern | Choice |
|---|---|
| Language | Python 3.12 or later (CI on 3.12 and 3.13) |
| Build | hatchling; version from `src/conduit/_version.txt` |
| Lint and format | Ruff (`ruff format`, `ruff check`) |
| Types | mypy `--strict` over `src`, `scripts` and `tests` |
| Tests | pytest with pytest-asyncio (`asyncio_mode = "auto"`); `integration` marker for live services, skipped by default |
| Licence policy | `scripts/licence_audit.py`: no strong copyleft; weak copyleft only as a named, reviewed exception; the core and every runtime extra audited |
| CI | GitHub Actions: lint, types, tests on Linux, macOS and Windows for 3.12 and 3.13, distribution build, licence audit |
| Runtime dependencies | The core depends on pydantic (MIT) alone; each other layer's dependencies arrive in its extra |

## Shared modules

These paths are shared across workstreams. Each is edited only by the session its kick-off prompt assigns it to; any other session that needs a change records the need (as a Skuld question and in its session summary) and stops on that point.

| Path | Why it is shared |
|---|---|
| `pyproject.toml` | Dependencies, extras, version source and tool configuration for every layer. |
| `src/conduit/core/` | The contracts every other layer builds on (owned by `feat/0.1.0-core` in Wave 1; shared once merged). |
| `src/conduit/__init__.py` | The package's top-level public surface. |
| `tests/tripwires/` | The dependency-direction and safety tripwires; never skipped or weakened. |
| `.github/` | CI and, later, the release workflow. |
| `dev/` | The worktree recipe and any sandbox Compose template. |
| `docs/design/` | This document and the architecture, against which the code is built. |

`CHANGELOG.md` is written by the integration session from the changelog lines each workstream records in its session summary.
