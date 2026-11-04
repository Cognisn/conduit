# conduit

# Project configuration
- Kvasir library: conduit
- Issue tracker: github
- GitHub repository: Cognisn/conduit
- Default branch: main
- Release branch style: release/v<version>
- Workstream branches: <kind>/<version>-<aspect>, cut from the release branch; the version never carries a `v` (for example `feat/0.1.0-core`)
- Merge to main: squash, only when the maintainer instructs it
- Commit scopes: core, providers, embeddings, agent, compaction, skills, mcp, stores, integrations, testing, cli, docs, ci, dev, repo
- Shared modules: pyproject.toml, src/conduit/core/, src/conduit/__init__.py, tests/tripwires/, .github/, dev/, docs/design/
- Worktree recipe: dev/worktree.json (no Compose template yet)

This file is the project-level working agreement for conduit. It sits under the workstation's global working agreement on any Cognisn development workstation; where the two overlap, the global agreement governs process (branching, versioning, commit format, worktrees, ports, Kvasir protocols, tagging) and this file governs what is specific to conduit. On a machine without the global agreement (for example a cloud session), the rules in "Minimum rules without the global agreement" still apply.

Project knowledge lives in the mwh-kvasir library `conduit`; pass `library: "conduit"` on every call. The active development plan, decisions and per-workstream session summaries are there. Releases are tracked in Skuld (project `conduit`).

## What conduit is

conduit is a reusable Python library that gives every Cognisn product one AI integration layer: model providers, a model catalogue with residency policy, streaming, prompt caching, normalised usage, an agent runner with budgets and hand-off, compaction, skills, an MCP client, embeddings, product-supplied storage, and audit and monitoring events. It supplies mechanisms; each product keeps its own policy (prompt layers, guardrails, approvals, credit ledgers, conversation management).

It is published publicly on PyPI as `cognisn-conduit` (import name `conduit`) under MIT.

`docs/design/architecture.md` is the authoritative statement of design and scope in the repository, and `docs/design/stack.md` lists the stack and the shared modules. Read both at the start of any session that touches behaviour.

## This is a public repository

Everything committed here is world-readable, including history.

- Never commit client names, client data, hostnames, internal paths, deployment addresses, credentials, keys, or anything received under NDA.
- Never commit internal business, commercial or legal discussion. Record that in Kvasir.
- Test fixtures are synthetic. Example domains use `example.com`, `example.org` and `example.net`; example addresses use the documentation ranges (`192.0.2.0/24`, `198.51.100.0/24`, `203.0.113.0/24`, `2001:db8::/32`). Recorded provider responses are synthetic or scrubbed of every account, key, region-specific identifier and real prompt.
- A security defect is never described in a commit message, public issue or issue title before it is fixed and released. Record it in Kvasir.

## Code origins

- **No Termograph code is ever copied into conduit.** Not a function, a prompt, a test or a data file. Do not open a Termograph clone from a conduit session. Termograph is a reference for lessons only, and those lessons reach conduit sessions written into their kick-off prompts. Termograph's routing classifier stays in Termograph, above conduit.
- **VOR and SPARK are Cognisn code.** VOR's AI layer is the seed and SPARK's LLM layer the second source; their code may be carried across and generalised when a kick-off prompt says so.
- **Reference clones are read-only.** VOR and SPARK are read from reference clones under `C:\Reference` (never from or as main clones). Never commit, build, install, run tests or run anything else in `C:\Reference`; read only. Before reading, fetch and check that the branch the kick-off names matches its origin.

## Architecture decisions

The full list is section 2 of `docs/design/architecture.md` (C1 to C20). These are standing decisions. Do not reopen them in a session; if a task cannot be completed within them, stop and say so.

- **Mechanism, not policy (C1).** conduit never hardcodes a product's prompts, guardrails, task classes, approvals or billing.
- **Layered, one way (C2).** The core is always installed; every other layer and every provider SDK is an optional extra. Upper layers depend on the core, never the reverse, and each layer is usable alone.
- **Async first (C3).** Every interface is async. The sync facade runs calls on a loop conduit owns and refuses to run on that loop's own thread.
- **Storage-agnostic (C11).** Products supply transcript store, event sink, skill source, secret resolver, approval handler and MCP token store; each interface ships a conformance suite.
- **Catalogue and residency (C18).** An undeclared model is refused. Capability floors and residency policy are enforced at start-up, never part-way through a task. No learned routing in conduit.
- **Usage and caching (C19).** Every usage event carries input, output, cache-read and cache-write tokens separately; usage is taken from the final stream event.
- **Content capture (C19).** Events carry identifiers, hashes, sizes and outcomes by default; prompt and tool content only when the product opts in.
- **Configuration (C15).** Typed settings models, buildable in code or from a mapping; credentials only as secret references. konfig is an optional extra. conduit logs through `logging` under the `conduit` logger.

## Package layout and dependency direction

```
src/conduit/
  core/          contracts: types, provider protocol, settings, catalogue, task classes,
                 residency, protocols for every product-supplied backend, events, sync facade
  providers/     bedrock, anthropic, openai_compat, gemini, ollama, stub
  embeddings/    interface and remote providers; local/ (local-embeddings extra)
  agent/         runner, budgets, hand-off, output contracts
  compaction/    strategies and cache-stable summaries
  skills/        SKILL.md skills and progressive disclosure
  mcp/           MCP client (mcp extra)
  stores/        memory and JSON backends; sql/ (sql extra)
  integrations/  konfig/, warden/, otel/, each behind its extra
  testing/       stub replay, conformance suites (pytest plugin), evaluation harness
  cli/           terminal interface (cli extra)
```

Rules, enforced by `tests/tripwires/` and never skipped or weakened:

- `conduit.core` imports nothing from any other conduit package, nor the top-level `conduit` package.
- Each optional third-party library (provider SDK or layer dependency) is imported only inside the layer of its extra. The table is `OPTIONAL_LIBRARIES` in `tests/tripwires/importscan.py`.
- Nothing outside `conduit.cli` imports `conduit.cli`.
- From Wave 1: no module constructs a provider client outside the provider registry; no event carries content unless the content-capture policy allows it.

## Shared modules

`pyproject.toml`, `src/conduit/core/`, `src/conduit/__init__.py`, `tests/tripwires/`, `.github/`, `dev/` and `docs/design/` are shared. Each is edited only by the session its kick-off prompt assigns it to. Any other session that needs a change there records the need as a Skuld question and in its session summary, and stops on that point.

## Development conventions

- Australian English in code comments, documentation, error messages and commit messages. Cyber Security is written as two words with capitals in prose. No long dashes in prose.
- No AI attribution anywhere: no `Co-Authored-By` trailers, no "generated with" footers, no session links in commits, pull requests, changelogs, code or documents.
- Branching follows the global agreement. `main` holds only the current released version (before the first release, the `0.1.0.dev0` scaffold). All work for a release happens on `release/v<version>`, through workstream branches cut from it and merged back by the integration session. Pre-release suffixes appear only on release branches. A release reaches `main` as a squash merge, only when the maintainer instructs it.
- Conventional Commits: `type(scope): lower-case imperative summary`, with the scopes in the project configuration above. A release stamp commit takes no scope: `chore: stamp 0.1.0a1`.
- Keep a Changelog: `CHANGELOG.md` under `[Unreleased]`. Workstream sessions do not edit it; they record their changelog lines in their session summary and the integration session writes them at merge.
- The version lives only in `src/conduit/_version.txt`, written only by `python scripts/stamp_version.py <version>` in the integration session on the release branch. Worktree sessions never bump the version.
- The public API is whatever is importable from `conduit` without a leading underscore. Anything else is internal.
- Python 3.12 or later. Do not add compatibility code for older versions.
- Type hints on every function; `mypy --strict` passes on `src`, `scripts` and `tests`.
- The core takes no runtime dependency beyond those the plan agrees. A new runtime dependency anywhere, including in an extra, needs the maintainer's agreement in the session and a licence check (`scripts/licence_audit.py`, with `--extra <name>` for an extra). No strong copyleft; weak copyleft only as a named, reviewed exception in `scripts/licence_audit.py`.
- Scripts that write files pass `newline="\n"`; the repository is `eol=lf`.

## Testing

- Write the failing test first, and watch it fail for the right reason.
- The default test run never calls a live model, a live MCP server or the network. Providers are tested against recorded responses and the stub; anything needing a live service is marked `integration` (skipped by default) and runs only on the maintainer's word.
- Concurrency claims are proven with genuinely concurrent tasks, not by reading the code.
- Every guard and tripwire is mutation-checked: break the property it protects, watch a test fail, restore the file byte-identically (compare sha256), and record the mutation and its failing count in the session summary. A guard that cannot fail is not a guard.
- Never weaken an assertion to make a test pass. Re-point it only when the behaviour has legitimately changed, and say so in the commit message.
- Database tests (the `sql` extra) run against a real Postgres through testcontainers, which counts against the workstation's heavy-workload cap.

## Integration procedure

The integration session works in the main clone on the release branch and merges workstreams one at a time:

1. Check the workstream tip's CI conclusion (`gh run list --branch <branch> --limit 1`). Never merge a red tip.
2. Merge with a merge commit (`git merge --no-ff`).
3. Run the full suite and every tripwire, then the checks (ruff format and check, `mypy --strict`, licence audit with and without `--all-extras`), each as its own command.
4. Write the workstream's changelog lines under `[Unreleased]`, stamp with `python scripts/stamp_version.py 0.1.0aN`, re-run the editable install, and re-run the suite. Commit as plain `chore: stamp 0.1.0aN`, as a separate step from the tests.
5. Push the release branch, confirm CI on the pushed commit, and record the merge and stamp in Skuld.

## Worktrees and the sandbox

`dev/worktree.json` is conduit's worktree recipe for the workstation's shared worktree setup script. It declares no services yet: the stores workstream adds Postgres (with its port offset and a Compose template) for the `sql` extra's manual checks. Every published port binds to `127.0.0.1` inside the worktree's own slot.

## Parallel sessions

When a session is running in a git worktree (`git rev-parse --git-common-dir` resolves outside the working tree), assume other sessions are active concurrently:

- Work only inside your own worktree, on your own branch. Never touch the main clone or another worktree.
- The Kvasir library is always `conduit`, and every document submitted from a worktree session carries that branch's `branch-` tag.
- Worktree sessions never bump the version and never create stamp commits.
- Shared modules are edited only by the session assigned them in its kick-off prompt.

## Minimum rules without the global agreement

If the global working agreement is not present on this machine, these still apply:

- Australian English; no AI attribution in commits, changelogs or release notes.
- The release-branch workflow above.
- Do not push, tag, publish to PyPI or open a pull request unless the maintainer instructs it in the session.
- Use the `conduit` Kvasir library for project knowledge, pass `library: "conduit"` on every call, and submit a `type-session` summary at the end of every session with the mandatory `project-`, `type-`, `status-` and `source-` tags.

## Session start checklist

1. Follow the global session start protocol (Kvasir library check, context retrieval, summary to the maintainer).
2. Read this file, `README.md`, `docs/design/architecture.md`, `docs/design/stack.md`, and the `[Unreleased]` section of `CHANGELOG.md`.
3. Confirm the task fits the architecture decisions above. If it does not, say so before writing code.
4. In worktree mode, confirm which shared modules, if any, the kick-off prompt assigns to this session.
