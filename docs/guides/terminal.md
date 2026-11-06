# The terminal interface

The `cli` extra adds `conduit`, a terminal interface that demonstrates conduit's public API end to end: provider adapters and credential checks (Bedrock's authentication modes and residency refusals included), the model catalogue, streaming chat with tool use under run budgets, skills, MCP servers with OAuth sign-in, forced compaction, and per-call usage with cache reads and writes. It is a demonstration and a diagnostic, not a product: it keeps no conversation or token unless asked, never prints a secret, and shows text from models, tools and servers without letting it drive the terminal.

Without a configuration file it runs on the stub provider, so everything can be tried with no credentials and no network.

## Installing and running

Run it without installing anything, with uvx:

```sh
uvx --from "cognisn-conduit[cli]" conduit chat
uvx --from "cognisn-conduit[cli,bedrock]" conduit check --config conduit.toml
```

Or install it into an environment, with pip or uv:

```sh
pip install "cognisn-conduit[cli,anthropic]"
conduit providers
python -m conduit.cli models
```

The `cli` extra brings rich and the `mcp` extra. Add the extra for each provider you configure (`bedrock`, `anthropic`, `openai`, `gemini`, `ollama`, `bedrock-mantle`); `conduit providers` names any that are missing. Without the `cli` extra, `conduit` says which extra to install and exits.

## Commands

| Command | Does |
|---|---|
| `conduit providers` | Lists every provider adapter with its extra and whether its SDK is installed, then each configured provider with its kind, region, authentication mode and whether its settings validate. |
| `conduit models [--provider NAME]` | Lists the model catalogue after the configuration's overlays: provider, model, kind, residency scope, regions and capabilities. |
| `conduit check` | Builds every configured provider through the provider registry (secrets resolved, clients made, a short-term Bedrock key presigned) and shows each as ready or refused, then resolves every assignment against the catalogue, its floor and the residency policy, showing each refusal on its own row. Calls no model. |
| `conduit mcp [--tokens DIR]` | Connects every configured MCP server (signing in with OAuth where needed), lists each server's bridged tools with their task classes, and closes. |
| `conduit chat [options]` | A streaming conversation (below). |

Every command takes `--config FILE`, a JSON or TOML configuration file. The global options come before the command:

- `--version` prints the version.
- `--log-level {debug,info,warning,error}` sets **conduit's own loggers** (`conduit` and below) and writes their records to standard error; the default is `warning`. Every other library stays at WARNING whatever you ask, because some log content or credentials at DEBUG (the MCP SDK logs messages, botocore logs signatures).

The exit status is 0 when everything succeeded, 1 when something was refused or failed (a `check` refusal, a configuration error), 2 for a usage error, and 130 when interrupted (Ctrl-C ends the interface at once, at the prompt too).

`check` is the command to run before deploying a configuration. For Bedrock it says what each authentication mode proved: a short-term key is presigned from the configured credentials, so missing credentials are refused there, while for the SigV4 modes AWS checks the credentials on the first call. Every assignment is resolved on its own, so all of a configuration's residency and capability refusals show at once, under a heading naming the policy's scopes and regions:

```python
from conduit.cli.main import main

assert main(["models", "--provider", "stub"]) == 0
assert main(["check"]) == 0  # the stub configuration: every provider and assignment ready
```

## Chat

`conduit chat` runs one agent run per message on the task class `chat`, streaming the answer as it arrives. Its tools are `add` and `word_count`, `use_skill` when a skills directory is given, and every connected MCP server's tools (named `<server>__<tool>`). After each model call it prints a `[usage]` line with the four token counts and a `[cache]` line, and it reports tool calls, budget exhaustion, compaction, skills, MCP calls and each run's totals on `[tool]`, `[budget]`, `[compaction]`, `[skill]`, `[mcp]` and `[run]` lines.

| Option | Does |
|---|---|
| `--config FILE` | The configuration file (default: the stub). |
| `--skills DIR` | A directory of `SKILL.md` skills, offered through `use_skill`. |
| `--transcripts DIR` | Keep the conversation in `DIR` as JSON transcripts. Without it, the conversation is kept in memory only and is gone when you leave. |
| `--tokens DIR` | Keep MCP OAuth tokens in `DIR`. Without it, tokens are held in memory and you sign in again next time. Protect the directory as you would any file holding credentials. |
| `--no-mcp` | Do not connect the configured MCP servers. |
| `--events` | Show every event as it is emitted. |
| `--max-model-calls N`, `--max-tool-calls N`, `--max-tokens N`, `--max-wall-time SECONDS` | Override the run budget (by default 8 model calls and 8 tool calls per message). |

Slash commands at the prompt:

| Command | Does |
|---|---|
| `/help` | Lists the commands. |
| `/tools` | Lists the tools the model may call. |
| `/skills` | Lists the skills, and which are loaded. |
| `/load NAME` | Loads a skill into the system prompt. |
| `/compact [STRATEGY]` | Forces a compaction before the next model call (with a named strategy, or the most aggressive), showing the conversation's size now and, when it runs, the tokens and messages before and after. |
| `/usage` | Shows the session's usage so far, and the conversation's size. |
| `/events` | Turns the event display on or off. |
| `/quit`, `/exit` | Leaves. |

On the stub, the model echoes what you type, and `call <tool> <JSON object>` makes it call a tool, so tool use, skills and MCP tools can be tried without a model:

```console
$ conduit chat
conduit chat on stub stub-echo (local). Type /help for commands, /quit to leave.
The conversation is kept in memory only, and is gone when you leave.
> hello
hello
[usage] stub stub-echo: input 37, output 2, cache read 0, cache write 92
[cache] read 0, written 92
[run] completed: input 37, output 2, cache read 0, cache write 92; 1 model calls, 0 tool calls
> call add {"a": 2, "b": 3}
[usage] stub stub-echo: input 62, output 4, cache read 92, cache write 0
[cache] read 92, written 0
[tool] add called
[tool] add: ok
...
> call use_skill {"name": "release-notes"}
> /compact
> /quit
```

The system prompt and tools are written to the cache on the first call and read from it on every later one, which the `[cache]` lines show.

## The configuration file

A configuration file is JSON (`.json`) or TOML (`.toml`). Every key is optional:

| Key | Holds |
|---|---|
| `conduit` | `ConduitSettings`, as conduit reads it: providers, assignments, residency, capture, catalogue overlays, `require_all_assignments`. Assign the task class `chat` (its floor needs tools), and optionally `summary`, which offers the summarising compaction strategy. |
| `system` | The system prompt. |
| `budget` | The run budget per message (`RunBudget`: `max_model_calls`, `max_tool_calls`, `max_tokens`, `counted_tokens`, `max_wall_time_s`). |
| `skills` | A skills directory, relative to the file. |
| `mcp` | `callback_port`, the OAuth sign-in's loopback port (default 33418), and `servers`, each an `McpServerSettings` whose `task_classes` include `chat`. |
| `compaction` | `keep_turns` (default 2), and the thresholds `drop_tool_results_at` (0.6), `summarise_at` (0.75, used when `summary` is assigned) and `keep_last_turns_at` (0.85). |

Credentials are only ever `secret://NAME` references, resolved from the environment variable `NAME` when a provider is built. A configuration holding a plain-text credential is refused, naming the field and never the value.

```toml
system = "You answer questions about example.com's services."

[conduit.providers.bedrock]
kind = "bedrock"
region = "eu-west-2"
auth = { mode = "short_term_api_key" }

[conduit.providers.claude]
kind = "anthropic"
api_key = "secret://ANTHROPIC_API_KEY"

[conduit.assignments.chat]
provider = "bedrock"
model = "anthropic.claude-sonnet-5"

[conduit.assignments.summary]
provider = "claude"
model = "claude-haiku-5-5"

[conduit.residency]
scopes = ["local", "in_region", "global"]

[budget]
max_model_calls = 6
max_tool_calls = 6

[[mcp.servers]]
name = "docs"
transport = { type = "streamable_http", url = "https://mcp.example.com/docs" }
auth = { type = "oauth", scopes = ["docs:read"] }
task_classes = ["chat"]
```

A refused assignment shows in `check`, which exits 1. Here a residency policy allowing only in-region processing in one Region refuses the stub's local model:

```python
import json
from pathlib import Path

Path("in-region-only.json").write_text(
    json.dumps(
        {
            "conduit": {
                "providers": {
                    "stub": {"kind": "stub", "script": "demo", "not_for_production": True}
                },
                "assignments": {"chat": {"provider": "stub", "model": "stub-echo"}},
                "residency": {"scopes": ["in_region"], "regions": ["eu-west-2"]},
            }
        }
    ),
    encoding="utf-8",
)

assert main(["check", "--config", "in-region-only.json"]) == 1
```

## MCP servers and OAuth sign-in

`conduit mcp`, and `conduit chat` unless `--no-mcp` is given, connect every server in the configuration's `mcp.servers`. An OAuth server with no usable tokens opens the browser at its authorisation server and waits for the redirect on `http://127.0.0.1:<callback_port>/callback`, by default port 33418. Register that redirect URI with the authorisation server (or let conduit register itself dynamically), and change `callback_port` if the port is taken. Tokens are held in memory, or in `--tokens DIR` to survive the session.

## What it shows and keeps

- Text from models, tools and MCP servers is shown as written: control, escape and invisible characters appear as visible markers, and nothing is read as terminal markup or emoji codes.
- Errors from conduit are shown by their message, which names configuration and never a value; any other failure is shown by its class name only.
- Nothing is written to disk unless `--transcripts` or `--tokens` names a directory.
- The interface uses conduit's public API only, so everything it does a product can do the same way.
