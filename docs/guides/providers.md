# Setting up providers

Each provider adapter is behind its own extra, and imports its SDK only when a client is built: installing conduit without an extra, or building `conduit.providers.builtin_factories()`, never needs one. This guide covers installing each extra, configuring each provider, and what each adapter does about authentication, residency, environment variables and logging.

Configure providers in `ConduitSettings.providers`, assign them to task classes, and pass `builtin_factories()` to `start` (see [Integrating conduit](integration.md)). Every example here validates its settings through a provider registry, as `start` does, without building a client, so none needs credentials or a network.

```python
from conduit import ConfigurationError, LiteralSecretResolver, ProviderRegistry
from conduit.providers import builtin_factories

registry = ProviderRegistry(secrets=LiteralSecretResolver({}), factories=builtin_factories())


def refused(settings: dict[str, object]) -> str:
    """The refusal ``start`` would raise for ``settings``."""
    try:
        registry.validate("example", settings)
    except ConfigurationError as refusal:
        return str(refusal)
    raise AssertionError("expected a refusal")
```

## Installing

The distribution is `cognisn-conduit`; the import name is `conduit`. Python 3.12 or later.

| Extra | Provider or layer |
|---|---|
| `bedrock` | Amazon Bedrock (bedrock-runtime: Converse and ConverseStream) |
| `bedrock-mantle` | Claude through Amazon Bedrock's mantle endpoint (single-Region) |
| `anthropic` | Anthropic direct, and Claude in Microsoft Foundry |
| `openai` | OpenAI, X.AI and Microsoft Foundry (one OpenAI-compatible adapter) |
| `gemini` | The Gemini API |
| `ollama` | Ollama |
| `all-providers` | Every provider above |
| `local-embeddings` | Local embeddings with sentence-transformers on CPU-only torch |

With pip, in the product's virtual environment:

```sh
pip install "cognisn-conduit[bedrock]"
pip install "cognisn-conduit[anthropic,openai]"
pip install "cognisn-conduit[all-providers]"
```

With uv, in a project (adds the dependency to `pyproject.toml`), or into a plain environment:

```sh
uv add "cognisn-conduit[bedrock]"
uv pip install "cognisn-conduit[bedrock]"
```

With uvx, to run the terminal interface without installing anything:

```sh
uvx --from "cognisn-conduit[cli,all-providers]" conduit
```

### Local embeddings and CPU-only torch

The `local-embeddings` extra runs sentence-transformers on the CPU. conduit never uses a GPU, so install torch's CPU-only build. On Linux the default torch from PyPI also installs NVIDIA CUDA packages, which are large and which conduit never uses; the macOS and Windows wheels on PyPI are CPU-only already.

With pip, install torch from the CPU index first, then conduit:

```sh
pip install "torch>=2.15,<3" --index-url https://download.pytorch.org/whl/cpu
pip install "cognisn-conduit[local-embeddings]"
```

A later `pip install --upgrade torch` without `--index-url` brings the CUDA build back.

With uv in a project, take torch from the CPU index in `pyproject.toml`, then add torch as a direct dependency alongside conduit, so the source applies to it:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
```

```sh
uv add torch "cognisn-conduit[local-embeddings]"
```

With uv in a plain environment, or uvx for a throwaway run:

```sh
uv pip install --torch-backend cpu "cognisn-conduit[local-embeddings]"
uvx --torch-backend cpu --from "cognisn-conduit[local-embeddings,cli]" conduit
```

## Amazon Bedrock

Kind `bedrock`, extra `bedrock`. Calls the bedrock-runtime Converse API through boto3 (ConverseStream by default; `use_converse_stream: false` uses Converse). boto3 is blocking, so its calls run in a worker thread.

`region` is required: it is the Region the bedrock-runtime endpoint is called in, and the region the residency policy checks.

### Authentication

`auth.mode` chooses one of four modes. No setting ever holds a credential: every credential is a `secret://` reference.

| Mode | How requests are authenticated |
|---|---|
| `credential_chain` (default) | The AWS credential chain (environment, shared profile, SSO, container or instance role), optionally a named `profile`. SigV4. |
| `access_keys` | An access key id, secret access key and optional session token, each a secret reference resolved through the product's resolver. SigV4. |
| `short_term_api_key` | conduit generates a short-term Bedrock API key from AWS credentials (`auth.source`: the credential chain or access keys) and sends it as a bearer token. A key lasts `lifetime_s` (15 minutes to 12 hours, 12 by default) or until its signing credentials expire, and is replaced `refresh_before_s` before it expires. |
| `long_term_api_key` | A long-term Bedrock API key held as a secret reference. AWS recommends long-term keys for exploration only, so they are refused unless `allow_long_term_api_keys` is set. |

```python
credential_chain = {"kind": "bedrock", "region": "eu-west-2"}
named_profile = {
    "kind": "bedrock",
    "region": "eu-west-2",
    "auth": {"mode": "credential_chain", "profile": "example-ai"},
}
access_keys = {
    "kind": "bedrock",
    "region": "eu-west-2",
    "auth": {
        "mode": "access_keys",
        "access_key_id": "secret://aws/access_key_id",
        "secret_access_key": "secret://aws/secret_access_key",
    },
}
short_term = {
    "kind": "bedrock",
    "region": "eu-west-2",
    "auth": {"mode": "short_term_api_key", "lifetime_s": 3_600, "refresh_before_s": 300},
}
long_term = {
    "kind": "bedrock",
    "region": "eu-west-2",
    "allow_long_term_api_keys": True,
    "auth": {"mode": "long_term_api_key", "api_key": "secret://bedrock/api_key"},
}

for mapping in (credential_chain, named_profile, access_keys, short_term, long_term):
    registry.validate("bedrock", mapping)

without_the_flag = {**long_term, "allow_long_term_api_keys": False}
assert "long-term Bedrock API keys are refused" in refused(without_the_flag)
```

With a short-term key, the first key is generated when the provider is built, so missing AWS credentials are refused at start-up, not on the first call. The process environment (`AWS_BEARER_TOKEN_BEDROCK`) is never read or written, so two providers in one process can use different keys.

A deployment that does not use Bedrock API keys can deny them outright in IAM: deny the `bedrock:CallWithBearerToken` action for its principals, and no key minted from those credentials will work.

### Residency

The default catalogue declares every current Claude model on bedrock-runtime and every other vendor's current chat model that Bedrock serves through Converse on demand (Amazon Nova, DeepSeek, Mistral AI, OpenAI, Qwen, MiniMax, Moonshot AI, NVIDIA, Writer and Z.AI), with each in-Region id and inference profile the model cards document, plus Titan Text Embeddings V2. Models in their legacy period with an end-of-life date are left out. Capabilities follow each model card's bedrock-runtime features: a model whose card lists tool calling only for bedrock-mantle is declared without tools, and prompt caching is declared only where Converse takes cache points (Claude and Amazon Nova). The non-Claude entries are tested against recorded responses until the next live verification pass. Each model id is declared with its scope:

- **In-region** model ids (`anthropic.claude-sonnet-5`) are processed in the Region the provider is called in, and are declared only for the Regions that serve them.
- **Geographic inference profiles** (`eu.`, `us.`, `au.`, `jp.`, `in.`, `ca.`, `apac.` and `us-gov.` prefixes) are `cross_region`: Bedrock may process a request in any Region of the profile, and the catalogue lists them.
- **Global inference profiles** (`global.` prefix) are `global`.

A deployment that must keep processing in one Region allows only `local` and `in_region` and names its Regions; an assignment to a profile is then refused at start-up (`ResidencyRefused`), before any client exists. A deployment that may use a geography allows `cross_region` and lists every Region of the profile; a profile reaching outside the list is refused.

```python
from conduit import ResidencyPolicy, default_catalogue

in_region = ResidencyPolicy(
    scopes=frozenset({"local", "in_region"}), regions=frozenset({"eu-west-2"})
)
europe = ResidencyPolicy(
    scopes=frozenset({"local", "in_region", "cross_region"}),
    regions=frozenset(
        default_catalogue().require("bedrock", "eu.anthropic.claude-sonnet-5").regions
    ),
)

catalogue = default_catalogue()
in_region.check(catalogue.require("bedrock", "anthropic.claude-sonnet-5"), "eu-west-2", task="chat")
europe.check(catalogue.require("bedrock", "eu.anthropic.claude-sonnet-5"), "eu-west-2", task="chat")
```

### Logging and credentials

botocore's signer logs every signature at DEBUG on `botocore.auth`, and a short-term key can be rebuilt from those records; botocore's endpoint logs each request object at DEBUG, which a buffering handler could render after the bearer key is set. When a provider uses API keys, conduit installs two filters on botocore's loggers, once per process: one drops `botocore.auth` records whose message starts with `Signature:`, and one renders every `botocore.endpoint` record when it is logged. Both are process-wide, so they also apply to the product's own botocore clients. Removing them gives back every signature (and with it every short-term key) and every bearer key to whatever the product's handlers write; do that only for local debugging with throwaway credentials. The presign step is `conduit.providers.bedrock.auth.short_term_keys`, shared by the bedrock and bedrock-mantle adapters.

conduit cannot filter everything botocore logs. When botocore refreshes temporary credentials (assume-role through STS, SSO, a container or instance role) it can log the responses that carry them at DEBUG, on loggers conduit does not touch. **Keep the `botocore` logger above DEBUG in production.**

### Other settings

`endpoint_url` replaces the regional endpoint (an interface VPC endpoint, say; https only). `connect_timeout_s`, `read_timeout_s` and `max_attempts` set botocore's client configuration. Cache breakpoints become Converse cache points where the catalogue says the model supports prompt caching.

## Claude through Bedrock's mantle endpoint

Kind `bedrock_mantle`, extra `bedrock-mantle` (the anthropic SDK and boto3). Calls Claude through the Anthropic Messages API at `https://bedrock-mantle.<region>.api.aws/anthropic`, which runs each request in the Region it is sent to. That makes it the route to single-Region Claude where bedrock-runtime offers only inference profiles: its catalogue entries are `in_region`, one for each Claude model the Bedrock model cards document on bedrock-mantle (gated Mythos models aside), declared for the Regions each card lists.

Authentication is by Bedrock API key only: `short_term_api_key` (the default, generated from the credential chain or access keys and refreshed) or `long_term_api_key` behind `allow_long_term_api_keys`. The SDK never signs requests or reads a key or Region from the environment, and the key is set at send time, so it never appears in the request options the SDK logs. `ANTHROPIC_LOG=debug` is refused while set, since it makes the SDK log whole request bodies.

```python
registry.validate("claude_regional", {"kind": "bedrock_mantle", "region": "eu-west-1"})
mantle = catalogue.require("bedrock_mantle", "anthropic.claude-sonnet-5")
assert mantle.residency_scope == "in_region" and "eu-west-1" in mantle.regions
```

## Anthropic

Kind `anthropic`, extra `anthropic`. Two endpoints:

- `endpoint: direct` (the default) calls Anthropic's API with `api_key`. `base_url` is optional, for a gateway in front of the API; it must be https (plain http only to loopback) and may not carry credentials.
- `endpoint: foundry` calls Claude in Microsoft Foundry through the resource's Anthropic endpoint, given as `base_url` (`https://<resource>.services.ai.azure.com/anthropic/`), with exactly one of `api_key` and `entra_id`.

```python
direct = {"kind": "anthropic", "api_key": "secret://anthropic/api_key"}
foundry_key = {
    "kind": "anthropic",
    "endpoint": "foundry",
    "base_url": "https://example-resource.services.ai.azure.com/anthropic/",
    "api_key": "secret://foundry/api_key",
}
foundry_entra = {
    "kind": "anthropic",
    "endpoint": "foundry",
    "base_url": "https://example-resource.services.ai.azure.com/anthropic/",
    "entra_id": {"mode": "entra_id"},
}

for mapping in (direct, foundry_key, foundry_entra):
    registry.validate("claude", mapping)

assert "settings refused" in refused(
    {"kind": "anthropic", "endpoint": "foundry", "entra_id": {"mode": "entra_id"}}
)  # a Foundry endpoint needs base_url
```

Microsoft Entra ID uses azure-identity's `DefaultAzureCredential` (environment, workload identity, managed identity, developer sign-ins) and a bearer token provider for the scope (`https://ai.azure.com/.default` by default). `managed_identity_client_id` picks a user-assigned managed identity.

The client takes its key, endpoint and headers only from the settings. Three environment variables would let the SDK override them or log content, so a client is refused while one is set:

- `ANTHROPIC_CUSTOM_HEADERS` (merged over the headers conduit sets, including the key);
- `ANTHROPIC_FOUNDRY_RESOURCE`, for a Foundry endpoint;
- `ANTHROPIC_LOG=debug` (`info` logs only retries and is allowed).

The SDK's own OpenTelemetry spans are switched off, since they can carry request content outside conduit's capture policy. The catalogue's `anthropic` entries describe the direct API (`global`): every current Claude model and every legacy model still available, as Anthropic's models overview lists them (Claude Mythos models, offered only to verified organisations, are left out). Claude in Foundry is a regional deployment: declare the models you deploy in a catalogue overlay with the resource's residency.

## OpenAI, X.AI and Microsoft Foundry

Kind `openai_compat`, extra `openai`. One adapter serves three services, chosen by `service.name`, through streaming Chat Completions with usage on the final chunk. Prompt caching on these services is automatic, so cache breakpoints are ignored, and cached prompt tokens come back as `cache_read_tokens`.

| Service | Endpoint | Authentication | Catalogue provider |
|---|---|---|---|
| `openai` | `https://api.openai.com/v1` | `api_key`; `organization` and `project` sent only when set | `openai` |
| `xai` | `https://api.x.ai/v1` | `api_key` | `xai` |
| `foundry` | the resource's `https://<resource>.openai.azure.com/openai/v1/` | an API key or Microsoft Entra ID | `openai`, or `xai` for Grok deployments |

"ChatGPT" in conduit means the OpenAI API through this adapter: signing in with a consumer ChatGPT subscription is not supported.

```python
openai = {
    "kind": "openai_compat",
    "service": {"name": "openai", "api_key": "secret://openai/api_key"},
}
xai = {"kind": "openai_compat", "service": {"name": "xai", "api_key": "secret://xai/api_key"}}
foundry = {
    "kind": "openai_compat",
    "service": {
        "name": "foundry",
        "endpoint": "https://example-resource.openai.azure.com/openai/v1/",
        "auth": {
            "mode": "entra_id",
            "managed_identity_client_id": "00000000-0000-0000-0000-000000000000",
        },
    },
}

for mapping in (openai, xai, foundry):
    registry.validate("gpt", mapping)
```

A Foundry model id is the **deployment name**, which the product declares in a catalogue overlay under the catalogue provider, with its own residency: Foundry deployments are regional, while the default catalogue's `openai` and `xai` entries describe the public APIs: each provider's current chat models with function calling through Chat Completions, and OpenAI's `text-embedding-3-small` and `text-embedding-3-large`. X.AI states no per-model output limit, so its entries carry the API's default of 128,000 tokens. OpenAI's `gpt-6-astra` and `gpt-6.1-sol` take tool calls only through the Responses API, so their entries declare no tools. Name deployments so they do not collide with a public model id, or the overlay would change that entry too.

```python
from conduit import CatalogueOverlay

foundry_overlay = CatalogueOverlay.model_validate(
    {
        "name": "example-foundry",
        "add": [
            {
                "provider": "openai",
                "model_id": "example-gpt5-swedencentral",
                "kind": "chat",
                "residency_scope": "in_region",
                "regions": ["swedencentral"],
                "capabilities": {"tools": True, "structured_output": True},
                "context_window": 400_000,
                "output_limit": 128_000,
            }
        ],
    }
)
declared = catalogue.apply(foundry_overlay).require("openai", "example-gpt5-swedencentral")
assert declared.residency_scope == "in_region"
```

Give the Foundry provider `region: swedencentral` in its settings so the residency policy can check it.

Environment variables: the key and base URL are always passed, so `OPENAI_API_KEY` and `OPENAI_BASE_URL` are never used, and `OPENAI_ORG_ID`, `OPENAI_PROJECT_ID`, `OPENAI_ADMIN_KEY` and `OPENAI_WEBHOOK_SECRET` are cleared on every client unless the settings give an organisation or project (an OpenAI organisation id is never sent to X.AI or Foundry). Two still reach the SDK, and deployments should know them:

- `OPENAI_LOG` (`info` or `debug`) makes the SDK log each request's options, including prompt and tool content, on its own `openai` logger, outside conduit's capture policy. Leave it unset in production.
- `HTTPS_PROXY`, `HTTP_PROXY`, `ALL_PROXY` and `NO_PROXY` route every call through a proxy (the TLS session still ends at the provider).

## Gemini

Kind `gemini`, extra `gemini`. The Gemini API with a Google AI Studio key. Vertex AI is not served by this adapter.

```python
registry.validate("gemini", {"kind": "gemini", "api_key": "secret://gemini/api_key"})
```

The key, the backend and the endpoint are pinned, so `GOOGLE_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_GENAI_USE_ENTERPRISE` and `GOOGLE_GEMINI_BASE_URL` are never read. The rest of the SDK's transport follows the environment as any HTTPS client would: proxy variables and `.netrc`, `SSL_CERT_FILE` and `SSL_CERT_DIR`, and `GOOGLE_GENAI_HTTP_CLIENT`. `base_url` may name an https proxy in front of the API (no credentials, query or fragment); `max_attempts` bounds the SDK's retries. The catalogue's Gemini entries are `global`: the current stable chat models (Pro tier `gemini-4-carbon` and `gemini-4-argon`, and the Flash and Flash-Lite models) and `gemini-embedding-001`.

## Ollama

Kind `ollama`, extra `ollama` (httpx). A locally hosted Ollama server, on loopback by default (`http://127.0.0.1:11434`). The catalogue's Ollama entries are `local`: the deployment's own hosts. A remote server is allowed, but its URL may not carry credentials, a query or a fragment.

```python
registry.validate("ollama", {"kind": "ollama"})
registry.validate(
    "ollama", {"kind": "ollama", "base_url": "http://192.0.2.40:11434", "num_ctx": 32_768}
)
assert "base_url" in refused({"kind": "ollama", "base_url": "http://user:pass@192.0.2.40:11434"})
```

The HTTP client ignores the environment (no proxy variables, no `.netrc`) and never follows redirects. Left unset, `num_ctx` requests the catalogue entry's full context window (131,072 tokens for `llama3.1:8b`), because Ollama's own default is far smaller; a full window needs memory on the Ollama host to match, so set `num_ctx` to request less. Declare any other model you pull in a catalogue overlay under provider `ollama` with scope `local`.

## The stub

Kind `stub`, no extra. A deterministic, scriptable provider for tests: it never reaches a network, streams like a real provider, and reports all four usage counts (including cache reads and writes for requests with cache breakpoints). It needs `not_for_production: true` in its settings and logs a warning when built. Scripts are registered by name with `StubFactory(scripts={...})`: `echo` (the default), `scripted` turns in order, and `answer_tool_results` (call a tool, then answer from its result).

## Building clients

Each adapter resolves its credential and builds its SDK client in one guarded step. The registry builds every client under the same guard (`guarded_client`, exported from `conduit`), a product's own factories included.

If a client cannot be built, `start()` raises `ConfigurationError` naming the provider, its kind and the exception's class, such as `provider 'primary': the anthropic client could not be built: TypeError`. The error has no cause or context, so the key never appears in a traceback with locals. conduit's own refusals (a missing extra, an unresolved secret, a refused environment variable) keep their own messages, with their traceback, cause, context and notes cleared.

```python
import asyncio
from collections.abc import Mapping
from typing import Any

from pydantic import SecretStr

from conduit import ConduitSettings, ProviderSettings, TaskClass, start
from conduit.providers.stub import StubFactory


class BrokenStubFactory(StubFactory):
    """A factory whose client step fails with the key in the library's message."""

    def create_client(self, settings: ProviderSettings, secrets: Mapping[str, SecretStr]) -> Any:
        raise TypeError("sk-example-not-real is not a valid key")


async def start_broken() -> ConfigurationError:
    settings = ConduitSettings.model_validate(
        {
            "providers": {"primary": {"kind": "stub", "not_for_production": True}},
            "assignments": {"triage": {"provider": "primary", "model": "stub-echo"}},
        }
    )
    try:
        await start(
            settings,
            task_classes=[TaskClass(name="triage")],
            factories=[BrokenStubFactory()],
            secrets=LiteralSecretResolver({}),
        )
    except ConfigurationError as refusal:
        return refusal
    raise AssertionError("expected a refusal")


refusal = asyncio.run(start_broken())
assert str(refusal) == "provider 'primary': the stub client could not be built: TypeError"
assert refusal.__cause__ is None and refusal.__context__ is None
```

This covers ordinary exceptions. Cancellation, `KeyboardInterrupt` and other `BaseException` propagate unchanged, with the library's frames.

The guard cannot stop a library logging its own arguments before it fails. botocore's signer, for example, logs the canonical request (with the AWS access key id and any session token) at DEBUG when conduit presigns a short-term Bedrock key. Keep SDK loggers above DEBUG in production.

## Writing a provider adapter

A product can add an adapter of its own by giving the registry a `ProviderFactory`: a `kind`, a `settings_model` (a subclass of `ProviderSettings`, whose credential fields must be typed `SecretRef`), `create_client(settings, secrets)` and `create_provider(name, settings, client, *, catalogue=None)`. Only the registry calls `create_client`, and every provider it builds is wrapped in metering, so calls through the adapter get admission checks, residency, usage events and capture enforcement like any other. Subclass `BaseProvider` and implement `stream`; declare the adapter's models in a catalogue overlay under its kind.

The registry already runs `create_client` under the construction guard. A factory that builds a client anywhere else (an embedder's client, a client it refreshes later) can run that step under the same guard with `guarded_client(kind, build, name=...)`: what `build` returns passes through, a conduit error goes on with its traceback, cause, context and notes cleared, and any other exception becomes the fixed `ConfigurationError`.

```python
from conduit import guarded_client


def build_acme_client() -> object:
    raise ValueError("https://api.example.com/?key=sk-example-not-real refused")


try:
    guarded_client("acme", build_acme_client, name="primary")
except ConfigurationError as refusal:
    assert str(refusal) == "provider 'primary': the acme client could not be built: ValueError"
    assert refusal.__cause__ is None and refusal.__context__ is None
else:
    raise AssertionError("expected a refusal")

assert guarded_client("acme", lambda: "client") == "client"
```
