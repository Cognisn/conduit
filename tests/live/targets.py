"""Provider settings for the live suite, built from ``CONDUIT_LIVE_`` variables.

Every credential is a ``secret://<NAME>`` reference, resolved by
``EnvironmentSecretResolver(prefix="CONDUIT_LIVE_")`` from ``CONDUIT_LIVE_<NAME>`` when the
provider is built. The settings never hold a key, and nothing here reads one.
Non-secret values (regions, model ids, URLs, deployment names) are read as plain values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from conduit.core.catalogue import Capabilities, CatalogueOverlay, ModelEntry, default_catalogue
from conduit.core.residency import ResidencyScope
from conduit.core.secrets import EnvironmentSecretResolver
from tests.live.needs import PREFIX, LiveEnv, present, value


def resolver(environ: LiveEnv) -> EnvironmentSecretResolver:
    """The only way the live suite reaches a credential."""
    return EnvironmentSecretResolver(prefix=PREFIX, environ=environ)


def ref(name: str) -> str:
    """A secret reference to ``CONDUIT_LIVE_<name>``."""
    return f"secret://{name}"


@dataclass(frozen=True, slots=True)
class Target:
    """One provider route and model: what a live check runs against."""

    label: str
    provider: dict[str, Any]
    model: str
    overlays: tuple[CatalogueOverlay, ...] = field(default=())

    @property
    def subject(self) -> str:
        return f"{self.label} {self.model}"


def offered_in(entry: ModelEntry, region: str) -> bool:
    """Whether ``entry`` can be called from ``region`` (as start-up resolution decides)."""
    if entry.residency_scope in (ResidencyScope.IN_REGION, ResidencyScope.CROSS_REGION):
        return region in entry.regions
    return True


def regions_list(environ: LiveEnv, many: str, one: str) -> list[str]:
    """The regions named in ``many`` (comma-separated), or the single region ``one``."""
    listed = value(environ, many)
    if listed:
        return [part.strip() for part in listed.split(",") if part.strip()]
    single = value(environ, one)
    return [single] if single else []


# Bedrock --------------------------------------------------------------------------------

#: The authentication modes the suite exercises, by label.
BEDROCK_AUTH_MODES = (
    "credential_chain",
    "access_keys",
    "short_term_from_chain",
    "short_term_from_access_keys",
    "long_term_api_key",
)


def _chain(environ: LiveEnv) -> dict[str, Any]:
    auth: dict[str, Any] = {"mode": "credential_chain"}
    profile = value(environ, "BEDROCK_PROFILE")
    if profile:
        auth["profile"] = profile
    return auth


def _access_keys(environ: LiveEnv) -> dict[str, Any]:
    auth: dict[str, Any] = {
        "mode": "access_keys",
        "access_key_id": ref("AWS_ACCESS_KEY_ID"),
        "secret_access_key": ref("AWS_SECRET_ACCESS_KEY"),
    }
    if present(environ, "AWS_SESSION_TOKEN"):
        auth["session_token"] = ref("AWS_SESSION_TOKEN")
    return auth


def bedrock_auth(mode: str, environ: LiveEnv) -> tuple[dict[str, Any], bool]:
    """The ``auth`` mapping for ``mode``, and whether long-term keys must be allowed."""
    if mode == "credential_chain":
        return _chain(environ), False
    if mode == "access_keys":
        return _access_keys(environ), False
    if mode == "short_term_from_chain":
        return {"mode": "short_term_api_key", "source": _chain(environ)}, False
    if mode == "short_term_from_access_keys":
        return {"mode": "short_term_api_key", "source": _access_keys(environ)}, False
    if mode == "long_term_api_key":
        return {"mode": "long_term_api_key", "api_key": ref("BEDROCK_API_KEY")}, True
    raise ValueError(f"unknown Bedrock authentication mode {mode!r}")


def first_bedrock_auth(environ: LiveEnv) -> str | None:
    """The first SigV4 or key mode whose variables are set, preferring SigV4."""
    if present(environ, "BEDROCK_CREDENTIAL_CHAIN"):
        return "credential_chain"
    if present(environ, "AWS_ACCESS_KEY_ID") and present(environ, "AWS_SECRET_ACCESS_KEY"):
        return "access_keys"
    if present(environ, "BEDROCK_API_KEY"):
        return "long_term_api_key"
    return None


def bedrock_settings(region: str, mode: str, environ: LiveEnv) -> dict[str, Any]:
    auth, long_term = bedrock_auth(mode, environ)
    settings: dict[str, Any] = {"kind": "bedrock", "region": region, "auth": auth}
    if long_term:
        settings["allow_long_term_api_keys"] = True
    return settings


#: The cheapest Claude on bedrock-runtime, by profile prefix.
BEDROCK_DEFAULT_FAMILY = "anthropic.claude-haiku-4-5-20251001-v1:0"


def default_bedrock_model(region: str) -> str | None:
    """A cross-region Haiku 4.5 profile that serves ``region``, else the global one."""
    candidates = [
        entry
        for entry in default_catalogue()
        if entry.provider == "bedrock" and entry.model_id.endswith(BEDROCK_DEFAULT_FAMILY)
    ]
    for entry in candidates:
        if entry.residency_scope is ResidencyScope.CROSS_REGION and region in entry.regions:
            return entry.model_id
    for entry in candidates:
        if entry.residency_scope is ResidencyScope.GLOBAL:
            return entry.model_id
    return None


def mantle_auth(environ: LiveEnv) -> tuple[dict[str, Any], bool] | None:
    """Short-term keys from AWS credentials if set, else a long-term key."""
    if present(environ, "BEDROCK_CREDENTIAL_CHAIN"):
        return bedrock_auth("short_term_from_chain", environ)
    if present(environ, "AWS_ACCESS_KEY_ID") and present(environ, "AWS_SECRET_ACCESS_KEY"):
        return bedrock_auth("short_term_from_access_keys", environ)
    if present(environ, "BEDROCK_API_KEY"):
        return bedrock_auth("long_term_api_key", environ)
    return None


def mantle_settings(region: str, auth: dict[str, Any], long_term: bool) -> dict[str, Any]:
    settings: dict[str, Any] = {"kind": "bedrock_mantle", "region": region, "auth": auth}
    if long_term:
        settings["allow_long_term_api_keys"] = True
    return settings


def default_mantle_model(region: str) -> str | None:
    """Haiku 4.5 on bedrock-mantle if offered in ``region``, else the first entry that is."""
    entries = [
        e
        for e in default_catalogue()
        if e.provider == "bedrock_mantle" and e.kind == "chat" and region in e.regions
    ]
    for entry in entries:
        if entry.model_id == "anthropic.claude-haiku-4-5":
            return entry.model_id
    return entries[0].model_id if entries else None


# Anthropic ------------------------------------------------------------------------------


def anthropic_direct() -> dict[str, Any]:
    return {"kind": "anthropic", "endpoint": "direct", "api_key": ref("ANTHROPIC_API_KEY")}


def anthropic_foundry(environ: LiveEnv, *, entra: bool) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "kind": "anthropic",
        "endpoint": "foundry",
        "base_url": value(environ, "FOUNDRY_ANTHROPIC_BASE_URL"),
    }
    if entra:
        settings["entra_id"] = _entra(environ)
    else:
        settings["api_key"] = ref("FOUNDRY_ANTHROPIC_API_KEY")
    return settings


def _entra(environ: LiveEnv) -> dict[str, Any]:
    entra: dict[str, Any] = {"mode": "entra_id"}
    client_id = value(environ, "FOUNDRY_MANAGED_IDENTITY_CLIENT_ID")
    if client_id:
        entra["managed_identity_client_id"] = client_id
    return entra


# OpenAI-compatible ----------------------------------------------------------------------


def openai_settings() -> dict[str, Any]:
    return {
        "kind": "openai_compat",
        "service": {"name": "openai", "api_key": ref("OPENAI_API_KEY")},
    }


def xai_settings() -> dict[str, Any]:
    return {"kind": "openai_compat", "service": {"name": "xai", "api_key": ref("XAI_API_KEY")}}


def foundry_settings(
    environ: LiveEnv, *, entra: bool, catalogue_provider: str = "openai"
) -> dict[str, Any]:
    auth: dict[str, Any] = (
        _entra(environ) if entra else {"mode": "api_key", "api_key": ref("FOUNDRY_OPENAI_API_KEY")}
    )
    return {
        "kind": "openai_compat",
        "service": {
            "name": "foundry",
            "endpoint": value(environ, "FOUNDRY_OPENAI_ENDPOINT"),
            "auth": auth,
            "catalogue_provider": catalogue_provider,
        },
    }


def _int(environ: LiveEnv, name: str, default: int) -> int:
    given = value(environ, name)
    return int(given) if given else default


def deployment_overlay(
    environ: LiveEnv, provider: str, deployment: str, *, kind: str = "chat"
) -> tuple[CatalogueOverlay, ...]:
    """A product overlay declaring a Foundry deployment, unless the catalogue has it.

    Foundry model ids are deployment names, which a product declares itself. The limits
    come from ``FOUNDRY_CONTEXT_WINDOW`` and ``FOUNDRY_OUTPUT_LIMIT`` (or, for an
    embedding deployment, ``FOUNDRY_EMBEDDING_DIMENSIONS``), with conservative defaults.
    """
    if default_catalogue().get(provider, deployment) is not None:
        return ()
    if kind == "embedding":
        entry = ModelEntry(
            model_id=deployment,
            provider=provider,
            kind="embedding",
            residency_scope=ResidencyScope.GLOBAL,
            context_window=8191,
            embedding_dimensions=_int(environ, "FOUNDRY_EMBEDDING_DIMENSIONS", 1536),
        )
    else:
        entry = ModelEntry(
            model_id=deployment,
            provider=provider,
            residency_scope=ResidencyScope.GLOBAL,
            capabilities=Capabilities(tools=True, prompt_caching=True, structured_output=True),
            context_window=_int(environ, "FOUNDRY_CONTEXT_WINDOW", 128_000),
            output_limit=_int(environ, "FOUNDRY_OUTPUT_LIMIT", 16_384),
        )
    return (CatalogueOverlay(name="live-foundry", add=(entry,)),)


# Gemini and Ollama ------------------------------------------------------------------------


def gemini_settings() -> dict[str, Any]:
    return {"kind": "gemini", "api_key": ref("GEMINI_API_KEY")}


#: A context length the suite asks Ollama for unless ``OLLAMA_NUM_CTX`` says otherwise:
#: enough for the caching prefix, far below the catalogue's full window.
OLLAMA_NUM_CTX = 16_384


def ollama_settings(environ: LiveEnv) -> dict[str, Any]:
    return {
        "kind": "ollama",
        "base_url": value(environ, "OLLAMA_URL"),
        "num_ctx": _int(environ, "OLLAMA_NUM_CTX", OLLAMA_NUM_CTX),
    }
