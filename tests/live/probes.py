"""Read-only metadata probes for the catalogue verification pass (Q0002, Q0004, Q0021).

conduit's providers expose model calls, not the providers' model metadata, so these
probes read what conduit cannot: the context window and output limit a provider's
model API states, the regions a Bedrock inference profile routes to, and the raw usage
arithmetic of an X.AI response. They are test code only. Each builds its SDK or HTTP
client from a key resolved through ``EnvironmentSecretResolver`` (never read from the
environment directly) and sends no prompt, except the X.AI probe, which sends one fixed
arithmetic question and keeps only the numbers of its usage.

A probe returns numbers, region names or ``None`` (nothing observable). It never returns
provider text; a failure is reported by its exception's class name alone, raised after
the handler so no provider message travels with it.
"""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from conduit.core.secrets import SecretRef
from conduit.providers.anthropic.settings import DEFAULT_BASE_URL as ANTHROPIC_BASE_URL
from conduit.providers.gemini.settings import DEFAULT_BASE_URL as GEMINI_BASE_URL
from conduit.providers.openai_compat.settings import XAI_BASE_URL
from tests.live.needs import present, value
from tests.live.targets import resolver


class ProbeFailed(Exception):
    """A probe could not observe: names the exception class only."""


@dataclass(frozen=True, slots=True)
class Limits:
    """What a model API states, where it states it."""

    context_window: int | None = None
    output_limit: int | None = None


async def _secret(name: str, environ: Mapping[str, str] | None) -> str:
    return (await resolver(environ).resolve(SecretRef(name))).get_secret_value()


def _count(found: Any) -> int | None:
    return found if isinstance(found, int) and not isinstance(found, bool) else None


async def _guarded(call: Any) -> Any:
    """Await ``call``; any failure becomes ``ProbeFailed`` naming its class, unchained."""
    failed = ""
    try:
        return await call
    except asyncio.CancelledError:
        raise
    except Exception as error:
        failed = type(error).__name__
    raise ProbeFailed(failed)


# Anthropic ------------------------------------------------------------------------------


async def anthropic_limits(model: str, environ: Mapping[str, str] | None = None) -> Limits:
    """The Models API's ``max_input_tokens`` and ``max_tokens`` for ``model``, if stated."""
    library: Any = importlib.import_module("anthropic")
    client = library.AsyncAnthropic(
        api_key=await _secret("ANTHROPIC_API_KEY", environ), base_url=ANTHROPIC_BASE_URL
    )
    try:
        info = await _guarded(client.models.retrieve(model))
    finally:
        await client.close()
    return Limits(
        context_window=_count(getattr(info, "max_input_tokens", None)),
        output_limit=_count(getattr(info, "max_tokens", None)),
    )


# Gemini ---------------------------------------------------------------------------------


async def gemini_limits(model: str, environ: Mapping[str, str] | None = None) -> Limits:
    """The Gemini API's ``input_token_limit`` and ``output_token_limit`` for ``model``."""
    genai: Any = importlib.import_module("google.genai")
    client = genai.Client(
        api_key=await _secret("GEMINI_API_KEY", environ),
        vertexai=False,
        http_options={"base_url": GEMINI_BASE_URL},
    )
    info = await _guarded(client.aio.models.get(model=model))
    return Limits(
        context_window=_count(getattr(info, "input_token_limit", None)),
        output_limit=_count(getattr(info, "output_token_limit", None)),
    )


# Ollama ---------------------------------------------------------------------------------


async def ollama_context(model: str, environ: Mapping[str, str]) -> int | None:
    """The model's trained context length from ``/api/show``, or ``None``."""
    httpx: Any = importlib.import_module("httpx")
    base = value(environ, "OLLAMA_URL")
    async with httpx.AsyncClient(base_url=base, trust_env=False, timeout=30) as client:
        response = await _guarded(client.post("/api/show", json={"model": model}))
        if response.status_code == 404:
            raise ProbeFailed("model not pulled on the Ollama server")
        if response.status_code != 200:
            raise ProbeFailed(f"HTTP {response.status_code}")
        info = response.json().get("model_info") or {}
    lengths = [_count(found) for key, found in info.items() if str(key).endswith(".context_length")]
    known = [length for length in lengths if length is not None]
    return known[0] if known else None


# Bedrock --------------------------------------------------------------------------------


def _aws_session(environ: Mapping[str, str]) -> tuple[Any, dict[str, str]]:
    """A boto3 session factory and the access keys to pass, from the live variables.

    Access keys are resolved through the secret resolver by the caller; the credential
    chain (and an optional ``BEDROCK_PROFILE``) is boto3's own.
    """
    boto3: Any = importlib.import_module("boto3")
    profile = value(environ, "BEDROCK_PROFILE")
    return boto3, ({"profile_name": profile} if profile else {})


async def bedrock_control_client(region: str, environ: Mapping[str, str]) -> Any:
    """A Bedrock control-plane client in ``region``, from AWS credentials (SigV4)."""
    boto3, options = _aws_session(environ)
    if present(environ, "BEDROCK_CREDENTIAL_CHAIN"):
        session = boto3.session.Session(**options)
    elif present(environ, "AWS_ACCESS_KEY_ID") and present(environ, "AWS_SECRET_ACCESS_KEY"):
        keys = {
            "aws_access_key_id": await _secret("AWS_ACCESS_KEY_ID", environ),
            "aws_secret_access_key": await _secret("AWS_SECRET_ACCESS_KEY", environ),
        }
        if present(environ, "AWS_SESSION_TOKEN"):
            keys["aws_session_token"] = await _secret("AWS_SESSION_TOKEN", environ)
        session = boto3.session.Session(**keys)
    else:
        raise ProbeFailed("no AWS credentials for the control plane")
    return session.client("bedrock", region_name=region)


async def bedrock_profile_regions(client: Any, profile_id: str) -> frozenset[str]:
    """The regions an inference profile routes to, from its models' ARNs."""

    def call() -> Any:
        return client.get_inference_profile(inferenceProfileIdentifier=profile_id)

    found = await _guarded(asyncio.to_thread(call))
    regions = set()
    for model in found.get("models") or ():
        arn = str(model.get("modelArn", ""))
        parts = arn.split(":")
        if len(parts) > 3 and parts[3]:
            regions.add(parts[3])
    return frozenset(regions)


async def bedrock_model_exists(client: Any, model_id: str) -> bool:
    """Whether the region's control plane knows the foundation model ``model_id``."""

    def call() -> Any:
        return client.get_foundation_model(modelIdentifier=model_id)

    found = await _guarded(asyncio.to_thread(call))
    return bool(found.get("modelDetails"))


# X.AI -----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RawUsage:
    prompt: int
    completion: int
    reasoning: int
    cached: int
    total: int | None

    @property
    def reasoning_outside_completion(self) -> bool | None:
        """True when ``total`` is prompt, completion and reasoning added together; False
        when it is prompt and completion only; ``None`` when neither (or no reasoning)."""
        if not self.reasoning or self.total is None:
            return None
        if self.total == self.prompt + self.completion + self.reasoning:
            return True
        if self.total == self.prompt + self.completion:
            return False
        return None


def _details(usage: Any, name: str) -> Any:
    return getattr(usage, name, None)


async def xai_raw_usage(
    model: str, max_completion_tokens: int, environ: Mapping[str, str] | None = None
) -> RawUsage:
    """One non-streaming X.AI call; only its usage numbers are kept."""
    openai: Any = importlib.import_module("openai")
    client = openai.AsyncOpenAI(
        api_key=await _secret("XAI_API_KEY", environ),
        base_url=XAI_BASE_URL,
        organization=None,
        project=None,
    )
    try:
        response = await _guarded(
            client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": "What is 17 multiplied by 23? Answer briefly."}
                ],
                max_completion_tokens=max_completion_tokens,
            )
        )
    finally:
        await client.close()
    usage = response.usage
    prompt_details = _details(usage, "prompt_tokens_details")
    completion_details = _details(usage, "completion_tokens_details")
    return RawUsage(
        prompt=_count(getattr(usage, "prompt_tokens", None)) or 0,
        completion=_count(getattr(usage, "completion_tokens", None)) or 0,
        reasoning=_count(getattr(completion_details, "reasoning_tokens", None)) or 0,
        cached=_count(getattr(prompt_details, "cached_tokens", None)) or 0,
        total=_count(getattr(usage, "total_tokens", None)),
    )


async def xai_accepts_output(
    model: str, max_completion_tokens: int, environ: Mapping[str, str] | None = None
) -> tuple[bool, str]:
    """Whether X.AI accepts ``max_completion_tokens`` for ``model`` (one short call)."""
    try:
        await xai_raw_usage(model, max_completion_tokens, environ)
    except ProbeFailed as failed:
        return False, str(failed)
    return True, ""
