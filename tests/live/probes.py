"""Read-only metadata probes for the catalogue verification pass (Q0002, Q0004, Q0021).

conduit's providers expose model calls, not the providers' model metadata, so these
probes read what conduit cannot: the context window and output limit a provider's
model API states, the regions a Bedrock inference profile routes to, and the raw usage
arithmetic of an X.AI response. They are test code only. Each builds its SDK or HTTP
client from a key resolved through ``EnvironmentSecretResolver`` (never read from the
environment directly) and sends no prompt, except the X.AI probe, which sends one fixed
arithmetic question and keeps only the numbers of its usage.

Every probe runs as one guarded step (``_guarded``): resolving the key, constructing the
client, the call and closing the client all happen inside it, so any failure, a
constructor's included, becomes ``ProbeFailed`` naming the exception's class alone,
raised after the handler with no cause and no context. A key stays a ``SecretStr`` until
the constructor call that needs it, and no frame outside the step ever holds it.

A probe returns numbers, region names or ``None`` (nothing observable). It never returns
provider text.
"""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import SecretStr

from conduit.core.secrets import SecretRef
from conduit.providers.anthropic.settings import DEFAULT_BASE_URL as ANTHROPIC_BASE_URL
from conduit.providers.gemini.settings import DEFAULT_BASE_URL as GEMINI_BASE_URL
from conduit.providers.openai_compat.settings import XAI_BASE_URL
from tests.live.needs import LiveEnv, present, value
from tests.live.targets import resolver


class ProbeFailed(Exception):
    """A probe could not observe: names the exception class (or the suite's own reason)."""


@dataclass(frozen=True, slots=True)
class Limits:
    """What a model API states, where it states it."""

    context_window: int | None = None
    output_limit: int | None = None


async def _secret(name: str, environ: LiveEnv) -> SecretStr:
    return await resolver(environ).resolve(SecretRef(name))


def _count(found: Any) -> int | None:
    return found if isinstance(found, int) and not isinstance(found, bool) else None


async def _guarded[T](step: Callable[[], Awaitable[T]]) -> T:
    """Run ``step``; any failure becomes ``ProbeFailed`` naming its class, unchained.

    A ``ProbeFailed`` the step raises itself (with the suite's own words) goes on as it
    is; cancellation propagates unchanged.
    """
    failed = ""
    try:
        return await step()
    except (asyncio.CancelledError, ProbeFailed):
        raise
    except Exception as error:
        failed = type(error).__name__
    raise ProbeFailed(failed)


# Anthropic ------------------------------------------------------------------------------


async def anthropic_limits(model: str, environ: LiveEnv) -> Limits:
    """The Models API's ``max_input_tokens`` and ``max_tokens`` for ``model``, if stated."""

    async def step() -> Limits:
        library: Any = importlib.import_module("anthropic")
        key = await _secret("ANTHROPIC_API_KEY", environ)
        client = library.AsyncAnthropic(api_key=key.get_secret_value(), base_url=ANTHROPIC_BASE_URL)
        try:
            info = await client.models.retrieve(model)
        finally:
            await client.close()
        return Limits(
            context_window=_count(getattr(info, "max_input_tokens", None)),
            output_limit=_count(getattr(info, "max_tokens", None)),
        )

    return await _guarded(step)


# Gemini ---------------------------------------------------------------------------------


async def gemini_limits(model: str, environ: LiveEnv) -> Limits:
    """The Gemini API's ``input_token_limit`` and ``output_token_limit`` for ``model``."""

    async def step() -> Limits:
        genai: Any = importlib.import_module("google.genai")
        key = await _secret("GEMINI_API_KEY", environ)
        client = genai.Client(
            api_key=key.get_secret_value(),
            vertexai=False,
            http_options={"base_url": GEMINI_BASE_URL},
        )
        try:
            info = await client.aio.models.get(model=model)
        finally:
            await client.aio.aclose()
            client.close()
        return Limits(
            context_window=_count(getattr(info, "input_token_limit", None)),
            output_limit=_count(getattr(info, "output_token_limit", None)),
        )

    return await _guarded(step)


# Ollama ---------------------------------------------------------------------------------


async def ollama_context(model: str, environ: LiveEnv) -> int | None:
    """The model's trained context length from ``/api/show``, or ``None``."""

    async def step() -> int | None:
        httpx: Any = importlib.import_module("httpx")
        base = value(environ, "OLLAMA_URL")
        async with httpx.AsyncClient(base_url=base, trust_env=False, timeout=30) as client:
            response = await client.post("/api/show", json={"model": model})
            if response.status_code == 404:
                raise ProbeFailed("model not pulled on the Ollama server")
            if response.status_code != 200:
                raise ProbeFailed(f"HTTP {response.status_code}")
            info = response.json().get("model_info") or {}
        lengths = [
            _count(found) for key, found in info.items() if str(key).endswith(".context_length")
        ]
        known = [length for length in lengths if length is not None]
        return known[0] if known else None

    return await _guarded(step)


# Bedrock --------------------------------------------------------------------------------


async def bedrock_control_client(region: str, environ: LiveEnv) -> Any:
    """A Bedrock control-plane client in ``region``, from AWS credentials (SigV4)."""

    async def step() -> Any:
        boto3: Any = importlib.import_module("boto3")
        if present(environ, "BEDROCK_CREDENTIAL_CHAIN"):
            profile = value(environ, "BEDROCK_PROFILE")
            session = boto3.session.Session(**({"profile_name": profile} if profile else {}))
        elif present(environ, "AWS_ACCESS_KEY_ID") and present(environ, "AWS_SECRET_ACCESS_KEY"):
            key_id = await _secret("AWS_ACCESS_KEY_ID", environ)
            secret = await _secret("AWS_SECRET_ACCESS_KEY", environ)
            token = (
                await _secret("AWS_SESSION_TOKEN", environ)
                if present(environ, "AWS_SESSION_TOKEN")
                else None
            )
            session = boto3.session.Session(
                aws_access_key_id=key_id.get_secret_value(),
                aws_secret_access_key=secret.get_secret_value(),
                aws_session_token=token.get_secret_value() if token is not None else None,
            )
        else:
            raise ProbeFailed("no AWS credentials for the control plane")
        return session.client("bedrock", region_name=region)

    return await _guarded(step)


async def bedrock_profile_regions(client: Any, profile_id: str) -> frozenset[str]:
    """The regions an inference profile routes to, from its models' ARNs."""

    async def step() -> frozenset[str]:
        found = await asyncio.to_thread(
            client.get_inference_profile, inferenceProfileIdentifier=profile_id
        )
        regions = set()
        for model in found.get("models") or ():
            parts = str(model.get("modelArn", "")).split(":")
            if len(parts) > 3 and parts[3]:
                regions.add(parts[3])
        return frozenset(regions)

    return await _guarded(step)


async def bedrock_model_exists(client: Any, model_id: str) -> bool:
    """Whether the region's control plane knows the foundation model ``model_id``."""

    async def step() -> bool:
        found = await asyncio.to_thread(client.get_foundation_model, modelIdentifier=model_id)
        return bool(found.get("modelDetails"))

    return await _guarded(step)


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


async def xai_raw_usage(model: str, max_completion_tokens: int, environ: LiveEnv) -> RawUsage:
    """One non-streaming X.AI call; only its usage numbers are kept."""

    async def step() -> RawUsage:
        openai: Any = importlib.import_module("openai")
        key = await _secret("XAI_API_KEY", environ)
        client = openai.AsyncOpenAI(
            api_key=key.get_secret_value(), base_url=XAI_BASE_URL, organization=None, project=None
        )
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": "What is 17 multiplied by 23? Answer briefly."}
                ],
                max_completion_tokens=max_completion_tokens,
            )
        finally:
            await client.close()
        usage = response.usage
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        return RawUsage(
            prompt=_count(getattr(usage, "prompt_tokens", None)) or 0,
            completion=_count(getattr(usage, "completion_tokens", None)) or 0,
            reasoning=_count(getattr(completion_details, "reasoning_tokens", None)) or 0,
            cached=_count(getattr(prompt_details, "cached_tokens", None)) or 0,
            total=_count(getattr(usage, "total_tokens", None)),
        )

    return await _guarded(step)


async def xai_accepts_output(
    model: str, max_completion_tokens: int, environ: LiveEnv
) -> tuple[bool, str]:
    """Whether X.AI accepts ``max_completion_tokens`` for ``model`` (one short call)."""
    try:
        await xai_raw_usage(model, max_completion_tokens, environ)
    except ProbeFailed as failed:
        return False, str(failed)
    return True, ""
