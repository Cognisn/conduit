"""Live: the OpenAI-compatible adapter on OpenAI, X.AI and Microsoft Foundry.

Foundry runs with an API key and with Microsoft Entra ID (azure-identity's
``DefaultAzureCredential``: sign in with the Azure CLI, or set the ``AZURE_`` variables
its environment credential reads). X.AI also settles Q0021 (its reasoning-token
arithmetic) and grok-4.3's output limit.
"""

from __future__ import annotations

import pytest

from conduit.core.catalogue import default_catalogue
from tests.live import needs
from tests.live.harness import (
    check_caching,
    check_chat,
    check_output_limit,
    check_tools,
    started,
)
from tests.live.needs import live_env, value
from tests.live.probes import ProbeFailed, xai_accepts_output, xai_raw_usage
from tests.live.report import REPORT, Verdict
from tests.live.targets import (
    Target,
    deployment_overlay,
    foundry_settings,
    openai_settings,
    xai_settings,
)

pytestmark = pytest.mark.integration

DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_XAI_MODEL = "grok-4.3"


def _openai() -> Target:
    model = value(live_env(), "OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    assert model is not None
    return Target("openai", openai_settings(), model)


def _xai() -> Target:
    model = value(live_env(), "XAI_MODEL", DEFAULT_XAI_MODEL)
    assert model is not None
    return Target("xai", xai_settings(), model)


def _foundry(entra: bool, family: str) -> Target:
    variable = "FOUNDRY_GROK_DEPLOYMENT" if family == "xai" else "FOUNDRY_OPENAI_DEPLOYMENT"
    deployment = value(live_env(), variable)
    assert deployment is not None  # the gate checked it
    label = f"foundry {'entra-id' if entra else 'api-key'} {family}"
    return Target(
        label,
        foundry_settings(live_env(), entra=entra, catalogue_provider=family),
        deployment,
        deployment_overlay(live_env(), family, deployment),
    )


# OpenAI ---------------------------------------------------------------------------------


@pytest.mark.live_needs(needs.OPENAI)
async def test_openai_streaming_chat() -> None:
    target = _openai()
    async with started(target) as binding:
        await check_chat(binding, target.subject)


@pytest.mark.live_needs(needs.OPENAI)
async def test_openai_tool_loop() -> None:
    target = _openai()
    async with started(target) as binding:
        await check_tools(binding, target.subject)


@pytest.mark.live_needs(needs.OPENAI)
async def test_openai_cached_second_call() -> None:
    target = _openai()
    async with started(target) as binding:
        await check_caching(binding, target.subject, explicit=False, strict=True)


# X.AI -----------------------------------------------------------------------------------


@pytest.mark.live_needs(needs.XAI)
async def test_xai_streaming_chat() -> None:
    target = _xai()
    async with started(target) as binding:
        await check_chat(binding, target.subject)


@pytest.mark.live_needs(needs.XAI)
async def test_xai_tool_loop() -> None:
    target = _xai()
    async with started(target) as binding:
        await check_tools(binding, target.subject)


@pytest.mark.live_needs(needs.XAI)
async def test_xai_cached_second_call() -> None:
    target = _xai()
    async with started(target) as binding:
        await check_caching(binding, target.subject, explicit=False, strict=True)


@pytest.mark.live_needs(needs.XAI)
async def test_xai_reasoning_token_arithmetic() -> None:
    """Q0021: whether X.AI's ``total_tokens`` counts reasoning beside the completion."""
    target = _xai()
    try:
        raw = await xai_raw_usage(target.model, 2048, live_env())
    except ProbeFailed as failed:
        pytest.fail(f"the X.AI usage probe failed: {failed}")
    outside = raw.reasoning_outside_completion
    numbers = (
        f"prompt {raw.prompt}, completion {raw.completion}, reasoning {raw.reasoning}, "
        f"cached {raw.cached}, total {raw.total}"
    )
    if outside is None:
        verdict, words = Verdict.UNOBSERVED, "neither arithmetic holds, or no reasoning"
    elif outside:
        verdict, words = Verdict.FINDING, "reasoning is outside completion_tokens (added)"
    else:
        verdict, words = Verdict.FINDING, "reasoning is inside completion_tokens"
    REPORT.record(
        "usage", target.subject, "Q0021 reasoning arithmetic", verdict, f"{words}; {numbers}"
    )


@pytest.mark.live_needs(needs.XAI)
async def test_xai_output_limit() -> None:
    """grok-4.3's catalogued output limit is accepted, and whether one more is refused."""
    target = _xai()
    entry = default_catalogue().require("xai", target.model)
    limit = entry.output_limit
    assert limit is not None
    async with started(target, max_output_tokens=limit) as binding:
        accepted, failure = await check_output_limit(binding, limit)
    REPORT.record(
        "catalogue",
        target.subject,
        f"output limit {limit} accepted",
        Verdict.MATCH if accepted else Verdict.MISMATCH,
        failure,
    )
    above, why = await xai_accepts_output(target.model, limit + 1, live_env())
    REPORT.record(
        "catalogue",
        target.subject,
        f"output limit {limit + 1} (one above)",
        Verdict.FINDING,
        "accepted: the catalogue may be low" if above else f"refused: {why}",
    )
    assert accepted, f"X.AI refused the catalogued output limit ({failure})"


# Foundry --------------------------------------------------------------------------------

FOUNDRY_AUTH = [
    pytest.param(False, marks=pytest.mark.live_needs(needs.FOUNDRY_OPENAI_KEY), id="api-key"),
    pytest.param(True, marks=pytest.mark.live_needs(needs.FOUNDRY_OPENAI_ENTRA), id="entra-id"),
]


@pytest.mark.parametrize("entra", FOUNDRY_AUTH)
async def test_foundry_streaming_chat(entra: bool) -> None:
    target = _foundry(entra, "openai")
    async with started(target) as binding:
        await check_chat(binding, target.subject)
    if entra:
        REPORT.record("auth", target.subject, "Entra ID token accepted", Verdict.MATCH)


@pytest.mark.parametrize("entra", FOUNDRY_AUTH)
async def test_foundry_tool_loop(entra: bool) -> None:
    target = _foundry(entra, "openai")
    async with started(target) as binding:
        await check_tools(binding, target.subject)


@pytest.mark.parametrize("entra", FOUNDRY_AUTH)
async def test_foundry_cached_second_call(entra: bool) -> None:
    """Foundry caches automatically where the deployment's model does: best effort."""
    target = _foundry(entra, "openai")
    async with started(target) as binding:
        await check_caching(binding, target.subject, explicit=False, strict=False)


@pytest.mark.parametrize("entra", FOUNDRY_AUTH)
@pytest.mark.live_needs(needs.FOUNDRY_GROK)
async def test_foundry_grok_streaming_chat_and_tools(entra: bool) -> None:
    target = _foundry(entra, "xai")
    async with started(target) as binding:
        await check_chat(binding, target.subject)
        await check_tools(binding, target.subject)
