"""Live: Anthropic, direct and Claude in Microsoft Foundry (API key and Entra ID)."""

from __future__ import annotations

import os

import pytest

from tests.live import needs
from tests.live.harness import check_caching, check_chat, check_tools, started
from tests.live.needs import value
from tests.live.report import REPORT, Verdict
from tests.live.targets import Target, anthropic_direct, anthropic_foundry

pytestmark = pytest.mark.integration

#: The cheapest Claude in the default catalogue.
DEFAULT_MODEL = "claude-haiku-4-5"
#: A model for the tool loop with thinking (Claude 4.6 and later think adaptively).
DEFAULT_THINKING_MODEL = "claude-sonnet-4-6"


def _direct(model_variable: str = "ANTHROPIC_MODEL", default: str = DEFAULT_MODEL) -> Target:
    model = value(os.environ, model_variable, default)
    assert model is not None
    return Target("anthropic direct", anthropic_direct(), model)


def _foundry(entra: bool) -> Target:
    model = value(os.environ, "FOUNDRY_ANTHROPIC_MODEL", DEFAULT_MODEL)
    assert model is not None
    label = "anthropic foundry " + ("entra-id" if entra else "api-key")
    return Target(label, anthropic_foundry(os.environ, entra=entra), model)


@pytest.mark.live_needs(needs.ANTHROPIC_DIRECT)
async def test_direct_streaming_chat() -> None:
    target = _direct()
    async with started(target) as binding:
        await check_chat(binding, target.subject)


@pytest.mark.live_needs(needs.ANTHROPIC_DIRECT)
async def test_direct_tool_loop() -> None:
    target = _direct()
    async with started(target) as binding:
        await check_tools(binding, target.subject)


@pytest.mark.live_needs(needs.ANTHROPIC_DIRECT)
async def test_direct_tool_loop_with_thinking() -> None:
    """A tool loop on a thinking model, its thinking blocks sent back unchanged.

    conduit's request has no field that turns extended thinking on, so this observes
    whether the model thinks by default; when it does not, the report says so and the
    loop is still checked.
    """
    target = _direct("ANTHROPIC_THINKING_MODEL", DEFAULT_THINKING_MODEL)
    async with started(target) as binding:
        state_blocks = await check_tools(binding, target.subject)
    REPORT.record(
        "tools",
        target.subject,
        "thinking state in a tool loop",
        Verdict.MATCH if state_blocks else Verdict.UNOBSERVED,
        f"{state_blocks} thinking block(s) returned and sent back",
    )


@pytest.mark.live_needs(needs.ANTHROPIC_DIRECT)
async def test_direct_cache_breakpoint_writes_then_reads() -> None:
    target = _direct()
    async with started(target) as binding:
        await check_caching(binding, target.subject, explicit=True, strict=True)


FOUNDRY_AUTH = [
    pytest.param(False, marks=pytest.mark.live_needs(needs.FOUNDRY_ANTHROPIC_KEY), id="api-key"),
    pytest.param(True, marks=pytest.mark.live_needs(needs.FOUNDRY_ANTHROPIC_ENTRA), id="entra-id"),
]


@pytest.mark.parametrize("entra", FOUNDRY_AUTH)
async def test_foundry_streaming_chat(entra: bool) -> None:
    target = _foundry(entra)
    async with started(target) as binding:
        await check_chat(binding, target.subject)
    if entra:
        REPORT.record("auth", target.subject, "Entra ID token accepted", Verdict.MATCH)


@pytest.mark.parametrize("entra", FOUNDRY_AUTH)
async def test_foundry_tool_loop(entra: bool) -> None:
    target = _foundry(entra)
    async with started(target) as binding:
        await check_tools(binding, target.subject)


@pytest.mark.parametrize("entra", FOUNDRY_AUTH)
async def test_foundry_cache_breakpoint_writes_then_reads(entra: bool) -> None:
    target = _foundry(entra)
    async with started(target) as binding:
        await check_caching(binding, target.subject, explicit=True, strict=True)
