"""Live: Claude on bedrock-mantle with short-term keys (and a long-term key, if allowed).

Confirms that bedrock-mantle accepts conduit's short-term Bedrock API keys, streams,
uses tools, and reports usage with a cached second call.
"""

from __future__ import annotations

import pytest

from conduit.core.catalogue import default_catalogue
from tests.live import needs
from tests.live.harness import check_caching, check_chat, check_tools, started
from tests.live.needs import live_env, value
from tests.live.report import REPORT, Verdict
from tests.live.targets import (
    Target,
    bedrock_auth,
    default_mantle_model,
    mantle_auth,
    mantle_settings,
)

pytestmark = [pytest.mark.integration, pytest.mark.live_needs(needs.MANTLE_REGION)]

KEY_SOURCES = {
    "short_term_from_chain": needs.BEDROCK_CHAIN,
    "short_term_from_access_keys": needs.BEDROCK_ACCESS_KEYS,
    "long_term_api_key": needs.BEDROCK_LONG_TERM_KEY,
}


def _model(region: str) -> str:
    model = value(live_env(), "BEDROCK_MANTLE_MODEL") or default_mantle_model(region)
    if model is None:
        pytest.skip(
            f"no catalogued bedrock-mantle model in {region}; set CONDUIT_LIVE_BEDROCK_MANTLE_MODEL"
        )
    return model


def _region() -> str:
    region = value(live_env(), "BEDROCK_MANTLE_REGION")
    assert region is not None  # the gate checked it
    return region


def _caching_model(region: str) -> str:
    """The run's model if the catalogue marks it for caching, else one that is."""
    chosen = _model(region)
    catalogue = default_catalogue()
    entry = catalogue.get("bedrock_mantle", chosen)
    if entry is not None and entry.capabilities.prompt_caching:
        return chosen
    for candidate in catalogue:
        if (
            candidate.provider == "bedrock_mantle"
            and candidate.capabilities.prompt_caching
            and region in candidate.regions
        ):
            return candidate.model_id
    pytest.skip(f"no bedrock-mantle model in {region} is catalogued for prompt caching")


def _target(*, caching: bool = False) -> Target:
    region = _region()
    chosen = mantle_auth(live_env())
    assert chosen is not None
    auth, long_term = chosen
    return Target(
        f"bedrock-mantle {region} {auth['mode']}",
        mantle_settings(region, auth, long_term),
        _caching_model(region) if caching else _model(region),
    )


@pytest.mark.parametrize(
    "source",
    [pytest.param(name, marks=pytest.mark.live_needs(need)) for name, need in KEY_SOURCES.items()],
)
async def test_streaming_chat_with_each_key_source(source: str) -> None:
    """Short-term keys from each AWS credential source are accepted (and a long-term key)."""
    region = _region()
    auth, long_term = bedrock_auth(source, live_env())
    target = Target(
        f"bedrock-mantle {region} {source}",
        mantle_settings(region, auth, long_term),
        _model(region),
    )
    async with started(target) as binding:
        await check_chat(binding, target.subject)
    if source.startswith("short_term"):
        REPORT.record("auth", target.subject, "short-term key accepted", Verdict.MATCH)


@pytest.mark.live_needs(needs.MANTLE_ANY_AUTH)
async def test_tool_loop_with_thinking_state_sent_back() -> None:
    target = _target()
    async with started(target) as binding:
        await check_tools(binding, target.subject)


@pytest.mark.live_needs(needs.MANTLE_ANY_AUTH)
async def test_cache_breakpoint_writes_then_reads() -> None:
    target = _target(caching=True)
    async with started(target) as binding:
        await check_caching(binding, target.subject, explicit=True, strict=True)
