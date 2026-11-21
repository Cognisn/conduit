"""Live: Amazon Bedrock (bedrock-runtime) in every authentication mode, residency, and Q0004.

Runs only with ``-m integration`` and the variables each test's needs name.
"""

from __future__ import annotations

import pytest

from conduit.core.catalogue import default_catalogue
from conduit.core.residency import ResidencyScope
from tests.live import needs
from tests.live.harness import (
    check_caching,
    check_chat,
    check_residency_refusal,
    check_tools,
    in_region_policy,
    started,
)
from tests.live.needs import live_env, value
from tests.live.report import REPORT, Verdict
from tests.live.targets import (
    BEDROCK_AUTH_MODES,
    Target,
    bedrock_settings,
    default_bedrock_model,
    first_bedrock_auth,
    regions_list,
)

pytestmark = pytest.mark.integration

#: Each authentication mode and what it needs.
AUTH_NEEDS = {
    "credential_chain": needs.BEDROCK_CHAIN,
    "access_keys": needs.BEDROCK_ACCESS_KEYS,
    "short_term_from_chain": needs.BEDROCK_CHAIN,
    "short_term_from_access_keys": needs.BEDROCK_ACCESS_KEYS,
    "long_term_api_key": needs.BEDROCK_LONG_TERM_KEY,
}

#: The in-region (bare id) Claude entries Q0004 asks about.
IN_REGION_SONNETS = ("anthropic.claude-sonnet-4-6", "anthropic.claude-sonnet-5")


def _region() -> str:
    region = value(live_env(), "BEDROCK_REGION")
    assert region is not None  # the gate checked it
    return region


def _target(mode: str) -> Target:
    region = _region()
    model = value(live_env(), "BEDROCK_MODEL") or default_bedrock_model(region)
    if model is None:
        pytest.skip(f"no catalogued Claude profile serves {region}; set CONDUIT_LIVE_BEDROCK_MODEL")
    return Target(f"bedrock {region} {mode}", bedrock_settings(region, mode, live_env()), model)


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param(mode, marks=pytest.mark.live_needs(needs.BEDROCK_REGION, AUTH_NEEDS[mode]))
        for mode in BEDROCK_AUTH_MODES
    ],
)
async def test_streaming_chat_in_each_authentication_mode(mode: str) -> None:
    target = _target(mode)
    async with started(target) as binding:
        await check_chat(binding, target.subject)


@pytest.mark.live_needs(needs.BEDROCK_REGION, needs.BEDROCK_ANY_AUTH)
async def test_tool_loop_with_reasoning_state_sent_back() -> None:
    mode = first_bedrock_auth(live_env())
    assert mode is not None
    target = _target(mode)
    async with started(target) as binding:
        await check_tools(binding, target.subject)


@pytest.mark.live_needs(needs.BEDROCK_REGION, needs.BEDROCK_ANY_AUTH)
async def test_cache_point_writes_then_reads() -> None:
    mode = first_bedrock_auth(live_env())
    assert mode is not None
    target = _target(mode)
    async with started(target) as binding:
        await check_caching(binding, target.subject, explicit=True, strict=True)


@pytest.mark.live_needs(needs.BEDROCK_REGION, needs.BEDROCK_ANY_AUTH)
async def test_cross_region_profile_is_refused_under_an_in_region_policy() -> None:
    region = _region()
    mode = first_bedrock_auth(live_env())
    assert mode is not None
    profiles = [
        entry
        for entry in default_catalogue()
        if entry.provider == "bedrock"
        and entry.residency_scope is ResidencyScope.CROSS_REGION
        and region in entry.regions
    ]
    if not profiles:
        pytest.skip(f"no catalogued cross-region profile serves {region}")
    target = Target(
        f"bedrock {region} {mode}", bedrock_settings(region, mode, live_env()), profiles[0].model_id
    )
    await check_residency_refusal(target, region, target.subject)


@pytest.mark.parametrize("model", IN_REGION_SONNETS)
@pytest.mark.live_needs(needs.BEDROCK_REGION, needs.BEDROCK_ANY_AUTH)
async def test_bare_id_in_region_invocation(model: str) -> None:
    """Q0004: the bare id is invoked in each catalogued region the run covers, under an
    in-region policy for that region."""
    entry = default_catalogue().require("bedrock", model)
    mode = first_bedrock_auth(live_env())
    assert mode is not None
    covered = [
        region
        for region in regions_list(live_env(), "BEDROCK_REGIONS", "BEDROCK_REGION")
        if region in entry.regions
    ]
    if not covered:
        pytest.skip(
            f"{model} is catalogued in {', '.join(sorted(entry.regions))}; name one in "
            "CONDUIT_LIVE_BEDROCK_REGIONS"
        )
    for region in covered:
        target = Target(
            f"bedrock {region} {mode}", bedrock_settings(region, mode, live_env()), model
        )
        async with started(target, residency=in_region_policy(region)) as binding:
            await check_chat(binding, target.subject)
        REPORT.record("catalogue", target.subject, "Q0004 bare id in region", Verdict.MATCH)
