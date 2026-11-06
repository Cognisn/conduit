"""Live: the default catalogue verification pass (Q0002), with Bedrock's region lists.

One test per declared model. Each calls the model through conduit, for every route the
run has credentials and regions for, and checks what it can observe against the entry:

- the model is invocable (in each catalogued region the run covers, for in-region
  entries: Q0004's bare ids among them);
- the provider accepts a request at the catalogued output limit;
- each declared capability: tools, prompt caching, structured output, vision;
- what a metadata API states: Anthropic's and Gemini's context window and output limit,
  Ollama's trained context length, and the regions a Bedrock inference profile routes
  to (compared with the entry's region list).

Everything observed goes into the report. A mismatch fails the entry's test, naming
the checks; what cannot be observed (OpenAI's and X.AI's limits, which no API states;
a model that chose not to call the tool) is reported as unobserved and never fails.

Opt in with ``CONDUIT_LIVE_CATALOGUE=1`` as well as each provider's credentials; select
providers with ``-k`` (``-k "bedrock and not mantle"``).
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Mapping

import pytest

from conduit.core.catalogue import ModelEntry, default_catalogue
from conduit.core.errors import ProviderError
from conduit.core.residency import ResidencyScope
from tests.live import needs
from tests.live.harness import (
    check_caching,
    check_chat,
    check_embeddings,
    check_output_limit,
    check_structured_output,
    check_tools,
    check_vision,
    started,
    started_embeddings,
)
from tests.live.needs import Need
from tests.live.probes import (
    Limits,
    ProbeFailed,
    anthropic_limits,
    bedrock_control_client,
    bedrock_model_exists,
    bedrock_profile_regions,
    gemini_limits,
    ollama_context,
)
from tests.live.report import REPORT, Verdict
from tests.live.targets import (
    Target,
    anthropic_direct,
    bedrock_settings,
    first_bedrock_auth,
    gemini_settings,
    mantle_auth,
    mantle_settings,
    offered_in,
    ollama_settings,
    openai_settings,
    regions_list,
    xai_settings,
)

pytestmark = pytest.mark.integration

#: What each catalogue provider needs, besides the opt-in.
PROVIDER_NEEDS: dict[str, tuple[Need, ...]] = {
    "anthropic": (needs.ANTHROPIC_DIRECT,),
    "bedrock": (needs.BEDROCK_REGION, needs.BEDROCK_ANY_AUTH),
    "bedrock_mantle": (needs.MANTLE_REGION, needs.MANTLE_ANY_AUTH),
    "openai": (needs.OPENAI,),
    "xai": (needs.XAI,),
    "gemini": (needs.GEMINI,),
    "ollama": (needs.OLLAMA,),
}

#: Providers whose caches are written through explicit breakpoints.
EXPLICIT_CACHES = frozenset({"anthropic", "bedrock", "bedrock_mantle"})

#: Capabilities the catalogue declares that an adapter refuses before any call: reported
#: as unobserved with the reason, since the provider is never asked (Skuld Q0042).
KNOWN_ADAPTER_GAPS: dict[tuple[str, str], str] = {
    ("bedrock", "structured output"): (
        "known adapter gap: Converse structured output is not mapped, refused before any call"
    ),
}

ENTRIES = [entry for entry in default_catalogue() if entry.provider != "stub"]


def _param(entry: ModelEntry) -> object:
    return pytest.param(
        entry,
        marks=pytest.mark.live_needs(needs.CATALOGUE_PASS, *PROVIDER_NEEDS[entry.provider]),
        id=f"{entry.provider}:{entry.model_id}",
    )


def routes(entry: ModelEntry, environ: Mapping[str, str]) -> list[tuple[Target, str | None]]:
    """The targets to call ``entry`` through, each with the region it is called in."""
    model = entry.model_id
    if entry.provider == "anthropic":
        return [(Target("anthropic direct", anthropic_direct(), model), None)]
    if entry.provider == "openai":
        return [(Target("openai", openai_settings(), model), None)]
    if entry.provider == "xai":
        return [(Target("xai", xai_settings(), model), None)]
    if entry.provider == "gemini":
        return [(Target("gemini", gemini_settings(), model), None)]
    if entry.provider == "ollama":
        return [(Target("ollama", ollama_settings(environ), model), None)]
    if entry.provider == "bedrock":
        mode = first_bedrock_auth(environ)
        assert mode is not None
        covered = [
            region
            for region in regions_list(environ, "BEDROCK_REGIONS", "BEDROCK_REGION")
            if offered_in(entry, region)
        ]
        if entry.residency_scope is not ResidencyScope.IN_REGION:
            covered = covered[:1]
        return [
            (
                Target(f"bedrock {region} {mode}", bedrock_settings(region, mode, environ), model),
                region,
            )
            for region in covered
        ]
    if entry.provider == "bedrock_mantle":
        chosen = mantle_auth(environ)
        assert chosen is not None
        auth, long_term = chosen
        covered = [
            region
            for region in regions_list(environ, "BEDROCK_MANTLE_REGIONS", "BEDROCK_MANTLE_REGION")
            if offered_in(entry, region)
        ]
        return [
            (
                Target(
                    f"bedrock-mantle {region} {auth['mode']}",
                    mantle_settings(region, auth, long_term),
                    model,
                ),
                region,
            )
            for region in covered
        ]
    return []


async def _guard(subject: str, check: str, coro: Awaitable[object]) -> bool:
    """Run one check; a provider refusal is a mismatch, a model's choice is unobserved."""
    try:
        await coro
    except ProviderError as error:
        REPORT.record(
            "catalogue", subject, check, Verdict.MISMATCH, f"ProviderError({error.failure.value})"
        )
        return False
    except AssertionError as error:
        REPORT.record("catalogue", subject, check, Verdict.UNOBSERVED, str(error))
        return False
    return True


async def verify_chat(entry: ModelEntry, target: Target) -> None:
    subject = target.subject
    caps = entry.capabilities
    limit = entry.output_limit or 1
    async with started(target, max_output_tokens=min(4096, limit)) as binding:
        invocable = False
        try:
            await check_chat(binding, subject)
            invocable = True
        except ProviderError as error:
            REPORT.record(
                "catalogue",
                subject,
                "invocable",
                Verdict.MISMATCH,
                f"ProviderError({error.failure.value})",
            )
        except AssertionError as error:
            REPORT.record("catalogue", subject, "invocable", Verdict.MISMATCH, str(error))
        if not invocable:
            return
        REPORT.record("catalogue", subject, "invocable", Verdict.MATCH)
        if caps.tools:
            await _guard(subject, "tools", check_tools(binding, subject))
        if caps.prompt_caching:
            explicit = entry.provider in EXPLICIT_CACHES
            try:
                await check_caching(binding, subject, explicit=explicit, strict=False)
            except ProviderError as error:
                REPORT.record(
                    "catalogue",
                    subject,
                    "prompt caching",
                    Verdict.MISMATCH,
                    f"ProviderError({error.failure.value})",
                )
            except AssertionError as error:
                REPORT.record("catalogue", subject, "prompt caching", Verdict.MISMATCH, str(error))
        for declared, name, check in (
            (caps.structured_output, "structured output", check_structured_output),
            (caps.vision, "vision", check_vision),
        ):
            gap = KNOWN_ADAPTER_GAPS.get((entry.provider, name))
            if declared and gap is not None:
                REPORT.record("catalogue", subject, name, Verdict.UNOBSERVED, gap)
            elif declared:
                ok, why = await check(binding)
                REPORT.record(
                    "catalogue", subject, name, Verdict.MATCH if ok else Verdict.MISMATCH, why
                )
    async with started(target, max_output_tokens=limit) as binding:
        accepted, failure = await check_output_limit(binding, limit)
    REPORT.record(
        "catalogue",
        subject,
        f"output limit {limit} accepted",
        Verdict.MATCH if accepted else Verdict.MISMATCH,
        failure,
    )


def _compare(subject: str, check: str, catalogued: int | None, stated: int | None) -> None:
    if stated is None:
        REPORT.record("catalogue", subject, check, Verdict.UNOBSERVED, "not stated by the API")
    elif stated == catalogued:
        REPORT.record("catalogue", subject, check, Verdict.MATCH, f"{stated}")
    else:
        REPORT.record(
            "catalogue", subject, check, Verdict.MISMATCH, f"catalogue {catalogued}, API {stated}"
        )


async def verify_metadata(
    entry: ModelEntry, target: Target, region: str | None, environ: Mapping[str, str]
) -> None:
    subject = target.subject
    try:
        if entry.provider in ("anthropic", "gemini"):
            probe = anthropic_limits if entry.provider == "anthropic" else gemini_limits
            limits: Limits = await probe(entry.model_id, environ)
            _compare(subject, "context window", entry.context_window, limits.context_window)
            if entry.kind == "chat":
                _compare(subject, "output limit", entry.output_limit, limits.output_limit)
        elif entry.provider == "ollama":
            stated = await ollama_context(entry.model_id, environ)
            _compare(subject, "context window", entry.context_window, stated)
        elif entry.provider == "bedrock" and region is not None:
            await _verify_bedrock(entry, subject, region, environ)
        else:
            REPORT.record("catalogue", subject, "limits", Verdict.UNOBSERVED, "no API states them")
    except ProbeFailed as failed:
        REPORT.record("catalogue", subject, "metadata", Verdict.UNOBSERVED, f"probe: {failed}")


async def _verify_bedrock(
    entry: ModelEntry, subject: str, region: str, environ: Mapping[str, str]
) -> None:
    client = await bedrock_control_client(region, environ)
    if entry.residency_scope is ResidencyScope.IN_REGION:
        exists = await bedrock_model_exists(client, entry.model_id)
        REPORT.record(
            "catalogue",
            subject,
            "foundation model listed in region",
            Verdict.MATCH if exists else Verdict.MISMATCH,
        )
        return
    routed = await bedrock_profile_regions(client, entry.model_id)
    if entry.residency_scope is ResidencyScope.GLOBAL:
        REPORT.record(
            "catalogue",
            subject,
            "global profile regions",
            Verdict.FINDING,
            f"routes to {len(routed)} region(s)",
        )
        return
    missing = sorted(routed - entry.regions)
    extra = sorted(entry.regions - routed)
    if not missing and not extra:
        REPORT.record(
            "catalogue", subject, "profile region list", Verdict.MATCH, f"{len(routed)} regions"
        )
    else:
        REPORT.record(
            "catalogue",
            subject,
            "profile region list",
            Verdict.MISMATCH,
            f"routed but not catalogued: {', '.join(missing) or 'none'}; "
            f"catalogued but not routed: {', '.join(extra) or 'none'}",
        )


@pytest.mark.parametrize("entry", [_param(entry) for entry in ENTRIES])
async def test_default_catalogue_entry(entry: ModelEntry) -> None:
    environ = os.environ
    found = routes(entry, environ)
    if not found:
        pytest.skip(
            f"{entry.provider}:{entry.model_id} is not offered in any region this run covers"
        )
    if entry.provider == "ollama":
        try:
            await ollama_context(entry.model_id, environ)
        except ProbeFailed as failed:
            pytest.skip(f"{entry.model_id} on the Ollama server: {failed}")
    for target, region in found:
        if entry.kind == "embedding":
            own = entry.embedding_dimensions
            assert own is not None
            try:
                async with started_embeddings(target) as binding:
                    await check_embeddings(binding, target.subject, own)
            except ProviderError as error:
                REPORT.record(
                    "catalogue",
                    target.subject,
                    "embeddings",
                    Verdict.MISMATCH,
                    f"ProviderError({error.failure.value})",
                )
        else:
            await verify_chat(entry, target)
        await verify_metadata(entry, target, region, environ)
    subjects = {target.subject for target, _ in found}
    mismatched = sorted(
        f"{line.subject} :: {line.check}"
        for line in REPORT.lines
        if line.verdict is Verdict.MISMATCH and line.subject in subjects
    )
    count = len(mismatched)
    assert count == 0, "catalogue mismatches: " + "; ".join(mismatched)
