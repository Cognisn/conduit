"""Live: the default catalogue verification pass (Q0002), on each provider's listing.

Bedrock and bedrock-mantle (C18, Q0061): one test per kind. For each configured Region
(``BEDROCK_REGIONS`` or ``BEDROCK_REGION``, ``BEDROCK_MANTLE_REGIONS`` or
``BEDROCK_MANTLE_REGION``) a connection is built with the run's Region choice
(``BEDROCK_PROCESSING_REGIONS``, ``BEDROCK_ALLOW_GLOBAL``) and listed through the
registry; every listed model it allows that a catalogue entry describes is verified
through its listed id (bare id, each geographic profile, the global profile when the
choice allows it), and the listing's catalogue drift and each profile's destination
Regions are recorded. No route is taken from Regions in catalogue entries.

The other providers: one test per declared model, through the provider's one route.

Every route checks what it can observe against the (listed) entry:

- the model is invocable;
- the provider accepts a request at the output limit;
- each declared capability: tools, prompt caching, structured output, vision;
- what a metadata API states: Anthropic's and Gemini's context window and output limit,
  and Ollama's trained context length.

Everything observed goes into the report. A mismatch fails the entry's test, naming
the checks; what cannot be observed (OpenAI's and X.AI's limits, which no API states;
a model that chose not to call the tool; an entry with no output limit) is reported as
unobserved and never fails. ``test_catalogue_logic.py`` proves this offline on the stub.

Opt in with ``CONDUIT_LIVE_CATALOGUE=1`` as well as each provider's credentials; select
providers with ``-k`` (``-k "bedrock and not mantle"``).
"""

from __future__ import annotations

from collections.abc import Awaitable, Iterable
from typing import Any

import pytest

from conduit.core.catalogue import ModelEntry, default_catalogue
from conduit.core.discovery import DiscoveryResult
from conduit.core.errors import ConfigurationError, ProviderError
from conduit.core.registry import ProviderFactory
from conduit.core.residency import ResidencyScope
from tests.live import needs
from tests.live.harness import (
    DEFAULT_OUTPUT_BUDGET,
    NoToolCall,
    check_caching,
    check_chat,
    check_embeddings,
    check_output_limit,
    check_structured_output,
    check_tools,
    check_vision,
    listing,
    started,
    started_embeddings,
)
from tests.live.needs import LiveEnv, Need, live_env
from tests.live.probes import (
    Limits,
    ProbeFailed,
    anthropic_limits,
    gemini_limits,
    ollama_context,
)
from tests.live.report import REPORT, Verdict
from tests.live.targets import (
    Route,
    Target,
    anthropic_direct,
    bedrock_settings,
    first_bedrock_auth,
    gemini_settings,
    listed_routes,
    mantle_auth,
    mantle_settings,
    ollama_settings,
    openai_settings,
    processing_choice,
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

#: The kinds whose routes come from each configured Region's listing.
LISTED_KINDS = ("bedrock", "bedrock_mantle")

ENTRIES = [entry for entry in default_catalogue() if entry.provider not in ("stub", *LISTED_KINDS)]


def _param(entry: ModelEntry) -> object:
    return pytest.param(
        entry,
        marks=pytest.mark.live_needs(needs.CATALOGUE_PASS, *PROVIDER_NEEDS[entry.provider]),
        id=f"{entry.provider}:{entry.model_id}",
    )


def routes(entry: ModelEntry, environ: LiveEnv) -> list[Target]:
    """The one target a non-listed-kind ``entry`` is called through."""
    model = entry.model_id
    if entry.provider == "anthropic":
        return [Target("anthropic direct", anthropic_direct(), model)]
    if entry.provider == "openai":
        return [Target("openai", openai_settings(), model)]
    if entry.provider == "xai":
        return [Target("xai", xai_settings(), model)]
    if entry.provider == "gemini":
        return [Target("gemini", gemini_settings(), model)]
    if entry.provider == "ollama":
        return [Target("ollama", ollama_settings(environ), model)]
    return []


def connections(kind: str, environ: LiveEnv) -> list[tuple[str, dict[str, Any], str]]:
    """One connection per configured Region of ``kind``: its label, its settings (with
    the run's Region choice, for Bedrock) and the Region."""
    if kind == "bedrock":
        mode = first_bedrock_auth(environ)
        if mode is None:
            return []
        choice = processing_choice(environ)
        return [
            (f"bedrock {mode}", bedrock_settings(region, mode, environ, processing=choice), region)
            for region in regions_list(environ, "BEDROCK_REGIONS", "BEDROCK_REGION")
        ]
    if kind == "bedrock_mantle":
        chosen = mantle_auth(environ)
        if chosen is None:
            return []
        auth, long_term = chosen
        return [
            (f"bedrock-mantle {auth['mode']}", mantle_settings(region, auth, long_term), region)
            for region in regions_list(environ, "BEDROCK_MANTLE_REGIONS", "BEDROCK_MANTLE_REGION")
        ]
    return []


def record_drift(subject: str, listed: DiscoveryResult) -> None:
    """Record each catalogue drift the listing found, as a finding (the listing is what
    conduit uses; drift is for the catalogue's maintainers)."""
    for drift in listed.drift:
        REPORT.record(
            "catalogue",
            subject,
            f"{drift.reason.value} {drift.model_id}",
            Verdict.FINDING,
            drift.detail,
        )


def record_destinations(subject: str, found: Iterable[Route]) -> None:
    """Record each geographic profile's destination Regions, as the listing gives them,
    and the widest (the API documents at most 5 models in a profile)."""
    widest = 0
    for route in found:
        if route.scope is not ResidencyScope.CROSS_REGION:
            continue
        widest = max(widest, len(route.regions))
        REPORT.record(
            "catalogue",
            subject,
            f"destinations {route.target.model}",
            Verdict.FINDING,
            ", ".join(sorted(route.regions)),
        )
    if widest:
        REPORT.record(
            "catalogue",
            subject,
            "widest geographic profile",
            Verdict.FINDING,
            f"{widest} destination Regions",
        )


def _failure(error: ProviderError) -> str:
    return f"ProviderError({error.failure.value})"


async def verify_tools(binding_subject: str, run: Awaitable[object]) -> None:
    """Record a tool loop: a model that chose not to call the tool is unobserved; any
    other defect (wrong tool, unfinished call, wrong stop, no answer, no usage, a refusal)
    is a mismatch."""
    check = "tools"
    try:
        await run
    except NoToolCall as error:
        REPORT.record("catalogue", binding_subject, check, Verdict.UNOBSERVED, str(error))
    except AssertionError as error:
        REPORT.record("catalogue", binding_subject, check, Verdict.MISMATCH, str(error))
    except ProviderError as error:
        REPORT.record("catalogue", binding_subject, check, Verdict.MISMATCH, _failure(error))


async def verify_chat(
    entry: ModelEntry, target: Target, *, factories: Iterable[ProviderFactory] | None = None
) -> None:
    """Invocation and every declared capability, then the output limit."""
    subject = target.subject
    caps = entry.capabilities
    budget = min(DEFAULT_OUTPUT_BUDGET, entry.output_limit or DEFAULT_OUTPUT_BUDGET)
    async with started(target, max_output_tokens=budget, factories=factories) as binding:
        detail = ""
        try:
            await check_chat(binding, subject)
        except ProviderError as error:
            detail = _failure(error)
        except AssertionError as error:
            detail = str(error) or "the chat check failed"
        if detail:
            REPORT.record("catalogue", subject, "invocable", Verdict.MISMATCH, detail)
            return
        REPORT.record("catalogue", subject, "invocable", Verdict.MATCH)
        if caps.tools:
            await verify_tools(subject, check_tools(binding, subject))
        if caps.prompt_caching:
            explicit = entry.provider in EXPLICIT_CACHES
            detail = ""
            try:
                await check_caching(binding, subject, explicit=explicit, strict=False)
            except ProviderError as error:
                detail = _failure(error)
            except AssertionError as error:
                detail = str(error) or "the caching check failed"
            if detail:
                REPORT.record("catalogue", subject, "prompt caching", Verdict.MISMATCH, detail)
        for declared, name, check in (
            (caps.structured_output, "structured output", check_structured_output),
            (caps.vision, "vision", check_vision),
        ):
            if declared:
                ok, why = await check(binding)
                REPORT.record(
                    "catalogue", subject, name, Verdict.MATCH if ok else Verdict.MISMATCH, why
                )
    await verify_output_limit(entry, target, factories=factories)


async def verify_output_limit(
    entry: ModelEntry, target: Target, *, factories: Iterable[ProviderFactory] | None = None
) -> None:
    """Whether the provider accepts a request at the catalogued output limit."""
    limit = entry.output_limit
    if limit is None:
        REPORT.record(
            "catalogue", target.subject, "output limit", Verdict.UNOBSERVED, "no output limit"
        )
        return
    async with started(target, max_output_tokens=limit, factories=factories) as binding:
        accepted, failure = await check_output_limit(binding, limit)
    REPORT.record(
        "catalogue",
        target.subject,
        f"output limit {limit} accepted",
        Verdict.MATCH if accepted else Verdict.MISMATCH,
        failure,
    )


def compare(subject: str, check: str, catalogued: int | None, stated: int | None) -> None:
    """Record a catalogued number against what an API states."""
    if stated is None:
        REPORT.record("catalogue", subject, check, Verdict.UNOBSERVED, "not stated by the API")
    elif stated == catalogued:
        REPORT.record("catalogue", subject, check, Verdict.MATCH, f"{stated}")
    else:
        REPORT.record(
            "catalogue", subject, check, Verdict.MISMATCH, f"catalogue {catalogued}, API {stated}"
        )


def compare_regions(subject: str, catalogued: frozenset[str], routed: frozenset[str]) -> None:
    """Record a cross-region entry's region list against where its profile routes."""
    missing = sorted(routed - catalogued)
    extra = sorted(catalogued - routed)
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


async def verify_metadata(entry: ModelEntry, target: Target, environ: LiveEnv) -> None:
    subject = target.subject
    try:
        if entry.provider in ("anthropic", "gemini"):
            probe = anthropic_limits if entry.provider == "anthropic" else gemini_limits
            limits: Limits = await probe(entry.model_id, environ)
            compare(subject, "context window", entry.context_window, limits.context_window)
            if entry.kind == "chat":
                compare(subject, "output limit", entry.output_limit, limits.output_limit)
        elif entry.provider == "ollama":
            stated = await ollama_context(entry.model_id, environ)
            compare(subject, "context window", entry.context_window, stated)
        else:
            REPORT.record("catalogue", subject, "limits", Verdict.UNOBSERVED, "no API states them")
    except ProbeFailed as failed:
        REPORT.record("catalogue", subject, "metadata", Verdict.UNOBSERVED, f"probe: {failed}")


def fail_on_mismatches(subjects: Iterable[str]) -> None:
    """Fail, naming each check, if any of ``subjects`` recorded a mismatch."""
    wanted = set(subjects)
    mismatched = sorted(
        f"{line.subject} :: {line.check}"
        for line in REPORT.lines
        if line.verdict is Verdict.MISMATCH and line.subject in wanted
    )
    count = len(mismatched)
    assert count == 0, "catalogue mismatches: " + "; ".join(mismatched)


#: Each route's listing, made once per run.
_LISTINGS: dict[str, DiscoveryResult] = {}


async def listed_by(target: Target, environ: LiveEnv) -> bool:
    """Whether the provider behind ``target`` lists its model (listed once per run)."""
    listed = _LISTINGS.get(target.label)
    if listed is None:
        listed = await listing(target.provider, environ=environ)
        _LISTINGS[target.label] = listed
    return not listed.supported or listed.get(target.model) is not None


@pytest.mark.parametrize("entry", [_param(entry) for entry in ENTRIES])
async def test_default_catalogue_entry(entry: ModelEntry) -> None:
    found = routes(entry, live_env())
    if not found:
        pytest.skip(
            f"{entry.provider}:{entry.model_id} is not offered in any region this run covers"
        )
    for target in found:
        if not await listed_by(target, live_env()):
            REPORT.record("catalogue", target.subject, "listed", Verdict.FINDING, "unlisted")
            pytest.skip(f"{entry.provider}:{entry.model_id} is not in the provider's listing")
    if entry.provider == "ollama":
        try:
            await ollama_context(entry.model_id, live_env())
        except ProbeFailed as failed:
            pytest.skip(f"{entry.model_id} on the Ollama server: {failed}")
    for target in found:
        await verify_entry(entry, target)
        await verify_metadata(entry, target, live_env())
    fail_on_mismatches(target.subject for target in found)


async def verify_entry(entry: ModelEntry, target: Target) -> None:
    """Embed at the entry's dimensions, or verify the chat entry."""
    if entry.kind == "embedding":
        own = entry.embedding_dimensions
        assert own is not None
        try:
            async with started_embeddings(target) as binding:
                await check_embeddings(binding, target.subject, own)
        except ProviderError as error:
            REPORT.record(
                "catalogue", target.subject, "embeddings", Verdict.MISMATCH, _failure(error)
            )
    else:
        await verify_chat(entry, target)


@pytest.mark.parametrize(
    "kind",
    [
        pytest.param(
            kind,
            marks=pytest.mark.live_needs(needs.CATALOGUE_PASS, *PROVIDER_NEEDS[kind]),
            id=kind,
        )
        for kind in LISTED_KINDS
    ],
)
async def test_listed_models(kind: str) -> None:
    """Every listed model the catalogue describes, in every configured Region."""
    environ = live_env()
    subjects: list[str] = []
    for label, settings, region in connections(kind, environ):
        subject = f"{label} {region}"
        subjects.append(subject)
        try:
            listed = await listing(settings, environ=environ)
        except (ProviderError, ConfigurationError) as error:
            why = _failure(error) if isinstance(error, ProviderError) else type(error).__name__
            REPORT.record("catalogue", subject, "listing", Verdict.MISMATCH, why)
            continue
        record_drift(subject, listed)
        found = listed_routes(label, settings, region, listed)
        REPORT.record("catalogue", subject, "routes", Verdict.FINDING, f"{len(found)} listed")
        record_destinations(subject, found)
        for route in found:
            await verify_entry(route.entry, route.target)
            subjects.append(route.target.subject)
    if not subjects:
        pytest.skip(f"no {kind} Region is configured")
    fail_on_mismatches(subjects)
