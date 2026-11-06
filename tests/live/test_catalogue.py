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
a model that chose not to call the tool; an entry with no output limit) is reported as
unobserved and never fails. ``test_catalogue_logic.py`` proves this offline on the stub.

Opt in with ``CONDUIT_LIVE_CATALOGUE=1`` as well as each provider's credentials; select
providers with ``-k`` (``-k "bedrock and not mantle"``).
"""

from __future__ import annotations

from collections.abc import Awaitable, Iterable

import pytest

from conduit.core.catalogue import ModelEntry, default_catalogue
from conduit.core.errors import ProviderError
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
    started,
    started_embeddings,
)
from tests.live.needs import LiveEnv, Need, live_env
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

ENTRIES = [entry for entry in default_catalogue() if entry.provider != "stub"]


def _param(entry: ModelEntry) -> object:
    return pytest.param(
        entry,
        marks=pytest.mark.live_needs(needs.CATALOGUE_PASS, *PROVIDER_NEEDS[entry.provider]),
        id=f"{entry.provider}:{entry.model_id}",
    )


def routes(entry: ModelEntry, environ: LiveEnv) -> list[tuple[Target, str | None]]:
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


async def verify_metadata(
    entry: ModelEntry, target: Target, region: str | None, environ: LiveEnv
) -> None:
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
        elif entry.provider == "bedrock" and region is not None:
            await _verify_bedrock(entry, subject, region, environ)
        else:
            REPORT.record("catalogue", subject, "limits", Verdict.UNOBSERVED, "no API states them")
    except ProbeFailed as failed:
        REPORT.record("catalogue", subject, "metadata", Verdict.UNOBSERVED, f"probe: {failed}")


async def _verify_bedrock(entry: ModelEntry, subject: str, region: str, environ: LiveEnv) -> None:
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
    compare_regions(subject, entry.regions, routed)


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


@pytest.mark.parametrize("entry", [_param(entry) for entry in ENTRIES])
async def test_default_catalogue_entry(entry: ModelEntry) -> None:
    found = routes(entry, live_env())
    if not found:
        pytest.skip(
            f"{entry.provider}:{entry.model_id} is not offered in any region this run covers"
        )
    if entry.provider == "ollama":
        try:
            await ollama_context(entry.model_id, live_env())
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
                    "catalogue", target.subject, "embeddings", Verdict.MISMATCH, _failure(error)
                )
        else:
            await verify_chat(entry, target)
        await verify_metadata(entry, target, region, live_env())
    fail_on_mismatches(target.subject for target, _ in found)
