"""The default catalogue's Bedrock entries: capabilities and limits by foundation model,
and residency decided at start-up on the Region's own listing (C18)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from pydantic import SecretStr

from conduit.core.catalogue import default_catalogue
from conduit.core.discovery import DiscoveredModel
from conduit.core.errors import AssignmentRefused, ResidencyRefused
from conduit.core.residency import ResidencyPolicy, ResidencyScope
from conduit.core.runtime import ConduitSettings, start
from conduit.core.secrets import LiteralSecretResolver
from conduit.core.settings import ProviderSettings
from conduit.core.tasks import TaskClass
from conduit.providers.bedrock import BedrockFactory

PROFILE_PREFIXES = ("us.", "eu.", "au.", "jp.", "ca.", "in.", "apac.", "us-gov.", "global.")


def bedrock_entries() -> list[Any]:
    return [e for e in default_catalogue() if e.provider == "bedrock"]


@pytest.mark.parametrize("entry", bedrock_entries(), ids=lambda e: e.model_id)
def test_each_entry_is_a_foundation_model_with_no_residency(entry: Any) -> None:
    # Where a Bedrock model is offered and processed is the listing's to say.
    assert not entry.model_id.startswith(PROFILE_PREFIXES)
    assert entry.residency_scope is None
    assert entry.regions == frozenset()


@pytest.mark.parametrize(
    ("model_id", "output_limit"),
    [
        # Sources: the Bedrock model cards for Claude Sonnet 4.6, Claude Sonnet 5 and
        # Claude Opus 5.5.
        ("anthropic.claude-sonnet-4-6", 64_000),
        ("anthropic.claude-sonnet-5", 128_000),
        ("anthropic.claude-opus-5-5", 128_000),
    ],
)
def test_the_claude_entries(model_id: str, output_limit: int) -> None:
    entry = default_catalogue().require("bedrock", model_id)
    assert entry.output_limit == output_limit
    assert entry.context_window == 1_000_000
    assert entry.capabilities.prompt_caching and entry.capabilities.tools


def test_a_profile_only_model_is_described_by_its_foundation_model() -> None:
    # Sonnet 5.5's card lists no In-Region Region on bedrock-runtime; its profiles take
    # this entry through the foundation model their listing names.
    assert default_catalogue().get("bedrock", "anthropic.claude-sonnet-5-5") is not None
    assert default_catalogue().get("bedrock", "global.anthropic.claude-sonnet-5-5") is None


def test_no_bedrock_chat_entry_declares_structured_output() -> None:
    # Q0042: the adapter refuses output_schema until Converse structured output is mapped.
    chat = [e for e in bedrock_entries() if e.kind == "chat"]
    assert chat and not any(e.capabilities.structured_output for e in chat)


# Sources (read 10 November 2026): the Bedrock model cards,
# docs.aws.amazon.com/bedrock/latest/userguide/model-card-<vendor>-<model>.html, for
# the model id, context window, maximum output ("nK" read as n x 1,000) and the
# bedrock-runtime feature list (client-side tool calling, explicit prompt caching,
# image input). Keyed by foundation model id.
#: model id -> (context window, output limit, tools, vision, prompt caching)
DOCUMENTED: dict[str, tuple[int, int, bool, bool, bool]] = {
    "anthropic.claude-fable-5-2": (1_000_000, 128_000, True, True, True),
    "anthropic.claude-fable-5-1": (1_000_000, 128_000, True, True, True),
    "anthropic.claude-fable-5": (1_000_000, 128_000, True, True, True),
    "anthropic.claude-opus-5": (1_000_000, 128_000, True, True, True),
    "anthropic.claude-opus-4-8": (1_000_000, 128_000, True, True, True),
    "anthropic.claude-opus-4-7": (1_000_000, 128_000, True, True, True),
    "anthropic.claude-opus-4-6-v1": (1_000_000, 128_000, True, True, True),
    "anthropic.claude-opus-4-5-20251101-v1:0": (200_000, 64_000, True, True, True),
    "amazon.nova-2-lite-v1:0": (1_000_000, 64_000, True, True, True),
    "amazon.nova-pro-v1:0": (300_000, 10_000, False, True, True),
    "amazon.nova-lite-v1:0": (300_000, 10_000, True, True, True),
    "amazon.nova-micro-v1:0": (128_000, 10_000, True, False, True),
    "deepseek.v3.2": (164_000, 8_000, False, False, False),
    "mistral.mistral-large-3-675b-instruct": (256_000, 32_000, False, True, False),
    "mistral.mistral-large-2402-v1:0": (32_000, 4_000, True, False, False),
    "mistral.pixtral-large-2502-v1:0": (128_000, 16_000, True, True, False),
    "openai.gpt-6-astra": (1_050_000, 128_000, False, True, False),
    "openai.gpt-6.1-sol": (1_000_000, 131_072, False, True, False),
    "openai.gpt-oss-120b-1:0": (128_000, 16_000, False, False, False),
    "qwen.qwen3-235b-a22b-2507-v1:0": (256_000, 8_000, True, False, False),
    "qwen.qwen3-vl-235b-a22b": (256_000, 8_000, False, True, False),
    "minimax.minimax-m2.5": (196_000, 8_000, False, False, False),
    "moonshotai.kimi-k2.5": (256_000, 16_000, False, True, False),
    "nvidia.nemotron-super-3-120b": (256_000, 32_000, False, False, False),
    "writer.palmyra-vision-7b": (4_000, 4_000, False, True, False),
    "zai.glm-5": (200_000, 128_000, False, False, False),
}


@pytest.mark.parametrize("model_id", sorted(DOCUMENTED))
def test_documented_limits_and_capabilities(model_id: str) -> None:
    context, output, tools, vision, caching = DOCUMENTED[model_id]
    entry = default_catalogue().require("bedrock", model_id)
    assert (entry.context_window, entry.output_limit) == (context, output)
    assert entry.capabilities.tools is tools
    assert entry.capabilities.vision is vision
    assert entry.capabilities.prompt_caching is caching


@pytest.mark.parametrize(
    "model_id",
    [
        # Legacy with an end-of-life date, end of life, gated, or with no stated output
        # limit: not described by the default catalogue. A Region that lists one serves
        # it on the conservative floor unless an overlay describes it.
        "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "anthropic.claude-opus-4-1-20250805-v1:0",
        "anthropic.claude-mythos-5-2",
        "meta.llama3-3-70b-instruct-v1:0",
        "ai21.jamba-1-5-large-v1:0",
        "cohere.command-r-plus-v1:0",
        "deepseek.r1-v1:0",
        "google.gemma-3-27b-it",
        "writer.palmyra-x5-v1:0",
        "xai.grok-4.7",
        "moonshotai.kimi-k3.1",
        "zai.glm-5.3",
    ],
)
def test_models_left_out(model_id: str) -> None:
    assert default_catalogue().get("bedrock", model_id) is None


LONDON = "eu-west-2"
EUROPE = frozenset({"eu-west-1", "eu-west-2", "eu-west-3", "eu-central-1"})


def _profile(profile_id: str, *regions: str) -> DiscoveredModel:
    base = profile_id.partition(".")[2]
    if profile_id.startswith("global."):
        return DiscoveredModel(profile_id, scope=ResidencyScope.GLOBAL, base_model_id=base)
    return DiscoveredModel(
        profile_id,
        scope=ResidencyScope.CROSS_REGION,
        regions=frozenset(regions),
        base_model_id=base,
    )


def _bare(model_id: str) -> DiscoveredModel:
    return DiscoveredModel(
        model_id,
        scope=ResidencyScope.IN_REGION,
        regions=frozenset({LONDON}),
        base_model_id=model_id,
    )


#: A synthetic London listing: three in-Region models and their profiles.
LONDON_LISTING: tuple[DiscoveredModel, ...] = (
    _bare("anthropic.claude-sonnet-4-6"),
    _bare("anthropic.claude-sonnet-5"),
    _bare("anthropic.claude-opus-5-5"),
    _profile("eu.anthropic.claude-sonnet-4-6", *EUROPE),
    _profile("eu.anthropic.claude-sonnet-5", *EUROPE),
    _profile("eu.anthropic.claude-opus-5-5", *EUROPE),
    _profile("global.anthropic.claude-sonnet-4-6"),
    _profile("global.anthropic.claude-sonnet-5-5"),
)


class OfflineBedrockFactory(BedrockFactory):
    """Lists a synthetic listing and builds providers around a placeholder client,
    recording each client it makes and each provider it builds."""

    def __init__(self, listing: Sequence[DiscoveredModel] = LONDON_LISTING) -> None:
        super().__init__()
        self.listing = tuple(listing)
        self.clients = 0
        self.built: list[str] = []

    def create_client(self, settings: ProviderSettings, secrets: Mapping[str, SecretStr]) -> Any:
        self.clients += 1
        return object()

    async def discover_models(
        self, settings: ProviderSettings, client: Any
    ) -> Sequence[DiscoveredModel]:
        return self.listing

    def create_provider(self, name: str, settings: ProviderSettings, client: Any, **kw: Any) -> Any:
        self.built.append(name)
        return super().create_provider(name, settings, client, **kw)


IN_REGION_LONDON = ResidencyPolicy(
    scopes=frozenset({ResidencyScope.IN_REGION}), regions=frozenset({LONDON})
)
WITHIN_EUROPE = ResidencyPolicy(
    scopes=frozenset({ResidencyScope.IN_REGION, ResidencyScope.CROSS_REGION}), regions=EUROPE
)


async def start_with(
    model: str, factory: OfflineBedrockFactory, residency: ResidencyPolicy = IN_REGION_LONDON
) -> Any:
    return await start(
        ConduitSettings.model_validate(
            {
                "providers": {"bedrock_london": {"kind": "bedrock", "region": LONDON}},
                "assignments": {"summarise": {"provider": "bedrock_london", "model": model}},
                "residency": residency,
            }
        ),
        task_classes=[TaskClass(name="summarise")],
        factories=[factory],
        secrets=LiteralSecretResolver({}),
    )


class TestResidencyAtStartUp:
    @pytest.mark.parametrize(
        "model",
        [
            "eu.anthropic.claude-sonnet-4-6",
            "eu.anthropic.claude-sonnet-5",
            "eu.anthropic.claude-opus-5-5",
            "global.anthropic.claude-sonnet-4-6",
            "global.anthropic.claude-sonnet-5-5",
        ],
    )
    async def test_an_in_region_policy_refuses_profiles_before_any_provider_is_built(
        self, model: str
    ) -> None:
        factory = OfflineBedrockFactory()
        with pytest.raises(ResidencyRefused):
            await start_with(model, factory)
        assert factory.clients == 1  # the listing's client only
        assert factory.built == ["bedrock_london"]  # the listing's holder only

    @pytest.mark.parametrize(
        "model",
        ["anthropic.claude-sonnet-4-6", "anthropic.claude-sonnet-5", "anthropic.claude-opus-5-5"],
    )
    async def test_it_accepts_the_bare_in_region_ids(self, model: str) -> None:
        factory = OfflineBedrockFactory()
        runtime = await start_with(model, factory)
        try:
            assert runtime.for_task("summarise").assignment.model == model
            assert factory.clients == 2  # the listing's, then the provider's
        finally:
            await runtime.aclose()

    async def test_a_geographic_profile_is_allowed_on_its_listed_regions(self) -> None:
        factory = OfflineBedrockFactory()
        runtime = await start_with("eu.anthropic.claude-sonnet-5", factory, WITHIN_EUROPE)
        try:
            entry = runtime.for_task("summarise").assignment.entry
            assert entry.regions == EUROPE
            assert entry.capabilities.prompt_caching  # from the foundation model's entry
        finally:
            await runtime.aclose()

    async def test_a_profile_routed_wider_than_the_policy_is_refused(self) -> None:
        wider = (_profile("eu.anthropic.claude-sonnet-5", *EUROPE, "eu-south-2"),)
        with pytest.raises(ResidencyRefused, match="eu-south-2"):
            await start_with(
                "eu.anthropic.claude-sonnet-5", OfflineBedrockFactory(wider), WITHIN_EUROPE
            )

    async def test_a_model_the_region_does_not_list_is_refused(self) -> None:
        with pytest.raises(AssignmentRefused, match="is not listed by provider"):
            await start_with("anthropic.claude-sonnet-5-5", OfflineBedrockFactory())
