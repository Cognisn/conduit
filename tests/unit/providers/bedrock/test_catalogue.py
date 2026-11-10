"""The default catalogue's Bedrock entries: residency scopes, the in-region Sonnet entries
(Q0004) and refusal of cross-region profiles at start-up."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import SecretStr

from conduit.core.catalogue import default_catalogue
from conduit.core.errors import ResidencyRefused
from conduit.core.residency import ResidencyPolicy, ResidencyScope
from conduit.core.runtime import ConduitSettings, start
from conduit.core.secrets import LiteralSecretResolver
from conduit.core.settings import ProviderSettings
from conduit.core.tasks import TaskClass
from conduit.providers.bedrock import BedrockFactory

GEOGRAPHIC_PREFIXES = ("us.", "eu.", "au.", "jp.", "ca.", "in.", "apac.", "us-gov.")


def bedrock_entries() -> list[Any]:
    return [e for e in default_catalogue() if e.provider == "bedrock"]


@pytest.mark.parametrize("entry", bedrock_entries(), ids=lambda e: e.model_id)
def test_each_entry_is_scoped_by_its_id(entry: Any) -> None:
    if entry.model_id.startswith("global."):
        assert entry.residency_scope is ResidencyScope.GLOBAL
    elif entry.model_id.startswith(GEOGRAPHIC_PREFIXES):
        assert entry.residency_scope is ResidencyScope.CROSS_REGION
        assert entry.regions
    else:
        assert entry.residency_scope is ResidencyScope.IN_REGION
        assert entry.regions


@pytest.mark.parametrize(
    ("model_id", "regions", "output_limit"),
    [
        # Sources: the Bedrock model cards for Claude Sonnet 4.6 and Claude Sonnet 5,
        # "Availability using the bedrock-runtime endpoint", In-Region column.
        ("anthropic.claude-sonnet-4-6", {"eu-west-2"}, 64_000),
        ("anthropic.claude-sonnet-5", {"eu-west-2", "ap-northeast-2", "ap-southeast-1"}, 128_000),
        # Claude Opus 5.5's card: bare-id in-region inference in eu-west-2 only.
        ("anthropic.claude-opus-5-5", {"eu-west-2"}, 128_000),
    ],
)
def test_the_in_region_entries(model_id: str, regions: set[str], output_limit: int) -> None:
    entry = default_catalogue().require("bedrock", model_id)
    assert entry.residency_scope is ResidencyScope.IN_REGION
    assert entry.regions == regions
    assert entry.output_limit == output_limit
    assert entry.context_window == 1_000_000
    assert entry.capabilities.prompt_caching and entry.capabilities.tools


def test_sonnet_5_5_has_no_in_region_entry() -> None:
    # Its card lists no In-Region Region on bedrock-runtime.
    assert default_catalogue().get("bedrock", "anthropic.claude-sonnet-5-5") is None


@pytest.mark.parametrize(
    ("model_id", "regions"),
    [
        (
            "us.anthropic.claude-sonnet-5",
            {"us-east-1", "us-east-2", "us-west-1", "us-west-2", "ca-central-1", "ca-west-1"},
        ),
        (
            "eu.anthropic.claude-sonnet-5",
            {
                "eu-central-1",
                "eu-central-2",
                "eu-north-1",
                "eu-south-1",
                "eu-south-2",
                "eu-west-1",
                "eu-west-2",
                "eu-west-3",
            },
        ),
        ("au.anthropic.claude-sonnet-5", {"ap-southeast-2", "ap-southeast-4"}),
        ("in.anthropic.claude-sonnet-5", {"ap-south-1", "ap-south-2"}),
    ],
)
def test_the_sonnet_5_geographic_profiles(model_id: str, regions: set[str]) -> None:
    entry = default_catalogue().require("bedrock", model_id)
    assert entry.residency_scope is ResidencyScope.CROSS_REGION
    assert entry.regions == regions
    assert default_catalogue().require("bedrock", "global.anthropic.claude-sonnet-5")


def test_no_bedrock_chat_entry_declares_structured_output() -> None:
    # Q0042: the adapter refuses output_schema until Converse structured output is mapped.
    chat = [e for e in bedrock_entries() if e.kind == "chat"]
    assert chat and not any(e.capabilities.structured_output for e in chat)


# Sources (read 10 November 2026): the Bedrock model cards,
# docs.aws.amazon.com/bedrock/latest/userguide/model-card-<vendor>-<model>.html, for
# the model id, context window, maximum output ("nK" read as n x 1,000), Regional
# availability (In-Region column, and each geo profile's destination Regions) and the
# bedrock-runtime feature list (client-side tool calling, explicit prompt caching,
# image input).
#: model id -> (context window, output limit, tools, vision, prompt caching)
DOCUMENTED: dict[str, tuple[int, int, bool, bool, bool]] = {
    "global.anthropic.claude-fable-5-2": (1_000_000, 128_000, True, True, True),
    "us.anthropic.claude-fable-5-1": (1_000_000, 128_000, True, True, True),
    "us.anthropic.claude-fable-5": (1_000_000, 128_000, True, True, True),
    "anthropic.claude-opus-5": (1_000_000, 128_000, True, True, True),
    "jp.anthropic.claude-opus-4-8": (1_000_000, 128_000, True, True, True),
    "au.anthropic.claude-opus-4-7": (1_000_000, 128_000, True, True, True),
    "anthropic.claude-opus-4-6-v1": (1_000_000, 128_000, True, True, True),
    "eu.anthropic.claude-opus-4-5-20251101-v1:0": (200_000, 64_000, True, True, True),
    "us.amazon.nova-2-lite-v1:0": (1_000_000, 64_000, True, True, True),
    "amazon.nova-pro-v1:0": (300_000, 10_000, False, True, True),
    "amazon.nova-lite-v1:0": (300_000, 10_000, True, True, True),
    "amazon.nova-micro-v1:0": (128_000, 10_000, True, False, True),
    "deepseek.v3.2": (164_000, 8_000, False, False, False),
    "mistral.mistral-large-3-675b-instruct": (256_000, 32_000, False, True, False),
    "mistral.mistral-large-2402-v1:0": (32_000, 4_000, True, False, False),
    "us.mistral.pixtral-large-2502-v1:0": (128_000, 16_000, True, True, False),
    "us.openai.gpt-6-astra": (1_050_000, 128_000, False, True, False),
    "global.openai.gpt-6.1-sol": (1_000_000, 131_072, False, True, False),
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
    ("model_id", "regions"),
    [
        ("anthropic.claude-opus-5", {"ap-northeast-2"}),
        ("anthropic.claude-opus-4-6-v1", {"eu-west-2"}),
        (
            "amazon.nova-pro-v1:0",
            {
                "us-east-1",
                "us-gov-west-1",
                "eu-west-2",
                "ap-southeast-2",
                "ap-southeast-3",
                "me-central-1",
            },
        ),
        ("mistral.mistral-small-2402-v1:0", {"us-east-1"}),
        ("qwen.qwen3-coder-next", {"us-east-1", "eu-west-2", "ap-southeast-2"}),
    ],
)
def test_more_in_region_entries(model_id: str, regions: set[str]) -> None:
    entry = default_catalogue().require("bedrock", model_id)
    assert entry.residency_scope is ResidencyScope.IN_REGION
    assert entry.regions == regions


@pytest.mark.parametrize(
    ("model_id", "regions"),
    [
        (
            "eu.amazon.nova-pro-v1:0",
            {
                "eu-central-1",
                "eu-north-1",
                "eu-south-1",
                "eu-south-2",
                "eu-west-1",
                "eu-west-3",
                "il-central-1",
            },
        ),
        ("au.anthropic.claude-opus-4-6-v1", {"ap-southeast-2", "ap-southeast-4", "ap-southeast-6"}),
        ("in.anthropic.claude-opus-5", {"ap-south-1", "ap-south-2"}),
        (
            "us.anthropic.claude-opus-4-5-20251101-v1:0",
            {"us-east-1", "us-east-2", "us-west-1", "us-west-2", "ca-central-1"},
        ),
        ("us-gov.openai.gpt-oss-120b-1:0", {"us-gov-east-1", "us-gov-west-1"}),
        ("ca.amazon.nova-lite-v1:0", {"ca-central-1", "ca-west-1"}),
    ],
)
def test_more_geographic_profiles(model_id: str, regions: set[str]) -> None:
    entry = default_catalogue().require("bedrock", model_id)
    assert entry.residency_scope is ResidencyScope.CROSS_REGION
    assert entry.regions == regions


@pytest.mark.parametrize(
    "model_id",
    [
        # Profile-only on bedrock-runtime (no In-Region Region on the card).
        "anthropic.claude-sonnet-5-5",
        "anthropic.claude-haiku-5-5",
        "anthropic.claude-fable-5-2",
        "amazon.nova-2-lite-v1:0",
        # Legacy with an end-of-life date, end of life, gated, or with no stated
        # output limit: left out of the default catalogue.
        "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "us.anthropic.claude-opus-4-1-20250805-v1:0",
        "global.anthropic.claude-mythos-5-2",
        "meta.llama3-3-70b-instruct-v1:0",
        "ai21.jamba-1-5-large-v1:0",
        "cohere.command-r-plus-v1:0",
        "us.deepseek.r1-v1:0",
        "google.gemma-3-27b-it",
        "us.writer.palmyra-x5-v1:0",
        "us.xai.grok-4.7",
        "us.moonshotai.kimi-k3.1",
        "us.zai.glm-5.3",
    ],
)
def test_models_left_out(model_id: str) -> None:
    assert default_catalogue().get("bedrock", model_id) is None


class OfflineBedrockFactory(BedrockFactory):
    """Builds providers around a placeholder client, recording each client it makes."""

    def __init__(self) -> None:
        super().__init__()
        self.clients = 0

    def create_client(self, settings: ProviderSettings, secrets: Mapping[str, SecretStr]) -> Any:
        self.clients += 1
        return object()


IN_REGION_LONDON = ResidencyPolicy(
    scopes=frozenset({ResidencyScope.IN_REGION}), regions=frozenset({"eu-west-2"})
)


async def start_with(model: str, factory: OfflineBedrockFactory) -> Any:
    return await start(
        ConduitSettings.model_validate(
            {
                "providers": {"bedrock_london": {"kind": "bedrock", "region": "eu-west-2"}},
                "assignments": {"summarise": {"provider": "bedrock_london", "model": model}},
                "residency": IN_REGION_LONDON,
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
    async def test_an_in_region_policy_refuses_profiles_before_any_client_exists(
        self, model: str
    ) -> None:
        factory = OfflineBedrockFactory()
        with pytest.raises(ResidencyRefused):
            await start_with(model, factory)
        assert factory.clients == 0

    @pytest.mark.parametrize(
        "model",
        ["anthropic.claude-sonnet-4-6", "anthropic.claude-sonnet-5", "anthropic.claude-opus-5-5"],
    )
    async def test_it_accepts_the_bare_in_region_ids(self, model: str) -> None:
        factory = OfflineBedrockFactory()
        runtime = await start_with(model, factory)
        try:
            assert runtime.for_task("summarise").assignment.model == model
            assert factory.clients == 1
        finally:
            await runtime.aclose()
