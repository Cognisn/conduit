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

GEOGRAPHIC_PREFIXES = ("us.", "eu.", "au.", "jp.", "ca.", "in.", "apac.")


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
