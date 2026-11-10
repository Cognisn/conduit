"""The default catalogue's bedrock-mantle entries: single-Region Claude, and residency.

Source for every entry: the Amazon Bedrock model card of each model
(docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-*.md, read
5 November 2026), section "Availability using the bedrock-mantle endpoint", In-Region
column; the model ID from the card's Programmatic Access table, bedrock-mantle row;
context window and maximum output from Model Details; prompt caching and structured
output from "Features supported using bedrock-mantle endpoint". bedrock-mantle offers
no geographic or global inference profiles, so every entry is in_region.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import SecretStr

from conduit.core.catalogue import default_catalogue
from conduit.core.errors import ResidencyRefused, UndeclaredModel
from conduit.core.residency import ResidencyPolicy, ResidencyScope
from conduit.core.runtime import ConduitSettings, start
from conduit.core.secrets import LiteralSecretResolver
from conduit.core.settings import ProviderSettings
from conduit.core.tasks import TaskClass
from conduit.providers.bedrock import BedrockFactory
from conduit.providers.bedrock_mantle import BedrockMantleFactory

MELBOURNE = "ap-southeast-4"

#: model id -> (regions, context window, output limit, prompt caching)
EXPECTED: dict[str, tuple[frozenset[str], int, int, bool]] = {
    "anthropic.claude-opus-5-5": (
        frozenset({"us-east-1", MELBOURNE, "us-gov-west-1"}),
        1_000_000,
        128_000,
        True,
    ),
    "anthropic.claude-sonnet-5-5": (frozenset({"us-gov-west-1"}), 1_000_000, 128_000, True),
    "anthropic.claude-haiku-5-5": (frozenset({"us-gov-west-1"}), 1_000_000, 128_000, True),
    "anthropic.claude-fable-5-1": (frozenset({"us-gov-west-1"}), 1_000_000, 128_000, True),
    # Added 10 November 2026 from its card (Active, legacy): mantle in us-east-1.
    "anthropic.claude-fable-5": (frozenset({"us-east-1"}), 1_000_000, 128_000, True),
    "anthropic.claude-opus-5": (
        frozenset({"us-east-1", "us-gov-west-1", "eu-north-1", "eu-west-1", MELBOURNE}),
        1_000_000,
        128_000,
        True,
    ),
    "anthropic.claude-sonnet-5": (
        frozenset({"us-east-1", "us-gov-west-1", "eu-north-1", "eu-west-1", MELBOURNE}),
        1_000_000,
        128_000,
        True,
    ),
    "anthropic.claude-opus-4-8": (
        frozenset(
            {"us-east-1", "us-gov-west-1", "eu-north-1", "eu-west-1", "ap-northeast-1", MELBOURNE}
        ),
        1_000_000,
        128_000,
        True,
    ),
    "anthropic.claude-opus-4-7": (
        frozenset({"us-east-1", "eu-north-1", "eu-west-1", "ap-northeast-1", MELBOURNE}),
        1_000_000,
        128_000,
        True,
    ),
    # Its mantle feature list leaves out prompt caching; the live suite is to confirm.
    "anthropic.claude-haiku-4-5": (
        frozenset(
            {
                "us-east-1",
                "us-east-2",
                "us-west-2",
                "us-gov-west-1",
                "eu-north-1",
                "eu-west-1",
                "ap-northeast-1",
                MELBOURNE,
            }
        ),
        200_000,
        64_000,
        False,
    ),
}


def mantle_entries() -> list[Any]:
    return [e for e in default_catalogue() if e.provider == "bedrock_mantle"]


def test_exactly_the_expected_models_are_declared() -> None:
    assert {e.model_id for e in mantle_entries()} == set(EXPECTED)


@pytest.mark.parametrize("model_id", sorted(EXPECTED))
def test_each_entry_matches_its_model_card(model_id: str) -> None:
    regions, context, output, caching = EXPECTED[model_id]
    entry = default_catalogue().require("bedrock_mantle", model_id)
    assert entry.kind == "chat"
    assert entry.residency_scope is ResidencyScope.IN_REGION
    assert entry.regions == regions
    assert entry.context_window == context
    assert entry.output_limit == output
    assert entry.capabilities.prompt_caching is caching
    assert entry.capabilities.tools
    assert entry.capabilities.vision
    assert entry.capabilities.structured_output


def test_melbourne_has_single_region_claude() -> None:
    in_melbourne = {e.model_id for e in mantle_entries() if MELBOURNE in e.regions}
    assert in_melbourne == {
        "anthropic.claude-opus-5-5",
        "anthropic.claude-opus-5",
        "anthropic.claude-sonnet-5",
        "anthropic.claude-opus-4-8",
        "anthropic.claude-opus-4-7",
        "anthropic.claude-haiku-4-5",
    }


@pytest.mark.parametrize(
    "model_id",
    [
        # Preview and gated to verified organisations.
        "anthropic.claude-mythos-5",
        # Not served on bedrock-mantle (their cards: bedrock-runtime only).
        "anthropic.claude-fable-5-2",
        "anthropic.claude-sonnet-4-6",
    ],
)
def test_gated_and_runtime_only_models_are_not_declared(model_id: str) -> None:
    assert default_catalogue().get("bedrock_mantle", model_id) is None


class OfflineMantleFactory(BedrockMantleFactory):
    """Builds providers around a placeholder client, counting the clients it makes."""

    def __init__(self) -> None:
        super().__init__()
        self.clients = 0

    def create_client(self, settings: ProviderSettings, secrets: Mapping[str, SecretStr]) -> Any:
        self.clients += 1
        return object()


class OfflineBedrockFactory(BedrockFactory):
    def __init__(self) -> None:
        super().__init__()
        self.clients = 0

    def create_client(self, settings: ProviderSettings, secrets: Mapping[str, SecretStr]) -> Any:
        self.clients += 1
        return object()


IN_REGION_MELBOURNE = ResidencyPolicy(
    scopes=frozenset({ResidencyScope.IN_REGION}), regions=frozenset({MELBOURNE})
)


async def start_with(kind: str, model: str, factory: Any) -> Any:
    return await start(
        ConduitSettings.model_validate(
            {
                "providers": {"claude_melbourne": {"kind": kind, "region": MELBOURNE}},
                "assignments": {"summarise": {"provider": "claude_melbourne", "model": model}},
                "residency": IN_REGION_MELBOURNE,
            }
        ),
        task_classes=[TaskClass(name="summarise")],
        factories=[factory],
        secrets=LiteralSecretResolver({}),
    )


class TestResidencyAtStartUp:
    @pytest.mark.parametrize(
        "model",
        sorted(m for m, (regions, *_) in EXPECTED.items() if MELBOURNE in regions),
    )
    async def test_an_in_region_melbourne_policy_accepts_the_mantle_entries(
        self, model: str
    ) -> None:
        factory = OfflineMantleFactory()
        runtime = await start_with("bedrock_mantle", model, factory)
        try:
            assert runtime.for_task("summarise").assignment.model == model
            assert factory.clients == 1
        finally:
            await runtime.aclose()

    @pytest.mark.parametrize(
        "model",
        [
            "au.anthropic.claude-sonnet-5",
            "au.anthropic.claude-opus-5-5",
            "au.anthropic.claude-haiku-4-5-20251001-v1:0",
            "global.anthropic.claude-sonnet-5",
        ],
    )
    async def test_it_refuses_bedrock_cross_region_profiles_before_any_client(
        self, model: str
    ) -> None:
        factory = OfflineBedrockFactory()
        with pytest.raises(ResidencyRefused):
            await start_with("bedrock", model, factory)
        assert factory.clients == 0

    @pytest.mark.parametrize(
        "model", ["au.anthropic.claude-sonnet-5", "global.anthropic.claude-opus-5-5"]
    )
    async def test_a_profile_id_on_mantle_is_undeclared(self, model: str) -> None:
        # bedrock-mantle has no cross-Region profiles, so none is declared for it.
        factory = OfflineMantleFactory()
        with pytest.raises(UndeclaredModel):
            await start_with("bedrock_mantle", model, factory)
        assert factory.clients == 0
