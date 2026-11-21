"""The default catalogue's bedrock-mantle entries: Claude's capabilities and limits, and
residency on the endpoint's own listing (C18).

Source for every entry: the Amazon Bedrock model card of each model
(docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-*.md, read
5 November 2026): the model ID from the card's Programmatic Access table, bedrock-mantle
row; context window and maximum output from Model Details; prompt caching and structured
output from "Features supported using bedrock-mantle endpoint". Which Regions serve a
model is the endpoint's listing's to say; bedrock-mantle offers no geographic or global
inference profiles, so every listed model is processed in the endpoint's Region.
"""

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
from conduit.providers.bedrock_mantle import BedrockMantleFactory

MELBOURNE = "ap-southeast-4"

#: model id -> (context window, output limit, prompt caching)
EXPECTED: dict[str, tuple[int, int, bool]] = {
    "anthropic.claude-opus-5-5": (1_000_000, 128_000, True),
    "anthropic.claude-sonnet-5-5": (1_000_000, 128_000, True),
    "anthropic.claude-haiku-5-5": (1_000_000, 128_000, True),
    "anthropic.claude-fable-5-1": (1_000_000, 128_000, True),
    # Added 10 November 2026 from its card (Active, legacy).
    "anthropic.claude-fable-5": (1_000_000, 128_000, True),
    "anthropic.claude-opus-5": (1_000_000, 128_000, True),
    "anthropic.claude-sonnet-5": (1_000_000, 128_000, True),
    "anthropic.claude-opus-4-8": (1_000_000, 128_000, True),
    "anthropic.claude-opus-4-7": (1_000_000, 128_000, True),
    # Its mantle feature list leaves out prompt caching; the live suite is to confirm.
    "anthropic.claude-haiku-4-5": (200_000, 64_000, False),
}


def mantle_entries() -> list[Any]:
    return [e for e in default_catalogue() if e.provider == "bedrock_mantle"]


def test_exactly_the_expected_models_are_declared() -> None:
    assert {e.model_id for e in mantle_entries()} == set(EXPECTED)


@pytest.mark.parametrize("model_id", sorted(EXPECTED))
def test_each_entry_matches_its_model_card(model_id: str) -> None:
    context, output, caching = EXPECTED[model_id]
    entry = default_catalogue().require("bedrock_mantle", model_id)
    assert entry.kind == "chat"
    assert entry.residency_scope is None  # the listing says where it is processed
    assert entry.regions == frozenset()
    assert entry.context_window == context
    assert entry.output_limit == output
    assert entry.capabilities.prompt_caching is caching
    assert entry.capabilities.tools
    assert entry.capabilities.vision
    assert entry.capabilities.structured_output


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


def _here(model_id: str) -> DiscoveredModel:
    return DiscoveredModel(model_id, scope=ResidencyScope.IN_REGION, regions=frozenset({MELBOURNE}))


#: A synthetic Melbourne listing of the endpoint.
MELBOURNE_LISTING = tuple(
    _here(model_id)
    for model_id in (
        "anthropic.claude-opus-5-5",
        "anthropic.claude-opus-5",
        "anthropic.claude-sonnet-5",
        "anthropic.claude-opus-4-8",
        "anthropic.claude-opus-4-7",
        "anthropic.claude-haiku-4-5",
    )
)


class OfflineMantleFactory(BedrockMantleFactory):
    """Lists a synthetic listing and builds providers around a placeholder client."""

    def __init__(self, listing: Sequence[DiscoveredModel] = MELBOURNE_LISTING) -> None:
        super().__init__()
        self.listing = tuple(listing)
        self.clients = 0

    def create_client(self, settings: ProviderSettings, secrets: Mapping[str, SecretStr]) -> Any:
        self.clients += 1
        return object()

    async def discover_models(
        self, settings: ProviderSettings, client: Any
    ) -> Sequence[DiscoveredModel]:
        return self.listing


IN_REGION_MELBOURNE = ResidencyPolicy(
    scopes=frozenset({ResidencyScope.IN_REGION}), regions=frozenset({MELBOURNE})
)


async def start_with(model: str, factory: Any, residency: ResidencyPolicy) -> Any:
    return await start(
        ConduitSettings.model_validate(
            {
                "providers": {"claude_melbourne": {"kind": "bedrock_mantle", "region": MELBOURNE}},
                "assignments": {"summarise": {"provider": "claude_melbourne", "model": model}},
                "residency": residency,
            }
        ),
        task_classes=[TaskClass(name="summarise")],
        factories=[factory],
        secrets=LiteralSecretResolver({}),
    )


class TestResidencyAtStartUp:
    @pytest.mark.parametrize("model", sorted(m.model_id for m in MELBOURNE_LISTING))
    async def test_an_in_region_melbourne_policy_accepts_the_listed_models(
        self, model: str
    ) -> None:
        factory = OfflineMantleFactory()
        runtime = await start_with(model, factory, IN_REGION_MELBOURNE)
        try:
            resolved = runtime.for_task("summarise").assignment
            assert resolved.model == model
            assert resolved.entry.regions == frozenset({MELBOURNE})
            assert factory.clients == 2  # the listing's, then the provider's
        finally:
            await runtime.aclose()

    async def test_a_policy_without_the_endpoint_region_refuses_it(self) -> None:
        sydney = ResidencyPolicy(
            scopes=frozenset({ResidencyScope.IN_REGION}), regions=frozenset({"ap-southeast-2"})
        )
        with pytest.raises(ResidencyRefused, match=MELBOURNE):
            await start_with("anthropic.claude-sonnet-5", OfflineMantleFactory(), sydney)

    @pytest.mark.parametrize(
        "model", ["au.anthropic.claude-sonnet-5", "anthropic.claude-sonnet-5-5"]
    )
    async def test_a_model_the_endpoint_does_not_list_is_refused(self, model: str) -> None:
        with pytest.raises(AssignmentRefused, match="is not listed by provider"):
            await start_with(model, OfflineMantleFactory(), IN_REGION_MELBOURNE)
