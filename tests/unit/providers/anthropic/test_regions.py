"""Anthropic Regions: the direct API's ``inference_geo`` and a Foundry deployment's type.

``inference_geo: us`` keeps inference in the US (``anthropic:us``) for the models that
support it (Claude 4.6 and later), and every request carries it; a response whose
``usage.inference_geo`` names any other geography fails the call. ``global`` (or no
choice) is ``global``. Every event and setting here is synthetic.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from conduit.core.errors import ProviderError, ProviderFailure, SettingsRefused
from conduit.core.registry import ProviderRegistry
from conduit.core.residency import RegionChoice, Residency, ResidencyScope
from conduit.core.secrets import LiteralSecretResolver
from conduit.core.streaming import StreamEvent, StreamStopped
from conduit.core.types import Message, ModelRequest
from conduit.providers.anthropic import AnthropicFactory, AnthropicSettings
from conduit.providers.anthropic.request import build_params
from conduit.providers.anthropic.settings import INFERENCE_GEO_MODELS, supports_inference_geo
from conduit.providers.anthropic.stream import translate

from .fakes import message_end, message_start, scripted_client, text_block, usage

KEY = "secret://anthropic/key"
US = Residency(ResidencyScope.CROSS_REGION, frozenset({"anthropic:us"}))
GLOBAL = Residency(ResidencyScope.GLOBAL)
FOUNDRY_URL = "https://example-resource.services.ai.azure.com/anthropic/"


def _direct(**fields: Any) -> AnthropicSettings:
    return AnthropicSettings.model_validate({"api_key": KEY, **fields})


def _refused(mapping: dict[str, Any]) -> list[str]:
    registry = ProviderRegistry(secrets=LiteralSecretResolver({}), factories=[AnthropicFactory()])
    with pytest.raises(SettingsRefused) as refused:
        registry.validate("connection", {"kind": "anthropic", **mapping})
    assert refused.value.__cause__ is None and refused.value.__context__ is None
    return list(refused.value.reasons)


class TestNoChoice:
    def test_an_old_setting_keeps_its_meaning(self) -> None:
        settings = _direct()
        assert settings.inference_geo is None and settings.region is None
        assert settings.region_choice() is None
        assert settings.declared_residency("claude-sonnet-5-5") is None


class TestInferenceGeo:
    def test_us_declares_and_chooses_the_us(self) -> None:
        settings = _direct(inference_geo="us")
        assert settings.region == "anthropic:us"
        assert settings.region_choice() == RegionChoice(regions=frozenset({"anthropic:us"}))
        assert settings.declared_residency("claude-sonnet-5-5") == US
        assert settings.declared_residency("claude-sonnet-4-6-20260101") == US

    def test_a_model_without_inference_geo_support_is_global(self) -> None:
        settings = _direct(inference_geo="us")
        for model in ("claude-haiku-4-5", "claude-sonnet-4-5-20250929", "claude-opus-4-5"):
            assert not supports_inference_geo(model)
            assert settings.declared_residency(model) == GLOBAL

    def test_the_supported_models_follow_the_documentation(self) -> None:
        assert {"claude-opus-5-5", "claude-sonnet-4-6", "claude-opus-4-6"} <= INFERENCE_GEO_MODELS
        assert not supports_inference_geo("claude-sonnet-4-60")

    def test_global_declares_global_and_chooses_nothing(self) -> None:
        settings = _direct(inference_geo="global")
        assert settings.region is None
        assert settings.region_choice() is None
        assert settings.declared_residency("claude-sonnet-5-5") == GLOBAL

    def test_an_unknown_geo_is_refused(self) -> None:
        _refused({"api_key": KEY, "inference_geo": "eu"})

    def test_us_through_a_gateway_is_refused(self) -> None:
        reasons = _refused(
            {"api_key": KEY, "inference_geo": "us", "base_url": "https://gateway.example.com"}
        )
        assert any("gateway" in reason for reason in reasons)

    def test_inference_geo_is_only_for_the_direct_api(self) -> None:
        reasons = _refused(
            {"endpoint": "foundry", "base_url": FOUNDRY_URL, "api_key": KEY, "inference_geo": "us"}
        )
        assert any("only for endpoint direct" in reason for reason in reasons)

    def test_a_region_that_disagrees_is_refused(self) -> None:
        reasons = _refused({"api_key": KEY, "inference_geo": "us", "region": "anthropic:eu"})
        assert any("region is set by the" in reason for reason in reasons)

    def test_the_settings_round_trip(self) -> None:
        settings = _direct(inference_geo="us")
        assert AnthropicSettings.model_validate(settings.model_dump()) == settings


class TestFoundryDeployment:
    def test_a_foundry_deployment_declares_its_type(self) -> None:
        settings = AnthropicSettings.model_validate(
            {
                "endpoint": "foundry",
                "base_url": FOUNDRY_URL,
                "api_key": KEY,
                "deployment": {"type": "data_zone", "azure_region": "eastus2"},
            }
        )
        zone = frozenset({"azure:eastus2", "azure:zone-us"})
        assert settings.region == "azure:eastus2"
        assert settings.declared_residency("claude-sonnet-5-5") == Residency(
            ResidencyScope.CROSS_REGION, zone
        )
        assert settings.region_choice() == RegionChoice(regions=zone)

    def test_a_deployment_is_only_for_foundry(self) -> None:
        reasons = _refused(
            {"api_key": KEY, "deployment": {"type": "global", "azure_region": "eastus"}}
        )
        assert any("only for endpoint foundry" in reason for reason in reasons)


class TestListings:
    def test_a_listing_for_the_direct_api_and_for_foundry(self) -> None:
        registry = ProviderRegistry(
            secrets=LiteralSecretResolver({}), factories=[AnthropicFactory()]
        )
        listings = {listing.service: listing for listing in registry.regions("anthropic")}
        assert set(listings) == {"direct", "foundry"}
        [us] = listings["direct"].regions
        assert (us.code, us.scope) == ("anthropic:us", ResidencyScope.CROSS_REGION)
        assert listings["direct"].global_option is not None
        assert listings["direct"].documentation.startswith("https://platform.claude.com/")


def _request(model: str = "claude-sonnet-5-5") -> ModelRequest:
    return ModelRequest(model=model, messages=(Message.user("hello"),), max_output_tokens=64)


class TestRequest:
    def test_no_geo_sends_nothing(self) -> None:
        assert "inference_geo" not in build_params(_request(), structured_output=False)

    @pytest.mark.parametrize("geo", ["us", "global"])
    def test_a_geo_is_sent_on_every_request(self, geo: str) -> None:
        params = build_params(_request(), structured_output=False, inference_geo=geo)
        assert params["inference_geo"] == geo


async def _translated(events: list[Any], geo: str | None) -> list[StreamEvent]:
    async def source() -> AsyncIterator[Any]:
        for event in events:
            yield event

    return [event async for event in translate(source(), inference_geo=geo)]


def _reply(start_geo: str | None, end_geo: str | None = None) -> list[Any]:
    start = usage(input_tokens=5, output_tokens=1)
    if start_geo is not None:
        start["inference_geo"] = start_geo
    end: dict[str, Any] = {"output_tokens": 3}
    if end_geo is not None:
        end["inference_geo"] = end_geo
    return [message_start(start), *text_block(0, "hi"), *message_end("end_turn", end)]


class TestResponseGeo:
    async def test_a_reply_inside_the_geo_completes(self) -> None:
        events = await _translated(_reply("us", "us"), "us")
        assert isinstance(events[-1], StreamStopped)

    async def test_a_reply_outside_the_geo_fails_the_call(self) -> None:
        with pytest.raises(ProviderError) as failed:
            await _translated(_reply("global"), "us")
        assert failed.value.failure is ProviderFailure.REJECTED
        assert failed.value.__cause__ is None and failed.value.__context__ is None

    async def test_a_later_report_outside_the_geo_fails_the_call(self) -> None:
        with pytest.raises(ProviderError):
            await _translated(_reply("us", "eu"), "us")

    async def test_no_choice_ignores_the_report(self) -> None:
        events = await _translated(_reply("global"), None)
        assert isinstance(events[-1], StreamStopped)

    async def test_global_accepts_any_report(self) -> None:
        events = await _translated(_reply("us"), "global")
        assert isinstance(events[-1], StreamStopped)

    async def test_a_reply_that_reports_no_geo_completes(self) -> None:
        events = await _translated(_reply(None), "us")
        assert isinstance(events[-1], StreamStopped)


def _built(settings: AnthropicSettings, *steps: Any) -> tuple[Any, Any]:
    client = scripted_client(*steps)
    provider = AnthropicFactory().create_provider("claude", settings, client)
    return provider, client


class TestProvider:
    async def test_a_us_connection_sends_its_geo_and_completes(self) -> None:
        provider, client = _built(_direct(inference_geo="us"), _reply("us", "us"))
        events = [event async for event in provider.stream(_request())]
        assert isinstance(events[-1], StreamStopped)
        assert client.messages.calls[-1]["inference_geo"] == "us"

    async def test_a_us_connection_fails_a_reply_from_another_geo(self) -> None:
        provider, _ = _built(_direct(inference_geo="us"), _reply("global"))
        with pytest.raises(ProviderError) as failed:
            _ = [event async for event in provider.stream(_request())]
        assert failed.value.failure is ProviderFailure.REJECTED
        assert failed.value.__cause__ is None and failed.value.__context__ is None

    async def test_a_connection_without_a_geo_sends_none(self) -> None:
        provider, client = _built(_direct(), _reply("global"))
        _ = [event async for event in provider.stream(_request())]
        assert "inference_geo" not in client.messages.calls[-1]
