"""The mantle factory: registered, SDK-free to import, and strict before any client exists."""

from __future__ import annotations

import subprocess
import sys
import types
from typing import Any

import pytest
from pydantic import SecretStr

from conduit.core.catalogue import CatalogueOverlay, default_catalogue
from conduit.core.errors import ConfigurationError
from conduit.core.registry import ProviderRegistry
from conduit.core.secrets import LiteralSecretResolver, SecretRef
from conduit.providers import builtin_factories
from conduit.providers.anthropic import AnthropicSettings
from conduit.providers.bedrock import LongTermApiKeyAuth
from conduit.providers.bedrock_mantle import (
    BedrockMantleFactory,
    BedrockMantleProvider,
    BedrockMantleSettings,
)

from ..anthropic.fakes import scripted_client

LONG_TERM_SECRETS = {"auth.api_key": SecretStr("synthetic-long-term-key")}


def test_builtin_factories_include_bedrock_mantle() -> None:
    found = [f for f in builtin_factories() if f.kind == "bedrock_mantle"]
    assert len(found) == 1
    assert isinstance(found[0], BedrockMantleFactory)
    assert found[0].settings_model is BedrockMantleSettings


def test_importing_the_adapter_does_not_import_any_sdk() -> None:
    code = (
        "import sys, conduit.providers, conduit.providers.bedrock_mantle;"
        "conduit.providers.builtin_factories();"
        "print(sorted(m for m in sys.modules"
        " if m.split('.')[0] in ('anthropic', 'boto3', 'botocore', 'httpx', 'httpx2')))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == "[]"


def test_other_settings_are_refused() -> None:
    with pytest.raises(ConfigurationError):
        BedrockMantleFactory().create_client(
            AnthropicSettings(api_key=SecretRef("k")), LONG_TERM_SECRETS
        )


class TestLongTermGate:
    """The client refuses a long-term key unless explicitly allowed, however the
    settings were made. Settings validation is one gate; create_client is the other."""

    def made_without_validation(self) -> list[BedrockMantleSettings]:
        allowed = BedrockMantleSettings(
            region="us-east-1",
            auth=LongTermApiKeyAuth(api_key=SecretRef("mantle/key")),
            allow_long_term_api_keys=True,
        )
        return [
            allowed.model_copy(update={"allow_long_term_api_keys": False}),
            BedrockMantleSettings.model_construct(
                region="us-east-1", auth=LongTermApiKeyAuth(api_key=SecretRef("mantle/key"))
            ),
        ]

    def test_a_truthy_flag_that_is_not_true_is_no_allowance(self) -> None:
        allowed = BedrockMantleSettings(
            region="us-east-1",
            auth=LongTermApiKeyAuth(api_key=SecretRef("mantle/key")),
            allow_long_term_api_keys=True,
        )
        settings = allowed.model_copy(update={"allow_long_term_api_keys": "yes"})
        with pytest.raises(ConfigurationError, match="long-term"):
            BedrockMantleFactory().create_client(settings, LONG_TERM_SECRETS)

    def test_create_client_refuses_before_anything_is_imported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # With the SDK made unimportable, a refusal naming the extra would mean the
        # gate came too late.
        monkeypatch.setitem(sys.modules, "anthropic.lib.bedrock", None)
        for settings in self.made_without_validation():
            with pytest.raises(ConfigurationError, match="long-term"):
                BedrockMantleFactory().create_client(settings, LONG_TERM_SECRETS)

    async def test_the_registry_refuses_them_too(self) -> None:
        for settings in self.made_without_validation():
            registry = ProviderRegistry(
                secrets=LiteralSecretResolver({"mantle/key": "synthetic-long-term-key"}),
                factories=[BedrockMantleFactory()],
            )
            with pytest.raises(ConfigurationError, match="long-term"):
                await registry.build("mantle", settings)


class TestEnvironment:
    def test_sdk_debug_logging_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # ANTHROPIC_LOG=debug makes the SDK log whole request bodies.
        monkeypatch.setenv("ANTHROPIC_LOG", "debug")
        settings = BedrockMantleSettings(region="us-east-1")
        with pytest.raises(ConfigurationError, match="ANTHROPIC_LOG") as caught:
            BedrockMantleFactory().create_client(settings, {})
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None

    def test_the_sdk_missing_is_a_configuration_error_naming_the_extra(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "anthropic.lib.bedrock", None)
        settings = BedrockMantleSettings(region="us-east-1")
        with pytest.raises(ConfigurationError, match=r"bedrock-mantle") as caught:
            BedrockMantleFactory().create_client(settings, {})
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None


class TestCreateProvider:
    def test_without_a_catalogue_the_default_is_used(self) -> None:
        settings = BedrockMantleSettings(region="ap-southeast-4")
        provider = BedrockMantleFactory().create_provider("mantle", settings, scripted_client())
        assert isinstance(provider, BedrockMantleProvider)
        assert provider.structured_output("anthropic.claude-sonnet-5")

    async def test_the_registry_passes_its_catalogue(self) -> None:
        entry = default_catalogue().require("bedrock_mantle", "anthropic.claude-sonnet-5")
        changed = entry.model_copy(
            update={"capabilities": entry.capabilities.model_copy(update={"prompt_caching": False})}
        )
        catalogue = default_catalogue().apply(CatalogueOverlay(override=(changed,)))
        registry = ProviderRegistry(
            secrets=LiteralSecretResolver({}),
            factories=[BedrockMantleFactory()],
            catalogue=catalogue,
        )
        built = await registry.build(
            "mantle", BedrockMantleSettings(region="ap-southeast-4"), client=scripted_client()
        )
        try:
            inner = built._inner
            assert isinstance(inner, BedrockMantleProvider)
            assert inner.prompt_caching("anthropic.claude-sonnet-5") is False
        finally:
            await registry.aclose()


class TestClientOptions:
    """The SDK client is made from conduit's settings alone (a stand-in SDK module)."""

    def made(self, monkeypatch: pytest.MonkeyPatch, **fields: Any) -> dict[str, Any]:
        calls: list[dict[str, Any]] = []
        module = types.ModuleType("anthropic.lib.bedrock")

        class StandIn:
            def __init__(self, **kwargs: Any) -> None:
                calls.append(kwargs)

            async def _prepare_request(self, request: Any) -> None:
                return None

        module.AsyncAnthropicBedrockMantle = StandIn  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "anthropic.lib.bedrock", module)
        settings = BedrockMantleSettings(
            region="ap-southeast-4",
            auth=LongTermApiKeyAuth(api_key=SecretRef("mantle/key")),
            allow_long_term_api_keys=True,
            **fields,
        )
        client = BedrockMantleFactory().create_client(settings, LONG_TERM_SECRETS)
        # A subclass of the SDK's client, which sets the key at send time.
        assert isinstance(client, StandIn)
        assert type(client) is not StandIn
        (kwargs,) = calls
        return kwargs

    def test_every_option_comes_from_the_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        kwargs = self.made(monkeypatch)
        assert kwargs == {
            # No SDK auth: no key from the environment, and no key held by the client.
            "skip_auth": True,
            "aws_region": "ap-southeast-4",
            "base_url": "https://bedrock-mantle.ap-southeast-4.api.aws/anthropic",
            "max_retries": 2,
            # The SDK's spans can carry request content outside the capture policy.
            "open_telemetry": False,
        }

    def test_an_endpoint_override_and_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        kwargs = self.made(
            monkeypatch, endpoint_url="https://vpce.example.com/anthropic", max_retries=0
        )
        assert kwargs["base_url"] == "https://vpce.example.com/anthropic"
        assert kwargs["max_retries"] == 0


class TestDestinationGate:
    """create_client checks the Region and endpoint again, however the settings were
    made, so a key is never sent to a host the validators would have refused."""

    @pytest.mark.parametrize(
        "region", ["evil.example.com#", "us-east-1.example.com/", "", "US-EAST-1"]
    )
    def test_a_region_that_is_not_an_aws_region_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, region: str
    ) -> None:
        monkeypatch.setitem(sys.modules, "anthropic.lib.bedrock", None)
        settings = BedrockMantleSettings(region="us-east-1").model_copy(update={"region": region})
        with pytest.raises(ConfigurationError, match="region") as caught:
            BedrockMantleFactory().create_client(settings, {})
        if region:
            assert region not in str(caught.value)
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None

    @pytest.mark.parametrize(
        "url", ["http://planted.example.com/anthropic", "planted.example.com", "ftp://x"]
    )
    def test_an_endpoint_that_is_not_https_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, url: str
    ) -> None:
        monkeypatch.setitem(sys.modules, "anthropic.lib.bedrock", None)
        settings = BedrockMantleSettings.model_construct(region="us-east-1", endpoint_url=url)
        with pytest.raises(ConfigurationError, match="https") as caught:
            BedrockMantleFactory().create_client(settings, {})
        assert url not in str(caught.value)
