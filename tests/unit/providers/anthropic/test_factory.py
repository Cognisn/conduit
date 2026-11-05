"""The Anthropic adapter's settings and factory: the direct API and Microsoft Foundry."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

from conduit.core.errors import ConfigurationError
from conduit.core.events import MemoryEventSink, ModelCallFinishedEvent
from conduit.core.registry import ProviderFactory, ProviderRegistry
from conduit.core.secrets import LiteralSecretResolver, SecretRef
from conduit.core.types import Message, ModelRequest
from conduit.providers import builtin_factories
from conduit.providers.anthropic import (
    DEFAULT_BASE_URL,
    AnthropicFactory,
    AnthropicProvider,
    AnthropicSettings,
)

from .fakes import message_end, message_start, scripted_client, text_block, usage

KEY = "synthetic-test-key-not-real"


class TestSettings:
    def test_direct_defaults(self) -> None:
        settings = AnthropicSettings.model_validate(
            {"kind": "anthropic", "api_key": "secret://ANTHROPIC_KEY"}
        )
        assert settings.endpoint == "direct"
        assert settings.api_key == SecretRef("ANTHROPIC_KEY")
        assert settings.base_url is None
        assert settings.max_retries == 2

    def test_the_key_must_be_a_reference(self) -> None:
        with pytest.raises(ValidationError) as caught:
            AnthropicSettings.model_validate({"kind": "anthropic", "api_key": KEY})
        assert KEY not in str(caught.value)

    def test_the_key_is_required(self) -> None:
        with pytest.raises(ValidationError):
            AnthropicSettings.model_validate({"kind": "anthropic"})

    def test_foundry_needs_its_endpoint(self) -> None:
        with pytest.raises(ValidationError, match="base_url"):
            AnthropicSettings.model_validate(
                {"kind": "anthropic", "endpoint": "foundry", "api_key": "secret://K"}
            )

    def test_foundry_with_its_endpoint(self) -> None:
        settings = AnthropicSettings.model_validate(
            {
                "kind": "anthropic",
                "endpoint": "foundry",
                "api_key": "secret://FOUNDRY_KEY",
                "base_url": "https://example-resource.services.ai.azure.example.com/anthropic/",
            }
        )
        assert settings.endpoint == "foundry"

    @pytest.mark.parametrize(
        "url",
        [
            "http://api.example.com",
            "ftp://api.example.com",
            "not a url",
            "https://",
            "https://user:password@api.example.com",
        ],
    )
    def test_base_url_must_be_https(self, url: str) -> None:
        with pytest.raises(ValidationError) as caught:
            AnthropicSettings.model_validate(
                {"kind": "anthropic", "api_key": "secret://K", "base_url": url}
            )
        assert "password" not in str(caught.value)

    @pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "[::1]"])
    def test_plain_http_is_allowed_to_loopback(self, host: str) -> None:
        settings = AnthropicSettings.model_validate(
            {"kind": "anthropic", "api_key": "secret://K", "base_url": f"http://{host}:8080"}
        )
        assert settings.base_url == f"http://{host}:8080"

    def test_unknown_fields_are_refused(self) -> None:
        with pytest.raises(ValidationError):
            AnthropicSettings.model_validate(
                {"kind": "anthropic", "api_key": "secret://K", "temperature": 0.1}
            )

    @pytest.mark.parametrize("retries", [-1, 11])
    def test_retries_are_bounded(self, retries: int) -> None:
        with pytest.raises(ValidationError):
            AnthropicSettings.model_validate(
                {"kind": "anthropic", "api_key": "secret://K", "max_retries": retries}
            )


class _FakeSdk:
    """Stands in for the anthropic module, recording the client constructions."""

    def __init__(self) -> None:
        self.made: list[tuple[str, dict[str, Any]]] = []
        self.module = types.ModuleType("anthropic")
        self.module.AsyncAnthropic = self._maker("AsyncAnthropic")  # type: ignore[attr-defined]
        self.module.AsyncAnthropicFoundry = self._maker("AsyncAnthropicFoundry")  # type: ignore[attr-defined]

    def _maker(self, name: str) -> Any:
        def make(**kwargs: Any) -> tuple[str, dict[str, Any]]:
            self.made.append((name, kwargs))
            return (name, kwargs)

        return make


@pytest.fixture
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> _FakeSdk:
    sdk = _FakeSdk()
    monkeypatch.setitem(sys.modules, "anthropic", sdk.module)
    return sdk


class TestFactory:
    def test_it_is_a_provider_factory_of_kind_anthropic(self) -> None:
        factory = AnthropicFactory()
        assert isinstance(factory, ProviderFactory)
        assert factory.kind == "anthropic"
        assert factory.settings_model is AnthropicSettings

    def test_builtin_factories_include_it(self) -> None:
        kinds = [f.kind for f in builtin_factories()]
        assert "anthropic" in kinds
        assert "stub" in kinds

    def test_direct_client_gets_explicit_key_and_endpoint(self, fake_sdk: _FakeSdk) -> None:
        settings = AnthropicSettings(kind="anthropic", api_key=SecretRef("K"), max_retries=4)
        AnthropicFactory().create_client(settings, {"api_key": SecretStr(KEY)})
        assert fake_sdk.made == [
            (
                "AsyncAnthropic",
                {
                    "api_key": KEY,
                    "base_url": DEFAULT_BASE_URL,
                    "max_retries": 4,
                    "open_telemetry": False,
                },
            )
        ]

    def test_direct_client_with_a_base_url(self, fake_sdk: _FakeSdk) -> None:
        settings = AnthropicSettings(
            kind="anthropic", api_key=SecretRef("K"), base_url="https://gateway.example.com/"
        )
        AnthropicFactory().create_client(settings, {"api_key": SecretStr(KEY)})
        assert fake_sdk.made[0][1]["base_url"] == "https://gateway.example.com/"

    def test_foundry_client(self, fake_sdk: _FakeSdk) -> None:
        url = "https://example-resource.services.ai.azure.example.com/anthropic/"
        settings = AnthropicSettings(
            kind="anthropic", endpoint="foundry", api_key=SecretRef("K"), base_url=url
        )
        AnthropicFactory().create_client(settings, {"api_key": SecretStr(KEY)})
        assert fake_sdk.made == [
            (
                "AsyncAnthropicFoundry",
                {"api_key": KEY, "base_url": url, "max_retries": 2, "open_telemetry": False},
            )
        ]

    def test_create_client_needs_anthropic_settings(self, fake_sdk: _FakeSdk) -> None:
        from conduit.providers.stub import StubSettings

        with pytest.raises(ConfigurationError):
            AnthropicFactory().create_client(
                StubSettings(not_for_production=True), {"api_key": SecretStr(KEY)}
            )
        assert fake_sdk.made == []

    def test_missing_sdk_is_a_configuration_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "anthropic", None)
        settings = AnthropicSettings(kind="anthropic", api_key=SecretRef("K"))
        with pytest.raises(ConfigurationError, match=r"cognisn-conduit\[anthropic\]"):
            AnthropicFactory().create_client(settings, {"api_key": SecretStr(KEY)})

    def test_create_provider(self) -> None:
        settings = AnthropicSettings(kind="anthropic", api_key=SecretRef("K"))
        provider = AnthropicFactory().create_provider("claude", settings, client=object())
        assert isinstance(provider, AnthropicProvider)
        assert provider.kind == "anthropic"


class TestThroughTheRegistry:
    async def test_build_resolves_the_key_and_meters_calls(self, fake_sdk: _FakeSdk) -> None:
        sink = MemoryEventSink()
        registry = ProviderRegistry(
            secrets=LiteralSecretResolver({"ANTHROPIC_KEY": KEY}),
            events=sink,
            factories=[AnthropicFactory()],
        )
        provider = await registry.build(
            "claude", {"kind": "anthropic", "api_key": "secret://ANTHROPIC_KEY"}
        )
        assert fake_sdk.made[0][1]["api_key"] == KEY

        client = scripted_client(
            [
                message_start(usage(12, 1, 4, 0)),
                *text_block(0, "hi"),
                *message_end("end_turn", {"output_tokens": 3}),
            ]
        )
        metered = await registry.build(
            "claude", {"kind": "anthropic", "api_key": "secret://ANTHROPIC_KEY"}, client=client
        )
        response = await metered.complete(
            ModelRequest(
                model="claude-haiku-5-5", messages=(Message.user("x"),), max_output_tokens=8
            )
        )
        assert response.text == "hi"
        finished = [e for e in sink.events if isinstance(e, ModelCallFinishedEvent)]
        assert finished[-1].usage is not None
        assert finished[-1].usage.cache_read_tokens == 4
        assert finished[-1].usage.output_tokens == 3
        assert provider.kind == "anthropic"
        await registry.aclose()


class TestWithTheRealSdk:
    """Runs only where the anthropic extra is installed; never reaches the network."""

    def test_direct_and_foundry_clients_are_built(self) -> None:
        sdk = pytest.importorskip("anthropic")
        factory = AnthropicFactory()
        direct = factory.create_client(
            AnthropicSettings(kind="anthropic", api_key=SecretRef("K")),
            {"api_key": SecretStr(KEY)},
        )
        assert isinstance(direct, sdk.AsyncAnthropic)
        assert str(direct.base_url).rstrip("/") == DEFAULT_BASE_URL
        foundry = factory.create_client(
            AnthropicSettings(
                kind="anthropic",
                endpoint="foundry",
                api_key=SecretRef("K"),
                base_url="https://example-resource.services.ai.azure.example.com/anthropic/",
            ),
            {"api_key": SecretStr(KEY)},
        )
        assert isinstance(foundry, sdk.AsyncAnthropicFoundry)
        assert "example-resource" in str(foundry.base_url)

    def test_the_environment_does_not_redirect_the_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("anthropic")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://elsewhere.example.org")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "another-synthetic-key")
        client = AnthropicFactory().create_client(
            AnthropicSettings(kind="anthropic", api_key=SecretRef("K")),
            {"api_key": SecretStr(KEY)},
        )
        assert "elsewhere" not in str(client.base_url)
        assert client.api_key == KEY

    def test_real_sdk_errors_are_classified(self) -> None:
        sdk = pytest.importorskip("anthropic")
        from conduit.core.errors import ProviderFailure
        from conduit.providers.anthropic.errors import failure_for

        assert (
            failure_for(sdk.APIConnectionError(request=_http_request()))
            is ProviderFailure.UNAVAILABLE
        )
        assert failure_for(sdk.APITimeoutError(request=_http_request())) is ProviderFailure.TIMEOUT


def _http_request() -> Any:
    httpx2 = pytest.importorskip("httpx2")
    return httpx2.Request("POST", "https://api.example.com/v1/messages")
