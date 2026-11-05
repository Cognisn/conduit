"""The Anthropic adapter's settings and factory: the direct API and Microsoft Foundry."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

from conduit.core.catalogue import CatalogueOverlay, default_catalogue
from conduit.core.errors import ConfigurationError, ProviderError, ProviderFailure
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

    def test_an_unparsable_url_is_refused_with_nothing_chained(self) -> None:
        # urlsplit's own error quotes the netloc, credential included.
        url = "https://user:SECRET-MARKER@ex\uff03ample.com/"
        with pytest.raises(ValidationError) as caught:
            AnthropicSettings.model_validate(
                {"kind": "anthropic", "api_key": "secret://K", "base_url": url}
            )
        assert "SECRET-MARKER" not in str(caught.value)
        inner = caught.value.errors()[0]["ctx"]["error"]
        assert isinstance(inner, ValueError)
        assert inner.__cause__ is None
        assert inner.__context__ is None
        for text in (str(inner), repr(inner), *(repr(a) for a in inner.args)):
            assert "SECRET-MARKER" not in text

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
        with pytest.raises(ConfigurationError, match=r"cognisn-conduit\[anthropic\]") as caught:
            AnthropicFactory().create_client(settings, {"api_key": SecretStr(KEY)})
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None

    def test_create_provider(self) -> None:
        settings = AnthropicSettings(kind="anthropic", api_key=SecretRef("K"))
        provider = AnthropicFactory().create_provider("claude", settings, client=object())
        assert isinstance(provider, AnthropicProvider)
        assert provider.kind == "anthropic"


FOUNDRY_URL = "https://example-resource.services.ai.azure.example.com/anthropic/"
ENVIRONMENT_VALUE = "x-api-key: env-synthetic-key\nanthropic-version: 2023-06-01"


def _settings(endpoint: str) -> AnthropicSettings:
    if endpoint == "foundry":
        return AnthropicSettings(
            kind="anthropic", endpoint="foundry", api_key=SecretRef("K"), base_url=FOUNDRY_URL
        )
    return AnthropicSettings(kind="anthropic", api_key=SecretRef("K"))


class TestEnvironmentCannotOverrideTheSettings:
    """The SDK reads some variables even when conduit passes every value explicitly."""

    @pytest.mark.parametrize("endpoint", ["direct", "foundry"])
    @pytest.mark.parametrize("value", [ENVIRONMENT_VALUE, ""])
    def test_custom_headers_are_refused(
        self, fake_sdk: _FakeSdk, monkeypatch: pytest.MonkeyPatch, endpoint: str, value: str
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", value)
        with pytest.raises(ConfigurationError, match="ANTHROPIC_CUSTOM_HEADERS") as caught:
            AnthropicFactory().create_client(_settings(endpoint), {"api_key": SecretStr(KEY)})
        assert "env-synthetic-key" not in str(caught.value)
        assert fake_sdk.made == []

    def test_a_foundry_resource_is_refused_for_foundry(
        self, fake_sdk: _FakeSdk, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_FOUNDRY_RESOURCE", "elsewhere-synthetic")
        with pytest.raises(ConfigurationError, match="ANTHROPIC_FOUNDRY_RESOURCE") as caught:
            AnthropicFactory().create_client(_settings("foundry"), {"api_key": SecretStr(KEY)})
        assert "elsewhere-synthetic" not in str(caught.value)
        assert fake_sdk.made == []

    def test_a_foundry_resource_does_not_affect_the_direct_api(
        self, fake_sdk: _FakeSdk, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_FOUNDRY_RESOURCE", "elsewhere-synthetic")
        AnthropicFactory().create_client(_settings("direct"), {"api_key": SecretStr(KEY)})
        assert fake_sdk.made[0][0] == "AsyncAnthropic"

    @pytest.mark.parametrize("endpoint", ["direct", "foundry"])
    def test_sdk_debug_logging_is_refused(
        self, fake_sdk: _FakeSdk, monkeypatch: pytest.MonkeyPatch, endpoint: str
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_LOG", "debug")
        with pytest.raises(ConfigurationError, match="ANTHROPIC_LOG"):
            AnthropicFactory().create_client(_settings(endpoint), {"api_key": SecretStr(KEY)})
        assert fake_sdk.made == []

    def test_sdk_info_logging_is_allowed(
        self, fake_sdk: _FakeSdk, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_LOG", "info")
        AnthropicFactory().create_client(_settings("direct"), {"api_key": SecretStr(KEY)})
        assert len(fake_sdk.made) == 1


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

    async def test_structured_output_follows_the_registry_catalogue(self) -> None:
        # A product overlay that withdraws structured output reaches the adapter.
        model = "claude-haiku-5-5"
        entry = default_catalogue().require("anthropic", model)
        withdrawn = entry.model_copy(
            update={
                "capabilities": entry.capabilities.model_copy(update={"structured_output": False})
            }
        )
        registry = ProviderRegistry(
            secrets=LiteralSecretResolver({}),
            catalogue=default_catalogue().apply(
                CatalogueOverlay(name="no-structured", override=(withdrawn,))
            ),
            factories=[AnthropicFactory()],
        )
        client = scripted_client()
        provider = await registry.build(
            "claude", {"kind": "anthropic", "api_key": "secret://ANTHROPIC_KEY"}, client=client
        )
        request = ModelRequest(
            model=model,
            messages=(Message.user("x"),),
            max_output_tokens=8,
            output_schema={"type": "object"},
        )
        with pytest.raises(ProviderError) as caught:
            await provider.complete(request)
        assert caught.value.failure is ProviderFailure.REJECTED
        assert client.messages.calls == []


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

    @pytest.mark.parametrize("endpoint", ["direct", "foundry"])
    def test_the_environment_does_not_redirect_the_client(
        self, monkeypatch: pytest.MonkeyPatch, endpoint: str
    ) -> None:
        pytest.importorskip("anthropic")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://elsewhere.example.org")
        monkeypatch.setenv("ANTHROPIC_FOUNDRY_BASE_URL", "https://elsewhere.example.org")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "another-synthetic-key")
        monkeypatch.setenv("ANTHROPIC_FOUNDRY_API_KEY", "another-synthetic-key")
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "another-synthetic-token")
        client = AnthropicFactory().create_client(_settings(endpoint), {"api_key": SecretStr(KEY)})
        assert "elsewhere" not in str(client.base_url)
        assert client.api_key == KEY
        headers = {k.lower(): v for k, v in client.default_headers.items()}
        assert "another-synthetic" not in repr(headers)
        assert headers["anthropic-version"] == "2026-10-21"
        if endpoint == "direct":
            assert headers["authorization"] == f"Bearer {KEY}"
        else:
            assert "authorization" not in headers

    @pytest.mark.parametrize("endpoint", ["direct", "foundry"])
    def test_custom_headers_never_reach_a_real_client(
        self, monkeypatch: pytest.MonkeyPatch, endpoint: str
    ) -> None:
        pytest.importorskip("anthropic")
        monkeypatch.setenv(
            "ANTHROPIC_CUSTOM_HEADERS",
            "x-api-key: env-synthetic\napi-key: env-synthetic\n"
            "Authorization: Bearer env-synthetic\nanthropic-version: 2023-06-01",
        )
        with pytest.raises(ConfigurationError, match="ANTHROPIC_CUSTOM_HEADERS") as caught:
            AnthropicFactory().create_client(_settings(endpoint), {"api_key": SecretStr(KEY)})
        assert "env-synthetic" not in str(caught.value)

    def test_a_foundry_resource_is_a_configuration_error_with_the_real_sdk(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("anthropic")
        monkeypatch.setenv("ANTHROPIC_FOUNDRY_RESOURCE", "elsewhere-synthetic")
        with pytest.raises(ConfigurationError, match="ANTHROPIC_FOUNDRY_RESOURCE"):
            AnthropicFactory().create_client(_settings("foundry"), {"api_key": SecretStr(KEY)})

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
