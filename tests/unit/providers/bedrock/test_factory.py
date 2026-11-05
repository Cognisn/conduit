"""The Bedrock factory is registered, and importing it never imports the SDK."""

from __future__ import annotations

import subprocess
import sys

import pytest
from pydantic import SecretStr

from conduit.core.errors import ConfigurationError
from conduit.core.registry import ProviderRegistry
from conduit.core.secrets import LiteralSecretResolver, SecretRef
from conduit.providers import builtin_factories
from conduit.providers.bedrock import BedrockFactory, BedrockSettings, LongTermApiKeyAuth


def test_builtin_factories_include_bedrock() -> None:
    assert [f for f in builtin_factories() if f.kind == "bedrock"]
    assert isinstance(next(f for f in builtin_factories() if f.kind == "bedrock"), BedrockFactory)


def test_importing_the_adapter_does_not_import_the_sdk() -> None:
    code = (
        "import sys, conduit.providers, conduit.providers.bedrock;"
        "conduit.providers.builtin_factories();"
        "print(sorted(m for m in sys.modules if m.split('.')[0] in ('boto3', 'botocore')))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == "[]"


class TestLongTermGate:
    """The client refuses a long-term key unless explicitly allowed, however the
    settings were made. Settings validation is one gate; create_client is the other."""

    def made_without_validation(self) -> list[BedrockSettings]:
        allowed = BedrockSettings(
            region="eu-west-2",
            auth=LongTermApiKeyAuth(api_key=SecretRef("bedrock/key")),
            allow_long_term_api_keys=True,
        )
        return [
            allowed.model_copy(update={"allow_long_term_api_keys": False}),
            BedrockSettings.model_construct(
                region="eu-west-2", auth=LongTermApiKeyAuth(api_key=SecretRef("bedrock/key"))
            ),
        ]

    def test_create_client_refuses_before_any_client_is_made(self) -> None:
        for settings in self.made_without_validation():
            assert settings.allow_long_term_api_keys is False
            with pytest.raises(ConfigurationError, match="long-term"):
                BedrockFactory().create_client(
                    settings, {"auth.api_key": SecretStr("synthetic-long-term-key")}
                )

    async def test_the_registry_refuses_them_too(self) -> None:
        for settings in self.made_without_validation():
            registry = ProviderRegistry(
                secrets=LiteralSecretResolver({"bedrock/key": "synthetic-long-term-key"}),
                factories=[BedrockFactory()],
            )
            with pytest.raises(ConfigurationError, match="long-term"):
                await registry.build("bedrock_london", settings)
