"""Settings for the bedrock-mantle adapter: region, Bedrock API keys and client options."""

from __future__ import annotations

from typing import ClassVar

import pytest
from pydantic import ValidationError

from conduit.core.secrets import SecretRef
from conduit.providers.bedrock import (
    AccessKeyAuth,
    CredentialChainAuth,
    LongTermApiKeyAuth,
    ShortTermApiKeyAuth,
)
from conduit.providers.bedrock_mantle import (
    MANTLE_BASE_URL,
    BedrockMantleSettings,
    mantle_base_url,
)


def test_the_defaults_are_short_term_keys_from_the_credential_chain() -> None:
    settings = BedrockMantleSettings.model_validate(
        {"kind": "bedrock_mantle", "region": "ap-southeast-4"}
    )
    assert settings.kind == "bedrock_mantle"
    assert settings.catalogue_provider == "bedrock_mantle"
    assert settings.region == "ap-southeast-4"
    assert isinstance(settings.auth, ShortTermApiKeyAuth)
    assert isinstance(settings.auth.source, CredentialChainAuth)
    assert settings.allow_long_term_api_keys is False
    assert settings.endpoint_url is None
    assert settings.max_retries == 2


def test_the_region_is_required() -> None:
    with pytest.raises(ValidationError):
        BedrockMantleSettings.model_validate({"kind": "bedrock_mantle"})


@pytest.mark.parametrize("region", ["Melbourne", "ap-southeast", "ap_southeast_4", ""])
def test_the_region_must_look_like_an_aws_region(region: str) -> None:
    with pytest.raises(ValidationError):
        BedrockMantleSettings.model_validate({"kind": "bedrock_mantle", "region": region})


def test_short_term_keys_signed_with_access_keys() -> None:
    settings = BedrockMantleSettings.model_validate(
        {
            "kind": "bedrock_mantle",
            "region": "us-east-1",
            "auth": {
                "mode": "short_term_api_key",
                "source": {
                    "mode": "access_keys",
                    "access_key_id": "secret://mantle/id",
                    "secret_access_key": "secret://mantle/secret",
                },
                "lifetime_s": 3_600,
                "refresh_before_s": 300,
            },
        }
    )
    assert isinstance(settings.auth, ShortTermApiKeyAuth)
    assert isinstance(settings.auth.source, AccessKeyAuth)
    assert settings.auth.source.access_key_id == SecretRef("mantle/id")
    assert settings.auth.lifetime_s == 3_600


class TestLongTermKeys:
    LONG_TERM: ClassVar[dict[str, str]] = {
        "mode": "long_term_api_key",
        "api_key": "secret://mantle/key",
    }

    def test_refused_unless_explicitly_allowed(self) -> None:
        with pytest.raises(ValidationError, match="long-term"):
            BedrockMantleSettings.model_validate(
                {"kind": "bedrock_mantle", "region": "us-east-1", "auth": self.LONG_TERM}
            )

    def test_accepted_when_allowed(self) -> None:
        settings = BedrockMantleSettings.model_validate(
            {
                "kind": "bedrock_mantle",
                "region": "us-east-1",
                "auth": self.LONG_TERM,
                "allow_long_term_api_keys": True,
            }
        )
        assert isinstance(settings.auth, LongTermApiKeyAuth)

    def test_the_key_must_be_a_reference(self) -> None:
        with pytest.raises(ValidationError) as caught:
            BedrockMantleSettings.model_validate(
                {
                    "kind": "bedrock_mantle",
                    "region": "us-east-1",
                    "auth": {"mode": "long_term_api_key", "api_key": "plain-key-MARKER"},
                    "allow_long_term_api_keys": True,
                }
            )
        assert "MARKER" not in str(caught.value)


@pytest.mark.parametrize("mode", ["credential_chain", "access_keys"])
def test_sigv4_modes_are_not_offered(mode: str) -> None:
    # The mantle adapter authenticates with Bedrock API keys only.
    auth: dict[str, str] = {"mode": mode}
    if mode == "access_keys":
        auth |= {"access_key_id": "secret://a", "secret_access_key": "secret://b"}
    with pytest.raises(ValidationError):
        BedrockMantleSettings.model_validate(
            {"kind": "bedrock_mantle", "region": "us-east-1", "auth": auth}
        )


@pytest.mark.parametrize(
    "url", ["http://bedrock-mantle.example.com", "ftp://example.com", "bedrock-mantle.example.com"]
)
def test_an_endpoint_override_must_be_https(url: str) -> None:
    with pytest.raises(ValidationError):
        BedrockMantleSettings.model_validate(
            {"kind": "bedrock_mantle", "region": "us-east-1", "endpoint_url": url}
        )


def test_unknown_fields_are_refused() -> None:
    with pytest.raises(ValidationError):
        BedrockMantleSettings.model_validate(
            {"kind": "bedrock_mantle", "region": "us-east-1", "temperature": 0.2}
        )


@pytest.mark.parametrize(
    ("region", "expected"),
    [
        ("ap-southeast-4", "https://bedrock-mantle.ap-southeast-4.api.aws/anthropic"),
        ("us-gov-west-1", "https://bedrock-mantle.us-gov-west-1.api.aws/anthropic"),
    ],
)
def test_the_endpoint_for_a_region(region: str, expected: str) -> None:
    # The model cards: https://bedrock-mantle.{region}.api.aws/anthropic/v1/messages;
    # the SDK appends /v1/messages to its base URL.
    assert mantle_base_url(region) == expected
    assert MANTLE_BASE_URL.format(region=region) == expected
