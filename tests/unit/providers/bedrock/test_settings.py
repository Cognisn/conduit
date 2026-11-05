"""Bedrock settings: region, the authentication modes, and the long-term key gate."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
from pydantic import SecretStr, ValidationError

from conduit.core.secrets import LiteralSecretResolver, SecretRef, resolve_secret_fields
from conduit.core.settings import ProviderSettings
from conduit.providers.bedrock import (
    AccessKeyAuth,
    BedrockSettings,
    CredentialChainAuth,
    LongTermApiKeyAuth,
    ShortTermApiKeyAuth,
)


def settings(**fields: Any) -> BedrockSettings:
    return BedrockSettings.model_validate({"region": "eu-west-2", **fields})


class TestRegion:
    def test_the_region_is_required(self) -> None:
        with pytest.raises(ValidationError):
            BedrockSettings.model_validate({})

    @pytest.mark.parametrize("region", ["eu-west-2", "ap-southeast-4", "us-gov-west-1"])
    def test_aws_region_names_are_accepted(self, region: str) -> None:
        assert settings(region=region).region == region

    @pytest.mark.parametrize("region", ["EU-WEST-2", "eu_west_2", "london", "eu-west-2a"])
    def test_anything_else_is_refused(self, region: str) -> None:
        with pytest.raises(ValidationError):
            settings(region=region)


class TestAuthenticationModes:
    def test_the_default_is_the_credential_chain_without_a_profile(self) -> None:
        built = settings()
        assert built.kind == "bedrock"
        assert built.auth == CredentialChainAuth()
        assert built.auth.profile is None

    def test_the_credential_chain_takes_a_named_profile(self) -> None:
        built = settings(auth={"mode": "credential_chain", "profile": "example-profile"})
        assert isinstance(built.auth, CredentialChainAuth)
        assert built.auth.profile == "example-profile"

    def test_access_keys_are_secret_references(self) -> None:
        built = settings(
            auth={
                "mode": "access_keys",
                "access_key_id": "secret://bedrock/access-key-id",
                "secret_access_key": "secret://bedrock/secret-access-key",
            }
        )
        assert isinstance(built.auth, AccessKeyAuth)
        assert built.auth.access_key_id == SecretRef("bedrock/access-key-id")
        assert built.auth.session_token is None

    def test_a_literal_access_key_is_refused_without_echoing_it(self) -> None:
        with pytest.raises(ValidationError) as raised:
            settings(
                auth={
                    "mode": "access_keys",
                    "access_key_id": "AKIAEXAMPLELITERAL00",
                    "secret_access_key": "secret://bedrock/secret-access-key",
                }
            )
        assert "AKIAEXAMPLELITERAL00" not in str(raised.value)

    def test_short_term_keys_default_to_twelve_hours_from_the_credential_chain(self) -> None:
        built = settings(auth={"mode": "short_term_api_key"})
        assert isinstance(built.auth, ShortTermApiKeyAuth)
        assert built.auth.source == CredentialChainAuth()
        assert built.auth.lifetime_s == 43_200
        assert built.auth.refresh_before_s < built.auth.lifetime_s

    def test_short_term_keys_can_be_signed_with_access_keys(self) -> None:
        built = settings(
            auth={
                "mode": "short_term_api_key",
                "source": {
                    "mode": "access_keys",
                    "access_key_id": "secret://a",
                    "secret_access_key": "secret://b",
                },
            }
        )
        assert isinstance(built.auth, ShortTermApiKeyAuth)
        assert isinstance(built.auth.source, AccessKeyAuth)

    @pytest.mark.parametrize("lifetime", [0, 899, 43_201])
    def test_a_short_term_lifetime_outside_fifteen_minutes_to_twelve_hours_is_refused(
        self, lifetime: int
    ) -> None:
        with pytest.raises(ValidationError):
            settings(auth={"mode": "short_term_api_key", "lifetime_s": lifetime})

    def test_the_refresh_margin_must_leave_part_of_the_lifetime(self) -> None:
        with pytest.raises(ValidationError, match="refresh_before_s"):
            settings(
                auth={"mode": "short_term_api_key", "lifetime_s": 900, "refresh_before_s": 900}
            )

    def test_an_unknown_mode_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            settings(auth={"mode": "password"})


class TestLongTermKeys:
    LONG_TERM: ClassVar[dict[str, str]] = {
        "mode": "long_term_api_key",
        "api_key": "secret://bedrock/api-key",
    }

    def test_a_long_term_key_is_refused_by_default(self) -> None:
        with pytest.raises(ValidationError, match="long-term Bedrock API keys are refused"):
            settings(auth=self.LONG_TERM)

    def test_a_long_term_key_is_accepted_only_when_explicitly_allowed(self) -> None:
        built = settings(auth=self.LONG_TERM, allow_long_term_api_keys=True)
        assert isinstance(built.auth, LongTermApiKeyAuth)
        assert built.auth.api_key == SecretRef("bedrock/api-key")

    def test_the_refusal_holds_when_the_settings_arrive_as_generic_provider_settings(
        self,
    ) -> None:
        # The provider registry re-validates ConduitSettings.providers entries against the
        # adapter's own model; the gate must hold on that path too.
        generic = ProviderSettings.model_validate(
            {"kind": "bedrock", "region": "eu-west-2", "auth": self.LONG_TERM}
        )
        with pytest.raises(ValidationError, match="long-term"):
            BedrockSettings.model_validate(generic.model_dump())


class TestSecretResolution:
    async def test_nested_references_resolve_by_dotted_path(self) -> None:
        built = settings(
            auth={
                "mode": "short_term_api_key",
                "source": {
                    "mode": "access_keys",
                    "access_key_id": "secret://a",
                    "secret_access_key": "secret://b",
                    "session_token": "secret://c",
                },
            }
        )
        resolver = LiteralSecretResolver({"a": "id", "b": "secret", "c": "token"})
        resolved = await resolve_secret_fields(built, resolver)
        assert {k: v.get_secret_value() for k, v in resolved.items()} == {
            "auth.source.access_key_id": "id",
            "auth.source.secret_access_key": "secret",
            "auth.source.session_token": "token",
        }
        assert all(isinstance(v, SecretStr) for v in resolved.values())


class TestOtherSettings:
    def test_unknown_fields_are_refused(self) -> None:
        with pytest.raises(ValidationError):
            settings(colour="blue")

    def test_converse_stream_is_the_default(self) -> None:
        assert settings().use_converse_stream is True

    def test_an_endpoint_override_must_be_https(self) -> None:
        assert settings(endpoint_url="https://vpce.example.com").endpoint_url == (
            "https://vpce.example.com"
        )
        with pytest.raises(ValidationError):
            settings(endpoint_url="http://vpce.example.com")
