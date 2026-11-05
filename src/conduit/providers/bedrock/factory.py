"""The Bedrock provider factory: the only place the Bedrock SDK is imported.

``create_client`` imports boto3 and botocore, builds a ``bedrock-runtime`` client for
the configured region and authentication mode, and returns it:

- ``credential_chain`` and ``access_keys``: a boto3 session (named profile, or the
  resolved access keys) signs every request with SigV4.
- ``short_term_api_key`` and ``long_term_api_key``: the client is unsigned, and a
  ``before-send`` handler sets ``Authorization: Bearer <key>`` on every attempt,
  retries included. Short-term keys are presigned here with botocore's SigV4 query
  signer and refreshed by ``ShortTermKeySource`` before they expire; the first key is
  generated at once, so missing credentials are refused at start-up.

The process environment (``AWS_BEARER_TOKEN_BEDROCK``) is never read or written: two
providers in one process can use different keys.

The factory takes the catalogue its providers consult for prompt-caching support. It
defaults to conduit's default catalogue; a product that overlays the catalogue passes
the applied catalogue here (see the Skuld question on passing it from the registry).
"""

from __future__ import annotations

import importlib
import time
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import SecretStr

from conduit.core.catalogue import Catalogue, default_catalogue
from conduit.core.errors import ConfigurationError
from conduit.core.provider import Provider
from conduit.core.settings import ProviderSettings
from conduit.providers.bedrock.auth import (
    KEY_ACTION,
    KEY_HOST,
    KEY_SIGNING_NAME,
    KEY_URL,
    IssuedKey,
    KeySource,
    ShortTermKeySource,
    StaticKeySource,
    encode_short_term_key,
)
from conduit.providers.bedrock.provider import BedrockProvider
from conduit.providers.bedrock.settings import (
    AccessKeyAuth,
    BedrockSettings,
    CredentialChainAuth,
    LongTermApiKeyAuth,
    ShortTermApiKeyAuth,
)

SERVICE = "bedrock-runtime"
USER_AGENT = "conduit"


def _bedrock(settings: ProviderSettings) -> BedrockSettings:
    if not isinstance(settings, BedrockSettings):
        raise ConfigurationError("the Bedrock factory needs BedrockSettings")
    return settings


class BedrockFactory:
    """Builds Bedrock clients and providers for the provider registry."""

    kind = "bedrock"
    settings_model = BedrockSettings

    def __init__(
        self,
        *,
        catalogue: Catalogue | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._catalogue = catalogue
        self._clock = clock

    def create_client(self, settings: ProviderSettings, secrets: Mapping[str, SecretStr]) -> Any:
        """A ``bedrock-runtime`` client for ``settings``. Imports the SDK here only."""
        bedrock = _bedrock(settings)
        # The settings validator refuses long-term keys too, but settings can be made
        # without validation (model_copy, model_construct); refuse them here as well,
        # before anything is imported or built.
        if (
            isinstance(bedrock.auth, LongTermApiKeyAuth)
            and bedrock.allow_long_term_api_keys is not True
        ):
            raise ConfigurationError(
                "long-term Bedrock API keys are refused unless allow_long_term_api_keys is set"
            )
        boto3 = importlib.import_module("boto3")
        botocore = importlib.import_module("botocore")
        botocore_auth = importlib.import_module("botocore.auth")
        botocore_config = importlib.import_module("botocore.config")
        botocore_exceptions = importlib.import_module("botocore.exceptions")
        awsrequest = importlib.import_module("botocore.awsrequest")
        auth = bedrock.auth
        region = bedrock.region
        clock = self._clock

        def secret(path: str) -> str:
            try:
                return secrets[path].get_secret_value()
            except KeyError:
                raise ConfigurationError(f"secret {path} was not resolved") from None

        def session_for(source: CredentialChainAuth | AccessKeyAuth, prefix: str) -> Any:
            if isinstance(source, AccessKeyAuth):
                token = secret(f"{prefix}session_token") if source.session_token else None
                return boto3.session.Session(
                    aws_access_key_id=secret(f"{prefix}access_key_id"),
                    aws_secret_access_key=secret(f"{prefix}secret_access_key"),
                    aws_session_token=token,
                    region_name=region,
                )
            return boto3.session.Session(profile_name=source.profile, region_name=region)

        key_source: KeySource | None = None
        if isinstance(auth, ShortTermApiKeyAuth):
            signing_session = session_for(auth.source, "auth.source.")

            def issue(lifetime_s: int) -> IssuedKey:
                credentials = signing_session.get_credentials()
                if credentials is None:
                    raise botocore_exceptions.NoCredentialsError()
                frozen = credentials.get_frozen_credentials()
                request = awsrequest.AWSRequest(
                    method="POST",
                    url=KEY_URL,
                    headers={"host": KEY_HOST},
                    params={"Action": KEY_ACTION},
                )
                signer = botocore_auth.SigV4QueryAuth(
                    frozen, KEY_SIGNING_NAME, region, expires=lifetime_s
                )
                signer.add_auth(request)
                expires_at = clock() + lifetime_s
                credentials_expiry = getattr(credentials, "_expiry_time", None)
                if credentials_expiry is not None:
                    expires_at = min(expires_at, credentials_expiry.timestamp())
                return IssuedKey(value=encode_short_term_key(request.url), expires_at=expires_at)

            key_source = ShortTermKeySource(
                issue,
                lifetime_s=auth.lifetime_s,
                refresh_before_s=auth.refresh_before_s,
                clock=clock,
            )
            refused: str | None = None
            try:
                key_source.current()  # refuse missing credentials at start-up
            except Exception as error:
                refused = type(error).__name__
            if refused is not None:
                raise ConfigurationError(
                    f"a short-term Bedrock API key cannot be generated: {refused}"
                )
            session = boto3.session.Session(region_name=region)
        elif isinstance(auth, LongTermApiKeyAuth):
            key_source = StaticKeySource(secret("auth.api_key"))
            session = boto3.session.Session(region_name=region)
        else:
            session = session_for(auth, "auth.")

        options: dict[str, Any] = {
            "region_name": region,
            "connect_timeout": bedrock.connect_timeout_s,
            "read_timeout": bedrock.read_timeout_s,
            "retries": {"max_attempts": bedrock.max_attempts, "mode": "standard"},
            "user_agent_extra": USER_AGENT,
        }
        if key_source is not None:
            options["signature_version"] = botocore.UNSIGNED
        client = session.client(
            SERVICE,
            region_name=region,
            endpoint_url=bedrock.endpoint_url,
            config=botocore_config.Config(**options),
        )
        if key_source is not None:
            bearer_source = key_source

            def add_bearer(request: Any, **_: Any) -> None:
                request.headers["Authorization"] = f"Bearer {bearer_source.current()}"

            client.meta.events.register(f"before-send.{SERVICE}", add_bearer)
        return client

    def create_provider(self, name: str, settings: ProviderSettings, client: Any) -> Provider:
        catalogue = self._catalogue if self._catalogue is not None else default_catalogue()
        return BedrockProvider(name, _bedrock(settings), client, catalogue=catalogue)
