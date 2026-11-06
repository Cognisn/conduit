"""The Bedrock provider factory: the only place the Bedrock SDK is imported.

``create_client`` imports boto3 and botocore, builds a ``bedrock-runtime`` client for
the configured region and authentication mode, and returns it:

- ``credential_chain`` and ``access_keys``: a boto3 session (named profile, or the
  resolved access keys) signs every request with SigV4.
- ``short_term_api_key`` and ``long_term_api_key``: the client is unsigned, and a
  ``before-send`` handler sets ``Authorization: Bearer <key>`` on every attempt,
  retries included. Short-term keys come from ``conduit.providers.bedrock.auth``'s
  presign step (``short_term_keys``), signed with credentials resolved here and
  replaced before they expire; the first key is generated at once, so missing
  credentials are refused at start-up. In both modes botocore's DEBUG records are kept
  free of the key (``keep_keys_out_of_logs``; see that module for what it covers).

The process environment (``AWS_BEARER_TOKEN_BEDROCK``) is never read or written: two
providers in one process can use different keys.

Credentials are resolved, any short-term key presigned and the client built in one
guarded step (see ``conduit.providers``): a constructor that raises becomes a fixed
``ConfigurationError`` naming the adapter and the exception's class.

Providers read prompt-caching support from the catalogue the registry passes to
``create_provider``: the one it applied the product's overlays to. Called without one,
outside a registry, the factory uses conduit's default catalogue.
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
from conduit.core.registry import guarded_client
from conduit.core.settings import ProviderSettings
from conduit.providers.bedrock.auth import (
    KeySource,
    StaticKeySource,
    keep_keys_out_of_logs,
    short_term_keys,
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

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
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
        botocore_config = importlib.import_module("botocore.config")
        auth = bedrock.auth
        region = bedrock.region

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

        clock = self._clock

        def build() -> Any:
            # Credentials and keys are resolved here, in the guarded step's own frames.
            key_source: KeySource | None = None
            if isinstance(auth, ShortTermApiKeyAuth):
                signing_session = session_for(auth.source, "auth.source.")
                key_source = short_term_keys(
                    signing_session.get_credentials,
                    region=region,
                    lifetime_s=auth.lifetime_s,
                    refresh_before_s=auth.refresh_before_s,
                    clock=clock,
                )
                session = boto3.session.Session(region_name=region)
            elif isinstance(auth, LongTermApiKeyAuth):
                keep_keys_out_of_logs()
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

        return guarded_client(self.kind, build)

    def create_provider(
        self,
        name: str,
        settings: ProviderSettings,
        client: Any,
        *,
        catalogue: Catalogue | None = None,
    ) -> Provider:
        return BedrockProvider(
            name,
            _bedrock(settings),
            client,
            catalogue=catalogue if catalogue is not None else default_catalogue(),
        )
