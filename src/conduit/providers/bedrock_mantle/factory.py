"""The bedrock-mantle factory: the only place this adapter imports an SDK.

``create_client`` builds the anthropic SDK's mantle client,
``anthropic.lib.bedrock.AsyncAnthropicBedrockMantle``, for the configured Region:

- with ``skip_auth=True``, so the SDK neither signs requests nor reads a key from the
  environment (``AWS_BEARER_TOKEN_BEDROCK``, ``ANTHROPIC_AWS_API_KEY``), and holds no
  key of its own;
- with the Region and base URL given explicitly, so ``AWS_REGION``,
  ``AWS_DEFAULT_REGION`` and ``ANTHROPIC_BEDROCK_MANTLE_BASE_URL`` are never read;
- with the bearer middleware (``conduit.providers.bedrock_mantle.auth``), which sets
  ``Authorization: Bearer <key>`` on every attempt, retries included;
- with the SDK's OpenTelemetry spans off, since they can carry request content outside
  conduit's capture policy.

The mantle client does not read ``ANTHROPIC_CUSTOM_HEADERS``. ``ANTHROPIC_LOG=debug``
makes the SDK log whole request bodies on the ``anthropic`` logger, so a client is
refused while it is set. The process environment is never written.

Short-term keys are presigned here with botocore's SigV4 query signer, exactly as the
bedrock adapter makes them (a ``CallWithBearerToken`` URL on ``bedrock.amazonaws.com``,
encoded by ``conduit.providers.bedrock.auth.encode_short_term_key``), and refreshed by
``ShortTermKeySource`` before they expire. The first key is generated at once, so
missing credentials are refused at start-up. Long-term keys are refused here unless
``allow_long_term_api_keys`` is exactly ``True``, however the settings were made.

Providers read capabilities from the catalogue the registry passes to
``create_provider``; called without one, outside a registry, the factory uses conduit's
default catalogue.
"""

from __future__ import annotations

import contextlib
import importlib
import os
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
from conduit.providers.bedrock.settings import (
    AccessKeyAuth,
    CredentialChainAuth,
    LongTermApiKeyAuth,
    ShortTermApiKeyAuth,
)
from conduit.providers.bedrock_mantle.auth import bearer_middleware
from conduit.providers.bedrock_mantle.provider import KIND, BedrockMantleProvider
from conduit.providers.bedrock_mantle.settings import BedrockMantleSettings, mantle_base_url

#: The module holding the SDK's mantle clients (not re-exported at the package top).
SDK_MODULE = "anthropic.lib.bedrock"
EXTRA = "pip install 'cognisn-conduit[bedrock-mantle]'"


def _mantle(settings: ProviderSettings) -> BedrockMantleSettings:
    if not isinstance(settings, BedrockMantleSettings):
        raise ConfigurationError("the bedrock-mantle factory needs BedrockMantleSettings")
    return settings


def _check_environment() -> None:
    """Refuse variables through which the SDK would act outside conduit's settings."""
    if os.environ.get("ANTHROPIC_LOG") == "debug":
        raise ConfigurationError(
            "ANTHROPIC_LOG=debug makes the anthropic SDK log request content outside "
            "conduit's content-capture policy. Unset it or use info"
        )


def _import(name: str) -> Any:
    """The module ``name``, or ``ConfigurationError`` naming the extra (nothing chained)."""
    module: Any = None
    with contextlib.suppress(ImportError):
        module = importlib.import_module(name)
    if module is None:
        raise ConfigurationError(f"the bedrock-mantle provider needs its extra: {EXTRA}")
    return module


class BedrockMantleFactory:
    """Builds bedrock-mantle clients and providers for the provider registry."""

    kind = KIND
    settings_model = BedrockMantleSettings

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock

    def create_client(self, settings: ProviderSettings, secrets: Mapping[str, SecretStr]) -> Any:
        """The SDK's async mantle client for ``settings``. Imports the SDK here only."""
        mantle = _mantle(settings)
        # The settings validator refuses long-term keys too, but settings can be made
        # without validation (model_copy, model_construct); refuse them here as well,
        # before anything is imported or built.
        if (
            isinstance(mantle.auth, LongTermApiKeyAuth)
            and mantle.allow_long_term_api_keys is not True
        ):
            raise ConfigurationError(
                "long-term Bedrock API keys are refused unless allow_long_term_api_keys is set"
            )
        _check_environment()
        sdk = _import(SDK_MODULE)
        auth = mantle.auth
        region = mantle.region
        clock = self._clock

        def secret(path: str) -> str:
            value = secrets.get(path)
            if value is None:
                raise ConfigurationError(f"secret {path} was not resolved")
            return value.get_secret_value()

        keys: KeySource
        if isinstance(auth, ShortTermApiKeyAuth):
            botocore_session = _import("botocore.session")
            botocore_auth = _import("botocore.auth")
            awsrequest = _import("botocore.awsrequest")
            source = auth.source

            def signing_credentials() -> Any:
                session = botocore_session.Session(
                    profile=source.profile if isinstance(source, CredentialChainAuth) else None
                )
                if isinstance(source, AccessKeyAuth):
                    token = secret("auth.source.session_token") if source.session_token else None
                    session.set_credentials(
                        secret("auth.source.access_key_id"),
                        secret("auth.source.secret_access_key"),
                        token,
                    )
                credentials = session.get_credentials()
                if credentials is None:
                    raise ConfigurationError("no AWS credentials were found")
                return credentials

            credentials: Any = None

            def issue(lifetime_s: int) -> IssuedKey:
                nonlocal credentials
                if credentials is None:
                    credentials = signing_credentials()
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

            keys = ShortTermKeySource(
                issue,
                lifetime_s=auth.lifetime_s,
                refresh_before_s=auth.refresh_before_s,
                clock=clock,
            )
            refused: str | None = None
            try:
                keys.current()  # refuse missing credentials at start-up
            except Exception as error:
                refused = type(error).__name__
            if refused is not None:
                raise ConfigurationError(
                    f"a short-term Bedrock API key cannot be generated: {refused}"
                )
        else:
            keys = StaticKeySource(secret("auth.api_key"))

        return sdk.AsyncAnthropicBedrockMantle(
            skip_auth=True,
            aws_region=region,
            base_url=mantle.endpoint_url or mantle_base_url(region),
            max_retries=mantle.max_retries,
            open_telemetry=False,
            middleware=[bearer_middleware(keys)],
        )

    def create_provider(
        self,
        name: str,
        settings: ProviderSettings,
        client: Any,
        *,
        catalogue: Catalogue | None = None,
    ) -> Provider:
        _mantle(settings)
        return BedrockMantleProvider(
            client,
            name=name,
            catalogue=catalogue if catalogue is not None else default_catalogue(),
        )
