"""The bedrock-mantle factory: the only place this adapter imports an SDK.

``create_client`` builds the anthropic SDK's mantle client,
``anthropic.lib.bedrock.AsyncAnthropicBedrockMantle``, for the configured Region:

- with ``skip_auth=True``, so the SDK neither signs requests nor reads a key from the
  environment (``AWS_BEARER_TOKEN_BEDROCK``, ``ANTHROPIC_AWS_API_KEY``), and holds no
  key of its own;
- with the Region and base URL given explicitly, so ``AWS_REGION``,
  ``AWS_DEFAULT_REGION`` and ``ANTHROPIC_BEDROCK_MANTLE_BASE_URL`` are never read;
- as a subclass (``conduit.providers.bedrock_mantle.auth.bearer_client_class``) that
  sets ``Authorization: Bearer <key>`` at send time on every attempt, retries included,
  so the key is never in the request options the SDK logs at DEBUG;
- with the SDK's OpenTelemetry spans off, since they can carry request content outside
  conduit's capture policy.

The mantle client does not read ``ANTHROPIC_CUSTOM_HEADERS``. ``ANTHROPIC_LOG=debug``
makes the SDK log whole request bodies on the ``anthropic`` logger, so a client is
refused while it is set. The process environment is never written.

Short-term keys come from the bedrock adapter's presign step,
``conduit.providers.bedrock.auth.short_term_keys``, signed with credentials resolved
here, and are replaced before they expire. The first key is generated at once, so
missing credentials are refused at start-up. The step keeps botocore's signature records
out of the logs, since a presigned key can be rebuilt from them. Long-term keys are refused here
unless ``allow_long_term_api_keys`` is exactly ``True``, and the Region and any
``endpoint_url`` are checked again, however the settings were made.

Providers read capabilities from the catalogue the registry passes to
``create_provider``; called without one, outside a registry, the factory uses conduit's
default catalogue.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import re
import time
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import SecretStr

from conduit.core.catalogue import Catalogue, default_catalogue
from conduit.core.errors import ConfigurationError
from conduit.core.provider import Provider
from conduit.core.settings import ProviderSettings
from conduit.providers.bedrock.auth import (
    KeySource,
    StaticKeySource,
    short_term_keys,
)
from conduit.providers.bedrock.settings import (
    REGION_PATTERN,
    AccessKeyAuth,
    CredentialChainAuth,
    LongTermApiKeyAuth,
    ShortTermApiKeyAuth,
)
from conduit.providers.bedrock_mantle.auth import bearer_client_class
from conduit.providers.bedrock_mantle.provider import KIND, BedrockMantleProvider
from conduit.providers.bedrock_mantle.settings import BedrockMantleSettings, mantle_base_url

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


def _missing_extra() -> ConfigurationError:
    return ConfigurationError(f"the bedrock-mantle provider needs its extra: {EXTRA}")


def _check_destination(mantle: BedrockMantleSettings) -> None:
    """Refuse a region or endpoint the settings validators would have refused.

    Settings can be made without validation (``model_construct``); the key must never
    go to a host they did not allow. The values are never echoed.
    """
    region = mantle.region
    if not isinstance(region, str) or re.fullmatch(REGION_PATTERN, region) is None:
        raise ConfigurationError("the bedrock-mantle region is not an AWS region")
    endpoint = mantle.endpoint_url
    if endpoint is not None and (
        not isinstance(endpoint, str) or not endpoint.startswith("https://")
    ):
        raise ConfigurationError("the bedrock-mantle endpoint_url must be an https URL")


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
        _check_destination(mantle)
        _check_environment()
        # Literal module names, so the client-construction tripwire can follow them.
        sdk: Any = None
        with contextlib.suppress(ImportError):
            sdk = importlib.import_module("anthropic.lib.bedrock")
        if sdk is None:  # raised after the handler, with nothing chained
            raise _missing_extra()
        auth = mantle.auth
        region = mantle.region

        def secret(path: str) -> str:
            value = secrets.get(path)
            if value is None:
                raise ConfigurationError(f"secret {path} was not resolved")
            return value.get_secret_value()

        keys: KeySource
        if isinstance(auth, ShortTermApiKeyAuth):
            botocore_session: Any = None
            with contextlib.suppress(ImportError):
                botocore_session = importlib.import_module("botocore.session")
            if botocore_session is None:
                raise _missing_extra()
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
                return session.get_credentials()

            keys = short_term_keys(
                signing_credentials,
                region=region,
                lifetime_s=auth.lifetime_s,
                refresh_before_s=auth.refresh_before_s,
                clock=self._clock,
            )
        else:
            keys = StaticKeySource(secret("auth.api_key"))

        client_class = bearer_client_class(sdk.AsyncAnthropicBedrockMantle, keys)
        return client_class(
            skip_auth=True,
            aws_region=region,
            base_url=mantle.endpoint_url or mantle_base_url(region),
            max_retries=mantle.max_retries,
            open_telemetry=False,
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
