"""The Bedrock provider factory: the only place the Bedrock SDK is imported.

``create_client`` imports boto3 and botocore and builds, for the configured region and
authentication mode, a ``bedrock-runtime`` client and, beside it, a ``bedrock``
control-plane client for model discovery (``conduit.providers.bedrock.discovery``). It
returns both as ``BedrockClients``, which behaves as the runtime client. With an
``endpoint_url`` (which replaces the runtime endpoint only) no control-plane client is
made, and discovery is reported unsupported. Both clients authenticate the same way:

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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import SecretStr

from conduit.core.catalogue import Catalogue, default_catalogue
from conduit.core.discovery import DiscoveredModel
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
from conduit.providers.bedrock.discovery import discover
from conduit.providers.bedrock.provider import BedrockProvider
from conduit.providers.bedrock.settings import (
    AccessKeyAuth,
    BedrockSettings,
    CredentialChainAuth,
    LongTermApiKeyAuth,
    ShortTermApiKeyAuth,
)

SERVICE = "bedrock-runtime"
#: The control-plane service, for model discovery.
CONTROL_SERVICE = "bedrock"
USER_AGENT = "conduit"

#: Why discovery is unsupported with a replaced runtime endpoint.
REPLACED_ENDPOINT = (
    "endpoint_url replaces the bedrock-runtime endpoint only; conduit makes no "
    "control-plane client beside it, so the models cannot be listed"
)


@dataclass(frozen=True, eq=False)
class BedrockClients:
    """The runtime client and the control-plane client of one configured provider.

    It behaves as the runtime client (every attribute not its own is the runtime
    client's), so code written against ``create_client``'s runtime client, an embedding
    adapter's included, works unchanged. ``control`` is ``None`` when ``endpoint_url``
    replaces the runtime endpoint.

    ``close`` is its own, so every close path (a provider's, an embedder's, discovery's)
    closes both clients.
    """

    runtime: Any
    control: Any

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes the bundle lacks. Its own fields and dunder names
        # are never forwarded: on a half-built bundle (copy, pickle) ``runtime`` is not
        # set yet, and forwarding it would recurse.
        if name in ("runtime", "control") or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.runtime, name)

    def close(self) -> None:
        """Close the runtime client and the control-plane client, each that exists.

        A failed close does not stop the other: both are attempted, then the first
        failure is raised, outside every handler, so nothing is chained.
        """
        failed: BaseException | None = None
        for client in (self.runtime, self.control):
            close = getattr(client, "close", None) if client is not None else None
            if close is None:
                continue
            try:
                close()
            except Exception as error:
                failed = failed if failed is not None else error
        if failed is not None:
            raise failed


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
        """The ``bedrock-runtime`` and ``bedrock`` clients for ``settings``, as
        ``BedrockClients``. Imports the SDK here only."""
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
            control = None
            if bedrock.endpoint_url is None:
                control = session.client(
                    CONTROL_SERVICE,
                    region_name=region,
                    config=botocore_config.Config(**options),
                )
            if key_source is not None:
                bearer_source = key_source

                def add_bearer(request: Any, **_: Any) -> None:
                    request.headers["Authorization"] = f"Bearer {bearer_source.current()}"

                client.meta.events.register(f"before-send.{SERVICE}", add_bearer)
                if control is not None:
                    control.meta.events.register(f"before-send.{CONTROL_SERVICE}", add_bearer)
            return BedrockClients(runtime=client, control=control)

        return guarded_client(self.kind, build)

    def discovery_unsupported(self, settings: ProviderSettings) -> str | None:
        if _bedrock(settings).endpoint_url is not None:
            return REPLACED_ENDPOINT
        return None

    async def discover_models(
        self, settings: ProviderSettings, client: Any
    ) -> Sequence[DiscoveredModel]:
        """Foundation models and inference profiles, on the control-plane client."""
        _bedrock(settings)
        control = client.control if isinstance(client, BedrockClients) else None
        if control is None:
            raise ConfigurationError("Bedrock discovery needs the control-plane client")
        return await discover(control)

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
