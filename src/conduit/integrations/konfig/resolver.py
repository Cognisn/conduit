"""A ``SecretResolver`` over konfig's secrets store.

``secret://<name>`` resolves to konfig's secret ``<name>``, read through
``konfig.Secrets``, whichever backend konfig has selected: its AES-encrypted file,
the OS keyring, an AWS Secrets Manager secret per name, or the designated AWS
Secrets Manager bundle (``KONFIG_AWS_SECRETS_MANAGER``). konfig owns the AWS
access; conduit imports no AWS library here.

konfig's backends are synchronous and may wait on a file lock or the network, so
each read runs in a worker thread and never blocks the event loop.

Errors name the reference, never a value.
"""

from __future__ import annotations

import asyncio
import importlib
from typing import Any, Protocol

from pydantic import SecretStr

from conduit.core.errors import SecretUnavailable
from conduit.core.secrets import SecretRef

#: The value konfig writes into an empty AWS bundle store on first boot, for the
#: operator to replace. A reference that still holds it is refused.
KONFIG_PLACEHOLDER = "CHANGEME"


class KonfigSecretsSource(Protocol):
    """The part of ``konfig.Secrets`` the resolver uses."""

    def get(self, key: str) -> str | None: ...


class KonfigSecretResolver:
    """Resolves ``secret://<name>`` through konfig's secrets store."""

    def __init__(self, secrets: KonfigSecretsSource) -> None:
        self._secrets = secrets

    @classmethod
    def from_settings(cls, settings: Any, *, service_name: str = "konfig") -> KonfigSecretResolver:
        """A resolver over ``konfig.Secrets(service_name, settings=settings)``.

        konfig chooses the backend from the ``KONFIG_AWS_SECRETS_MANAGER``
        environment variable or the settings' ``secrets.*`` keys, as it documents.
        """
        konfig = importlib.import_module("konfig")
        return cls(konfig.Secrets(service_name, settings=settings))

    async def resolve(self, ref: SecretRef) -> SecretStr:
        failure: str | None = None
        value: object = None
        try:
            value = await asyncio.to_thread(self._secrets.get, ref.name)
        except Exception as exc:
            # Only the exception's type is kept: its message is the store's, not ours.
            failure = type(exc).__name__
        if failure is not None:
            # Raised here, after the handler, so the store's exception is neither the
            # cause nor the context.
            raise SecretUnavailable(f"{ref} could not be read from konfig ({failure})")
        if value is None or value == "":
            raise SecretUnavailable(f"{ref} is not held in konfig's secrets store")
        if not isinstance(value, str):
            raise SecretUnavailable(f"{ref} does not hold text in konfig's secrets store")
        if value == KONFIG_PLACEHOLDER:
            raise SecretUnavailable(f"{ref} still holds konfig's unpopulated placeholder")
        return SecretStr(value)

    def __repr__(self) -> str:
        backend = getattr(self._secrets, "backend", self._secrets)
        return f"KonfigSecretResolver({type(backend).__name__})"
