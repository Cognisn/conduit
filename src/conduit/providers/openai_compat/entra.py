"""Microsoft Entra ID for Microsoft Foundry, shared by the adapters that call Foundry.

``entra_id_token_provider`` builds azure-identity's ``DefaultAzureCredential`` (with
the settings' user-assigned managed identity, if any) and a bearer token provider for
the settings' scope. It is the only place azure-identity is imported, through
``importlib``, so importing conduit never needs it. The OpenAI-compatible adapter (the
``openai`` extra) and the Anthropic adapter's Foundry endpoint (the ``anthropic``
extra) both use it, with the same settings shape, ``FoundryEntraId``.

A credential holds an HTTP session, so it is closed whenever the client it was made
for cannot be built: ``close_soon`` closes it on the running loop if there is one,
otherwise straight away. A failing close is logged by class name only, never raised.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from conduit.providers.openai_compat.settings import FoundryEntraId

logger = logging.getLogger("conduit.providers.openai_compat")

TokenProvider = Callable[[], Awaitable[str]]

#: Credential closes scheduled on a running loop, kept until they finish.
_CLOSING: set[asyncio.Task[None]] = set()


async def close_quietly(credential: Any) -> None:
    """Close ``credential``; a failure is logged by its class name and not raised."""
    try:
        await credential.close()
    except Exception as error:
        logger.warning("closing an Entra ID credential failed: %s", type(error).__name__)


def close_soon(credential: Any) -> None:
    """Close ``credential`` from synchronous code, on the running loop if there is one."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(close_quietly(credential))
        return
    task = loop.create_task(close_quietly(credential))
    _CLOSING.add(task)
    task.add_done_callback(_CLOSING.discard)


def entra_id_token_provider(auth: FoundryEntraId) -> tuple[Any, TokenProvider]:
    """The credential and an async bearer token provider for ``auth``'s scope.

    If the token provider cannot be made, the credential is closed before the error
    goes on.
    """
    identity = importlib.import_module("azure.identity.aio")
    options: dict[str, Any] = {}
    if auth.managed_identity_client_id is not None:
        options["managed_identity_client_id"] = auth.managed_identity_client_id
    credential = identity.DefaultAzureCredential(**options)
    try:
        token_provider: TokenProvider = identity.get_bearer_token_provider(credential, auth.scope)
    except Exception:
        close_soon(credential)
        raise
    return credential, token_provider
