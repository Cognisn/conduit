"""The bearer middleware: the current Bedrock API key on every attempt the SDK makes.

The anthropic SDK runs client middleware once per HTTP attempt, inside its retry loop,
so a key set here is set again on every retry. The key comes from a ``KeySource``
(``conduit.providers.bedrock.auth``): a short-term key that is replaced before it
expires, or a long-term key. Replacing a short-term key may refresh AWS credentials,
which can block, so the source is read in a worker thread, never on the event loop.

Any other credential header on the request is dropped, so only conduit's key is sent.
A key that cannot be had is ``ProviderError(REJECTED)``, raised after the handler so
nothing is chained to it: the underlying error can name a profile or carry
credential text. Cancellation is never caught.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from conduit.core.errors import ProviderError, ProviderFailure
from conduit.providers.bedrock.auth import KeySource

#: Header names that carry a credential, compared in lower case.
CREDENTIAL_HEADERS = frozenset({"authorization", "x-api-key"})

CallNext = Callable[[Any], Awaitable[Any]]
Middleware = Callable[[Any, CallNext], Awaitable[Any]]


async def current_key(keys: KeySource) -> str:
    """The key to send now, read off the event loop; ``REJECTED`` if there is none."""
    key: str | None = None
    try:
        key = await asyncio.to_thread(keys.current)
    except Exception:
        key = None  # refused below, after the handler
    if not key:
        raise ProviderError(ProviderFailure.REJECTED)
    return key


def bearer_middleware(keys: KeySource) -> Middleware:
    """SDK middleware that sends ``Authorization: Bearer <key>`` on every attempt."""

    async def bearer(request: Any, call_next: CallNext) -> Any:
        key = await current_key(keys)
        headers = {
            name: value
            for name, value in dict(request.headers).items()
            if name.lower() not in CREDENTIAL_HEADERS
        }
        headers["Authorization"] = f"Bearer {key}"
        return await call_next(request.copy(headers=headers))

    return bearer
