"""Bearer keys on the mantle client: set at send time, and kept out of every log record.

The key goes on the HTTP request in the SDK's ``_prepare_request`` hook, which the SDK
calls on every attempt (retries included) after it has built, and logged, the request
options. So the key is never in the options the SDK writes to the ``anthropic`` logger at
DEBUG, which a product enables simply by setting its root logger to DEBUG. The hook is
installed by ``bearer_client_class``, a subclass of the SDK's mantle client; the SDK's
``copy`` and ``with_options`` keep the subclass.

The key comes from a ``KeySource`` (``conduit.providers.bedrock.auth``): a short-term key
that is replaced before it expires, or a long-term key. Replacing a short-term key may
refresh AWS credentials, which can block, so the source is read in a worker thread, never
on the event loop. Any other credential header on the request is dropped, so only
conduit's key is sent. A key that cannot be had is ``ProviderError(REJECTED)``, raised
after the handler so nothing is chained to it (the underlying error can name a profile or
carry credential text); the SDK passes it through unchanged and does not retry it.
Cancellation is never caught.

Short-term keys are presigned by ``conduit.providers.bedrock.auth.short_term_keys``, the
step the bedrock adapter uses too, which keeps botocore's signature records out of the
logs.
"""

from __future__ import annotations

import asyncio
from typing import Any

from conduit.core.errors import ProviderError, ProviderFailure
from conduit.providers.bedrock.auth import KeySource

#: Header names that carry a credential, compared in lower case.
CREDENTIAL_HEADERS = frozenset({"authorization", "x-api-key"})


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


async def set_bearer(request: Any, keys: KeySource) -> None:
    """Put the current key on the HTTP ``request`` as its only credential header."""
    key = await current_key(keys)
    for name in [n for n in request.headers if n.lower() in CREDENTIAL_HEADERS]:
        del request.headers[name]
    request.headers["Authorization"] = f"Bearer {key}"


def bearer_client_class(base: type[Any], keys: KeySource) -> type[Any]:
    """A subclass of the SDK client ``base`` that sets the bearer key at send time."""

    async def _prepare_request(self: Any, request: Any) -> None:
        await base._prepare_request(self, request)  # a no-op with skip_auth
        await set_bearer(request, keys)

    return type("BearerMantleClient", (base,), {"_prepare_request": _prepare_request})
