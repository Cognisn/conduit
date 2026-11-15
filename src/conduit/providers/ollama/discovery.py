"""Model discovery for Ollama: the server's local models, from ``GET /api/tags``.

Ollama's API reference (https://docs.ollama.com/api/tags) documents ``GET /api/tags`` as
the list of locally available models: one JSON body, ``{"models": [...]}``, each with
``name`` and ``model`` (both the model name), and no pagination. Each listed ``name``
becomes a ``DiscoveredModel`` with no label (the listing gives no kind of entry); sizes,
digests and details are not kept.

The request goes through the registry-built httpx client, with a fixed timeout. A
status other than 200 is a failure (``failure_for_status``), an httpx error is classified
as the provider's are, and a body that is not the documented shape is ``PROTOCOL``. Every
failure is raised as ``ProviderError`` after its handler, so nothing is chained.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

from conduit.core.discovery import DiscoveredModel
from conduit.core.errors import ProviderError, ProviderFailure
from conduit.providers.ollama.mapping import failure_for_status

#: Ollama's listing of local models.
TAGS_PATH = "/api/tags"
#: How long the listing may take, connecting and reading.
DISCOVERY_TIMEOUT_S = 30.0


def _classify(error: Exception) -> ProviderFailure:
    httpx = importlib.import_module("httpx")
    if isinstance(error, httpx.TimeoutException):
        return ProviderFailure.TIMEOUT
    return ProviderFailure.UNAVAILABLE


def parse_tags(body: object) -> tuple[DiscoveredModel, ...]:
    """The models in an ``/api/tags`` body. Anything else is ``PROTOCOL``."""
    models = body.get("models") if isinstance(body, dict) else None
    if not isinstance(models, list):
        raise ProviderError(ProviderFailure.PROTOCOL)
    found: list[DiscoveredModel] = []
    for item in models:
        name = item.get("name") if isinstance(item, dict) else None
        if not isinstance(name, str) or not name:
            raise ProviderError(ProviderFailure.PROTOCOL)
        found.append(DiscoveredModel(name))
    return tuple(found)


async def list_local_models(client: Any) -> Sequence[DiscoveredModel]:
    """The server's local models, through the registry-built ``client``."""
    failure: ProviderFailure | None = None
    response: Any = None
    try:
        response = await client.get(TAGS_PATH, timeout=DISCOVERY_TIMEOUT_S)
    except Exception as error:
        failure = _classify(error)
    if failure is None and response.status_code != 200:
        failure = failure_for_status(response.status_code)
    body: Any = None
    if failure is None:
        try:
            body = response.json()
        except Exception:
            failure = ProviderFailure.PROTOCOL
    if failure is not None:
        raise ProviderError(failure)  # outside every handler: nothing chained
    return parse_tags(body)
