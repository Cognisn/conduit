"""Model discovery for Ollama: the server's local models, from ``GET /api/tags``.

Ollama's API reference (https://docs.ollama.com/api/tags) documents ``GET /api/tags`` as
the list of locally available models: one JSON body, ``{"models": [...]}``, each with
``name`` and ``model`` (both the model name), and no pagination. Each listed ``name``
becomes a ``DiscoveredModel`` with no label (the listing gives no kind of entry); sizes,
digests and details are not kept.

Model names are ``model:tag``, and "the tag is optional and, if not provided, will
default to ``latest``" (Ollama's API documentation, "Model names",
https://github.com/ollama/ollama/blob/main/docs/api.md#model-names). A model pulled
without a tag is listed as ``name:latest``, while the catalogue declares it, and the
provider calls it, by its bare name; so a listed ``:latest`` tag is reported under the
bare name, and every other tag is kept as listed.

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
from conduit.core.residency import ResidencyScope
from conduit.providers.ollama.mapping import failure_for_status

#: Ollama's listing of local models.
TAGS_PATH = "/api/tags"
#: How long the listing may take, connecting and reading.
DISCOVERY_TIMEOUT_S = 30.0
#: The tag Ollama gives a name pulled or called without one.
DEFAULT_TAG = ":latest"


def bare_name(name: str) -> str:
    """``name`` as conduit addresses it: without the default ``:latest`` tag."""
    if name.endswith(DEFAULT_TAG) and len(name) > len(DEFAULT_TAG):
        return name[: -len(DEFAULT_TAG)]
    return name


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
        # A remote model (either remote field set) is processed on its upstream host,
        # not the deployment's own.
        remote = item.get("remote_host") or item.get("remote_model")
        scope = ResidencyScope.GLOBAL if remote else ResidencyScope.LOCAL
        found.append(DiscoveredModel(bare_name(name), scope=scope))
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
