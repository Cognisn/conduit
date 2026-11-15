"""Model discovery for the Anthropic adapter: the Models API on the direct endpoint.

Anthropic's API reference (https://platform.claude.com/docs/en/api/models/list)
documents ``GET /v1/models``: pages of at most 1000 (``limit``), each with ``has_more``
and ``last_id``, the next page asked for with ``after_id``. It lists active and deprecated
models by default. Each model's ``id`` is the id and its ``type`` (``model``) the label;
nothing else is kept.

Claude in Microsoft Foundry is not listed: the Foundry endpoint documents the Messages
and token-counting APIs only
(https://learn.microsoft.com/azure/foundry/foundry-models/concepts/claude-models#api-overview),
and its model ids are deployment names, which only the Azure management API lists.

The SDK's errors become ``ProviderError`` through ``failure_for``, raised after the
handler, so nothing is chained; an answer that is not the documented shape, or that
pages more than ``MAX_PAGES`` times, is ``PROTOCOL``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from typing import Any

from conduit.core.discovery import DiscoveredModel
from conduit.core.errors import ProviderError, ProviderFailure
from conduit.providers.anthropic.errors import failure_for

#: Models asked for per page (the API's maximum).
PAGE_SIZE = 1000
#: The most pages read before the listing is refused as never ending.
MAX_PAGES = 100

#: Why Claude in Foundry cannot be listed.
FOUNDRY_UNSUPPORTED = (
    "Claude in Microsoft Foundry documents no models listing on its Anthropic endpoint "
    "(Messages and token counting only), and its model ids are deployment names"
)


def _models(page: Any) -> tuple[list[DiscoveredModel], bool, str | None]:
    data = getattr(page, "data", None)
    more = getattr(page, "has_more", None)
    last = getattr(page, "last_id", None)
    if not isinstance(data, list) or not isinstance(more, bool | None):
        raise ProviderError(ProviderFailure.PROTOCOL)
    found: list[DiscoveredModel] = []
    for model in data:
        model_id, kind = getattr(model, "id", None), getattr(model, "type", None)
        listed: DiscoveredModel | None = None
        if isinstance(model_id, str) and isinstance(kind, str | None):
            with contextlib.suppress(ValueError):
                listed = DiscoveredModel(model_id, label=kind or None)
        if listed is None:
            raise ProviderError(ProviderFailure.PROTOCOL)
        found.append(listed)
    if more and (not isinstance(last, str) or not last):
        raise ProviderError(ProviderFailure.PROTOCOL)
    return found, bool(more), last if isinstance(last, str) else None


async def _page(client: Any, after: str | None) -> Any:
    failure: ProviderFailure | None = None
    page: Any = None
    params: dict[str, Any] = {"limit": PAGE_SIZE}
    if after is not None:
        params["after_id"] = after
    try:
        page = await client.models.list(**params)
    except Exception as error:
        failure = failure_for(error)
    if failure is not None:
        raise ProviderError(failure)  # outside the handler: nothing chained
    return page


async def list_models(client: Any) -> Sequence[DiscoveredModel]:
    """Every model the key can use, through the registry-built SDK ``client``."""
    found: list[DiscoveredModel] = []
    after: str | None = None
    for _ in range(MAX_PAGES):
        models, more, after = _models(await _page(client, after))
        found.extend(models)
        if not more:
            return found
    raise ProviderError(ProviderFailure.PROTOCOL)
