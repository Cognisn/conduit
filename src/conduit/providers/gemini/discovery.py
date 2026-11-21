"""Model discovery for Gemini: ``models.list``, chat-capable models only.

The Gemini API reference (https://aistudio.google.com/api-reference/models/list-models)
documents ``GET /v1beta/models``: pages of at most 1000 (``pageSize``), the next asked
for with ``pageToken`` while ``nextPageToken`` is given, each model named
``models/{model}`` with the generation methods it supports. Its own example lists the
models that support ``generateContent``; discovery keeps exactly those (embedding-only
and other models are left out), with the ``models/`` prefix removed so the id is the one
a request names. No label is kept.

The listing runs through the registry-built ``google.genai.Client`` (its async client,
whose pager follows the page tokens), at most ``MAX_PAGES`` pages. The SDK's errors
become ``ProviderError`` through ``classify``, raised after the handler, so nothing is
chained; a model that is not the documented shape is ``PROTOCOL``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from typing import Any

from conduit.core.discovery import DiscoveredModel
from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.residency import ResidencyScope
from conduit.providers.gemini.errors import classify

#: Models asked for per page (the API's maximum).
PAGE_SIZE = 1000
#: The most pages read before the listing is refused as never ending.
MAX_PAGES = 100
#: The generation method of a chat-capable model.
CHAT_METHOD = "generateContent"
_PREFIX = "models/"


def _positive(value: object) -> int | None:
    """A reported limit, or ``None`` for anything that is not a positive integer."""
    return value if type(value) is int and value > 0 else None


def _chat_models(page: object) -> list[DiscoveredModel]:
    if not isinstance(page, list):
        raise ProviderError(ProviderFailure.PROTOCOL)
    found: list[DiscoveredModel] = []
    for model in page:
        name = getattr(model, "name", None)
        actions = getattr(model, "supported_actions", None) or []
        listed: DiscoveredModel | None = None
        if isinstance(name, str) and isinstance(actions, list):
            with contextlib.suppress(ValueError):
                listed = DiscoveredModel(
                    name.removeprefix(_PREFIX),
                    kind="chat",
                    scope=ResidencyScope.GLOBAL,
                    context_window=_positive(getattr(model, "input_token_limit", None)),
                    output_limit=_positive(getattr(model, "output_token_limit", None)),
                )
        if listed is None:
            raise ProviderError(ProviderFailure.PROTOCOL)
        if CHAT_METHOD in actions:
            found.append(listed)
    return found


async def _call(step: Any) -> Any:
    failure: ProviderFailure | None = None
    answer: Any = None
    try:
        answer = await step()
    except Exception as error:
        failure = classify(error)
    if failure is not None:
        raise ProviderError(failure)  # outside the handler: nothing chained
    return answer


async def list_models(client: Any) -> Sequence[DiscoveredModel]:
    """Every chat-capable model the key can use, through the registry-built ``client``."""
    pager = await _call(lambda: client.aio.models.list(config={"page_size": PAGE_SIZE}))
    found = _chat_models(pager.page)
    for _ in range(MAX_PAGES - 1):
        if not pager.config.get("page_token"):
            return found
        found.extend(_chat_models(await _call(pager.next_page)))
    raise ProviderError(ProviderFailure.PROTOCOL)
