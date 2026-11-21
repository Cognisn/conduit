"""Model discovery for Gemini: ``models.list``, chat and embedding models.

The Gemini API reference (https://aistudio.google.com/api-reference/models/list-models)
documents ``GET /v1beta/models``: pages of at most 1000 (``pageSize``), the next asked
for with ``pageToken`` while ``nextPageToken`` is given, each model named
``models/{model}`` with the generation methods it supports. Discovery keeps the models
that support ``generateContent`` (chat) and those that support ``embedContent`` without
it (embedding), so the chat and the embedding adapter (which inherits this listing) each
find theirs; other models are left out. The ``models/`` prefix is removed so the id is
the one a request names. No label is kept. Every model is ``global`` (the API has no
Region notion), with the token limits the listing gives. The API's ``baseModelId`` is not
on the SDK's ``Model``; a tuned model's base is its ``tuned_model_info.base_model``, and
any other model is its own base.

The listing runs through the registry-built ``google.genai.Client`` (its async client,
whose pager follows the page tokens), at most ``MAX_PAGES`` pages. The SDK's errors
become ``ProviderError`` through ``classify``, raised after the handler, so nothing is
chained; a model that is not the documented shape is ``PROTOCOL``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from typing import Any, Literal

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
#: The generation method of an embedding model.
EMBED_METHOD = "embedContent"
_PREFIX = "models/"


def _positive(value: object) -> int | None:
    """A reported limit, or ``None`` for anything that is not a positive integer."""
    return value if type(value) is int and value > 0 else None


def _base(model: object, model_id: str) -> str:
    """The base model a listed model names: a tuned model's base, else itself."""
    base = getattr(getattr(model, "tuned_model_info", None), "base_model", None)
    if isinstance(base, str) and base.removeprefix(_PREFIX):
        return base.removeprefix(_PREFIX)
    return model_id


def _models(page: object) -> list[DiscoveredModel]:
    if not isinstance(page, list):
        raise ProviderError(ProviderFailure.PROTOCOL)
    found: list[DiscoveredModel] = []
    for model in page:
        name = getattr(model, "name", None)
        actions = getattr(model, "supported_actions", None) or []
        listed: DiscoveredModel | None = None
        kind: Literal["chat", "embedding"] = "chat"
        if isinstance(actions, list) and CHAT_METHOD not in actions:
            kind = "embedding"
        if isinstance(name, str) and isinstance(actions, list):
            model_id = name.removeprefix(_PREFIX)
            with contextlib.suppress(ValueError):
                listed = DiscoveredModel(
                    model_id,
                    kind=kind,
                    scope=ResidencyScope.GLOBAL,
                    base_model_id=_base(model, model_id),
                    context_window=_positive(getattr(model, "input_token_limit", None)),
                    output_limit=_positive(getattr(model, "output_token_limit", None)),
                )
        if listed is None:
            raise ProviderError(ProviderFailure.PROTOCOL)
        if CHAT_METHOD in actions or EMBED_METHOD in actions:
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
    """Every chat and embedding model the key can use, through the registry-built
    ``client``."""
    pager = await _call(lambda: client.aio.models.list(config={"page_size": PAGE_SIZE}))
    found = _models(pager.page)
    for _ in range(MAX_PAGES - 1):
        if not pager.config.get("page_token"):
            return found
        found.extend(_models(await _call(pager.next_page)))
    raise ProviderError(ProviderFailure.PROTOCOL)
