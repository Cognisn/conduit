"""Model discovery for the OpenAI-compatible adapter: the Models API of OpenAI and X.AI.

OpenAI's API reference (https://developers.openai.com/api/reference/resources/models)
documents ``GET /v1/models``, one list of model objects (``id``, ``object``, ``created``,
``owned_by``) with no pagination; X.AI's REST reference
(https://docs.x.ai/developers/rest-api-reference/inference/models) documents the same
``GET /v1/models`` shape. Each model's ``id`` is the id and its ``object`` (``model``) the
label; nothing else is kept. The same shape is read from bedrock-mantle's
OpenAI-compatible listing (``conduit.providers.bedrock_mantle.discovery``).

Microsoft Foundry is not listed. Its model ids are deployment names; the resource's
OpenAI v1 ``GET /openai/v1/models`` lists the models available to the resource, not its
deployments, which only the Azure management API lists
(https://learn.microsoft.com/azure/foundry/openai/how-to/working-with-models).

The SDK's errors become ``ProviderError`` through ``classify``, raised after the
handler, so nothing is chained; an entry without an id is ``PROTOCOL``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from typing import Any

from conduit.core.discovery import DiscoveredModel
from conduit.core.errors import ProviderError, ProviderFailure
from conduit.providers.openai_compat.errors import classify

#: Why a Foundry endpoint cannot be listed.
FOUNDRY_UNSUPPORTED = (
    "Microsoft Foundry model ids are deployment names, and the OpenAI v1 endpoint lists "
    "the resource's models, not its deployments (only the Azure management API does)"
)


def _field(item: object, name: str) -> object:
    return item.get(name) if isinstance(item, dict) else getattr(item, name, None)


def models_from(items: object) -> list[DiscoveredModel]:
    """The models in a Models API ``data`` list (mappings or SDK objects)."""
    if not isinstance(items, list):
        raise ProviderError(ProviderFailure.PROTOCOL)
    found: list[DiscoveredModel] = []
    for item in items:
        model_id, kind = _field(item, "id"), _field(item, "object")
        listed: DiscoveredModel | None = None
        if isinstance(model_id, str) and isinstance(kind, str | None):
            with contextlib.suppress(ValueError):
                listed = DiscoveredModel(model_id, label=kind or None)
        if listed is None:
            raise ProviderError(ProviderFailure.PROTOCOL)
        found.append(listed)
    return found


async def list_models(sdk: Any) -> Sequence[DiscoveredModel]:
    """Every model the key can use, through the registry-built SDK client ``sdk``."""
    failure: ProviderFailure | None = None
    page: Any = None
    try:
        page = await sdk.models.list()
    except Exception as error:
        failure = classify(error)
    if failure is not None:
        raise ProviderError(failure)  # outside the handler: nothing chained
    return models_from(getattr(page, "data", None))
