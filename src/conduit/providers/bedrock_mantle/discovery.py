"""Model discovery for bedrock-mantle: the endpoint's OpenAI-compatible model listing.

The Amazon Bedrock User Guide documents a model listing on the ``bedrock-mantle``
endpoint: the OpenAI-compatible Models API, ``GET /v1/models`` on
``https://bedrock-mantle.{region}.api.aws``, with a Bedrock API key as the bearer token
(https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html, "List models
on bedrock-mantle"). It documents no listing on the Anthropic Messages path
(``/anthropic/v1/...``) this adapter calls
(https://docs.aws.amazon.com/bedrock/latest/userguide/inference-messages-api.html).

So discovery asks the documented listing, on the configured Region's mantle host,
through the registry-built mantle client (which sets conduit's bearer key on every
request). The listing covers every model the endpoint serves; only Claude models
(``anthropic.`` ids), which the Messages route serves, are kept. Each id is kept as
listed, with its ``object`` (``model``) as the label, as the OpenAI-compatible adapter
reads the same shape.

With an ``endpoint_url`` (an interface VPC endpoint, say) the listing's address is not
documented, and no key is sent anywhere it might not belong: discovery is reported
unsupported. The SDK's errors become ``ProviderError`` through the anthropic adapter's
``failure_for``, raised after the handler, so nothing is chained; an answer that is not
the documented shape is ``PROTOCOL``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from conduit.core.discovery import DiscoveredModel
from conduit.core.errors import ProviderError, ProviderFailure
from conduit.providers.anthropic.errors import failure_for
from conduit.providers.openai_compat.discovery import models_from

#: The documented OpenAI-compatible model listing on the mantle endpoint.
MODELS_URL = "https://bedrock-mantle.{region}.api.aws/v1/models"
#: The id prefix of the Claude models the Messages route serves.
CLAUDE_PREFIX = "anthropic."

#: Why discovery is unsupported with a replaced endpoint.
REPLACED_ENDPOINT = (
    "endpoint_url replaces the bedrock-mantle endpoint, and the address of its model "
    "listing there is not documented, so the models cannot be listed"
)


def models_url(region: str) -> str:
    """The documented model listing of the mantle endpoint in ``region``."""
    return MODELS_URL.format(region=region)


async def list_models(client: Any, region: str) -> Sequence[DiscoveredModel]:
    """The Claude models the mantle endpoint in ``region`` lists, through ``client``."""
    failure: ProviderFailure | None = None
    body: Any = None
    try:
        body = await client.get(models_url(region), cast_to=object)
    except ProviderError as error:
        failure = error.failure  # no key could be had (set_bearer)
    except Exception as error:
        failure = failure_for(error)
    if failure is not None:
        raise ProviderError(failure)  # outside the handler: nothing chained
    data = body.get("data") if isinstance(body, dict) else None
    return [m for m in models_from(data) if m.model_id.startswith(CLAUDE_PREFIX)]
