"""Model discovery for Bedrock: foundation models and inference profiles in the Region.

Both listings are control-plane operations of the ``bedrock`` service, made on the
control-plane client ``BedrockFactory.create_client`` builds beside the runtime client,
for the same Region and authentication mode (SigV4, or a Bedrock API key as a bearer
token: Bedrock API keys cover Amazon Bedrock and Amazon Bedrock Runtime actions):

- ``ListFoundationModels``
  (https://docs.aws.amazon.com/bedrock/latest/APIReference/API_ListFoundationModels.html),
  filtered with ``byInferenceType=ON_DEMAND``, so a model that can be called only
  through an inference profile is not listed under its bare id. It does not page. A
  model is kept only if its ``outputModalities`` include ``TEXT`` or ``EMBEDDING`` (the
  kinds conduit calls); its ``modelId`` is the id, with no label.
- ``ListInferenceProfiles``
  (https://docs.aws.amazon.com/bedrock/latest/APIReference/API_ListInferenceProfiles.html),
  paged with ``nextToken`` (at most ``MAX_PAGES`` pages). Each profile's
  ``inferenceProfileId`` is the id, and its ``type`` (``SYSTEM_DEFINED`` or
  ``APPLICATION``) the label.

boto3 is synchronous, so the listing runs on its own daemon thread, never on the event
loop or in its default executor, bounded by the client's own connect and read timeouts.
Any SDK error becomes ``ProviderError`` (``conduit.providers.bedrock.errors.classify``)
and an answer that is not the documented shape ``PROTOCOL``, each raised after its
handler, so nothing is chained.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Callable, Sequence
from typing import Any

from conduit.core.discovery import DiscoveredModel
from conduit.core.errors import ProviderError, ProviderFailure
from conduit.providers.bedrock.errors import classify

#: The output modalities of a model conduit can call (chat and embeddings).
CALLABLE_OUTPUTS = frozenset({"TEXT", "EMBEDDING"})
#: Inference profiles asked for per page (the API's maximum).
PROFILE_PAGE_SIZE = 1000
#: The most profile pages read before the listing is refused as never ending.
MAX_PAGES = 100


def _protocol() -> ProviderError:
    return ProviderError(ProviderFailure.PROTOCOL)


def _model(model_id: object, label: object = None) -> DiscoveredModel:
    """A listed model; an id or label outside the documented shape is ``PROTOCOL``."""
    if not isinstance(model_id, str) or not isinstance(label, str | None):
        raise _protocol()
    found: DiscoveredModel | None = None
    with contextlib.suppress(ValueError):
        found = DiscoveredModel(model_id, label=label or None)
    if found is None:
        raise _protocol()
    return found


def _foundation_models(answer: object) -> list[DiscoveredModel]:
    summaries = answer.get("modelSummaries") if isinstance(answer, dict) else None
    if not isinstance(summaries, list):
        raise _protocol()
    found: list[DiscoveredModel] = []
    for summary in summaries:
        model_id = summary.get("modelId") if isinstance(summary, dict) else None
        outputs = summary.get("outputModalities", []) if isinstance(summary, dict) else None
        if not isinstance(outputs, list):
            raise _protocol()
        model = _model(model_id)
        if CALLABLE_OUTPUTS & {o for o in outputs if isinstance(o, str)}:
            found.append(model)
    return found


def _profiles(answer: object) -> tuple[list[DiscoveredModel], str | None]:
    summaries = answer.get("inferenceProfileSummaries") if isinstance(answer, dict) else None
    if not isinstance(summaries, list) or not isinstance(answer, dict):
        raise _protocol()
    found: list[DiscoveredModel] = []
    for summary in summaries:
        profile_id = summary.get("inferenceProfileId") if isinstance(summary, dict) else None
        kind = summary.get("type") if isinstance(summary, dict) else None
        found.append(_model(profile_id, kind))
    token = answer.get("nextToken")
    if token is not None and (not isinstance(token, str) or not token):
        raise _protocol()
    return found, token


def list_models(control: Any) -> list[DiscoveredModel]:
    """Every callable foundation model and inference profile. Runs the SDK: off the loop."""
    found = _foundation_models(control.list_foundation_models(byInferenceType="ON_DEMAND"))
    token: str | None = None
    for _ in range(MAX_PAGES):
        params: dict[str, Any] = {"maxResults": PROFILE_PAGE_SIZE}
        if token is not None:
            params["nextToken"] = token
        page, token = _profiles(control.list_inference_profiles(**params))
        found.extend(page)
        if token is None:
            return found
    raise _protocol()


async def off_loop[T](call: Callable[[], T], *, name: str) -> T:
    """``call()`` on its own daemon thread; its error comes back as ``ProviderError``."""
    loop = asyncio.get_running_loop()
    answered: asyncio.Future[tuple[Any, ProviderFailure | None]] = loop.create_future()

    def settle(outcome: tuple[Any, ProviderFailure | None]) -> None:
        if not answered.done():
            answered.set_result(outcome)

    def run() -> None:
        outcome: tuple[Any, ProviderFailure | None]
        try:
            outcome = (call(), None)
        except ProviderError as error:
            outcome = (None, error.failure)
        except Exception as error:
            outcome = (None, classify(error))
        with contextlib.suppress(RuntimeError):  # the loop has closed: nobody waits
            loop.call_soon_threadsafe(settle, outcome)

    threading.Thread(target=run, name=name, daemon=True).start()
    result, failure = await answered
    if failure is not None:
        raise ProviderError(failure)  # outside every handler: nothing chained
    found: T = result
    return found


async def discover(control: Any) -> Sequence[DiscoveredModel]:
    """The Region's callable models, through the control-plane client ``control``."""
    return await off_loop(lambda: list_models(control), name="conduit-bedrock-discovery")
