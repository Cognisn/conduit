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
  kinds conduit calls, ``EMBEDDING`` making it an embedding model); its ``modelId`` is
  the id, with no label. It is processed in the Region its ``modelArn`` names (the
  Region listed in): ``in_region``.
- ``ListInferenceProfiles``
  (https://docs.aws.amazon.com/bedrock/latest/APIReference/API_ListInferenceProfiles.html),
  paged with ``nextToken`` (at most ``MAX_PAGES`` pages). Each profile's
  ``inferenceProfileId`` is the id, and its ``type`` (``SYSTEM_DEFINED`` or
  ``APPLICATION``) the label. A profile's ``models`` are its ``InferenceProfileModel``
  values (https://docs.aws.amazon.com/bedrock/latest/APIReference/API_InferenceProfileModel.html),
  each the ARN of a foundation model in one destination Region, so the profile is
  ``cross_region`` over the Regions those ARNs name, and its base model is the one model
  they all name. A ``global.`` profile routes to any commercial Region (the global
  cross-Region inference guide), and a profile whose ARNs do not all name a Region is
  unbounded too: both are ``global``. A profile whose ARNs name different models has no
  base model. A profile whose ``status`` is present and not ``ACTIVE`` is left out.

The listing is the source of truth for where a model is processed (C18): no Region data
is taken from the catalogue.

boto3 is synchronous, so the listing runs on its own daemon thread, never on the event
loop or in its default executor, bounded by the client's own connect and read timeouts.
Any SDK error becomes ``ProviderError`` (``conduit.providers.bedrock.errors.classify``)
and an answer that is not the documented shape ``PROTOCOL``, each raised after its
handler, so nothing is chained.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import threading
from collections.abc import Callable, Sequence
from typing import Any

from conduit.core.discovery import DiscoveredModel
from conduit.core.errors import ProviderError, ProviderFailure
from conduit.core.residency import ResidencyScope
from conduit.providers.bedrock.errors import classify

#: The output modalities of a model conduit can call (chat and embeddings).
CALLABLE_OUTPUTS = frozenset({"TEXT", "EMBEDDING"})
#: Inference profiles asked for per page (the API's maximum).
PROFILE_PAGE_SIZE = 1000
#: The most profile pages read before the listing is refused as never ending.
MAX_PAGES = 100
#: The prefix of a global inference profile's id.
GLOBAL_PREFIX = "global."
#: A foundation model ARN: its partition, Region (possibly empty) and model id.
_FOUNDATION_ARN = re.compile(
    r"arn:aws(?:-[a-z-]+)?:bedrock:(?P<region>[a-z0-9-]*):(?:[0-9]{12})?:"
    r"foundation-model/(?P<model>[A-Za-z0-9.:_-]+)"
)


def _protocol() -> ProviderError:
    return ProviderError(ProviderFailure.PROTOCOL)


def _model(model_id: object, label: object = None, **listed: Any) -> DiscoveredModel:
    """A listed model; an id or label outside the documented shape is ``PROTOCOL``."""
    if not isinstance(model_id, str) or not isinstance(label, str | None):
        raise _protocol()
    found: DiscoveredModel | None = None
    with contextlib.suppress(ValueError):
        found = DiscoveredModel(model_id, label=label or None, **listed)
    if found is None:
        raise _protocol()
    return found


def _arn(value: object) -> tuple[str, str] | None:
    """The Region and model id a foundation model ARN names, or ``None``: an ARN that
    does not name both bounds nothing. A value that is not a string is ``PROTOCOL``."""
    if not isinstance(value, str):
        raise _protocol()
    match = _FOUNDATION_ARN.fullmatch(value)
    if match is None or not match["region"]:
        return None
    return match["region"], match["model"]


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
        _model(model_id)  # an id outside the documented shape is PROTOCOL, kept or not
        kinds = {o for o in outputs if isinstance(o, str)} & CALLABLE_OUTPUTS
        if not kinds:
            continue
        named = _arn(summary.get("modelArn")) if isinstance(summary, dict) else None
        residency: dict[str, Any] = {"scope": ResidencyScope.GLOBAL}
        if named is not None:
            residency = {"scope": ResidencyScope.IN_REGION, "regions": frozenset({named[0]})}
        found.append(
            _model(
                model_id,
                kind="embedding" if "EMBEDDING" in kinds else "chat",
                base_model_id=model_id if isinstance(model_id, str) and model_id else None,
                **residency,
            )
        )
    return found


def _destinations(profile_id: object, models: object) -> dict[str, Any]:
    """What a profile's ``models`` say: its scope, Regions and base model."""
    if not isinstance(models, list):
        raise _protocol()
    named = []
    for model in models:
        if not isinstance(model, dict):
            raise _protocol()
        named.append(_arn(model.get("modelArn")))
    bases = {found[1] for found in named if found is not None}
    base = bases.pop() if len(bases) == 1 else None
    if (
        not named
        or None in named
        or (isinstance(profile_id, str) and profile_id.startswith(GLOBAL_PREFIX))
    ):
        return {"scope": ResidencyScope.GLOBAL, "base_model_id": base}
    regions = frozenset(found[0] for found in named if found is not None)
    return {"scope": ResidencyScope.CROSS_REGION, "regions": regions, "base_model_id": base}


def _profiles(answer: object) -> tuple[list[DiscoveredModel], str | None]:
    summaries = answer.get("inferenceProfileSummaries") if isinstance(answer, dict) else None
    if not isinstance(summaries, list) or not isinstance(answer, dict):
        raise _protocol()
    found: list[DiscoveredModel] = []
    for summary in summaries:
        if not isinstance(summary, dict):
            raise _protocol()
        profile_id = summary.get("inferenceProfileId")
        kind = summary.get("type")
        listed = _destinations(profile_id, summary.get("models"))
        if summary.get("status", "ACTIVE") != "ACTIVE":
            continue
        found.append(_model(profile_id, kind, **listed))
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
