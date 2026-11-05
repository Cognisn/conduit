"""The provider registry: the only place provider clients are constructed.

Each provider adapter supplies a ``ProviderFactory``: its settings model, a
``create_client`` that builds the SDK client from validated settings and resolved
secrets, and a ``create_provider`` that wraps the client. The registry validates the
settings, resolves every secret reference, calls ``create_client`` (nothing else in
conduit may; a tripwire checks this) and wraps the result in a ``MeteredProvider``.

``create_provider`` receives the registry's catalogue as the keyword ``catalogue``: the
catalogue ``start`` applied the product's overlays to, so an adapter's capability checks
(structured output, prompt caching, context windows) follow the overlays. A factory
written against the Wave 2 signature, without ``catalogue``, is still called the old
way for this release, with a ``DeprecationWarning`` when it is registered; the next
release requires the keyword.

``validate`` re-validates every settings instance it is given against its adapter's
own model, so one made with ``model_construct`` or changed with ``model_copy(update=...)``
is held to its validators. Settings that fail raise ``SettingsRefused``, which names the
provider and the failing fields only: never pydantic's text, never a value.

``MeteredProvider`` is the single funnel every model call passes through. Before any
client call it refuses a model the catalogue does not declare for the provider, and
refuses a request whose ``max_output_tokens`` is above the model's output limit, and
holds the call to what the provider was built for:

- A provider built for resolved assignments (``start`` builds every runtime provider
  this way) serves only those assignments: each task class its own model, and a call
  without a task class one of the assigned models. Anything else raises
  ``ModelNotAssigned``, so no call can skip the floor and residency checks made at
  start-up by naming another declared model. Every assigned model is also checked
  against the registry's residency policy when the provider is built.
- A provider built without assignments (``ProviderRegistry.build`` used directly) is
  held to the registry's residency policy on every call. The default policy allows
  every scope and region, as ``ConduitSettings`` does.

It then emits a ``ModelCallStartedEvent`` and a ``ModelCallFinishedEvent`` (with the
final usage, all four counts) for every call, under the content-capture policy. The
adapter it wraps is private: there is no way to call it around these checks.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import re
import time
import warnings
from collections.abc import AsyncGenerator, AsyncIterator, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, SecretStr, ValidationError

from conduit.core.catalogue import Catalogue, default_catalogue
from conduit.core.errors import (
    ConfigurationError,
    ModelNotAssigned,
    ProviderError,
    ProviderFailure,
    SettingsRefused,
)
from conduit.core.events import (
    CapturedContent,
    CapturePolicy,
    ContentKind,
    EventSink,
    FanOutSink,
    ModelCallFinishedEvent,
    ModelCallStartedEvent,
)
from conduit.core.identifiers import new_id
from conduit.core.provider import Provider
from conduit.core.residency import ResidencyPolicy
from conduit.core.secrets import SecretResolver, resolve_secret_fields
from conduit.core.settings import KIND_PATTERN, ProviderSettings, refusals
from conduit.core.streaming import (
    StreamEvent,
    StreamStopped,
    TextDelta,
    ToolCallFinished,
    UsageReported,
    collect,
)
from conduit.core.types import ModelRequest, ModelResponse, ToolResultBlock, ToolUseBlock, Usage

logger = logging.getLogger("conduit.registry")

#: conduit's own source tree: a deprecation warning names the first frame outside it.
_CONDUIT_SOURCE = str(Path(__file__).resolve().parents[1])

_UNSET: Any = object()

#: ``ModelCallFinishedEvent.error_type`` for a call cancelled part-way (a timeout, say).
CANCELLED = "cancelled"
#: ``ModelCallFinishedEvent.error_type`` for a stream its consumer closed before the end.
ABANDONED = "abandoned"


async def _aclose(stream: AsyncIterator[StreamEvent]) -> None:
    closer = getattr(stream, "aclose", None)
    if closer is not None:
        await closer()


@runtime_checkable
class ProviderFactory(Protocol):
    """What a provider adapter gives the registry."""

    @property
    def kind(self) -> str:
        """The provider kind this adapter serves (``bedrock``, ``stub``)."""
        ...

    @property
    def settings_model(self) -> type[ProviderSettings]:
        """The adapter's own settings model."""
        ...

    def create_client(self, settings: ProviderSettings, secrets: Mapping[str, SecretStr]) -> Any:
        """The SDK client (or ``None`` for an adapter that needs none)."""
        ...

    def create_provider(self, name: str, settings: ProviderSettings, client: Any) -> Provider:
        """The provider for one configured name, around ``client``.

        The registry also passes ``catalogue=`` (its applied catalogue) to a factory whose
        ``create_provider`` accepts that keyword; declare it as
        ``*, catalogue: Catalogue | None = None`` for this release. The Wave 2 form
        without it is deprecated and is called without the catalogue.
        """
        ...


def _takes_catalogue(factory: ProviderFactory) -> bool:
    """Whether ``factory.create_provider`` accepts the ``catalogue`` keyword."""
    try:
        parameters = inspect.signature(factory.create_provider).parameters.values()
    except (TypeError, ValueError):
        return False
    named = (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    return any(
        p.kind is inspect.Parameter.VAR_KEYWORD or (p.name == "catalogue" and p.kind in named)
        for p in parameters
    )


def _has_tool_content(request: ModelRequest) -> bool:
    return any(
        isinstance(block, ToolResultBlock | ToolUseBlock)
        for message in request.messages
        for block in message.content
    )


def _request_text(request: ModelRequest, *, with_tool_content: bool) -> str:
    body: dict[str, Any] = request.model_dump(
        mode="json", include={"system", "messages", "tools", "output_schema"}
    )
    if not with_tool_content:
        for message in body["messages"]:
            for block in message["content"]:
                if block["type"] == "tool_result":
                    block["content"] = []
                elif block["type"] == "tool_use":
                    block["input"] = {}
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def capture_request(request: ModelRequest, policy: CapturePolicy) -> CapturedContent:
    """The request as an event carries it. Hash and size always describe the whole request;
    the text, when prompts are captured, omits tool content unless that is captured too."""
    full = _request_text(request, with_tool_content=True)
    captured = policy.capture(ContentKind.PROMPT, full)
    if captured.text is not None and not policy.tool_content and _has_tool_content(request):
        return captured.model_copy(update={"text": _request_text(request, with_tool_content=False)})
    return captured


@dataclass(frozen=True, slots=True)
class _CallContext:
    call_id: str
    provider: str
    kind: str
    model: str
    run_id: str | None
    correlation_id: str | None
    task_class: str | None

    def common(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "provider": self.provider,
            "kind": self.kind,
            "model": self.model,
            "run_id": self.run_id,
            "correlation_id": self.correlation_id,
            "task_class": self.task_class,
        }


class MeteredProvider:
    """A built provider, as the registry hands it out: checked, metered and recorded."""

    def __init__(
        self,
        inner: Provider,
        *,
        name: str,
        settings: ProviderSettings,
        catalogue: Catalogue,
        events: FanOutSink,
        assigned: Mapping[str, str] | None = None,
        residency: ResidencyPolicy | None = None,
    ) -> None:
        self._inner = inner
        self.name = name
        self.kind = inner.kind
        self.settings = settings
        self._catalogue = catalogue
        self._events = events
        self._assigned = dict(assigned) if assigned is not None else None
        self._models = frozenset(self._assigned.values()) if self._assigned is not None else None
        self._residency = residency if residency is not None else ResidencyPolicy()

    @property
    def assigned_models(self) -> frozenset[str] | None:
        """The models resolved for this provider, or ``None`` when it was built without
        assignments (and is held to the registry's residency policy instead)."""
        return self._models

    def _admit(self, request: ModelRequest) -> None:
        """Refuse, before any client call, a call outside what this provider serves."""
        entry = self._catalogue.require(self.settings.catalogue_provider, request.model)
        if entry.output_limit is not None and request.max_output_tokens > entry.output_limit:
            raise ConfigurationError(
                f"max_output_tokens {request.max_output_tokens} exceeds model "
                f"{request.model!r}'s output limit of {entry.output_limit}"
            )
        if self._assigned is None or self._models is None:
            self._residency.check(entry, self.settings.region, task=request.task_class or self.name)
            return
        if request.task_class is not None:
            if self._assigned.get(request.task_class) != request.model:
                raise ModelNotAssigned(self.name, request.model, request.task_class)
        elif request.model not in self._models:
            raise ModelNotAssigned(self.name, request.model, None)

    async def stream(self, request: ModelRequest) -> AsyncGenerator[StreamEvent, None]:
        self._admit(request)
        policy = self._events.policy
        context = _CallContext(
            call_id=new_id("call"),
            provider=self.name,
            kind=self.kind,
            model=request.model,
            run_id=request.run_id,
            correlation_id=request.correlation_id,
            task_class=request.task_class,
        )
        began = time.perf_counter()
        text: list[str] = []
        tool_calls = 0
        stopped: StreamStopped | None = None
        beyond_stop = False  # an event arrived after the stop: the stream broke its contract
        usage: Usage | None = None
        inner: AsyncIterator[StreamEvent] | None = None

        async def finished(**fields: Any) -> None:
            await self._events.emit(
                ModelCallFinishedEvent(
                    **context.common(),
                    duration_ms=(time.perf_counter() - began) * 1000,
                    tool_call_count=tool_calls,
                    **fields,
                )
            )

        def completed(stop: StreamStopped) -> dict[str, Any]:
            return {
                "outcome": "ok",
                "stop_reason": stop.stop_reason,
                "usage": stop.usage,
                "response": policy.capture(ContentKind.PROMPT, "".join(text)),
            }

        # Every path out of the call, from the started event on, is metered exactly once,
        # with the last usage seen, and only after the inner stream has been closed.
        try:
            try:
                await self._events.emit(
                    ModelCallStartedEvent(
                        **context.common(),
                        request=capture_request(request, policy),
                        message_count=len(request.messages),
                        tool_count=len(request.tools),
                        max_output_tokens=request.max_output_tokens,
                    )
                )
                inner = self._inner.stream(request)
                async for event in inner:
                    if stopped is not None:
                        beyond_stop = True
                    if isinstance(event, TextDelta):
                        text.append(event.text)
                    elif isinstance(event, ToolCallFinished):
                        tool_calls += 1
                    elif isinstance(event, UsageReported):
                        usage = event.usage
                    elif isinstance(event, StreamStopped):
                        stopped = event
                        usage = event.usage
                    yield event
            finally:
                if inner is not None:
                    await _aclose(inner)
        except ProviderError as error:
            await finished(outcome="failed", failure=error.failure, usage=usage)
            raise
        except Exception as error:
            await finished(outcome="failed", error_type=type(error).__name__, usage=usage)
            raise
        except asyncio.CancelledError:
            await finished(outcome="failed", error_type=CANCELLED, usage=usage)
            raise
        except GeneratorExit:
            # Closed by its consumer: complete if the stop had arrived (and nothing after it).
            if stopped is not None and not beyond_stop:
                await finished(**completed(stopped))
            else:
                await finished(outcome="failed", error_type=ABANDONED, usage=usage)
            raise
        except BaseException as error:
            await finished(outcome="failed", error_type=type(error).__name__, usage=usage)
            raise
        if stopped is None:
            await finished(outcome="failed", failure=ProviderFailure.PROTOCOL, usage=usage)
            raise ProviderError(ProviderFailure.PROTOCOL)
        await finished(**completed(stopped))

    async def complete(self, request: ModelRequest) -> ModelResponse:
        # Closed deterministically, so a call collect gives up on is metered at once.
        async with contextlib.aclosing(self.stream(request)) as events:
            return await collect(events, model=request.model)

    async def aclose(self) -> None:
        """Close the adapter. A close that raises is logged (the adapter's class and the
        exception's class only) and never raised, so it cannot replace or chain another
        failure. Cancellation propagates unchanged."""
        failed: str | None = None
        try:
            await self._inner.aclose()
        except Exception as error:
            failed = type(error).__name__
        if failed is not None:
            logger.warning(
                "closing provider %s (%s) failed: %s",
                self.name,
                type(self._inner).__name__,
                failed,
            )

    def __repr__(self) -> str:
        return f"MeteredProvider(name={self.name!r}, kind={self.kind!r})"


class ProviderRegistry:
    """Provider adapters by kind, and every provider built from them."""

    def __init__(
        self,
        *,
        secrets: SecretResolver,
        catalogue: Catalogue | None = None,
        events: EventSink | None = None,
        capture: CapturePolicy | None = None,
        residency: ResidencyPolicy | None = None,
        factories: Iterable[ProviderFactory] = (),
    ) -> None:
        self._secrets = secrets
        self.catalogue = catalogue if catalogue is not None else default_catalogue()
        #: Held on every call of a provider built without assignments.
        self.residency = residency if residency is not None else ResidencyPolicy()
        self._events = FanOutSink([events] if events is not None else [], policy=capture)
        self._factories: dict[str, ProviderFactory] = {}
        self._with_catalogue: dict[str, bool] = {}
        self._built: list[MeteredProvider] = []
        for factory in factories:
            self.register(factory)

    @property
    def kinds(self) -> frozenset[str]:
        return frozenset(self._factories)

    def register(self, factory: ProviderFactory) -> None:
        if not re.fullmatch(KIND_PATTERN, factory.kind):
            raise ConfigurationError(f"provider kind {factory.kind!r} is not a valid kind")
        if factory.kind in self._factories:
            raise ConfigurationError(f"provider kind {factory.kind!r} is registered twice")
        takes = _takes_catalogue(factory)
        if not takes:
            warnings.warn(
                f"provider factory for kind {factory.kind!r}: create_provider without the "
                "catalogue keyword is deprecated and will be refused in the next release",
                DeprecationWarning,
                skip_file_prefixes=(_CONDUIT_SOURCE,),
            )
        self._factories[factory.kind] = factory
        self._with_catalogue[factory.kind] = takes

    def _factory(self, kind: str) -> ProviderFactory:
        if kind not in self._factories:
            # The kind may come from an unvalidated mapping: name it only if it is a kind.
            named = f"kind {kind!r}" if re.fullmatch(KIND_PATTERN, kind) else "that kind"
            raise ConfigurationError(f"no provider adapter for {named}")
        return self._factories[kind]

    def validate(
        self, name: str, settings: ProviderSettings | Mapping[str, Any]
    ) -> ProviderSettings:
        """``settings`` re-validated against its adapter's own model.

        Every instance is validated afresh from its fields, whoever built it, so its
        validators always hold. An instance of a subclass of the adapter's model is
        validated against that subclass. Raises ``SettingsRefused`` naming the failing
        fields, or ``ConfigurationError`` for an instance whose fields cannot be read;
        neither keeps the exception it was raised for.
        """
        data: dict[str, Any] | None = None
        if isinstance(settings, BaseModel):
            try:
                # warnings=False: a serialiser warning would quote the offending value.
                data = settings.model_dump(round_trip=True, warnings=False)
            except Exception:
                data = None  # refused below, after the handler
            if data is None:
                raise ConfigurationError(f"provider {name!r}: settings cannot be read")
        else:
            data = dict(settings)
        factory = self._factory(str(data.get("kind")))
        model = factory.settings_model
        if isinstance(settings, model):
            model = type(settings)
        found: tuple[list[str], list[str]] = ([], [])
        try:
            return model.model_validate(data)
        except ValidationError as error:
            found = refusals(error, data)
        raise SettingsRefused(name, tuple(found[0]), tuple(found[1]))

    async def build(
        self,
        name: str,
        settings: ProviderSettings | Mapping[str, Any],
        *,
        client: Any = _UNSET,
        assigned: Mapping[str, str] | None = None,
    ) -> MeteredProvider:
        """The provider for one configured name. ``client`` replaces the SDK client in tests.

        ``assigned`` maps each task class this provider serves to its resolved model;
        the provider then serves those assignments only. Each assigned model must be
        declared and within the registry's residency policy, checked here before the
        client is made. Without ``assigned``, every call is held to the policy.
        """
        validated = self.validate(name, settings)
        factory = self._factory(validated.kind)
        if assigned is not None:
            if not assigned:
                raise ConfigurationError(f"provider {name!r} is built for no assignment")
            for task in sorted(assigned):
                entry = self.catalogue.require(validated.catalogue_provider, assigned[task])
                self.residency.check(entry, validated.region, task=task)
        if client is _UNSET:
            secrets = await resolve_secret_fields(validated, self._secrets)
            client = factory.create_client(validated, secrets)
        if self._with_catalogue.get(factory.kind, False):
            inner = factory.create_provider(
                name,
                validated,
                client,
                catalogue=self.catalogue,  # type: ignore[call-arg]
            )
        else:
            inner = factory.create_provider(name, validated, client)
        provider = MeteredProvider(
            inner,
            name=name,
            settings=validated,
            catalogue=self.catalogue,
            events=self._events,
            assigned=assigned,
            residency=self.residency,
        )
        self._built.append(provider)
        logger.debug("built provider %s (%s)", name, validated.kind)
        return provider

    async def aclose(self) -> None:
        built, self._built = self._built, []
        for provider in reversed(built):
            try:
                await provider.aclose()
            except Exception as error:
                logger.warning(
                    "closing provider %s failed: %s", provider.name, type(error).__name__
                )
