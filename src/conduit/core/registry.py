"""The provider registry: the only place provider clients are constructed.

Each provider adapter supplies a ``ProviderFactory``: its settings model, a
``create_client`` that builds the SDK client from validated settings and resolved
secrets, and a ``create_provider`` that wraps the client. The registry validates the
settings, resolves every secret reference, calls ``create_client`` (nothing else in
conduit may; a tripwire checks this) and wraps the result in a ``MeteredProvider``.

``MeteredProvider`` is the single funnel every model call passes through. Before any
client call it refuses a model the catalogue does not declare for the provider, and
holds the call to what the provider was built for:

- A provider built for resolved assignments (``start`` builds every runtime provider
  this way) serves only those assignments: each task class its own model, and a call
  without a task class one of the assigned models. Anything else raises
  ``ModelNotAssigned``, so no call can skip the floor and residency checks made at
  start-up by naming another declared model.
- A provider built without assignments (``ProviderRegistry.build`` used directly) is
  held to the registry's residency policy on every call. The default policy allows
  every scope and region, as ``ConduitSettings`` does.

It then emits a ``ModelCallStartedEvent`` and a ``ModelCallFinishedEvent`` (with the
final usage, all four counts) for every call, under the content-capture policy.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, SecretStr, ValidationError

from conduit.core.catalogue import Catalogue, default_catalogue
from conduit.core.errors import ConduitError, ConfigurationError, ProviderError, ProviderFailure
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
from conduit.core.settings import KIND_PATTERN, ProviderSettings
from conduit.core.streaming import StreamEvent, StreamStopped, TextDelta, ToolCallFinished, collect
from conduit.core.types import ModelRequest, ModelResponse, ToolResultBlock, ToolUseBlock

logger = logging.getLogger("conduit.registry")

_UNSET: Any = object()


class ModelNotAssigned(ConduitError, LookupError):
    """A call named a model, or a task class, outside the provider's resolved assignments.

    Raised before any client call. The message names the provider, model and task
    class, which are configuration, never content.
    """

    def __init__(self, provider: str, model: str, task_class: str | None) -> None:
        where = f" to task class {task_class!r}" if task_class is not None else ""
        super().__init__(f"model {model!r} is not assigned{where} on provider {provider!r}")
        self.provider = provider
        self.model = model
        self.task_class = task_class


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
        """The provider for one configured name, around ``client``."""
        ...


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
        self.inner = inner
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
        if self._assigned is None or self._models is None:
            self._residency.check(entry, self.settings.region, task=request.task_class or self.name)
            return
        if request.task_class is not None:
            if self._assigned.get(request.task_class) != request.model:
                raise ModelNotAssigned(self.name, request.model, request.task_class)
        elif request.model not in self._models:
            raise ModelNotAssigned(self.name, request.model, None)

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
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
        await self._events.emit(
            ModelCallStartedEvent(
                **context.common(),
                request=capture_request(request, policy),
                message_count=len(request.messages),
                tool_count=len(request.tools),
                max_output_tokens=request.max_output_tokens,
            )
        )
        began = time.perf_counter()
        text: list[str] = []
        tool_calls = 0
        stopped: StreamStopped | None = None

        async def finished(**fields: Any) -> None:
            await self._events.emit(
                ModelCallFinishedEvent(
                    **context.common(),
                    duration_ms=(time.perf_counter() - began) * 1000,
                    tool_call_count=tool_calls,
                    **fields,
                )
            )

        try:
            async for event in self.inner.stream(request):
                if isinstance(event, TextDelta):
                    text.append(event.text)
                elif isinstance(event, ToolCallFinished):
                    tool_calls += 1
                elif isinstance(event, StreamStopped):
                    stopped = event
                yield event
        except ProviderError as error:
            await finished(outcome="failed", failure=error.failure)
            raise
        except Exception as error:
            await finished(outcome="failed", error_type=type(error).__name__)
            raise
        if stopped is None:
            await finished(outcome="failed", failure=ProviderFailure.PROTOCOL)
            raise ProviderError(ProviderFailure.PROTOCOL)
        await finished(
            outcome="ok",
            stop_reason=stopped.stop_reason,
            usage=stopped.usage,
            response=policy.capture(ContentKind.PROMPT, "".join(text)),
        )

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return await collect(self.stream(request), model=request.model)

    async def aclose(self) -> None:
        await self.inner.aclose()

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
        self._factories[factory.kind] = factory

    def _factory(self, kind: str) -> ProviderFactory:
        try:
            return self._factories[kind]
        except KeyError:
            raise ConfigurationError(f"no provider adapter for kind {kind!r}") from None

    def validate(
        self, name: str, settings: ProviderSettings | Mapping[str, Any]
    ) -> ProviderSettings:
        """``settings`` re-validated against its adapter's own model."""
        data = settings.model_dump() if isinstance(settings, BaseModel) else dict(settings)
        factory = self._factory(str(data.get("kind")))
        if isinstance(settings, factory.settings_model):
            return settings
        try:
            return factory.settings_model.model_validate(data)
        except ValidationError as error:
            raise ConfigurationError(f"provider {name!r}: {error}") from None

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
        the provider then serves those assignments only. Without it, every call is
        held to the registry's residency policy.
        """
        validated = self.validate(name, settings)
        factory = self._factory(validated.kind)
        if assigned is not None:
            if not assigned:
                raise ConfigurationError(f"provider {name!r} is built for no assignment")
            for model in sorted(set(assigned.values())):
                self.catalogue.require(validated.catalogue_provider, model)
        if client is _UNSET:
            secrets = await resolve_secret_fields(validated, self._secrets)
            client = factory.create_client(validated, secrets)
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
