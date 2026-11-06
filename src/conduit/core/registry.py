"""The provider registry: the only place provider clients are constructed.

Each provider adapter supplies a ``ProviderFactory``: its settings model, a
``create_client`` that builds the SDK client from validated settings and resolved
secrets, and a ``create_provider`` that wraps the client. The registry validates the
settings, resolves every secret reference, calls ``create_client`` (nothing else in
conduit may; a tripwire checks this) and wraps the result in a ``MeteredProvider``.

The registry calls ``create_client`` under ``guarded_client``, the construction guard
every built-in factory also runs its own client step under. A factory that raises
surfaces at start-up as that factory's own conduit error, or as a fixed
``ConfigurationError`` naming the provider, its kind and the exception's class only;
never as the library's exception, whose frames could hold the resolved credential.

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

Embedding providers (C13) are built by the same registry, from an
``EmbeddingProviderFactory`` registered with ``register_embedding``: it normally shares
the chat adapter's settings model and ``create_client`` (so authentication is the same)
and supplies ``create_embedder``. ``build_embedder`` validates, resolves and makes the
client exactly as ``build`` does, and wraps the embedder in a ``MeteredEmbedder``, which
only the registry can make. Before any client call it refuses a model the catalogue does
not declare, a chat model, a model the adapter has no request shape for, and dimensions
above the model's own or that the model does not offer; it holds the call to its
resolved assignments, or to the residency policy, as ``MeteredProvider`` does. It checks
every response (one finite vector per text, each of the expected length, and usage of
input tokens only), raising ``ProviderError(PROTOCOL)`` for anything else, and emits an
``EmbeddingCallStartedEvent`` and an ``EmbeddingCallFinishedEvent`` for every call. The
texts are prompt content, captured only when the policy captures prompts.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import math
import os
import re
import sys
import time
import warnings
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, NonNegativeFloat, NonNegativeInt, SecretStr, ValidationError

from conduit.core.catalogue import Catalogue, ModelEntry, default_catalogue
from conduit.core.errors import (
    CapabilityRefused,
    ConduitError,
    ConfigurationError,
    ModelNotAssigned,
    ProviderError,
    ProviderFailure,
    SettingsRefused,
)
from conduit.core.events import (
    CallOutcome,
    CapturedContent,
    CapturePolicy,
    ContentKind,
    Event,
    EventSink,
    FanOutSink,
    ModelCallFinishedEvent,
    ModelCallStartedEvent,
)
from conduit.core.identifiers import Identifier, new_id
from conduit.core.provider import (
    EmbeddingProvider,
    EmbeddingRequest,
    EmbeddingResponse,
    Provider,
)
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


def _source_prefixes() -> tuple[str, ...]:
    """conduit's source tree, as imported and as resolved, each ending in a separator
    so a sibling directory (``conduit_x``) does not match."""
    imported = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    resolved = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    return tuple(dict.fromkeys(path.rstrip(os.sep) + os.sep for path in (imported, resolved)))


#: conduit's own source tree: a deprecation warning names the first frame outside it.
#: Frame filenames keep the form the code was imported under (a short name, a mapped
#: drive, a junction), so the unresolved form is used, with the resolved one beside it.
_CONDUIT_SOURCES = _source_prefixes()

_UNSET: Any = object()

#: ``ModelCallFinishedEvent.error_type`` for a call cancelled part-way (a timeout, say).
CANCELLED = "cancelled"
#: ``ModelCallFinishedEvent.error_type`` for a stream its consumer closed before the end.
ABANDONED = "abandoned"


async def _aclose(stream: AsyncIterator[StreamEvent]) -> None:
    closer = getattr(stream, "aclose", None)
    if closer is not None:
        await closer()


def guarded_client[T](kind: str, build: Callable[[], T], *, name: str | None = None) -> T:
    """Run ``build``, a factory's whole client step, under the construction guard.

    The step resolves the credential and constructs the SDK or HTTP client inside its
    own frame, which has ended by the time anything is raised here. What it returns
    passes through. If it raises:

    - a conduit error (a ``ConfigurationError`` the factory raises on purpose, say)
      goes on as it is, with its traceback, cause, context and notes cleared, so no
      frame of the step, no exception it handled and no note added on the way travels
      with it;
    - any other exception becomes ``ConfigurationError`` naming the adapter ``kind``
      (and the configured provider ``name``, when given) and the exception's class
      only, raised after the handler so its cause and context are both ``None``;
    - cancellation and any other ``BaseException`` propagate unchanged, so they still
      carry the step's frames: the promise about frames covers ordinary exceptions only.

    It cannot keep out what the library itself writes before it raises: an SDK that
    logs its own arguments at DEBUG does so before the guard sees anything.
    """
    failed: str | None = None
    refused: ConduitError | None = None
    try:
        return build()
    except ConduitError as error:
        refused = error
    except Exception as error:
        failed = type(error).__name__
    if refused is not None:
        refused.__cause__ = None
        refused.__context__ = None
        refused.__suppress_context__ = False
        if hasattr(refused, "__notes__"):
            del refused.__notes__
        raise refused.with_traceback(None)
    where = f"provider {name!r}: " if name is not None else ""
    raise ConfigurationError(f"{where}the {kind} client could not be built: {failed}")


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


class EmbeddingCallStartedEvent(Event):
    """An embedding call, as it starts. ``input`` is the batch of texts, as content."""

    type: Literal["embedding_call_started"] = "embedding_call_started"
    call_id: Identifier
    provider: Identifier  # the configured provider's name
    kind: Identifier
    model: Identifier
    input: CapturedContent
    text_count: NonNegativeInt
    dimensions: NonNegativeInt


class EmbeddingCallFinishedEvent(Event):
    """One embedding call's outcome and usage: the metering point for embeddings.

    Every call has exactly one, however it ends. ``usage`` is ``None`` for a failed call
    and for a provider that reports no count. ``error_type`` is the exception's class
    name, or ``cancelled`` for a call cancelled part-way.
    """

    type: Literal["embedding_call_finished"] = "embedding_call_finished"
    call_id: Identifier
    provider: Identifier
    kind: Identifier
    model: Identifier
    outcome: CallOutcome
    usage: Usage | None = None
    failure: ProviderFailure | None = None
    error_type: Identifier | None = None
    duration_ms: NonNegativeFloat
    vector_count: NonNegativeInt = 0
    dimensions: NonNegativeInt


@runtime_checkable
class EmbeddingProviderFactory(Protocol):
    """What an embedding adapter gives the registry.

    ``settings_model`` and ``create_client`` are normally the chat adapter's own (an
    embedding factory subclasses the chat factory), so an embedding provider shares the
    chat provider's settings and authentication.
    """

    @property
    def kind(self) -> str:
        """The provider kind this adapter serves: the kind of its settings."""
        ...

    @property
    def settings_model(self) -> type[ProviderSettings]: ...

    def create_client(self, settings: ProviderSettings, secrets: Mapping[str, SecretStr]) -> Any:
        """The SDK client (or ``None`` for an adapter that needs none)."""
        ...

    def create_embedder(
        self,
        name: str,
        settings: ProviderSettings,
        client: Any,
        *,
        catalogue: Catalogue | None = None,
    ) -> EmbeddingProvider:
        """The embedding provider for one configured name, around ``client``. The
        registry always passes its applied catalogue."""
        ...


#: The largest magnitude a vector value may have.
_FLOAT_MAX = sys.float_info.max

#: Only the registry holds this, so only the registry can make a ``MeteredEmbedder``.
_EMBEDDER_TOKEN = object()


def _capture_texts(request: EmbeddingRequest, policy: CapturePolicy) -> CapturedContent:
    text = json.dumps(list(request.texts), ensure_ascii=False)
    return policy.capture(ContentKind.PROMPT, text)


def _well_formed(response: object, count: int, dimensions: int) -> bool:
    """Whether ``response`` holds ``count`` finite vectors of ``dimensions`` each, and
    usage of input tokens only (or none)."""
    if not isinstance(response, EmbeddingResponse):
        return False
    usage = response.usage
    if usage is not None and (
        not isinstance(usage, Usage)
        or usage.output_tokens
        or usage.cache_read_tokens
        or usage.cache_write_tokens
    ):
        return False
    vectors = response.vectors
    if not isinstance(vectors, tuple) or len(vectors) != count:
        return False
    for vector in vectors:
        if not isinstance(vector, tuple) or len(vector) != dimensions:
            return False
        if not all(_finite_number(value) for value in vector):
            return False
    return True


def _finite_number(value: object) -> bool:
    """Whether ``value`` is a finite float, or an int a float can hold. Never raises:
    each value is checked on its own, so no sum can overflow."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return abs(value) <= _FLOAT_MAX
    return isinstance(value, float) and math.isfinite(value)


class MeteredEmbedder:
    """A built embedding provider, as the registry hands it out: checked and metered.

    Made only by ``ProviderRegistry.build_embedder``; the adapter it wraps is private.
    """

    def __init__(
        self,
        inner: EmbeddingProvider,
        *,
        token: object,
        name: str,
        settings: ProviderSettings,
        catalogue: Catalogue,
        events: FanOutSink,
        assigned: Mapping[str, str] | None = None,
        dimensions: Mapping[str, int] | None = None,
        residency: ResidencyPolicy | None = None,
    ) -> None:
        if token is not _EMBEDDER_TOKEN:
            raise TypeError("a MeteredEmbedder is made only by ProviderRegistry.build_embedder")
        self._inner = inner
        self.name = name
        self.kind = inner.kind
        self.settings = settings
        self._catalogue = catalogue
        self._events = events
        self._assigned = dict(assigned) if assigned is not None else None
        self._models = frozenset(self._assigned.values()) if self._assigned is not None else None
        self._dimensions = dict(dimensions or {})
        self._residency = residency if residency is not None else ResidencyPolicy()

    @property
    def assigned_models(self) -> frozenset[str] | None:
        """The models resolved for this provider, or ``None`` when it was built without
        assignments (and is held to the registry's residency policy instead)."""
        return self._models

    def serves(self, model: str) -> bool:
        return self._inner.serves(model)

    def accepts_dimensions(self, model: str, dimensions: int) -> bool:
        return self._inner.accepts_dimensions(model, dimensions)

    def _admit(self, request: EmbeddingRequest) -> tuple[int, EmbeddingRequest]:
        """The dimensions expected, and the request as the adapter receives it.

        Refuses, before any client call, a call outside what this provider serves.
        """
        model = request.model
        entry = self._catalogue.require(self.settings.catalogue_provider, model)
        if entry.kind != "embedding":
            raise CapabilityRefused(f"model {model!r} is a chat model, not an embedding model")
        wanted = request.dimensions
        if self._assigned is None or self._models is None:
            task = request.task_class or self.name
            self._residency.check(entry, self.settings.region, task=task)
        elif request.task_class is not None:
            if self._assigned.get(request.task_class) != model:
                raise ModelNotAssigned(self.name, model, request.task_class)
            fixed = self._dimensions.get(request.task_class)
            if fixed is not None:
                if wanted is not None and wanted != fixed:
                    raise ConfigurationError(
                        f"{request.task_class}: dimensions {wanted} differ from the task "
                        f"class's {fixed}"
                    )
                wanted = fixed
        elif model not in self._models:
            raise ModelNotAssigned(self.name, model, None)
        if not self._inner.serves(model):
            raise ConfigurationError(f"the {self.kind} embedding adapter cannot call {model!r}")
        own = entry.embedding_dimensions or 0
        wanted = wanted if wanted is not None else own
        if wanted > own:
            raise ConfigurationError(
                f"dimensions {wanted} exceed model {model!r}'s {own} dimensions"
            )
        if wanted != own and not self._inner.accepts_dimensions(model, wanted):
            raise ConfigurationError(f"model {model!r} does not offer {wanted} dimensions")
        forwarded = request.model_copy(update={"dimensions": None if wanted == own else wanted})
        return wanted, forwarded

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        dimensions, forwarded = self._admit(request)
        policy = self._events.policy
        common: dict[str, Any] = {
            "call_id": new_id("call"),
            "provider": self.name,
            "kind": self.kind,
            "model": request.model,
            "run_id": request.run_id,
            "correlation_id": request.correlation_id,
            "task_class": request.task_class,
        }
        began = time.perf_counter()

        async def finished(**fields: Any) -> None:
            await self._events.emit(
                EmbeddingCallFinishedEvent(
                    **common,
                    duration_ms=(time.perf_counter() - began) * 1000,
                    dimensions=dimensions,
                    **fields,
                )
            )

        await self._events.emit(
            EmbeddingCallStartedEvent(
                **common,
                input=_capture_texts(request, policy),
                text_count=len(request.texts),
                dimensions=dimensions,
            )
        )
        try:
            response = await self._inner.embed(forwarded)
        except ProviderError as error:
            await finished(outcome="failed", failure=error.failure)
            raise
        except Exception as error:
            await finished(outcome="failed", error_type=type(error).__name__)
            raise
        except asyncio.CancelledError:
            await finished(outcome="failed", error_type=CANCELLED)
            raise
        except BaseException as error:
            await finished(outcome="failed", error_type=type(error).__name__)
            raise
        # Checked and raised outside any handler: the error has no cause or context.
        if not _well_formed(response, len(request.texts), dimensions):
            await finished(outcome="failed", failure=ProviderFailure.PROTOCOL)
            raise ProviderError(ProviderFailure.PROTOCOL)
        await finished(outcome="ok", usage=response.usage, vector_count=len(response.vectors))
        return response

    async def aclose(self) -> None:
        """Close the adapter. A close that raises is logged (the adapter's class and the
        exception's class only) and never raised. Cancellation propagates unchanged."""
        await _close_quietly(self._inner, self.name)

    def __repr__(self) -> str:
        return f"MeteredEmbedder(name={self.name!r}, kind={self.kind!r})"


async def _close_quietly(inner: EmbeddingProvider, name: str) -> None:
    failed: str | None = None
    try:
        await inner.aclose()
    except Exception as error:
        failed = type(error).__name__
    if failed is not None:
        logger.warning(
            "closing embedding provider %s (%s) failed: %s", name, type(inner).__name__, failed
        )


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
        embedding_factories: Iterable[EmbeddingProviderFactory] = (),
    ) -> None:
        self._secrets = secrets
        self.catalogue = catalogue if catalogue is not None else default_catalogue()
        #: Held on every call of a provider built without assignments.
        self.residency = residency if residency is not None else ResidencyPolicy()
        self._events = FanOutSink([events] if events is not None else [], policy=capture)
        self._factories: dict[str, ProviderFactory] = {}
        self._with_catalogue: dict[str, bool] = {}
        self._built: list[MeteredProvider] = []
        self._embedding_factories: dict[str, EmbeddingProviderFactory] = {}
        self._embedders: list[MeteredEmbedder] = []
        for factory in factories:
            self.register(factory)
        for embedding_factory in embedding_factories:
            self.register_embedding(embedding_factory)

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
                skip_file_prefixes=_CONDUIT_SOURCES,
            )
        self._factories[factory.kind] = factory
        self._with_catalogue[factory.kind] = takes

    @property
    def embedding_kinds(self) -> frozenset[str]:
        return frozenset(self._embedding_factories)

    def register_embedding(self, factory: EmbeddingProviderFactory) -> None:
        """Register an embedding adapter. Its kind may also have a chat adapter."""
        if not re.fullmatch(KIND_PATTERN, factory.kind):
            raise ConfigurationError(
                f"embedding provider kind {factory.kind!r} is not a valid kind"
            )
        if factory.kind in self._embedding_factories:
            raise ConfigurationError(
                f"embedding provider kind {factory.kind!r} is registered twice"
            )
        self._embedding_factories[factory.kind] = factory

    def _embedding_factory(self, kind: str) -> EmbeddingProviderFactory:
        if kind not in self._embedding_factories:
            named = f"kind {kind!r}" if re.fullmatch(KIND_PATTERN, kind) else "that kind"
            raise ConfigurationError(f"no embedding adapter for {named}")
        return self._embedding_factories[kind]

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
        return self._validated(name, settings, lambda kind: self._factory(kind).settings_model)

    def validate_embedding(
        self, name: str, settings: ProviderSettings | Mapping[str, Any]
    ) -> ProviderSettings:
        """``settings`` re-validated against its embedding adapter's own model, as
        ``validate`` does for a chat adapter."""
        return self._validated(
            name, settings, lambda kind: self._embedding_factory(kind).settings_model
        )

    def _validated(
        self,
        name: str,
        settings: ProviderSettings | Mapping[str, Any],
        settings_model: Callable[[str], type[ProviderSettings]],
    ) -> ProviderSettings:
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
        model = settings_model(str(data.get("kind")))
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
            client = guarded_client(
                validated.kind, lambda: factory.create_client(validated, secrets), name=name
            )
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

    async def build_embedder(
        self,
        name: str,
        settings: ProviderSettings | Mapping[str, Any],
        *,
        client: Any = _UNSET,
        assigned: Mapping[str, str] | None = None,
        dimensions: Mapping[str, int] | None = None,
    ) -> MeteredEmbedder:
        """The embedding provider for one configured name. ``client`` replaces the SDK
        client in tests.

        ``assigned`` maps each embedding task class this provider serves to its model;
        each must be declared, an embedding model, and within the residency policy,
        checked before the client is made. ``dimensions`` fixes a task class's vector
        length below its model's own; the model must offer it. Without ``assigned``,
        every call is held to the policy.
        """
        validated = self.validate_embedding(name, settings)
        factory = self._embedding_factory(validated.kind)
        entries: dict[str, ModelEntry] = {}
        if assigned is not None:
            if not assigned:
                raise ConfigurationError(f"embedding provider {name!r} is built for no assignment")
            for task in sorted(assigned):
                entry = self.catalogue.require(validated.catalogue_provider, assigned[task])
                if entry.kind != "embedding":
                    raise CapabilityRefused(
                        f"{task}: model {assigned[task]!r} is a chat model, not an embedding model"
                    )
                self.residency.check(entry, validated.region, task=task)
                entries[task] = entry
        fixed = dict(dimensions or {})
        for task in sorted(fixed):
            if task not in entries:
                raise ConfigurationError(
                    f"dimensions are given for task class {task!r}, which is not assigned "
                    f"to {name!r}"
                )
            own = entries[task].embedding_dimensions or 0
            if fixed[task] < 1:
                raise CapabilityRefused(f"{task}: dimensions must be at least 1")
            if fixed[task] > own:
                raise CapabilityRefused(
                    f"{task}: {fixed[task]} dimensions exceed model "
                    f"{entries[task].model_id!r}'s {own} dimensions"
                )
        if client is _UNSET:
            secrets = await resolve_secret_fields(validated, self._secrets)
            client = guarded_client(
                validated.kind, lambda: factory.create_client(validated, secrets), name=name
            )
        inner = factory.create_embedder(name, validated, client, catalogue=self.catalogue)
        refusal: ConfigurationError | None = None
        for task in sorted(entries):
            model = entries[task].model_id
            wanted = fixed.get(task)
            if not inner.serves(model):
                refusal = ConfigurationError(
                    f"{task}: the {validated.kind} embedding adapter cannot call {model!r}"
                )
            elif (
                wanted is not None
                and wanted != entries[task].embedding_dimensions
                and not inner.accepts_dimensions(model, wanted)
            ):
                refusal = CapabilityRefused(
                    f"{task}: model {model!r} does not offer {wanted} dimensions"
                )
            if refusal is not None:
                break
        if refusal is not None:
            await _close_quietly(inner, name)
            raise refusal
        reduced = {
            task: wanted
            for task, wanted in fixed.items()
            if wanted != entries[task].embedding_dimensions
        }
        embedder = MeteredEmbedder(
            inner,
            token=_EMBEDDER_TOKEN,
            name=name,
            settings=validated,
            catalogue=self.catalogue,
            events=self._events,
            assigned=assigned,
            dimensions=reduced,
            residency=self.residency,
        )
        self._embedders.append(embedder)
        logger.debug("built embedding provider %s (%s)", name, validated.kind)
        return embedder

    async def aclose(self) -> None:
        embedders, self._embedders = self._embedders, []
        for embedder in reversed(embedders):
            await embedder.aclose()
        built, self._built = self._built, []
        for provider in reversed(built):
            try:
                await provider.aclose()
            except Exception as error:
                logger.warning(
                    "closing provider %s failed: %s", provider.name, type(error).__name__
                )
