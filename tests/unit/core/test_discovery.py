"""Model discovery: listed through the registry, joined to the catalogue and the policy.

Discovery never makes a model usable. Every listed id is reported as allowed, refused
(with the residency or Region reason start-up gives) or undeclared, and an undeclared
model is still refused by the provider afterwards.
"""

from __future__ import annotations

import ast
import asyncio
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import SecretStr

import conduit
from conduit.core.catalogue import Catalogue, ModelEntry
from conduit.core.discovery import (
    DiscoveredModel,
    DiscoveryOutcome,
    DiscoveryResult,
    join_discovered,
)
from conduit.core.errors import (
    ConduitError,
    ConfigurationError,
    FacadeError,
    ProviderError,
    ProviderFailure,
    SettingsRefused,
    UndeclaredModel,
)
from conduit.core.events import (
    CapturePolicy,
    ContentKind,
    DiscoveryFinishedEvent,
    MemoryEventSink,
)
from conduit.core.facade import LoopThread, SyncRuntime, start_sync, sync
from conduit.core.registry import DiscoveringFactory, ProviderRegistry, _capture_listing
from conduit.core.residency import ResidencyPolicy, ResidencyScope
from conduit.core.runtime import ConduitSettings
from conduit.core.secrets import LiteralSecretResolver
from conduit.core.settings import ProviderSettings
from conduit.core.tasks import TaskClass
from conduit.core.types import Message, ModelRequest
from tests.unit.core.fakes import FakeFactory, FakeSettings

SECRETS = LiteralSecretResolver({"fake/key": "not-a-real-key"})
SOURCE_ROOT = Path(conduit.__file__).parent
DISCOVER_METHOD = "discover_models"


def _entry(
    model_id: str,
    scope: str,
    regions: Sequence[str] = (),
    kind: Literal["chat", "embedding"] = "chat",
) -> ModelEntry:
    fields: dict[str, Any] = {
        "model_id": model_id,
        "provider": "fake",
        "kind": kind,
        "residency_scope": scope,
        "regions": frozenset(regions),
        "context_window": 8_000,
    }
    if kind == "chat":
        fields["output_limit"] = 1_000
    else:
        fields["embedding_dimensions"] = 256
    return ModelEntry.model_validate(fields)


CATALOGUE = Catalogue(
    [
        _entry("fake-local", "local"),
        _entry("fake-global", "global"),
        _entry("fake-here", "in_region", ["xx-home-1"]),
        _entry("fake-elsewhere", "in_region", ["xx-away-1"]),
        _entry("fake-spread", "cross_region", ["xx-home-1", "xx-away-1"]),
        _entry("fake-embed", "local", kind="embedding"),
    ],
    version="discovery-1",
)

LISTED = (
    DiscoveredModel("fake-local"),
    DiscoveredModel("fake-global", label="foundation"),
    DiscoveredModel("fake-here"),
    DiscoveredModel("fake-elsewhere"),
    DiscoveredModel("fake-spread", label="profile"),
    DiscoveredModel("fake-embed"),
    DiscoveredModel("fake-unknown", label="foundation"),
)
SETTINGS = {"kind": "fake", "api_key": "secret://fake/key", "region": "xx-home-1"}


class ListingFactory(FakeFactory):
    """The fake adapter, with discovery: lists ``listing`` or raises ``failure``."""

    def __init__(
        self,
        listing: Sequence[Any] = LISTED,
        *,
        failure: BaseException | None = None,
        unsupported: str | None = None,
    ) -> None:
        super().__init__()
        self.listing = tuple(listing)
        self.failure = failure
        self.unsupported = unsupported
        self.discovered_with: list[tuple[ProviderSettings, Any]] = []

    def discovery_unsupported(self, settings: ProviderSettings) -> str | None:
        return self.unsupported

    async def discover_models(
        self, settings: ProviderSettings, client: Any
    ) -> Sequence[DiscoveredModel]:
        self.discovered_with.append((settings, client))
        if self.failure is not None:
            raise self.failure
        return self.listing


def _registry(
    factory: FakeFactory,
    *,
    residency: ResidencyPolicy | None = None,
    sink: MemoryEventSink | None = None,
    capture: CapturePolicy | None = None,
) -> ProviderRegistry:
    return ProviderRegistry(
        secrets=SECRETS,
        catalogue=CATALOGUE,
        residency=residency,
        events=sink,
        capture=capture,
        factories=[factory],
    )


def _outcomes(result: DiscoveryResult) -> dict[str, DiscoveryOutcome]:
    return {model.model_id: model.outcome for model in result.models}


def _finished(sink: MemoryEventSink) -> list[DiscoveryFinishedEvent]:
    return [e for e in sink.events if isinstance(e, DiscoveryFinishedEvent)]


class TestContract:
    def test_a_listing_factory_is_a_discovering_factory(self) -> None:
        assert isinstance(ListingFactory(), DiscoveringFactory)
        assert not isinstance(FakeFactory(), DiscoveringFactory)

    def test_a_discovered_model_holds_an_id_and_a_label_only(self) -> None:
        model = DiscoveredModel("fake-local", label="foundation")
        assert (model.model_id, model.label) == ("fake-local", "foundation")
        assert DiscoveredModel("fake-local").label is None
        with pytest.raises(ValueError):
            DiscoveredModel("")


class TestUnsupported:
    async def test_a_factory_without_discovery_is_reported_unsupported(self) -> None:
        sink = MemoryEventSink()
        factory = FakeFactory()
        result = await _registry(factory, sink=sink).discover("main", SETTINGS)
        assert not result.supported
        assert result.reason is not None and "fake" in result.reason
        assert result.models == ()
        assert factory.clients == []  # nothing was built, no secret resolved
        [event] = _finished(sink)
        assert event.outcome == "unsupported"
        assert (event.provider, event.kind) == ("main", "fake")

    async def test_a_factory_may_say_these_settings_cannot_be_listed(self) -> None:
        factory = ListingFactory(unsupported="this endpoint offers no listing")
        result = await _registry(factory).discover("main", SETTINGS)
        assert not result.supported
        assert result.reason == "this endpoint offers no listing"
        assert factory.clients == []
        assert factory.discovered_with == []

    async def test_an_unsupported_kind_is_never_an_error_at_start_up(self) -> None:
        # Registering a factory without discovery is accepted, as before.
        registry = _registry(FakeFactory())
        assert "fake" in registry.kinds


class TestThroughTheRegistry:
    async def test_the_client_is_built_by_create_client_with_resolved_secrets(self) -> None:
        factory = ListingFactory()
        await _registry(factory).discover("main", SETTINGS)
        [client] = factory.clients
        assert client.key.get_secret_value() == "not-a-real-key"
        [(settings, given)] = factory.discovered_with
        assert given is client
        assert isinstance(settings, FakeSettings)

    async def test_the_client_is_closed_afterwards(self) -> None:
        factory = ListingFactory()
        await _registry(factory).discover("main", SETTINGS)
        [provider] = factory.providers
        assert provider.closed

    async def test_the_client_is_closed_when_the_listing_fails(self) -> None:
        factory = ListingFactory(failure=ProviderError(ProviderFailure.REJECTED))
        with pytest.raises(ProviderError):
            await _registry(factory).discover("main", SETTINGS)
        [provider] = factory.providers
        assert provider.closed

    async def test_settings_are_validated_first(self) -> None:
        factory = ListingFactory()
        with pytest.raises(SettingsRefused):
            await _registry(factory).discover("main", {"kind": "fake"})
        assert factory.clients == []

    async def test_a_client_given_for_a_test_replaces_create_client(self) -> None:
        factory = ListingFactory()
        marker = object()
        await _registry(factory).discover("main", SETTINGS, client=marker)
        assert factory.clients == []
        assert factory.discovered_with[0][1] is marker


class TestJoin:
    async def test_each_id_is_allowed_refused_or_undeclared(self) -> None:
        result = await _registry(ListingFactory()).discover("main", SETTINGS)
        assert result.supported
        assert _outcomes(result) == {
            "fake-local": DiscoveryOutcome.ALLOWED,
            "fake-global": DiscoveryOutcome.ALLOWED,
            "fake-here": DiscoveryOutcome.ALLOWED,
            "fake-elsewhere": DiscoveryOutcome.REFUSED,
            "fake-spread": DiscoveryOutcome.ALLOWED,
            "fake-embed": DiscoveryOutcome.ALLOWED,
            "fake-unknown": DiscoveryOutcome.UNDECLARED,
        }

    async def test_a_model_not_offered_in_the_region_carries_the_start_up_reason(self) -> None:
        result = await _registry(ListingFactory()).discover("main", SETTINGS)
        [elsewhere] = [m for m in result.models if m.model_id == "fake-elsewhere"]
        assert elsewhere.reason == "main: model 'fake-elsewhere' is not offered in region xx-home-1"
        assert elsewhere.entry is not None and elsewhere.entry.model_id == "fake-elsewhere"

    async def test_residency_refusals_carry_the_policy_reason(self) -> None:
        policy = ResidencyPolicy(
            scopes=frozenset(ResidencyScope) - {ResidencyScope.LOCAL},
            regions=frozenset({"xx-home-1"}),
        )
        result = await _registry(ListingFactory(), residency=policy).discover("main", SETTINGS)
        reasons = {m.model_id: m.reason for m in result.models}
        assert _outcomes(result)["fake-global"] is DiscoveryOutcome.REFUSED
        assert reasons["fake-global"] == (
            "main: model 'fake-global' (global): global processing is refused when regions "
            "are restricted"
        )
        assert reasons["fake-local"] == (
            "main: model 'fake-local' (local): scope local is not allowed "
            "(cross_region, global, in_region)"
        )
        assert reasons["fake-spread"] == (
            "main: model 'fake-spread' (cross_region): may be processed in xx-away-1, outside "
            "the allowed regions"
        )
        assert _outcomes(result)["fake-here"] is DiscoveryOutcome.ALLOWED
        assert reasons["fake-here"] is None

    async def test_undeclared_and_allowed_models_carry_no_reason(self) -> None:
        result = await _registry(ListingFactory()).discover("main", SETTINGS)
        for model in result.models:
            if model.outcome is not DiscoveryOutcome.REFUSED:
                assert model.reason is None
        [unknown] = [m for m in result.models if m.model_id == "fake-unknown"]
        assert unknown.entry is None
        assert unknown.label == "foundation"

    async def test_counts_and_views(self) -> None:
        result = await _registry(ListingFactory()).discover("main", SETTINGS)
        assert [m.model_id for m in result.undeclared] == ["fake-unknown"]
        assert [m.model_id for m in result.refused] == ["fake-elsewhere"]
        assert len(result.allowed) == 5
        assert (result.provider, result.kind, result.catalogue_provider) == ("main", "fake", "fake")
        assert result.region == "xx-home-1"

    async def test_a_repeated_id_is_reported_once(self) -> None:
        listing = (DiscoveredModel("fake-local"), DiscoveredModel("fake-local", label="again"))
        result = await _registry(ListingFactory(listing)).discover("main", SETTINGS)
        assert [(m.model_id, m.label) for m in result.models] == [("fake-local", None)]

    async def test_an_empty_listing(self) -> None:
        result = await _registry(ListingFactory(())).discover("main", SETTINGS)
        assert result.supported
        assert result.models == ()

    def test_join_is_usable_on_its_own(self) -> None:
        joined = join_discovered(
            [DiscoveredModel("fake-elsewhere"), DiscoveredModel("nope")],
            CATALOGUE,
            catalogue_provider="fake",
            region="xx-home-1",
            residency=ResidencyPolicy(),
            provider="main",
        )
        assert [m.outcome for m in joined] == [
            DiscoveryOutcome.REFUSED,
            DiscoveryOutcome.UNDECLARED,
        ]


class TestNeverPermits:
    async def test_an_undeclared_model_is_still_refused_after_discovery(self) -> None:
        factory = ListingFactory()
        registry = _registry(factory)
        result = await registry.discover("main", SETTINGS)
        assert "fake-unknown" in {m.model_id for m in result.undeclared}
        provider = await registry.build("main", SETTINGS)
        request = ModelRequest(
            model="fake-unknown", messages=(Message.user("Hi"),), max_output_tokens=10
        )
        with pytest.raises(UndeclaredModel):
            await provider.complete(request)
        assert factory.providers[-1].requests == []

    async def test_discovery_leaves_the_catalogue_unchanged(self) -> None:
        registry = _registry(ListingFactory())
        before = list(registry.catalogue)
        await registry.discover("main", SETTINGS)
        assert list(registry.catalogue) == before
        assert registry.catalogue.get("fake", "fake-unknown") is None


class TestFailures:
    async def test_a_provider_failure_is_raised_with_nothing_chained(self) -> None:
        sink = MemoryEventSink()
        factory = ListingFactory(failure=ProviderError(ProviderFailure.REJECTED))
        with pytest.raises(ProviderError) as raised:
            await _registry(factory, sink=sink).discover("main", SETTINGS)
        assert raised.value.failure is ProviderFailure.REJECTED
        assert raised.value.__cause__ is None
        [event] = _finished(sink)
        assert event.outcome == "failed"
        assert event.failure is ProviderFailure.REJECTED

    async def test_any_other_exception_becomes_a_protocol_failure(self) -> None:
        sink = MemoryEventSink()
        factory = ListingFactory(failure=RuntimeError("provider text at example.com"))
        with pytest.raises(ProviderError) as raised:
            await _registry(factory, sink=sink).discover("main", SETTINGS)
        assert raised.value.failure is ProviderFailure.PROTOCOL
        assert raised.value.__cause__ is None
        assert raised.value.__context__ is None
        assert "example.com" not in str(raised.value)
        [event] = _finished(sink)
        assert event.error_type == "RuntimeError"

    async def test_a_conduit_error_goes_on_as_it_is(self) -> None:
        factory = ListingFactory(failure=ConduitError("refused on purpose"))
        with pytest.raises(ConduitError, match="refused on purpose"):
            await _registry(factory).discover("main", SETTINGS)

    @pytest.mark.parametrize("bad", ["fake-local", None, 3])
    async def test_a_listing_that_is_not_discovered_models_is_a_protocol_failure(
        self, bad: Any
    ) -> None:
        factory = ListingFactory([bad])
        with pytest.raises(ProviderError) as raised:
            await _registry(factory).discover("main", SETTINGS)
        assert raised.value.failure is ProviderFailure.PROTOCOL

    async def test_cancellation_propagates(self) -> None:
        factory = ListingFactory(failure=asyncio.CancelledError())
        with pytest.raises(asyncio.CancelledError):
            await _registry(factory).discover("main", SETTINGS)
        assert factory.providers[0].closed


class TestEvent:
    async def test_the_event_carries_counts_and_no_model_list_by_default(self) -> None:
        sink = MemoryEventSink()
        await _registry(ListingFactory(), sink=sink).discover("main", SETTINGS)
        [event] = _finished(sink)
        assert event.outcome == "ok"
        assert (event.allowed, event.refused, event.undeclared) == (5, 1, 1)
        assert event.duration_ms >= 0
        assert event.models is not None
        assert event.models.text is None
        assert event.models.size > 0
        assert "fake-unknown" not in event.model_dump_json()

    async def test_the_model_list_is_captured_only_when_prompts_are(self) -> None:
        sink = MemoryEventSink()
        registry = _registry(ListingFactory(), sink=sink, capture=CapturePolicy(prompts=True))
        await registry.discover("main", SETTINGS)
        [event] = _finished(sink)
        assert event.models is not None and event.models.text is not None
        listed = json.loads(event.models.text)
        assert {"model_id": "fake-unknown", "outcome": "undeclared"} in listed

    async def test_tool_content_capture_alone_does_not_carry_the_list(self) -> None:
        sink = MemoryEventSink()
        registry = _registry(ListingFactory(), sink=sink, capture=CapturePolicy(tool_content=True))
        await registry.discover("main", SETTINGS)
        [event] = _finished(sink)
        assert event.models is not None and event.models.text is None


class _NoSecrets(FakeFactory):
    def create_client(self, settings: ProviderSettings, secrets: Mapping[str, SecretStr]) -> Any:
        raise RuntimeError("cannot build at example.net")


class _Both(_NoSecrets, ListingFactory):
    pass


async def test_a_client_that_cannot_be_built_is_a_configuration_error() -> None:
    with pytest.raises(ConfigurationError) as raised:
        await _registry(_Both()).discover("main", SETTINGS)
    assert "example.net" not in str(raised.value)
    assert raised.value.__cause__ is None


class TestOnlyTheRegistryLists:
    """Discovery is reached only through the registry: no other module calls an
    adapter's ``discover_models``, so every listing is made with a registry-built client,
    joined to the catalogue and recorded."""

    @staticmethod
    def _callers(source_root: Path) -> list[str]:
        found: list[str] = []
        for path in sorted(source_root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == DISCOVER_METHOD
                ):
                    found.append(path.relative_to(source_root).as_posix())
        return found

    def test_only_the_registry_calls_discover_models(self) -> None:
        callers = self._callers(SOURCE_ROOT)
        assert callers == ["core/registry.py"], callers

    def test_the_scan_sees_a_call_elsewhere(self, tmp_path: Path) -> None:
        # Guards against a vacuous pass: a call in another module is found.
        (tmp_path / "rogue.py").write_text(
            "async def f(factory, s, c):\n    return await factory.discover_models(s, c)\n",
            encoding="utf-8",
        )
        assert self._callers(tmp_path) == ["rogue.py"]


class TestSyncFacade:
    @staticmethod
    def _start(loop: LoopThread, factory: FakeFactory) -> SyncRuntime:
        settings = ConduitSettings.model_validate(
            {
                "providers": {"main": SETTINGS},
                "assignments": {"chat": {"provider": "main", "model": "fake-local"}},
            }
        )
        return start_sync(
            settings,
            task_classes=[TaskClass(name="chat")],
            factories=[factory],
            secrets=SECRETS,
            catalogue=CATALOGUE,
            loop=loop,
        )

    @pytest.fixture
    def loop(self) -> Iterator[LoopThread]:
        owned = LoopThread(name="conduit-discovery-loop")
        yield owned
        owned.close()

    def test_a_sync_runtime_discovers_a_built_provider(self, loop: LoopThread) -> None:
        factory = ListingFactory()
        with self._start(loop, factory) as runtime:
            result = runtime.discover("main")
        assert isinstance(result, DiscoveryResult)
        assert [m.model_id for m in result.undeclared] == ["fake-unknown"]
        # The listing ran on the loop that owns the runtime's clients.
        assert len(factory.discovered_with) == 1

    def test_a_sync_runtime_discovers_other_configured_settings(self, loop: LoopThread) -> None:
        factory = ListingFactory()
        with self._start(loop, factory) as runtime:
            result = runtime.discover("spare", {**SETTINGS, "region": "xx-away-1"})
        assert result.provider == "spare"
        assert _outcomes(result)["fake-elsewhere"] is DiscoveryOutcome.ALLOWED

    def test_an_unbuilt_provider_needs_its_settings(self, loop: LoopThread) -> None:
        with (
            self._start(loop, ListingFactory()) as runtime,
            pytest.raises(ConfigurationError, match="spare"),
        ):
            runtime.discover("spare")

    def test_discovery_is_refused_on_the_loops_own_thread(self, loop: LoopThread) -> None:
        runtime = self._start(loop, ListingFactory())
        try:

            async def nested() -> DiscoveryResult:
                return runtime.discover("main")

            with pytest.raises(FacadeError):
                loop.run(nested())
        finally:
            runtime.close()

    def test_the_general_facade_reaches_the_registry(self, loop: LoopThread) -> None:
        registry = loop.run(_built_registry())
        result = sync(registry, loop=loop).discover("main", SETTINGS)
        assert isinstance(result, DiscoveryResult) and result.supported


async def _built_registry() -> ProviderRegistry:
    return _registry(ListingFactory())


class TestListingCapture:
    """The registry's own capture of the listing, before the fan-out enforces it again."""

    MODELS = join_discovered(
        LISTED,
        CATALOGUE,
        catalogue_provider="fake",
        region="xx-home-1",
        residency=ResidencyPolicy(),
        provider="main",
    )

    def test_no_text_unless_prompts_are_captured(self) -> None:
        for policy in (CapturePolicy(), CapturePolicy(tool_content=True)):
            captured = _capture_listing(self.MODELS, policy)
            assert captured.text is None
            assert captured.kind is ContentKind.PROMPT
            assert captured.size > 0

    def test_the_text_with_prompts_captured(self) -> None:
        captured = _capture_listing(self.MODELS, CapturePolicy(prompts=True))
        assert captured.text is not None and "fake-unknown" in captured.text


class ClosableClient:
    """A client with a close of its own, sync or async."""

    def __init__(self, *, asynchronous: bool = False) -> None:
        self.closed = False
        self.asynchronous = asynchronous

    def close(self) -> Any:
        if self.asynchronous:

            async def closing() -> None:
                self.closed = True

            return closing()
        self.closed = True
        return None


class _NoProvider(ListingFactory):
    """Makes its client, then fails to wrap it in a provider."""

    def __init__(self, *, asynchronous: bool = False) -> None:
        super().__init__()
        self.asynchronous = asynchronous
        self.made: list[ClosableClient] = []

    def create_client(self, settings: ProviderSettings, secrets: Mapping[str, SecretStr]) -> Any:
        client = ClosableClient(asynchronous=self.asynchronous)
        self.made.append(client)
        return client

    def create_provider(
        self,
        name: str,
        settings: ProviderSettings,
        client: Any,
        *,
        catalogue: Catalogue | None = None,
    ) -> Any:
        raise ConfigurationError("no provider for this client")


class TestClientWithoutAProvider:
    @pytest.mark.parametrize("asynchronous", [False, True])
    async def test_discovery_closes_a_client_no_provider_took(self, asynchronous: bool) -> None:
        factory = _NoProvider(asynchronous=asynchronous)
        with pytest.raises(ConfigurationError, match="no provider for this client"):
            await _registry(factory).discover("main", SETTINGS)
        [client] = factory.made
        assert client.closed
        assert factory.discovered_with == []

    @pytest.mark.parametrize("asynchronous", [False, True])
    async def test_build_closes_a_client_no_provider_took(self, asynchronous: bool) -> None:
        factory = _NoProvider(asynchronous=asynchronous)
        with pytest.raises(ConfigurationError, match="no provider for this client"):
            await _registry(factory).build("main", SETTINGS)
        [client] = factory.made
        assert client.closed


class TestEveryDiscoveryIsRecorded:
    async def test_a_client_that_cannot_be_built_is_a_failed_discovery(self) -> None:
        sink = MemoryEventSink()
        with pytest.raises(ConfigurationError):
            await _registry(_Both(), sink=sink).discover("main", SETTINGS)
        [event] = _finished(sink)
        assert event.outcome == "failed"
        assert event.error_type == "ConfigurationError"

    async def test_a_secret_that_cannot_be_resolved_is_a_failed_discovery(self) -> None:
        sink = MemoryEventSink()
        registry = ProviderRegistry(
            secrets=LiteralSecretResolver({}),
            catalogue=CATALOGUE,
            events=sink,
            factories=[ListingFactory()],
        )
        with pytest.raises(ConduitError) as raised:
            await registry.discover("main", SETTINGS)
        [event] = _finished(sink)
        assert event.outcome == "failed"
        assert event.error_type == type(raised.value).__name__

    async def test_a_client_no_provider_took_is_a_failed_discovery(self) -> None:
        sink = MemoryEventSink()
        with pytest.raises(ConfigurationError):
            await _registry(_NoProvider(), sink=sink).discover("main", SETTINGS)
        [event] = _finished(sink)
        assert (event.outcome, event.error_type) == ("failed", "ConfigurationError")

    async def test_settings_that_do_not_validate_emit_no_event(self) -> None:
        sink = MemoryEventSink()
        with pytest.raises(SettingsRefused):
            await _registry(ListingFactory(), sink=sink).discover("main", {"kind": "fake"})
        assert _finished(sink) == []
