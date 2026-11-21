"""The registry's Wave 3 behaviour.

- Every settings instance it is given is re-validated against its adapter's model, so
  one made with ``model_construct`` or changed with ``model_copy(update=...)`` is held to
  its validators (Q0019).
- Settings that fail validation raise ``SettingsRefused`` naming the failing fields,
  never pydantic's text or an input value, with no cause or context (Q0022).
- ``MeteredProvider.aclose`` holds every adapter's close to the rule: a close that
  raises is logged by class name only and never raised (Q0022).
- ``create_provider`` receives the registry's applied catalogue, overlays included;
  a factory written against the Wave 2 signature still works for this release, with a
  ``DeprecationWarning`` (Q0010, Q0012).
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from conduit.core.catalogue import Catalogue, CatalogueOverlay
from conduit.core.errors import ConfigurationError, SettingsRefused
from conduit.core.events import CapturePolicy
from conduit.core.provider import Provider
from conduit.core.registry import ProviderRegistry
from conduit.core.runtime import ConduitSettings, start
from conduit.core.secrets import LiteralSecretResolver, SecretRef
from conduit.core.settings import AssignmentSettings, ProviderSettings
from conduit.core.tasks import TaskClass
from tests.unit.core.fakes import CATALOGUE, FakeFactory, FakeProvider, FakeSettings

SECRETS = LiteralSecretResolver({"fake/key": "not-a-real-key"})
PLAINTEXT = "zz-plaintext-credential-sk-4417zz"


def _registry(factory: Any = None) -> tuple[ProviderRegistry, Any]:
    factory = factory or FakeFactory()
    registry = ProviderRegistry(secrets=SECRETS, catalogue=CATALOGUE, capture=CapturePolicy())
    registry.register(factory)
    return registry, factory


def _no_leak(error: BaseException) -> None:
    assert error.__cause__ is None
    assert error.__context__ is None
    for text in (str(error), repr(error), *(str(a) for a in error.args)):
        assert PLAINTEXT not in text
        assert "Value error" not in text  # pydantic's own wording
        assert "type=" not in text


class TestRevalidation:
    async def test_a_constructed_instance_is_held_to_its_validators(self) -> None:
        registry, factory = _registry()
        forged = FakeSettings.model_construct(kind="fake", api_key=PLAINTEXT)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # a serialiser warning would quote the value
            with pytest.raises(SettingsRefused) as caught:
                await registry.build("p", forged)
        assert caught.value.fields == ("api_key",)
        _no_leak(caught.value)
        assert factory.clients == []

    async def test_a_copied_instance_is_held_to_its_validators(self) -> None:
        registry, factory = _registry()
        good = FakeSettings(api_key=SecretRef("fake/key"))
        changed = good.model_copy(update={"behaviour": "nonsense"})
        with pytest.raises(SettingsRefused) as caught:
            await registry.build("p", changed)
        assert caught.value.fields == ("behaviour",)
        assert factory.clients == []

    async def test_a_valid_instance_is_built_from_its_validated_copy(self) -> None:
        registry, factory = _registry()
        given = FakeSettings(api_key=SecretRef("fake/key"))
        provider = await registry.build("p", given)
        assert provider.settings == given
        assert isinstance(factory.providers[0].settings, FakeSettings)

    async def test_an_instance_of_a_subclass_keeps_its_own_model(self) -> None:
        class Stricter(FakeSettings):
            region: str = "example-1"

        registry, _ = _registry()
        provider = await registry.build("p", Stricter(api_key=SecretRef("fake/key")))
        assert type(provider.settings) is Stricter

    async def test_an_unreadable_instance_is_refused_without_its_content(self) -> None:
        class Unprintable:
            def __str__(self) -> str:
                raise RuntimeError(PLAINTEXT)

        registry, _ = _registry()
        forged = FakeSettings.model_construct(kind="fake", api_key=Unprintable())
        with pytest.raises(ConfigurationError, match="cannot be read") as caught:
            await registry.build("p", forged)
        _no_leak(caught.value)


class TestValidationErrors:
    async def test_failing_fields_are_named_and_values_never_echoed(self) -> None:
        registry, _ = _registry()
        with pytest.raises(SettingsRefused) as caught:
            await registry.build(
                "primary", {"kind": "fake", "api_key": PLAINTEXT, "behaviour": PLAINTEXT}
            )
        error = caught.value
        assert error.provider == "primary"
        assert error.fields == ("api_key", "behaviour")
        assert "api_key" in str(error) and "behaviour" in str(error)
        _no_leak(error)

    async def test_unknown_fields_are_named(self) -> None:
        registry, _ = _registry()
        with pytest.raises(SettingsRefused) as caught:
            await registry.build(
                "p", {"kind": "fake", "api_key": "secret://fake/key", "surplus": PLAINTEXT}
            )
        assert caught.value.fields == ("surplus",)
        _no_leak(caught.value)

    async def test_start_refuses_the_same_way(self) -> None:
        settings = ConduitSettings.model_validate(
            {"providers": {"main": {"kind": "fake", "api_key": "secret://fake/key", "x": 1}}}
        )
        with pytest.raises(SettingsRefused) as caught:
            await start(settings, task_classes=[], factories=[FakeFactory()], secrets=SECRETS)
        assert caught.value.fields == ("x",)
        _no_leak(caught.value)


class _ClosingFails(FakeProvider):
    async def aclose(self) -> None:
        raise RuntimeError(f"SDK close failed for {PLAINTEXT}")


class _CloseCancelled(FakeProvider):
    async def aclose(self) -> None:
        raise asyncio.CancelledError


class _FactoryOf(FakeFactory):
    def __init__(self, provider_class: type[FakeProvider]) -> None:
        super().__init__()
        self._class = provider_class

    def create_provider(
        self,
        name: str,
        settings: ProviderSettings,
        client: Any,
        *,
        catalogue: Catalogue | None = None,
    ) -> Provider:
        assert isinstance(settings, FakeSettings)
        provider = self._class(name, settings, client)
        self.providers.append(provider)
        return provider


class TestClosing:
    async def test_a_failing_adapter_close_is_logged_by_class_only(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        registry, _ = _registry(_FactoryOf(_ClosingFails))
        provider = await registry.build("p", {"kind": "fake", "api_key": "secret://fake/key"})
        with caplog.at_level(logging.WARNING, logger="conduit"):
            await provider.aclose()  # never raised
        assert "RuntimeError" in caplog.text
        assert "_ClosingFails" in caplog.text
        assert PLAINTEXT not in caplog.text
        assert all(record.exc_info is None for record in caplog.records)

    async def test_a_cancelled_close_propagates(self) -> None:
        registry, _ = _registry(_FactoryOf(_CloseCancelled))
        provider = await registry.build("p", {"kind": "fake", "api_key": "secret://fake/key"})
        with pytest.raises(asyncio.CancelledError):
            await provider.aclose()


class _Legacy:
    """A factory written against the Wave 2 signature, without ``catalogue``."""

    kind = "fake"
    settings_model = FakeSettings

    def __init__(self) -> None:
        self.built: list[str] = []

    def create_client(self, settings: ProviderSettings, secrets: Mapping[str, SecretStr]) -> Any:
        return None

    def create_provider(self, name: str, settings: ProviderSettings, client: Any) -> Provider:
        assert isinstance(settings, FakeSettings)
        self.built.append(name)
        return FakeProvider(name, settings, client)


def _overlay() -> CatalogueOverlay:
    return CatalogueOverlay.model_validate(
        {
            "add": [
                {
                    "model_id": "fake-overlaid",
                    "provider": "fake",
                    "residency_scope": "local",
                    "context_window": 2_000,
                    "output_limit": 500,
                }
            ]
        }
    )


class TestCatalogue:
    async def test_create_provider_receives_the_registry_catalogue(self) -> None:
        registry, factory = _registry()
        await registry.build("p", {"kind": "fake", "api_key": "secret://fake/key"})
        assert factory.catalogues == [registry.catalogue]

    async def test_start_passes_the_applied_catalogue_with_overlays(self) -> None:
        factory = FakeFactory()
        settings = ConduitSettings(
            providers={"main": FakeSettings(api_key=SecretRef("fake/key"))},
            assignments={"chat": AssignmentSettings(provider="main", model="fake-overlaid")},
            catalogue=(_overlay(),),
        )
        runtime = await start(
            settings,
            task_classes=[TaskClass(name="chat")],
            factories=[factory],
            secrets=SECRETS,
            catalogue=CATALOGUE,
        )
        try:
            (given,) = factory.catalogues
            assert given is runtime.catalogue
            assert given.get("fake", "fake-overlaid") is not None
            assert CATALOGUE.get("fake", "fake-overlaid") is None
        finally:
            await runtime.aclose()

    async def test_a_wave_2_factory_still_builds_with_a_deprecation_warning(self) -> None:
        legacy = _Legacy()
        registry = ProviderRegistry(secrets=SECRETS, catalogue=CATALOGUE)
        with pytest.warns(DeprecationWarning, match="'fake'.*catalogue"):
            registry.register(legacy)
        await registry.build("p", {"kind": "fake", "api_key": "secret://fake/key"})
        assert legacy.built == ["p"]


class TestNoWarningEchoes:
    async def test_reading_a_constructed_instance_warns_nothing_about_its_values(self) -> None:
        registry, _ = _registry()
        forged = FakeSettings.model_construct(
            kind="fake", api_key=SecretRef("fake/key"), region=PLAINTEXT.encode()
        )
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            with pytest.raises(SettingsRefused):
                await registry.build("p", forged.model_copy(update={"behaviour": 3}))
        assert not [w for w in seen if PLAINTEXT in str(w.message)]


class TestDeprecationAttribution:
    """The warning names the product's code, not conduit's, so default filters show it."""

    def test_registered_through_the_constructor(self) -> None:
        with pytest.warns(DeprecationWarning) as seen:
            ProviderRegistry(secrets=SECRETS, catalogue=CATALOGUE, factories=[_Legacy()])
        assert [Path(w.filename).resolve() for w in seen] == [Path(__file__).resolve()]

    async def test_registered_through_start(self) -> None:
        settings = ConduitSettings()
        with pytest.warns(DeprecationWarning) as seen:
            runtime = await start(settings, task_classes=[], factories=[_Legacy()], secrets=SECRETS)
        await runtime.aclose()
        assert [Path(w.filename).resolve() for w in seen] == [Path(__file__).resolve()]


class TestUnvalidatedKind:
    @pytest.mark.parametrize("kind", [PLAINTEXT + " with spaces", "Upper", 42])
    async def test_a_kind_that_is_not_a_kind_is_not_quoted(self, kind: object) -> None:
        registry, _ = _registry()
        with pytest.raises(ConfigurationError) as caught:
            await registry.build("p", {"kind": kind})
        assert str(kind) not in str(caught.value)
        assert caught.value.__cause__ is None and caught.value.__context__ is None

    async def test_a_well_formed_kind_is_named(self) -> None:
        registry, _ = _registry()
        with pytest.raises(ConfigurationError, match="'absent'"):
            await registry.build("p", {"kind": "absent"})


class TestConduitSourcePrefixes:
    """The prefixes skipped for the warning match frames as they were imported."""

    def test_the_prefixes_cover_this_module_as_imported(self) -> None:
        import os

        from conduit.core import registry

        imported = os.path.abspath(registry.__file__)
        assert any(imported.startswith(p) for p in registry._CONDUIT_SOURCES)

    def test_a_sibling_directory_does_not_match(self) -> None:
        import os

        from conduit.core import registry

        assert all(p.endswith(os.sep) for p in registry._CONDUIT_SOURCES)
        for prefix in registry._CONDUIT_SOURCES:
            sibling = prefix.rstrip(os.sep) + "_x" + os.sep + "product.py"
            assert not any(sibling.startswith(p) for p in registry._CONDUIT_SOURCES)

    def test_the_imported_form_is_kept_when_resolving_changes_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import os

        from conduit.core import registry

        imported = os.path.dirname(os.path.dirname(os.path.abspath(registry.__file__)))
        elsewhere = os.path.join(os.path.dirname(imported), "resolved-elsewhere", "conduit")
        monkeypatch.setattr(
            os.path, "realpath", lambda path: os.path.join(elsewhere, "core", "registry.py")
        )
        prefixes = registry._source_prefixes()
        assert prefixes == (imported + os.sep, elsewhere + os.sep)
