"""The event-content tripwire's handling of a missing extra, and its switch.

Locally a listed module whose extra is missing is skipped with the extra named. With
``CONDUIT_ALL_EXTRAS=require`` (set where every runtime extra is installed) it fails.
"""

from __future__ import annotations

import importlib
import types
from importlib import metadata

import pytest

from conduit.core.events import Event
from tests.tripwires import test_event_content as tripwire
from tests.tripwires.importscan import OPTIONAL_LIBRARIES

MODULE = "conduit.synthetic.events"


def _missing(name: str) -> ImportError:
    return ImportError(f"No module named {name!r}", name=name)


class TestExcuse:
    def test_locally_a_missing_extra_is_named(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(tripwire.EVERY_EXTRA_SWITCH, raising=False)
        reason = tripwire.excuse(MODULE, _missing("mcp.client"))
        assert reason == f"{MODULE} needs the mcp extra (mcp is not installed)"

    def test_a_library_no_extra_installs_is_named(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(tripwire.EVERY_EXTRA_SWITCH, raising=False)
        reason = tripwire.excuse(MODULE, _missing("aioboto3"))
        assert reason == f"{MODULE} needs aioboto3, which no extra installs"

    def test_with_every_extra_required_a_missing_extra_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(tripwire.EVERY_EXTRA_SWITCH, "require")
        with pytest.raises(pytest.fail.Exception, match="needs the sql extra"):
            tripwire.excuse(MODULE, _missing("sqlalchemy"))

    @pytest.mark.parametrize("value", ["", "require"])
    def test_any_other_import_error_is_raised(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv(tripwire.EVERY_EXTRA_SWITCH, value)
        error = _missing("conduit.no_such_module")
        with pytest.raises(ImportError) as raised:
            tripwire.excuse(MODULE, error)
        assert raised.value is error

    @pytest.mark.parametrize("value", ["1", "true", "REQUIRE", "required"])
    def test_a_misspelt_switch_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv(tripwire.EVERY_EXTRA_SWITCH, value)
        with pytest.raises(ValueError, match=tripwire.EVERY_EXTRA_SWITCH):
            tripwire.excuse(MODULE, _missing("mcp"))


class TestTheScan:
    def _break_import(self, monkeypatch: pytest.MonkeyPatch, broken: str) -> None:
        """Only ``broken`` fails to import, for want of the mcp extra. A module this
        environment genuinely cannot import (a job installing only some extras) stands
        in as an empty module, so the result does not depend on what is installed."""
        real = importlib.import_module

        def fake(name: str, package: str | None = None) -> object:
            if name == broken:
                raise _missing("mcp")
            try:
                return real(name, package)
            except ImportError:
                return types.ModuleType(name)

        monkeypatch.setattr(importlib, "import_module", fake)

    def test_the_listed_module_check_skips_locally_with_the_extra_named(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(tripwire.EVERY_EXTRA_SWITCH, raising=False)
        self._break_import(monkeypatch, "conduit.mcp.events")
        check = tripwire.TestEveryEventClass()
        with pytest.raises(pytest.skip.Exception, match="needs the mcp extra"):
            check.test_the_scan_sees_every_class_a_listed_module_defines("conduit.mcp.events")

    def test_the_listed_module_check_fails_when_every_extra_is_required(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(tripwire.EVERY_EXTRA_SWITCH, "require")
        self._break_import(monkeypatch, "conduit.mcp.events")
        check = tripwire.TestEveryEventClass()
        with pytest.raises(pytest.fail.Exception, match="needs the mcp extra"):
            check.test_the_scan_sees_every_class_a_listed_module_defines("conduit.mcp.events")

    def test_the_class_scan_fails_when_every_extra_is_required(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(tripwire.EVERY_EXTRA_SWITCH, "require")
        self._break_import(monkeypatch, "conduit.mcp.client")
        with pytest.raises(pytest.fail.Exception, match=r"conduit\.mcp\.client needs the mcp"):
            tripwire._event_classes()

    def test_the_class_scan_passes_over_a_missing_extra_locally(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(tripwire.EVERY_EXTRA_SWITCH, raising=False)
        self._break_import(monkeypatch, "conduit.mcp.client")
        assert Event in tripwire._event_classes()


class TestTheTable:
    def test_every_optional_library_has_an_entry(self) -> None:
        roots = {name.split(".")[0] for name in OPTIONAL_LIBRARIES}
        assert roots == set(tripwire.EXTRA_FOR_ROOT)

    def test_every_named_extra_is_one_the_package_provides(self) -> None:
        provided = set(metadata.metadata("cognisn-conduit").get_all("Provides-Extra") or [])
        named = {extra for extra in tripwire.EXTRA_FOR_ROOT.values() if extra is not None}
        assert named <= provided, sorted(named - provided)
