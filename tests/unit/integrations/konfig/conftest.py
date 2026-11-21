"""Fixtures for the konfig integration tests.

konfig is the konfig extra's dependency: these tests skip when it is not installed.
Every konfig object here is built in memory or in a temporary directory. Before each
test, the environment variables that would point konfig at AWS or a real key are
cleared, and so is every variable konfig's unprefixed environment layer would read
for the sections these tests use (``CONDUIT__*``, ``AI__*``, ``SECRETS__*``) or the
prefix one test sets (``CDTTEST__*``). So the environment cannot change what a test
loads, and nothing reaches the network or a real secrets store.
"""

from __future__ import annotations

import os
from types import ModuleType

import pytest

konfig: ModuleType = pytest.importorskip("konfig")

#: Environment variables konfig reads that would change what a test touches.
KONFIG_ENVIRONMENT = (
    "KONFIG_AWS_SECRETS_MANAGER",
    "KONFIG_AWS_SETTINGS",
    "KONFIG_AWS_SEED",
    "KONFIG_CONFIG_FORMAT",
    "KONFIG_MASTER_KEY",
)

#: Prefixes of the variables konfig's environment layer maps onto the settings these
#: tests read (``SECRETS__FILE_PATH`` is ``secrets.file_path``, and so on).
SETTINGS_ENVIRONMENT_PREFIXES = ("CONDUIT__", "AI__", "SECRETS__", "CDTTEST__")


@pytest.fixture(autouse=True)
def _isolated_konfig(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in KONFIG_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    for name in list(os.environ):
        if name.upper().startswith(SETTINGS_ENVIRONMENT_PREFIXES):
            monkeypatch.delenv(name)


@pytest.fixture
def konfig_module() -> ModuleType:
    return konfig
