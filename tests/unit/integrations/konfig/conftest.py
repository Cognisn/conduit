"""Fixtures for the konfig integration tests.

konfig is the konfig extra's dependency: these tests skip when it is not installed.
Every konfig object here is built in memory or in a temporary directory, and the
environment variables that would point konfig at AWS or a real key are cleared, so
nothing reaches the network or a real secrets store.
"""

from __future__ import annotations

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


@pytest.fixture(autouse=True)
def _isolated_konfig(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in KONFIG_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def konfig_module() -> ModuleType:
    return konfig
