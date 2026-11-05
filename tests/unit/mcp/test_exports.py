"""conduit.mcp's public names, which load on first use."""

from __future__ import annotations

import subprocess
import sys

import pytest

import conduit.mcp


def test_every_public_name_resolves() -> None:
    for name in conduit.mcp.__all__:
        assert getattr(conduit.mcp, name) is not None


def test_an_unknown_name_is_an_attribute_error() -> None:
    with pytest.raises(AttributeError):
        _ = conduit.mcp.NoSuchName


def test_importing_the_package_does_not_load_the_sdk() -> None:
    probe = "import sys, conduit.mcp; print('mcp' in sys.modules, 'httpx' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout.split()
    assert out == ["False", "False"]
