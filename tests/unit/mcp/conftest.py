"""The MCP client's tests need the mcp extra; CI's test jobs install every runtime extra."""

from __future__ import annotations

import pytest

pytest.importorskip("mcp")
