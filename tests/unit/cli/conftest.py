"""The terminal interface's tests need the cli extra; CI's test jobs install every extra."""

from __future__ import annotations

import pytest

pytest.importorskip("rich")
