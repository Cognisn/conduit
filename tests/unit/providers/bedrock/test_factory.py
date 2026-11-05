"""The Bedrock factory is registered, and importing it never imports the SDK."""

from __future__ import annotations

import subprocess
import sys

from conduit.providers import builtin_factories
from conduit.providers.bedrock import BedrockFactory


def test_builtin_factories_include_bedrock() -> None:
    assert [f for f in builtin_factories() if f.kind == "bedrock"]
    assert isinstance(next(f for f in builtin_factories() if f.kind == "bedrock"), BedrockFactory)


def test_importing_the_adapter_does_not_import_the_sdk() -> None:
    code = (
        "import sys, conduit.providers, conduit.providers.bedrock;"
        "conduit.providers.builtin_factories();"
        "print(sorted(m for m in sys.modules if m.split('.')[0] in ('boto3', 'botocore')))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == "[]"
