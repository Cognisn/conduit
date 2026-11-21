"""The Gemini factory is built in, and building it never needs the gemini extra."""

from __future__ import annotations

import subprocess
import sys

from conduit.core.registry import ProviderFactory
from conduit.providers import builtin_factories
from conduit.providers.gemini import GeminiFactory, GeminiSettings


def test_the_factory_is_built_in() -> None:
    factories = {factory.kind: factory for factory in builtin_factories()}
    assert isinstance(factories["gemini"], GeminiFactory)
    assert isinstance(factories["gemini"], ProviderFactory)
    assert factories["gemini"].settings_model is GeminiSettings


def test_building_the_factories_does_not_import_the_sdk() -> None:
    # google.genai is made unimportable; listing the factories and validating
    # settings must still work, so the extra is needed only for a client.
    code = (
        "import sys\n"
        "sys.modules['google.genai'] = None\n"
        "from conduit.providers import builtin_factories\n"
        "kinds = {f.kind: f for f in builtin_factories()}\n"
        "kinds['gemini'].settings_model.model_validate(\n"
        "    {'kind': 'gemini', 'api_key': 'secret://k'})\n"
        "assert 'google.genai' not in sys.modules or sys.modules['google.genai'] is None\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
