"""The Ollama factory is built in, and building it never needs the ollama extra."""

from __future__ import annotations

import subprocess
import sys

from conduit.core.registry import ProviderFactory
from conduit.providers import builtin_factories
from conduit.providers.ollama import OllamaFactory, OllamaSettings


def test_the_factory_is_built_in() -> None:
    factories = {factory.kind: factory for factory in builtin_factories()}
    assert isinstance(factories["ollama"], OllamaFactory)
    assert isinstance(factories["ollama"], ProviderFactory)
    assert factories["ollama"].settings_model is OllamaSettings


def test_building_the_factories_does_not_import_httpx() -> None:
    # httpx is made unimportable; listing the factories and validating settings
    # must still work, so the extra is needed only when a client is created.
    code = (
        "import sys\n"
        "sys.modules['httpx'] = None\n"
        "from conduit.providers import builtin_factories\n"
        "kinds = {f.kind: f for f in builtin_factories()}\n"
        "kinds['ollama'].settings_model.model_validate({'kind': 'ollama'})\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
