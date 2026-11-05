"""The konfig tests are isolated from the environment they run in.

konfig's environment layer reads unprefixed variables (``CONDUIT__CAPTURE__PROMPTS``
for ``conduit.capture.prompts``, ``SECRETS__FILE_PATH`` for ``secrets.file_path``), so
a variable left in a developer's or CI's environment could change what a test loads,
where its secrets store is written, or which backend konfig selects. The module
fixture below sets such variables before the conftest's per-test isolation runs; the
tests prove that isolation removes them.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

from conduit.core.secrets import SecretRef
from conduit.integrations.konfig import KonfigSecretResolver, load_settings

VALUE = "synthetic-secret-value-0003"
SECTION = {
    "providers": {"primary": {"kind": "fake", "api_key": "secret://fake/key"}},
    "capture": {"prompts": True},
    "require_all_assignments": False,
}


@pytest.fixture(scope="module")
def hostile_environment(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    redirected = tmp_path_factory.mktemp("hostile") / "redirected.enc"
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CONDUIT__CAPTURE__PROMPTS", "false")
        patch.setenv("CONDUIT__REQUIRE_ALL_ASSIGNMENTS", "true")
        patch.setenv("AI__CONDUIT__CAPTURE__PROMPTS", "false")
        patch.setenv("SECRETS__FILE_PATH", str(redirected))
        patch.setenv("SECRETS__MASTER_KEY", "hostile-master-key-dev-only")
        patch.setenv("KONFIG_MASTER_KEY", "hostile-master-key-dev-only")
        yield redirected


def test_conduit_settings_ignore_the_environment(
    hostile_environment: Path, konfig_module: ModuleType
) -> None:
    settings = load_settings(konfig_module.Settings(defaults={"conduit": SECTION}))
    assert settings.capture.prompts is True
    assert settings.require_all_assignments is False
    nested = load_settings(
        konfig_module.Settings(defaults={"ai": {"conduit": SECTION}}), section="ai.conduit"
    )
    assert nested.capture.prompts is True


async def test_the_secrets_store_ignores_the_environment(
    hostile_environment: Path, konfig_module: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / "secrets.enc"
    settings = konfig_module.Settings(
        defaults={
            "secrets": {
                "backend": "encrypted_file",
                "file_path": str(path),
                "master_key": "conduit-tests-master-key-dev-only",
            }
        }
    )
    konfig_module.Secrets("conduit-tests", settings=settings).set("fake/key", VALUE)
    resolver = KonfigSecretResolver.from_settings(settings, service_name="conduit-tests")
    assert (await resolver.resolve(SecretRef("fake/key"))).get_secret_value() == VALUE
    assert path.is_file()
    assert not hostile_environment.exists()
