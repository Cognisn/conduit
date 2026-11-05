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
        patch.setenv("CDTTEST__CONDUIT__CAPTURE__PROMPTS", "false")
        patch.setenv("SECRETS__BACKEND", "aws_secrets_manager")
        # If SECRETS__BACKEND ever got through, konfig would build an AWS Secrets
        # Manager client. These keep that client on a closed local port with
        # synthetic credentials, so even a broken isolation fails here and never
        # reaches AWS. They are not cleared by the isolation, and need not be.
        unused = tmp_path_factory.mktemp("aws") / "absent"
        patch.setenv("AWS_ENDPOINT_URL", "http://127.0.0.1:9")
        patch.setenv("AWS_ACCESS_KEY_ID", "synthetic-dev-only")
        patch.setenv("AWS_SECRET_ACCESS_KEY", "synthetic-dev-only")
        patch.setenv("AWS_DEFAULT_REGION", "ap-southeast-2")
        patch.setenv("AWS_MAX_ATTEMPTS", "1")
        patch.setenv("AWS_EC2_METADATA_DISABLED", "true")
        patch.setenv("AWS_CONFIG_FILE", str(unused))
        patch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(unused))
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
    prefixed = load_settings(
        konfig_module.Settings(defaults={"conduit": SECTION}, env_prefix="CDTTEST")
    )
    assert prefixed.capture.prompts is True


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
    assert repr(resolver) == "KonfigSecretResolver(EncryptedFileBackend)"
    assert (await resolver.resolve(SecretRef("fake/key"))).get_secret_value() == VALUE
    assert path.is_file()
    assert not hostile_environment.exists()
