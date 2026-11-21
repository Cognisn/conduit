"""conduit started from a konfig configuration and konfig's secrets store, end to end."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

from conduit.core.runtime import start
from conduit.core.tasks import CapabilityFloor, TaskClass
from conduit.integrations.konfig import KonfigSecretResolver, load_settings
from tests.unit.core.fakes import CATALOGUE, FakeFactory

VALUE = "synthetic-secret-value-0002"
CONFIG = """\
conduit:
  providers:
    primary:
      kind: fake
      api_key: secret://fake/key
  assignments:
    chat:
      provider: primary
      model: fake-large
secrets:
  backend: encrypted_file
  master_key: conduit-tests-master-key-dev-only
"""


async def test_start_resolves_konfig_secrets_for_the_provider(
    konfig_module: ModuleType, tmp_path: Path
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(CONFIG, encoding="utf-8", newline="\n")
    settings = konfig_module.Settings(config_file=config)
    settings.set("secrets.file_path", str(tmp_path / "secrets.enc"))
    konfig_module.Secrets("conduit-tests", settings=settings).set("fake/key", VALUE)

    factory = FakeFactory()
    runtime = await start(
        load_settings(settings),
        task_classes=[TaskClass(name="chat", floor=CapabilityFloor(tools=True))],
        factories=[factory],
        secrets=KonfigSecretResolver.from_settings(settings, service_name="conduit-tests"),
        catalogue=CATALOGUE,
    )
    assert [c.key.get_secret_value() for c in factory.clients] == [VALUE]
    await runtime.aclose()
