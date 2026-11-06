"""The registry builds every client under the construction guard.

``guarded_client`` runs a factory's client step: what it returns passes through; a
conduit error passes through with no traceback, cause or context; any other exception
becomes a fixed ``ConfigurationError`` naming the adapter (and the configured provider)
and the exception's class only. Cancellation and other ``BaseException`` propagate
unchanged. The registry calls every ``create_client`` this way, so a product's own
factory that raises surfaces at ``start()`` as the fixed error, with the key it
resolved nowhere in a traceback, a frame or a log record.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import SecretStr

from conduit.core.errors import ConfigurationError, SecretUnavailable
from conduit.core.registry import ProviderRegistry, guarded_client
from conduit.core.runtime import ConduitSettings, start
from conduit.core.secrets import LiteralSecretResolver
from conduit.core.settings import ProviderSettings
from conduit.core.tasks import CapabilityFloor, TaskClass
from conduit.embeddings import start_embeddings
from tests.unit.core.fakes import CATALOGUE, FakeFactory
from tests.unit.embeddings.fakes import CATALOGUE as EMBED_CATALOGUE
from tests.unit.embeddings.fakes import FAKE as FAKE_EMBED
from tests.unit.embeddings.fakes import FakeEmbedFactory
from tests.unit.providers.construction_cases import frame_locals, rendered

PLACEHOLDER = "conduit-placeholder-not-a-credential-0b9d"
SECRETS = LiteralSecretResolver({"fake/key": PLACEHOLDER})
CHAT = TaskClass(name="chat")
SEARCH = TaskClass(name="search", floor=CapabilityFloor(kind="embedding"))


class SyntheticFailure(RuntimeError):
    pass


class RaisingFactory(FakeFactory):
    """A product's factory that resolves its key and then fails to build its client."""

    def create_client(self, settings: ProviderSettings, secrets: Mapping[str, SecretStr]) -> Any:
        key = secrets["api_key"].get_secret_value()
        raise SyntheticFailure(f"client refused key {key}")


class RaisingEmbedFactory(FakeEmbedFactory):
    def create_client(self, settings: ProviderSettings, secrets: Mapping[str, SecretStr]) -> Any:
        key = secrets["api_key"].get_secret_value()
        raise SyntheticFailure(f"client refused key {key}")


def hidden(error: BaseException, caplog: pytest.LogCaptureFixture) -> bool:
    texts = [str(error), repr(error), repr(error.args), rendered(error), frame_locals(error)]
    texts += [record.getMessage() + repr(record.args) for record in caplog.records]
    return not any(PLACEHOLDER in text for text in texts)


def check_fixed(error: BaseException, *, kind: str) -> None:
    assert type(error) is ConfigurationError
    assert (
        str(error) == f"provider 'primary': the {kind} client could not be built: SyntheticFailure"
    )
    assert error.__cause__ is None
    assert error.__context__ is None


class TestGuardedClient:
    def test_the_built_client_passes_through(self) -> None:
        client = object()
        assert guarded_client("fake", lambda: client) is client

    def test_a_failure_becomes_the_fixed_error(self) -> None:
        def build() -> Any:
            key = PLACEHOLDER
            raise SyntheticFailure(f"refused {key}")

        with pytest.raises(ConfigurationError) as caught:
            guarded_client("fake", build)
        error = caught.value
        assert type(error) is ConfigurationError
        assert str(error) == "the fake client could not be built: SyntheticFailure"
        assert error.args == ("the fake client could not be built: SyntheticFailure",)
        assert error.__cause__ is None
        assert error.__context__ is None
        assert error.__suppress_context__ is False
        assert PLACEHOLDER not in rendered(error)
        assert PLACEHOLDER not in frame_locals(error)

    def test_the_configured_name_is_given_when_known(self) -> None:
        def build() -> Any:
            raise SyntheticFailure("refused")

        with pytest.raises(ConfigurationError, match=r"^provider 'primary': the fake client"):
            guarded_client("fake", build, name="primary")

    def test_a_conduit_error_passes_through_with_nothing_kept(self) -> None:
        def build() -> Any:
            key = PLACEHOLDER
            try:
                raise KeyError(key)
            except KeyError:
                raise SecretUnavailable("secret://fake/key is not held") from None

        with pytest.raises(SecretUnavailable, match="is not held") as caught:
            guarded_client("fake", build)
        error = caught.value
        assert error.__cause__ is None
        assert error.__context__ is None
        assert PLACEHOLDER not in rendered(error)
        assert PLACEHOLDER not in frame_locals(error)
        # The frames the error holds start at the guard: the step's frame is gone.
        tb = error.__traceback__
        assert tb is not None
        names = []
        while tb is not None:
            names.append(tb.tb_frame.f_code.co_name)
            tb = tb.tb_next
        assert "build" not in names

    def test_a_conduit_error_passes_through_without_its_notes(self) -> None:
        def build() -> Any:
            error = ConfigurationError("refused on purpose")
            error.add_note(f"while building with {PLACEHOLDER}")
            raise error

        with pytest.raises(ConfigurationError, match="refused on purpose") as caught:
            guarded_client("fake", build)
        error = caught.value
        assert not hasattr(error, "__notes__")
        assert PLACEHOLDER not in rendered(error)

    @pytest.mark.parametrize("raised", [asyncio.CancelledError, KeyboardInterrupt, SystemExit])
    def test_base_exceptions_propagate_unchanged(self, raised: type[BaseException]) -> None:
        signal = raised()

        def build() -> Any:
            raise signal

        with pytest.raises(raised) as caught:
            guarded_client("fake", build)
        assert caught.value is signal


class TestRegistry:
    async def test_build_refuses_with_the_fixed_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.DEBUG)
        registry = ProviderRegistry(
            secrets=SECRETS, catalogue=CATALOGUE, factories=[RaisingFactory()]
        )
        with pytest.raises(ConfigurationError) as caught:
            await registry.build("primary", {"kind": "fake", "api_key": "secret://fake/key"})
        check_fixed(caught.value, kind="fake")
        assert hidden(caught.value, caplog)

    async def test_start_refuses_with_the_fixed_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.DEBUG)
        settings = ConduitSettings.model_validate(
            {
                "providers": {"primary": {"kind": "fake", "api_key": "secret://fake/key"}},
                "assignments": {"chat": {"provider": "primary", "model": "fake-small"}},
            }
        )
        with pytest.raises(ConfigurationError) as caught:
            await start(
                settings,
                task_classes=[CHAT],
                factories=[RaisingFactory()],
                secrets=SECRETS,
                catalogue=CATALOGUE,
            )
        check_fixed(caught.value, kind="fake")
        assert hidden(caught.value, caplog)

    async def test_build_embedder_refuses_with_the_fixed_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.DEBUG)
        registry = ProviderRegistry(
            secrets=SECRETS,
            catalogue=EMBED_CATALOGUE,
            embedding_factories=[RaisingEmbedFactory()],
        )
        with pytest.raises(ConfigurationError) as caught:
            await registry.build_embedder("primary", FAKE_EMBED)
        check_fixed(caught.value, kind="fakeembed")
        assert hidden(caught.value, caplog)

    async def test_start_embeddings_refuses_with_the_fixed_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.DEBUG)
        settings = ConduitSettings.model_validate(
            {
                "providers": {"primary": FAKE_EMBED},
                "assignments": {"search": {"provider": "primary", "model": "fake-embed"}},
            }
        )
        with pytest.raises(ConfigurationError) as caught:
            await start_embeddings(
                settings,
                task_classes=[SEARCH],
                factories=[RaisingEmbedFactory()],
                secrets=SECRETS,
                catalogue=EMBED_CATALOGUE,
            )
        check_fixed(caught.value, kind="fakeembed")
        assert hidden(caught.value, caplog)
