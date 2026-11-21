"""Local embeddings: the factory, the CPU-only loader and the embedder, with a fake library.

These tests never load a real model. ``test_tiny_model.py`` runs the real library on a
tiny model built on disk, when the local-embeddings extra is installed.
"""

from __future__ import annotations

import asyncio
import importlib
import threading
from typing import Any

import pytest

from conduit.core.catalogue import CatalogueOverlay, ModelEntry, default_catalogue
from conduit.core.errors import ConfigurationError, ProviderError, ProviderFailure, SettingsRefused
from conduit.core.events import MemoryEventSink
from conduit.core.provider import EmbeddingRequest, embedding_usage
from conduit.core.registry import EmbeddingCallFinishedEvent, ProviderRegistry
from conduit.core.residency import ResidencyScope
from conduit.core.secrets import LiteralSecretResolver
from conduit.embeddings.local import (
    DEVICE,
    KIND,
    LocalEmbedder,
    LocalEmbeddingFactory,
    LocalEmbeddingSettings,
    LocalModels,
)
from conduit.embeddings.local import factory as factory_module

MODEL = "tiny-local"
TEXTS = ("A local note about host-a.example.com", "Another note for example.org")
SECRETS = LiteralSecretResolver({"hub/token": "not-a-real-token"})
CATALOGUE = default_catalogue().apply(
    CatalogueOverlay(
        name="local-test",
        add=(
            ModelEntry(
                model_id=MODEL,
                provider=KIND,
                kind="embedding",
                residency_scope=ResidencyScope.LOCAL,
                context_window=64,
                embedding_dimensions=4,
            ),
        ),
    )
)


class FakeArray:
    def __init__(self, rows: Any) -> None:
        self.rows = rows

    def tolist(self) -> Any:
        return self.rows

    def sum(self) -> int:
        return sum(sum(row) for row in self.rows)


class FakeModel:
    def __init__(self, source: str, **options: Any) -> None:
        self.source = source
        self.options = options
        self.encoded: list[dict[str, Any]] = []
        self.fail = False
        self.rows: Any = None
        self.release: threading.Event | None = None

    def encode(self, texts: list[str], **options: Any) -> FakeArray:
        self.encoded.append({"texts": texts, **options})
        if self.release is not None:
            self.release.wait(5)
        if self.fail:
            raise RuntimeError(f"synthetic failure on {texts[0]}")
        if self.rows is not None:
            return FakeArray(self.rows)
        return FakeArray([[float(len(t) % 5), 0.25, 0.5, 0.75] for t in texts])

    def tokenize(self, texts: list[str]) -> dict[str, FakeArray]:
        return {"attention_mask": FakeArray([[1] * len(t.split()) for t in texts])}


class FakeLibrary:
    def __init__(self) -> None:
        self.made: list[FakeModel] = []
        self.refuse = False

    def SentenceTransformer(self, source: str, **options: Any) -> FakeModel:
        if self.refuse:
            raise OSError(f"synthetic: {source} is not a model directory")
        model = FakeModel(source, **options)
        self.made.append(model)
        return model


@pytest.fixture
def library(monkeypatch: pytest.MonkeyPatch) -> FakeLibrary:
    fake = FakeLibrary()
    real = importlib.import_module

    def import_module(name: str, package: str | None = None) -> Any:
        if name == factory_module.LIBRARY:
            return fake
        return real(name, package)

    monkeypatch.setattr(importlib, "import_module", import_module)
    return fake


def _registry(sink: MemoryEventSink | None = None) -> ProviderRegistry:
    return ProviderRegistry(
        secrets=SECRETS,
        catalogue=CATALOGUE,
        events=sink,
        embedding_factories=[LocalEmbeddingFactory()],
    )


def _clean(error: BaseException) -> None:
    assert error.__cause__ is None
    assert error.__context__ is None
    for text in (*TEXTS, "not-a-real-token", "not a model directory"):
        assert text not in str(error)
        assert all(text not in str(arg) for arg in error.args)


# Settings ----------------------------------------------------------------------------


def test_the_settings_take_the_hub_token_as_a_secret_reference_only() -> None:
    with pytest.raises(SettingsRefused) as caught:
        _registry().validate_embedding("local", {"kind": KIND, "hub_token": "hf_plaintext"})
    assert "hf_plaintext" not in str(caught.value)
    settings = _registry().validate_embedding(
        "local", {"kind": KIND, "hub_token": "secret://hub/token"}
    )
    assert isinstance(settings, LocalEmbeddingSettings)


def test_the_settings_refuse_unknown_fields() -> None:
    with pytest.raises(SettingsRefused):
        _registry().validate_embedding("local", {"kind": KIND, "device": "cuda"})


def test_the_settings_have_no_region() -> None:
    settings = LocalEmbeddingSettings()
    assert settings.region is None
    assert settings.catalogue_provider == KIND


# Loading -----------------------------------------------------------------------------


async def test_models_load_on_the_cpu_without_remote_code_or_a_token(library: FakeLibrary) -> None:
    embedder = await _registry().build_embedder("local", {"kind": KIND})
    await embedder.embed(EmbeddingRequest(model=MODEL, texts=TEXTS))
    (made,) = library.made
    assert made.source == MODEL
    assert made.options["device"] == DEVICE == "cpu"
    assert made.options["trust_remote_code"] is False
    assert made.options["token"] is False
    assert made.options["local_files_only"] is False


async def test_the_settings_reach_the_loader(library: FakeLibrary) -> None:
    settings = {
        "kind": KIND,
        "model_sources": {MODEL: "D:/Models/tiny"},
        "cache_folder": "D:/Models/cache",
        "local_files_only": True,
        "revision": "v1.0",
        "hub_token": "secret://hub/token",
    }
    embedder = await _registry().build_embedder("local", settings)
    await embedder.embed(EmbeddingRequest(model=MODEL, texts=TEXTS))
    (made,) = library.made
    assert made.source == "D:/Models/tiny"
    assert made.options["cache_folder"] == "D:/Models/cache"
    assert made.options["local_files_only"] is True
    assert made.options["revision"] == "v1.0"
    assert made.options["token"] == "not-a-real-token"
    assert made.options["device"] == "cpu"


async def test_a_model_is_loaded_once(library: FakeLibrary) -> None:
    embedder = await _registry().build_embedder("local", {"kind": KIND})
    await asyncio.gather(
        *(embedder.embed(EmbeddingRequest(model=MODEL, texts=TEXTS)) for _ in range(4))
    )
    assert len(library.made) == 1


def test_a_missing_library_is_a_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def import_module(name: str, package: str | None = None) -> Any:
        raise ImportError(f"No module named {name!r}")

    monkeypatch.setattr(importlib, "import_module", import_module)
    with pytest.raises(ConfigurationError, match="local-embeddings extra") as caught:
        LocalEmbeddingFactory().create_client(LocalEmbeddingSettings(), {})
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_an_unresolved_hub_token_is_refused(library: FakeLibrary) -> None:
    settings = LocalEmbeddingSettings.model_validate({"hub_token": "secret://hub/token"})
    with pytest.raises(ConfigurationError, match="hub_token"):
        LocalEmbeddingFactory().create_client(settings, {})


def test_the_embedder_needs_the_factory_s_own_models() -> None:
    with pytest.raises(ConfigurationError):
        LocalEmbeddingFactory().create_embedder("local", LocalEmbeddingSettings(), object())


async def test_a_model_that_cannot_load_is_rejected_and_carries_nothing(
    library: FakeLibrary,
) -> None:
    library.refuse = True
    embedder = await _registry().build_embedder("local", {"kind": KIND})
    with pytest.raises(ProviderError) as caught:
        await embedder.embed(EmbeddingRequest(model=MODEL, texts=TEXTS))
    assert caught.value.failure is ProviderFailure.REJECTED
    _clean(caught.value)


# Embedding ---------------------------------------------------------------------------


async def test_vectors_and_token_usage(library: FakeLibrary) -> None:
    sink = MemoryEventSink()
    embedder = await _registry(sink).build_embedder("local", {"kind": KIND, "normalise": True})
    response = await embedder.embed(EmbeddingRequest(model=MODEL, texts=TEXTS))
    assert response.dimensions == 4
    assert response.usage == embedding_usage(9)
    encoded = library.made[0].encoded[0]
    assert encoded["texts"] == list(TEXTS)
    assert encoded["normalize_embeddings"] is True
    assert encoded["batch_size"] == 32
    assert encoded["show_progress_bar"] is False
    finished = sink.events[-1]
    assert isinstance(finished, EmbeddingCallFinishedEvent)
    assert finished.usage == embedding_usage(9)


async def test_an_encoding_failure_is_unavailable_and_carries_nothing(
    library: FakeLibrary,
) -> None:
    embedder = await _registry().build_embedder("local", {"kind": KIND})
    await embedder.embed(EmbeddingRequest(model=MODEL, texts=TEXTS))
    library.made[0].fail = True
    with pytest.raises(ProviderError) as caught:
        await embedder.embed(EmbeddingRequest(model=MODEL, texts=TEXTS))
    assert caught.value.failure is ProviderFailure.UNAVAILABLE
    _clean(caught.value)


@pytest.mark.parametrize(
    "rows", [[[0.1, 0.2, 0.3, "x"], [0.1] * 4], [[float("nan")] * 4, [0.1] * 4], ["ab", "cd"]]
)
async def test_malformed_vectors_are_a_protocol_failure(library: FakeLibrary, rows: Any) -> None:
    embedder = await _registry().build_embedder("local", {"kind": KIND})
    await embedder.embed(EmbeddingRequest(model=MODEL, texts=TEXTS))
    library.made[0].rows = rows
    with pytest.raises(ProviderError) as caught:
        await embedder.embed(EmbeddingRequest(model=MODEL, texts=TEXTS))
    assert caught.value.failure is ProviderFailure.PROTOCOL
    _clean(caught.value)


async def test_a_vector_of_the_wrong_length_is_caught_by_the_funnel(library: FakeLibrary) -> None:
    embedder = await _registry().build_embedder("local", {"kind": KIND})
    await embedder.embed(EmbeddingRequest(model=MODEL, texts=TEXTS))
    library.made[0].rows = [[0.1] * 5, [0.1] * 5]
    with pytest.raises(ProviderError) as caught:
        await embedder.embed(EmbeddingRequest(model=MODEL, texts=TEXTS))
    assert caught.value.failure is ProviderFailure.PROTOCOL


async def test_encoding_runs_off_the_loop_and_times_out_cleanly(library: FakeLibrary) -> None:
    embedder = await _registry().build_embedder("local", {"kind": KIND})
    await embedder.embed(EmbeddingRequest(model=MODEL, texts=TEXTS))
    release = threading.Event()
    library.made[0].release = release
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.001)

    tick = asyncio.create_task(ticker())
    try:
        with pytest.raises(ProviderError) as caught:
            await embedder.embed(EmbeddingRequest(model=MODEL, texts=TEXTS, timeout_s=0.2))
    finally:
        tick.cancel()
        release.set()
    assert caught.value.failure is ProviderFailure.TIMEOUT
    _clean(caught.value)
    assert ticks > 10


async def test_cancellation_propagates(library: FakeLibrary) -> None:
    embedder = await _registry().build_embedder("local", {"kind": KIND})
    await embedder.embed(EmbeddingRequest(model=MODEL, texts=TEXTS))
    release = threading.Event()
    library.made[0].release = release
    task = asyncio.create_task(embedder.embed(EmbeddingRequest(model=MODEL, texts=TEXTS)))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()


async def test_one_encoding_runs_at_a_time(library: FakeLibrary) -> None:
    embedder = await _registry().build_embedder("local", {"kind": KIND})
    await embedder.embed(EmbeddingRequest(model=MODEL, texts=TEXTS))
    model = library.made[0]
    inside = 0
    most = 0
    lock = threading.Lock()
    original = model.encode

    def encode(texts: list[str], **options: Any) -> FakeArray:
        nonlocal inside, most
        with lock:
            inside += 1
            most = max(most, inside)
        threading.Event().wait(0.05)
        with lock:
            inside -= 1
        return original(texts, **options)

    model.encode = encode  # type: ignore[method-assign]
    await asyncio.gather(
        *(embedder.embed(EmbeddingRequest(model=MODEL, texts=TEXTS)) for _ in range(3))
    )
    assert most == 1


async def test_no_shorter_vectors_are_offered(library: FakeLibrary) -> None:
    embedder = await _registry().build_embedder("local", {"kind": KIND})
    assert not embedder.accepts_dimensions(MODEL, 2)
    with pytest.raises(ConfigurationError, match="does not offer"):
        await embedder.embed(EmbeddingRequest(model=MODEL, texts=TEXTS, dimensions=2))


async def test_an_undeclared_local_model_is_refused_before_loading(library: FakeLibrary) -> None:
    from conduit.core.errors import UndeclaredModel

    embedder = await _registry().build_embedder("local", {"kind": KIND})
    with pytest.raises(UndeclaredModel):
        await embedder.embed(EmbeddingRequest(model="all-minilm-l6-v2", texts=TEXTS))
    assert library.made == []


async def test_closing_drops_the_models(library: FakeLibrary) -> None:
    registry = _registry()
    embedder = await registry.build_embedder("local", {"kind": KIND})
    await embedder.embed(EmbeddingRequest(model=MODEL, texts=TEXTS))
    inner = embedder._inner
    assert isinstance(inner, LocalEmbedder)
    models = inner._models
    assert isinstance(models, LocalModels)
    await registry.aclose()
    assert models._models == {}
