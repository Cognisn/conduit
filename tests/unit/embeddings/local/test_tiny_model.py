"""The real local-embeddings stack on a tiny model built on disk: no download, no network.

The model is a randomly initialised two-layer BERT with a 32-word vocabulary, written to
the test's temporary folder and loaded with ``local_files_only``. It shows the library,
torch and the CPU-only handling work together; it says nothing about embedding quality.
Skipped, with the reason, when the extra is not installed or cannot be imported here
(torch on Windows needs the Microsoft Visual C++ runtime, for example), unless
``CONDUIT_LOCAL_EMBEDDINGS_TESTS=require``, which CI's local-embeddings job sets: then a
missing or broken install fails instead.

The NVIDIA check reads package metadata only, so it runs wherever torch is installed.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import math
import os
from pathlib import Path
from typing import Any

import pytest

from conduit.core.catalogue import CatalogueOverlay, ModelEntry, default_catalogue
from conduit.core.provider import EmbeddingRequest
from conduit.core.registry import ProviderRegistry
from conduit.core.residency import ResidencyScope
from conduit.core.secrets import LiteralSecretResolver
from conduit.embeddings.local import KIND, LocalEmbeddingFactory

REQUIRED = os.environ.get("CONDUIT_LOCAL_EMBEDDINGS_TESTS") == "require"


def _unavailable() -> str | None:
    """Why the real library cannot run here, or ``None`` when it can."""
    if importlib.util.find_spec("sentence_transformers") is None:
        return "the local-embeddings extra (sentence-transformers) is not installed"
    failed: str | None = None
    try:
        importlib.import_module("sentence_transformers")
    except Exception as error:  # torch's native libraries may not load on this host
        failed = type(error).__name__
    if failed is not None:
        return f"sentence-transformers cannot be imported on this host ({failed})"
    return None


UNAVAILABLE = _unavailable()


def _need_library() -> None:
    if UNAVAILABLE is None:
        return
    if REQUIRED:
        pytest.fail(f"CONDUIT_LOCAL_EMBEDDINGS_TESTS=require, but {UNAVAILABLE}")
    pytest.skip(UNAVAILABLE)


HIDDEN = 16
WORDS = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
    "a", "about", "another", "for", "host", "local", "note", "example", "com", "org",
    "the", "mail", "relay", "change", "record", "-", ".", "inventory", "answers", "on",
    "is", "up",
]  # fmt: skip


def _tiny_model(folder: Path) -> Path:
    import torch  # the extra is installed: see pytestmark

    transformers: Any = importlib.import_module("transformers")
    st_models: Any = importlib.import_module("sentence_transformers.models")
    st: Any = importlib.import_module("sentence_transformers")
    torch.manual_seed(1234)
    base = folder / "bert"
    base.mkdir()
    (base / "vocab.txt").write_text("\n".join(WORDS) + "\n", encoding="utf-8", newline="\n")
    config = transformers.BertConfig(
        vocab_size=len(WORDS),
        hidden_size=HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
    )
    transformers.BertModel(config).save_pretrained(base)
    transformers.BertTokenizerFast(vocab_file=str(base / "vocab.txt")).save_pretrained(base)
    encoder = st_models.Transformer(str(base), max_seq_length=32)
    pooling = st_models.Pooling(HIDDEN, pooling_mode="mean")
    target = folder / "tiny-sentence-model"
    st.SentenceTransformer(modules=[encoder, pooling], device="cpu").save(str(target))
    return target


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> Path:
    _need_library()
    return _tiny_model(tmp_path_factory.mktemp("local-embeddings"))


def _registry() -> ProviderRegistry:
    catalogue = default_catalogue().apply(
        CatalogueOverlay(
            add=(
                ModelEntry(
                    model_id="tiny-local",
                    provider=KIND,
                    kind="embedding",
                    residency_scope=ResidencyScope.LOCAL,
                    context_window=32,
                    embedding_dimensions=HIDDEN,
                ),
            )
        )
    )
    return ProviderRegistry(
        secrets=LiteralSecretResolver({}),
        catalogue=catalogue,
        embedding_factories=[LocalEmbeddingFactory()],
    )


async def test_the_tiny_model_embeds_on_the_cpu(tiny: Path) -> None:
    registry = _registry()
    embedder = await registry.build_embedder(
        "local",
        {
            "kind": KIND,
            "model_sources": {"tiny-local": str(tiny)},
            "local_files_only": True,
            "normalise": True,
        },
    )
    texts = ("a local note about example.com", "another note for the mail relay")
    first = await embedder.embed(EmbeddingRequest(model="tiny-local", texts=texts))
    second = await embedder.embed(EmbeddingRequest(model="tiny-local", texts=texts[:1]))
    assert first.dimensions == HIDDEN
    assert first.vectors[0] == pytest.approx(second.vectors[0], abs=1e-6)
    assert first.vectors[0] != first.vectors[1]
    assert all(
        math.isclose(math.fsum(v * v for v in vector), 1.0, rel_tol=1e-4)
        for vector in first.vectors
    )
    assert first.usage is not None
    assert first.usage.input_tokens > len(texts)  # word pieces plus [CLS] and [SEP]
    inner: Any = embedder._inner
    loaded = inner._models._models["tiny-local"]
    assert str(loaded.device) == "cpu"
    await registry.aclose()


def test_torch_is_the_cpu_only_build() -> None:
    _need_library()
    torch: Any = importlib.import_module("torch")
    assert torch.version.cuda is None, "torch was installed with CUDA: use the CPU index"


def test_no_nvidia_package_is_installed() -> None:
    if importlib.util.find_spec("torch") is None and not REQUIRED:
        pytest.skip("torch is not installed")
    names = {
        (dist.metadata["Name"] or "").lower().replace("_", "-")
        for dist in importlib.metadata.distributions()
    }
    nvidia = sorted(name for name in names if name.startswith("nvidia-"))
    assert not nvidia, f"NVIDIA packages are installed: {', '.join(nvidia)}"
