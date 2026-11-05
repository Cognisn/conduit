"""Local embeddings with sentence-transformers on CPU-only torch (the local-embeddings extra)."""

# sentence-transformers is imported only inside ``LocalEmbeddingFactory.create_client``,
# so importing this package never needs the extra installed.
#
# Install torch's CPU-only build. On Linux the default torch from PyPI also installs about
# 2.4 GB of NVIDIA CUDA packages that conduit never uses; macOS and Windows wheels from
# PyPI are CPU-only already. With each installer:
#
# - pip: ``pip install "torch>=2.15,<3" --index-url https://download.pytorch.org/whl/cpu``, then
#   ``pip install "cognisn-conduit[local-embeddings]"``. A later ``pip install --upgrade
#   torch`` without ``--index-url`` brings the CUDA build back.
# - uv, in a project: ``uv add "cognisn-conduit[local-embeddings]"`` with torch taken
#   from the CPU index in ``pyproject.toml`` (a ``[[tool.uv.index]]`` named ``pytorch-cpu``
#   at ``https://download.pytorch.org/whl/cpu`` with ``explicit = true``, and
#   ``[tool.uv.sources]`` ``torch = { index = "pytorch-cpu" }``); in a plain environment,
#   ``uv pip install --torch-backend cpu "cognisn-conduit[local-embeddings]"``.
# - uvx, a throwaway tool run: ``uvx --torch-backend cpu --from
#   "cognisn-conduit[local-embeddings,cli]" conduit``.

from __future__ import annotations

from conduit.embeddings.local.factory import LocalEmbeddingFactory
from conduit.embeddings.local.provider import DEVICE, LocalEmbedder, LocalModels
from conduit.embeddings.local.settings import KIND, LocalEmbeddingSettings

__all__ = [
    "DEVICE",
    "KIND",
    "LocalEmbedder",
    "LocalEmbeddingFactory",
    "LocalEmbeddingSettings",
    "LocalModels",
]
