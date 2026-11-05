"""The local embedding factory: the only place sentence-transformers is imported.

``create_client`` imports sentence-transformers (which brings torch and transformers)
and returns the provider's ``LocalModels``; nothing is loaded until a model is first
used. Importing this module, or building ``builtin_embedding_factories()``, never needs
the local-embeddings extra.

conduit runs local models on the CPU only (C13): every model is loaded with
``device="cpu"``. Install torch's CPU-only build: on Linux the default torch from PyPI
also brings about 2.4 GB of NVIDIA CUDA packages that are never used; macOS and Windows
wheels from PyPI are CPU-only already (see ``conduit.embeddings.local``).

Environment variables still read by the libraries underneath, which deployments should
know (D0037):

- ``HF_ENDPOINT`` changes where models are downloaded from; ``HF_HUB_OFFLINE=1``
  stops downloads (as ``local_files_only`` does).
- ``HF_HOME``, ``HF_HUB_CACHE`` and ``SENTENCE_TRANSFORMERS_HOME`` move the model cache
  when ``cache_folder`` is not set.
- ``HF_TOKEN`` is never sent: conduit passes ``token=False`` unless ``hub_token`` is set.
- ``HTTPS_PROXY`` and the other proxy variables apply to downloads.
- ``TRANSFORMERS_VERBOSITY`` and ``TOKENIZERS_PARALLELISM`` change logging and tokenizer
  threads. Neither library logs the texts being encoded.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from pydantic import SecretStr

from conduit.core.catalogue import Catalogue, default_catalogue
from conduit.core.errors import ConfigurationError
from conduit.core.provider import EmbeddingProvider
from conduit.core.settings import ProviderSettings
from conduit.embeddings.local.provider import LocalEmbedder, LocalModels
from conduit.embeddings.local.settings import KIND, LocalEmbeddingSettings

#: The library the extra installs.
LIBRARY = "sentence_transformers"


def _local(settings: ProviderSettings) -> LocalEmbeddingSettings:
    if not isinstance(settings, LocalEmbeddingSettings):
        raise ConfigurationError("the local embedding factory needs LocalEmbeddingSettings")
    return settings


class LocalEmbeddingFactory:
    """Builds local embedding providers for the provider registry."""

    kind = KIND
    settings_model = LocalEmbeddingSettings

    def create_client(self, settings: ProviderSettings, secrets: Mapping[str, SecretStr]) -> Any:
        """The provider's ``LocalModels``. Imports sentence-transformers here only."""
        local = _local(settings)
        token: str | None = None
        if local.hub_token is not None:
            resolved = secrets.get("hub_token")
            if resolved is None:
                raise ConfigurationError("secret hub_token was not resolved")
            token = resolved.get_secret_value()
        missing = False
        try:
            library = importlib.import_module("sentence_transformers")
        except ImportError:
            missing = True
        if missing:
            raise ConfigurationError(
                "local embeddings need the local-embeddings extra (sentence-transformers)"
            )
        return LocalModels(library, local, token)

    def create_embedder(
        self,
        name: str,
        settings: ProviderSettings,
        client: Any,
        *,
        catalogue: Catalogue | None = None,
    ) -> EmbeddingProvider:
        if not isinstance(client, LocalModels):
            raise ConfigurationError("the local embedding factory needs its own LocalModels")
        return LocalEmbedder(
            name,
            _local(settings),
            client,
            catalogue=catalogue if catalogue is not None else default_catalogue(),
        )
