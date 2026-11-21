"""Settings for local embeddings with sentence-transformers (the local-embeddings extra)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, PositiveInt, field_validator

from conduit.core.secrets import SecretRef
from conduit.core.settings import ProviderSettings

KIND = "sentence_transformers"


class LocalEmbeddingSettings(ProviderSettings):
    """One configured local embedding provider. Models run on the CPU, in this process.

    - ``model_sources`` maps a catalogue model id to what sentence-transformers loads for
      it: a local directory or a Hugging Face repository id. A model id not listed is
      loaded by its own name. Declare each model in a catalogue overlay under provider
      ``sentence_transformers`` with residency scope ``local``.
    - ``cache_folder`` is where downloaded models are kept (sentence-transformers' own
      default otherwise, which follows ``SENTENCE_TRANSFORMERS_HOME`` and ``HF_HOME``).
    - ``local_files_only`` loads only models already on disk: nothing is downloaded.
    - ``revision`` pins the repository revision (a tag, branch or commit).
    - ``hub_token`` is a secret reference to a Hugging Face token, for gated models.
      Without it no token is sent, whatever ``HF_TOKEN`` holds.
    - ``batch_size`` is sentence-transformers' encoding batch size.
    - ``normalise`` scales each vector to unit length.
    """

    kind: Literal["sentence_transformers"] = "sentence_transformers"
    model_sources: dict[str, str] = Field(default_factory=dict)
    cache_folder: str | None = Field(default=None, min_length=1, max_length=1024)
    local_files_only: bool = False
    revision: str | None = Field(default=None, min_length=1, max_length=128)
    hub_token: SecretRef | None = None
    batch_size: PositiveInt = Field(default=32, le=4096)
    normalise: bool = False

    @field_validator("model_sources")
    @classmethod
    def _sources(cls, value: dict[str, str]) -> dict[str, str]:
        for model, source in value.items():
            if not model or not source or len(model) > 256 or len(source) > 1024:
                raise ValueError("model_sources needs model ids and sources of sensible length")
        return value
