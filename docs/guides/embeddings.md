# Embeddings

conduit's embeddings layer turns text into vectors through the same machinery as chat: the same settings, catalogue, residency policy, secret resolution and metering. Vector stores and retrieval are the product's.

Embedding task classes have a floor of kind `embedding`, so a chat model can never serve one (and an embedding model never serves a chat task class). `start_embeddings` resolves them from the same `ConduitSettings` as `start`, and builds their providers through the provider registry. Assignments in the settings for other task classes are left for `start`.

```python
import asyncio

from conduit import CapabilityFloor, ConduitSettings, LiteralSecretResolver, TaskClass
from conduit.embeddings import builtin_embedding_factories, start_embeddings

settings = ConduitSettings.model_validate(
    {
        "providers": {"local": {"kind": "stub", "not_for_production": True}},
        "assignments": {"search_index": {"provider": "local", "model": "stub-embed"}},
    }
)

SEARCH_INDEX = TaskClass(
    name="search_index",
    floor=CapabilityFloor(kind="embedding", min_embedding_dimensions=8),
)


async def embed() -> list[list[float]]:
    runtime = await start_embeddings(
        settings,
        task_classes=[SEARCH_INDEX],
        factories=builtin_embedding_factories(),
        secrets=LiteralSecretResolver({}),
    )
    try:
        binding = runtime.for_task("search_index")
        response = await binding.embed(["disk full on 192.0.2.10", "login failures"])
        assert binding.dimensions == 8
        # Embedding usage is input tokens only; None when a provider reports no count.
        assert response.usage is None or response.usage.output_tokens == 0
        return [list(vector) for vector in response.vectors]
    finally:
        await runtime.aclose()


vectors = asyncio.run(embed())
assert len(vectors) == 2 and all(len(v) == 8 for v in vectors)
```

Pass `dimensions={"search_index": 256}` to `start_embeddings` to ask a model that offers shorter vectors for them; the length must meet the floor's `min_embedding_dimensions` and be one the model offers, and is fixed at start-up. Every response is checked (one finite vector of the expected length per text), and each call emits an `embedding_call_started` and an `embedding_call_finished` event. The texts are prompt content, captured only when the capture policy captures prompts.

## Providers

Remote embedding providers share their chat adapter's settings and authentication: configure the provider once and assign an embedding model to an embedding task class.

| Provider kind | Extra | Model in the default catalogue |
|---|---|---|
| `bedrock` | `bedrock` | `amazon.titan-embed-text-v2:0` (in-region) |
| `openai_compat` (OpenAI and Microsoft Foundry) | `openai` | `text-embedding-3-small` |
| `gemini` | `gemini` | `gemini-embedding-001` |
| `ollama` | `ollama` | `nomic-embed-text` (local) |
| `sentence_transformers` | `local-embeddings` | none: declare the models you use |
| `stub` | none | `stub-embed` |

## Local embeddings

Kind `sentence_transformers`, extra `local-embeddings`. Models run on the CPU, in the product's process, with residency scope `local`. Install torch's CPU-only build first, as [Providers](providers.md#local-embeddings-and-cpu-only-torch) shows for each installer.

Declare each model in a catalogue overlay under provider `sentence_transformers`, and say where sentence-transformers loads it from:

```python
from conduit import ProviderRegistry

local_settings = ConduitSettings.model_validate(
    {
        "providers": {
            "minilm": {
                "kind": "sentence_transformers",
                "model_sources": {"all-minilm-l6": "sentence-transformers/all-MiniLM-L6-v2"},
                "cache_folder": "models",
                "local_files_only": True,
                "normalise": True,
            }
        },
        "assignments": {"search_index": {"provider": "minilm", "model": "all-minilm-l6"}},
        "catalogue": [
            {
                "name": "example-local-models",
                "add": [
                    {
                        "provider": "sentence_transformers",
                        "model_id": "all-minilm-l6",
                        "kind": "embedding",
                        "residency_scope": "local",
                        "context_window": 256,
                        "embedding_dimensions": 384,
                    }
                ],
            }
        ],
    }
)

registry = ProviderRegistry(
    secrets=LiteralSecretResolver({}), embedding_factories=builtin_embedding_factories()
)
registry.validate_embedding("minilm", local_settings.providers["minilm"])
```

- `model_sources` maps a catalogue model id to a local directory or a Hugging Face repository id; an id not listed is loaded by its own name.
- `local_files_only` loads only models already on disk, so nothing is downloaded at run time. Set it in production, and download models as part of the build.
- `revision` pins a repository revision; `hub_token` is a secret reference to a Hugging Face token for gated models (without it no token is sent, whatever `HF_TOKEN` holds).
- `batch_size` is the encoding batch size, and `normalise` scales each vector to unit length.

Building the provider loads the model, which imports torch: allow for the memory and start-up time.
