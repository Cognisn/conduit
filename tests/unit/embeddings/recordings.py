"""Synthetic embedding recordings: what each adapter sends, and what it got back.

The files in ``tests/recordings/embeddings/`` are written by this module through
``conduit.testing.replay.save_recording``, which refuses anything unscrubbed. They are
synthetic: the texts, hosts and numbers are invented, and each response follows the
provider's documented shape. Regenerate them with

    python -m tests.unit.embeddings.recordings

A test checks that the committed files equal what this module builds.

Requests and responses, per provider:

- bedrock: ``{"modelId", "body"}`` (the InvokeModel body as JSON) and the response body.
- openai_compat: the ``embeddings.create`` parameters and the response JSON.
- gemini: the ``embed_content`` arguments and the response (``embeddings[].values``).
- ollama: ``{"method", "path", "body"}`` and ``{"status", "body"}``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from conduit.testing.replay import Interaction, Recording, save_recording

DIRECTORY = Path(__file__).resolve().parents[2] / "recordings" / "embeddings"

TEXTS = (
    "Inventory note: host-a.example.com answers on 192.0.2.10.",
    "Change record for the example.org mail relay.",
)
TITAN = "amazon.titan-embed-text-v2:0"
OPENAI = "text-embedding-3-small"
GEMINI = "gemini-embedding-001"
OLLAMA = "nomic-embed-text"


def synthetic_vector(seed: int, dimensions: int) -> list[float]:
    """Invented values in [-0.5, 0.5), deterministic for a seed."""
    return [round(((seed * 7919 + i * 104729) % 1000) / 1000 - 0.5, 4) for i in range(dimensions)]


def titan(dimensions: int | None, length: int) -> Recording:
    interactions = []
    for seed, text in enumerate(TEXTS):
        body: dict[str, Any] = {"inputText": text}
        if dimensions is not None:
            body["dimensions"] = dimensions
        interactions.append(
            Interaction(
                request={"modelId": TITAN, "body": body},
                response={
                    "embedding": synthetic_vector(seed, length),
                    "embeddingsByType": {"float": synthetic_vector(seed, length)},
                    "inputTextTokenCount": 11 + seed,
                },
            )
        )
    name = f"titan-v2-{length}"
    return Recording(name=name, provider="bedrock", interactions=tuple(interactions))


def openai(dimensions: int) -> Recording:
    request: dict[str, Any] = {
        "model": OPENAI,
        "input": list(TEXTS),
        "encoding_format": "float",
        "dimensions": dimensions,
    }
    response = {
        "object": "list",
        "model": OPENAI,
        # Deliberately out of order: the adapter reads each item's index.
        "data": [
            {"object": "embedding", "index": 1, "embedding": synthetic_vector(1, dimensions)},
            {"object": "embedding", "index": 0, "embedding": synthetic_vector(0, dimensions)},
        ],
        "usage": {"prompt_tokens": 21, "total_tokens": 21},
    }
    return Recording(
        name=f"openai-small-{dimensions}",
        provider="openai_compat",
        interactions=(Interaction(request=request, response=response),),
    )


def gemini(dimensions: int) -> Recording:
    request = {
        "model": GEMINI,
        "contents": list(TEXTS),
        "config": {"output_dimensionality": dimensions},
    }
    response = {"embeddings": [{"values": synthetic_vector(s, dimensions)} for s in (0, 1)]}
    return Recording(
        name=f"gemini-001-{dimensions}",
        provider="gemini",
        interactions=(Interaction(request=request, response=response),),
    )


def ollama() -> Recording:
    request = {
        "method": "POST",
        "path": "/api/embed",
        "body": {
            "model": OLLAMA,
            "input": list(TEXTS),
            "truncate": False,
            "options": {"num_ctx": 2048},
        },
    }
    response = {
        "status": 200,
        "body": {
            "model": OLLAMA,
            "embeddings": [synthetic_vector(s, 768) for s in (0, 1)],
            "total_duration": 14143917,
            "load_duration": 1019500,
            "prompt_eval_count": 23,
        },
    }
    return Recording(
        name="nomic-embed-text",
        provider="ollama",
        interactions=(Interaction(request=request, response=response),),
    )


def all_recordings() -> list[Recording]:
    return [titan(None, 1024), titan(256, 256), openai(256), gemini(768), ollama()]


def main() -> None:
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    for recording in all_recordings():
        save_recording(recording, DIRECTORY / f"{recording.name}.json")


if __name__ == "__main__":
    main()
