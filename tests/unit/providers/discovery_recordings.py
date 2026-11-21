"""Synthetic model-listing recordings for every adapter's discovery.

The files ``tests/recordings/<provider>/discovery-*.json`` are written by this module
through ``conduit.testing.replay.save_recording``, which refuses anything unscrubbed.
They are synthetic: every model id not in conduit's catalogue is invented
(``example-...``), account numbers are the documentation account 123456789012, and each
body follows the shape the provider's API reference documents (cited in each adapter's
discovery module). Regenerate them with

    python -m tests.unit.providers.discovery_recordings

A test checks that the committed files equal what this module builds.

Each recorded request is the operation and its parameters as the adapter sent them; each
response is ``{"status", "body"}``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from conduit.testing.replay import Interaction, Recording, save_recording

ROOT = Path(__file__).resolve().parents[2] / "recordings"
ACCOUNT = "123456789012"
REGION = "ap-southeast-2"
#: The other Region of the synthetic geographic profiles.
OTHER_REGION = "ap-southeast-4"


def _recording(name: str, provider: str, *pages: tuple[Any, int, Any]) -> Recording:
    return Recording(
        name=name,
        provider=provider,
        interactions=tuple(
            Interaction(request=request, response={"status": status, "body": body})
            for request, status, body in pages
        ),
    )


# Ollama: GET /api/tags, one body, no pagination.


def _ollama_model(name: str, family: str, size: str) -> dict[str, Any]:
    return {
        "name": name,
        "model": name,
        "modified_at": "2026-10-01T09:30:00.000000+10:00",
        "size": 4_920_000_000,
        "digest": "0" * 64,
        "details": {
            "format": "gguf",
            "family": family,
            "families": [family],
            "parameter_size": size,
            "quantization_level": "Q4_K_M",
        },
    }


OLLAMA_TAGS = {"method": "GET", "path": "/api/tags"}


def ollama_tags() -> Recording:
    return _recording(
        "ollama-discovery-tags",
        "ollama",
        (
            OLLAMA_TAGS,
            200,
            {
                "models": [
                    _ollama_model("llama3.1:8b", "llama", "8.0B"),
                    _ollama_model("example-local:3b", "example", "3.2B"),
                    _ollama_model("nomic-embed-text:latest", "nomic-bert", "137M"),
                    {
                        **_ollama_model("example-cloud:120b", "example", "120B"),
                        "remote_model": "example-cloud:120b",
                        "remote_host": "https://ollama.example.com:443",
                    },
                ]
            },
        ),
    )


def ollama_empty() -> Recording:
    return _recording("ollama-discovery-empty", "ollama", (OLLAMA_TAGS, 200, {"models": []}))


# Bedrock: ListFoundationModels (on-demand models, one body) and ListInferenceProfiles
# (paged with nextToken), on the control-plane client.

BEDROCK_FOUNDATION = {
    "operation": "ListFoundationModels",
    "params": {"byInferenceType": "ON_DEMAND"},
}


def _bedrock_profiles(token: str | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {"maxResults": 1000}
    if token is not None:
        params["nextToken"] = token
    return {"operation": "ListInferenceProfiles", "params": params}


def _foundation(model_id: str, provider: str, outputs: list[str]) -> dict[str, Any]:
    return {
        "modelArn": f"arn:aws:bedrock:{REGION}::foundation-model/{model_id}",
        "modelId": model_id,
        "modelName": model_id,
        "providerName": provider,
        "inputModalities": ["TEXT"],
        "outputModalities": outputs,
        "responseStreamingSupported": "TEXT" in outputs,
        "customizationsSupported": [],
        "inferenceTypesSupported": ["ON_DEMAND"],
        "modelLifecycle": {"status": "ACTIVE"},
    }


def _model_arn(region: str, model_id: str) -> dict[str, str]:
    """An ``InferenceProfileModel``: a foundation model in one destination Region."""
    return {"modelArn": f"arn:aws:bedrock:{region}::foundation-model/{model_id}"}


def _profile(
    profile_id: str, kind: str, models: list[dict[str, str]] | None = None
) -> dict[str, Any]:
    """A profile routing to ``models``; by default its base model in both Regions."""
    base = profile_id.partition(".")[2] or profile_id
    return {
        "inferenceProfileName": profile_id,
        "inferenceProfileArn": (
            f"arn:aws:bedrock:{REGION}:{ACCOUNT}:inference-profile/{profile_id}"
        ),
        "inferenceProfileId": profile_id,
        "models": (
            models
            if models is not None
            else [_model_arn(REGION, base), _model_arn(OTHER_REGION, base)]
        ),
        "status": "ACTIVE",
        "type": kind,
    }


def bedrock_listing() -> Recording:
    return _recording(
        "bedrock-discovery-listing",
        "bedrock",
        (
            BEDROCK_FOUNDATION,
            200,
            {
                "modelSummaries": [
                    _foundation("amazon.nova-lite-v1:0", "Amazon", ["TEXT"]),
                    _foundation("amazon.titan-embed-text-v2:0", "Amazon", ["EMBEDDING"]),
                    _foundation("example.image-model-v1:0", "Example", ["IMAGE"]),
                    _foundation("example.text-model-v1:0", "Example", ["TEXT"]),
                    _foundation("anthropic.claude-opus-5-5", "Anthropic", ["TEXT"]),
                ]
            },
        ),
        (
            _bedrock_profiles(),
            200,
            {
                "inferenceProfileSummaries": [
                    _profile("au.anthropic.claude-opus-5-5", "SYSTEM_DEFINED"),
                    _profile("global.anthropic.claude-opus-5-5", "SYSTEM_DEFINED"),
                ],
                "nextToken": "page-2-token",
            },
        ),
        (
            _bedrock_profiles("page-2-token"),
            200,
            {
                "inferenceProfileSummaries": [
                    _profile(
                        "example-application-1",
                        "APPLICATION",
                        [_model_arn(REGION, "example.text-model-v1:0")],
                    )
                ]
            },
        ),
    )


#: A foundation model ARN that names no Region.
NO_REGION_ARN = "arn:aws:bedrock:::foundation-model/example.text-model-v1:0"


def bedrock_drift() -> Recording:
    """Catalogue drift against a live listing: a profile routed to fewer Regions than a
    stale catalogue said, a profile with no catalogue entry, a profile whose models
    disagree, and one whose model ARN names no Region."""
    return _recording(
        "bedrock-discovery-drift",
        "bedrock",
        (BEDROCK_FOUNDATION, 200, {"modelSummaries": []}),
        (
            _bedrock_profiles(),
            200,
            {
                "inferenceProfileSummaries": [
                    _profile("au.anthropic.claude-sonnet-4-6", "SYSTEM_DEFINED"),
                    _profile("au.anthropic.example-claude-unlisted-v1:0", "SYSTEM_DEFINED"),
                    _profile(
                        "apac.example.mixed-v1:0",
                        "SYSTEM_DEFINED",
                        [
                            _model_arn(REGION, "example.mixed-v1:0"),
                            _model_arn(OTHER_REGION, "example.other-v1:0"),
                        ],
                    ),
                    _profile(
                        "example.no-region-v1:0",
                        "APPLICATION",
                        [{"modelArn": NO_REGION_ARN}],
                    ),
                ]
            },
        ),
    )


def bedrock_empty() -> Recording:
    return _recording(
        "bedrock-discovery-empty",
        "bedrock",
        (BEDROCK_FOUNDATION, 200, {"modelSummaries": []}),
        (_bedrock_profiles(), 200, {"inferenceProfileSummaries": []}),
    )


def http(path: str, **query: str) -> dict[str, Any]:
    """An HTTP listing request as recorded: method, path and query."""
    return {"method": "GET", "path": path, "query": dict(sorted(query.items()))}


# Anthropic: GET /v1/models, paged with after_id while has_more.


def _anthropic_model(model_id: str, name: str) -> dict[str, Any]:
    return {
        "type": "model",
        "id": model_id,
        "display_name": name,
        "created_at": "2026-09-21T00:00:00Z",
        "lifecycle": "active",
        "max_input_tokens": 1_000_000,
        "max_tokens": 128_000,
        "capabilities": {
            "batch": {"supported": True},
            "image_input": {"supported": True},
            "pdf_input": {"supported": True},
            "structured_outputs": {"supported": True},
        },
    }


def anthropic_models() -> Recording:
    first = [
        _anthropic_model("claude-opus-5-5", "Claude Opus 5.5"),
        _anthropic_model("example-claude-preview", "Example preview"),
    ]
    second = [_anthropic_model("claude-haiku-4-5", "Claude Haiku 4.5")]
    return _recording(
        "anthropic-discovery-models",
        "anthropic",
        (
            http("/v1/models", limit="1000"),
            200,
            {
                "data": first,
                "has_more": True,
                "first_id": "claude-opus-5-5",
                "last_id": "example-claude-preview",
            },
        ),
        (
            http("/v1/models", limit="1000", after_id="example-claude-preview"),
            200,
            {
                "data": second,
                "has_more": False,
                "first_id": "claude-haiku-4-5",
                "last_id": "claude-haiku-4-5",
            },
        ),
    )


def anthropic_empty() -> Recording:
    return _recording(
        "anthropic-discovery-empty",
        "anthropic",
        (
            http("/v1/models", limit="1000"),
            200,
            {"data": [], "has_more": False, "first_id": None, "last_id": None},
        ),
    )


# OpenAI-compatible Models API (OpenAI, X.AI, and the bedrock-mantle endpoint): one list.


def _openai_model(model_id: str, owner: str) -> dict[str, Any]:
    return {"id": model_id, "object": "model", "created": 1_790_000_000, "owned_by": owner}


def openai_models() -> Recording:
    return _recording(
        "openai-discovery-models",
        "openai_compat",
        (
            http("/v1/models"),
            200,
            {
                "object": "list",
                "data": [
                    _openai_model("gpt-5.6-sol", "openai"),
                    _openai_model("example-gpt-tuned", "example-org"),
                    _openai_model("text-embedding-3-small", "openai"),
                ],
            },
        ),
    )


def xai_models() -> Recording:
    return _recording(
        "xai-discovery-models",
        "openai_compat",
        (
            http("/v1/models"),
            200,
            {
                "object": "list",
                "data": [
                    _openai_model("grok-4.7", "xai"),
                    {**_openai_model("example-grok-preview", "xai"), "context_length": 131_072},
                ],
            },
        ),
    )


def openai_empty() -> Recording:
    return _recording(
        "openai-discovery-empty",
        "openai_compat",
        (http("/v1/models"), 200, {"object": "list", "data": []}),
    )


def mantle_models() -> Recording:
    return _recording(
        "mantle-discovery-models",
        "bedrock_mantle",
        (
            http("/v1/models"),
            200,
            {
                "object": "list",
                "data": [
                    _openai_model("anthropic.claude-opus-5-5", "anthropic"),
                    _openai_model("anthropic.claude-sonnet-5", "anthropic"),
                    _openai_model("anthropic.example-claude-preview", "anthropic"),
                    _openai_model("openai.gpt-oss-120b", "openai"),
                ],
            },
        ),
    )


def mantle_empty() -> Recording:
    return _recording(
        "mantle-discovery-empty",
        "bedrock_mantle",
        (http("/v1/models"), 200, {"object": "list", "data": []}),
    )


# Gemini: GET /v1beta/models, paged with pageToken while nextPageToken is given.


def _gemini_model(model_id: str, actions: list[str]) -> dict[str, Any]:
    return {
        "name": f"models/{model_id}",
        "baseModelId": model_id,
        "version": "001",
        "displayName": model_id,
        "inputTokenLimit": 1_048_576,
        "outputTokenLimit": 65_536,
        "supportedGenerationMethods": actions,
    }


def gemini_models() -> Recording:
    return _recording(
        "gemini-discovery-models",
        "gemini",
        (
            http("/v1beta/models", pageSize="1000"),
            200,
            {
                "models": [
                    _gemini_model("gemini-4-argon", ["generateContent", "countTokens"]),
                    _gemini_model("gemini-embedding-001", ["embedContent"]),
                ],
                "nextPageToken": "page-2-token",
            },
        ),
        (
            http("/v1beta/models", pageSize="1000", pageToken="page-2-token"),
            200,
            {
                "models": [
                    _gemini_model("example-gemini-preview", ["generateContent"]),
                    _gemini_model("example-image-model", ["predict"]),
                ]
            },
        ),
    )


def gemini_empty() -> Recording:
    return _recording(
        "gemini-discovery-empty",
        "gemini",
        (http("/v1beta/models", pageSize="1000"), 200, {}),
    )


RECORDINGS: dict[str, Callable[[], Recording]] = {
    "anthropic/discovery-models.json": anthropic_models,
    "anthropic/discovery-empty.json": anthropic_empty,
    "bedrock/discovery-listing.json": bedrock_listing,
    "bedrock/discovery-drift.json": bedrock_drift,
    "bedrock/discovery-empty.json": bedrock_empty,
    "bedrock_mantle/discovery-models.json": mantle_models,
    "bedrock_mantle/discovery-empty.json": mantle_empty,
    "gemini/discovery-models.json": gemini_models,
    "gemini/discovery-empty.json": gemini_empty,
    "openai_compat/discovery-openai.json": openai_models,
    "openai_compat/discovery-xai.json": xai_models,
    "openai_compat/discovery-empty.json": openai_empty,
    "ollama/discovery-tags.json": ollama_tags,
    "ollama/discovery-empty.json": ollama_empty,
}


def main() -> None:
    for filename, build in RECORDINGS.items():
        path = ROOT / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        save_recording(build(), path)


if __name__ == "__main__":
    main()
