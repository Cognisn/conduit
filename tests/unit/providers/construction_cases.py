"""Shared cases for the guarded client step: every adapter, chat and embedding.

Each case replaces one SDK (or HTTP client) constructor with ``RaisingClient``, whose
exception quotes everything it was given, so a key that reached it would be in the
exception's text and in its frame's locals. The provider is then built through the
registry with ``PLACEHOLDER`` as every resolved secret. The placeholder is obviously
fake and is never a real credential. Nothing here reaches a network.

``tests/unit/providers/test_guarded_construction.py`` and
``tests/unit/embeddings/test_guarded_construction.py`` run the cases in process; the
first also runs them all in a child pytest under ``-vvv -l --tb=long``.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import traceback
import types
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from conduit.core.errors import ConfigurationError
from conduit.core.registry import ProviderRegistry
from conduit.core.secrets import LiteralSecretResolver, resolve_secret_fields
from conduit.embeddings import (
    BedrockEmbeddingFactory,
    GeminiEmbeddingFactory,
    LocalEmbeddingFactory,
    OllamaEmbeddingFactory,
    OpenAICompatEmbeddingFactory,
)
from conduit.providers.anthropic import AnthropicFactory
from conduit.providers.bedrock import BedrockFactory, auth
from conduit.providers.bedrock_mantle import BedrockMantleFactory
from conduit.providers.gemini import GeminiFactory
from conduit.providers.ollama import OllamaFactory
from conduit.providers.openai_compat import OpenAICompatFactory

#: Every resolved secret in these cases. Obviously fake; never a real credential.
PLACEHOLDER = "conduit-placeholder-not-a-credential-5e7a"
#: Text the raising constructor puts in its own exception, as an SDK might.
SDK_TEXT = "synthetic-sdk-text-c41d"
#: Reference names resolving to the placeholder: every API key, token and secret key.
REFS = ("key", "secret", "hub")
#: An AWS access key id and session token go into the canonical request botocore's
#: signer logs at DEBUG when the presign step signs a short-term key, which no guard
#: around construction can change (see ``conduit.providers.bedrock.auth``). They get
#: their own obviously fake values, so the placeholder test stays about the guard.
ACCESS_KEY_ID = "AKIAEXAMPLESYNTHETIC"
SESSION_TOKEN = "synthetic-session-token-not-real"
SECRETS = LiteralSecretResolver(
    dict.fromkeys(REFS, PLACEHOLDER) | {"access": ACCESS_KEY_ID, "token": SESSION_TOKEN}
)
#: The loggers a test watches at DEBUG: the root, conduit's and every SDK's.
LOGGERS = (
    "",
    "conduit",
    "anthropic",
    "openai",
    "httpx",
    "httpcore",
    "boto3",
    "botocore",
    "google_genai",
    "google",
    "azure",
    "sentence_transformers",
    "urllib3",
)
FOUNDRY_OPENAI = "https://example-resource.openai.azure.example.com/openai/v1/"
FOUNDRY_ANTHROPIC = "https://example-resource.services.ai.azure.example.com/anthropic/"
ACCESS_KEYS = {
    "mode": "access_keys",
    "access_key_id": "secret://access",
    "secret_access_key": "secret://secret",
    "session_token": "secret://token",
}


#: The short-term Bedrock keys presigned in the current case, in this process.
ISSUED: list[str] = []
#: Set by the parent of a child pytest run: the file (in the parent's pytest tmp path,
#: never the repository) the child appends each key it presigns to, one per line.
ISSUED_FILE_VARIABLE = "CONDUIT_TEST_ISSUED_KEYS_FILE"
#: How much of the end of a key's body is looked for on its own.
BODY_TAIL = 24


def record_issued_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Record every short-term key the presign step makes, so a test can look for it.

    In a child run, each key is also appended to the file the parent named, so the
    parent can look for it in the child's output.
    """
    ISSUED.clear()
    encode = auth.encode_short_term_key
    shared = os.environ.get(ISSUED_FILE_VARIABLE)

    def recorded(presigned_url: str) -> str:
        key = encode(presigned_url)
        ISSUED.append(key)
        if shared:
            with open(shared, "a", encoding="utf-8", newline="\n") as found:
                found.write(key + "\n")
        return key

    monkeypatch.setattr(auth, "encode_short_term_key", recorded)


class SyntheticConstructorFailure(RuntimeError):
    """What ``RaisingClient`` raises."""


class RaisingClient:
    """Stands in for an SDK client constructor (or a class an adapter subclasses).

    Its exception quotes every argument, and its frame holds them as locals.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise SyntheticConstructorFailure(f"{SDK_TEXT}: refused {args!r} {kwargs!r}")


@dataclass(frozen=True)
class Case:
    """One adapter and authentication mode, with the constructor that will raise."""

    id: str
    kind: str
    settings: dict[str, Any]
    #: ``(module, attribute)`` replaced by ``RaisingClient``.
    target: tuple[str, str]
    factory: Callable[[], Any]
    embedding: bool = False
    #: The modules the case needs installed (its extra).
    needs: tuple[str, ...] = field(default=())
    #: Modules replaced by empty stand-ins, so the case runs where the library cannot
    #: load (torch's native libraries, for one).
    stand_ins: tuple[str, ...] = field(default=())
    #: Whether a short-term Bedrock key is presigned before the constructor raises.
    presigns: bool = False

    def patch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for module in self.needs:
            pytest.importorskip(module)
        for module in self.stand_ins:
            monkeypatch.setitem(sys.modules, module, types.ModuleType(module))
        record_issued_keys(monkeypatch)
        owner = importlib.import_module(self.target[0])
        monkeypatch.setattr(owner, self.target[1], RaisingClient)

    def registry(self) -> ProviderRegistry:
        if self.embedding:
            return ProviderRegistry(secrets=SECRETS, embedding_factories=[self.factory()])
        return ProviderRegistry(secrets=SECRETS, factories=[self.factory()])

    async def build(self) -> Any:
        """Build through the registry, as ``start`` does; expected to raise."""
        registry = self.registry()
        if self.embedding:
            return await registry.build_embedder("primary", self.settings)
        return await registry.build("primary", self.settings)

    async def create_client(self) -> Any:
        """Call the factory's ``create_client`` directly; expected to raise."""
        registry = self.registry()
        if self.embedding:
            validated = registry.validate_embedding("primary", self.settings)
        else:
            validated = registry.validate("primary", self.settings)
        secrets = await resolve_secret_fields(validated, SECRETS)
        return self.factory().create_client(validated, secrets)


ANTHROPIC = ("anthropic",)
OPENAI = ("openai",)
AZURE = ("openai", "azure.identity.aio")
BOTO = ("boto3", "botocore")
GENAI = ("google.genai",)
HTTPX = ("httpx",)
MANTLE = ("anthropic", "botocore")
LOCAL = ("sentence_transformers",)
CREDENTIAL = ("azure.identity.aio", "DefaultAzureCredential")
SESSION = ("boto3.session", "Session")
#: The client's configuration, built after any short-term key has been presigned.
CONFIG = ("botocore.config", "Config")
SHORT_TERM = {
    "kind": "bedrock",
    "region": "eu-west-2",
    "auth": {"mode": "short_term_api_key", "source": ACCESS_KEYS},
}

CHAT_CASES: tuple[Case, ...] = (
    Case(
        "anthropic-direct",
        "anthropic",
        {"kind": "anthropic", "api_key": "secret://key"},
        ("anthropic", "AsyncAnthropic"),
        AnthropicFactory,
        needs=ANTHROPIC,
    ),
    Case(
        "anthropic-foundry-key",
        "anthropic",
        {
            "kind": "anthropic",
            "endpoint": "foundry",
            "api_key": "secret://key",
            "base_url": FOUNDRY_ANTHROPIC,
        },
        ("anthropic", "AsyncAnthropicFoundry"),
        AnthropicFactory,
        needs=ANTHROPIC,
    ),
    Case(
        "anthropic-foundry-entra-id",
        "anthropic",
        {"kind": "anthropic", "endpoint": "foundry", "entra_id": {}, "base_url": FOUNDRY_ANTHROPIC},
        CREDENTIAL,
        AnthropicFactory,
        needs=("anthropic", "azure.identity.aio"),
    ),
    Case(
        "openai-key",
        "openai_compat",
        {"kind": "openai_compat", "service": {"name": "openai", "api_key": "secret://key"}},
        ("openai", "AsyncOpenAI"),
        OpenAICompatFactory,
        needs=OPENAI,
    ),
    Case(
        "xai-key",
        "openai_compat",
        {"kind": "openai_compat", "service": {"name": "xai", "api_key": "secret://key"}},
        ("openai", "AsyncOpenAI"),
        OpenAICompatFactory,
        needs=OPENAI,
    ),
    Case(
        "foundry-key",
        "openai_compat",
        {
            "kind": "openai_compat",
            "service": {
                "name": "foundry",
                "endpoint": FOUNDRY_OPENAI,
                "auth": {"mode": "api_key", "api_key": "secret://key"},
            },
        },
        ("openai", "AsyncOpenAI"),
        OpenAICompatFactory,
        needs=OPENAI,
    ),
    Case(
        "foundry-entra-id",
        "openai_compat",
        {
            "kind": "openai_compat",
            "service": {
                "name": "foundry",
                "endpoint": FOUNDRY_OPENAI,
                "auth": {"mode": "entra_id"},
            },
        },
        CREDENTIAL,
        OpenAICompatFactory,
        needs=AZURE,
    ),
    Case(
        "gemini",
        "gemini",
        {"kind": "gemini", "api_key": "secret://key"},
        ("google.genai", "Client"),
        GeminiFactory,
        needs=GENAI,
    ),
    Case(
        "ollama",
        "ollama",
        {"kind": "ollama"},
        ("httpx", "AsyncClient"),
        OllamaFactory,
        needs=HTTPX,
    ),
    Case(
        "bedrock-credential-chain",
        "bedrock",
        {"kind": "bedrock", "region": "eu-west-2"},
        SESSION,
        BedrockFactory,
        needs=BOTO,
    ),
    Case(
        "bedrock-access-keys",
        "bedrock",
        {"kind": "bedrock", "region": "eu-west-2", "auth": ACCESS_KEYS},
        SESSION,
        BedrockFactory,
        needs=BOTO,
    ),
    Case(
        "bedrock-short-term-key",
        "bedrock",
        {
            "kind": "bedrock",
            "region": "eu-west-2",
            "auth": {"mode": "short_term_api_key", "source": ACCESS_KEYS},
        },
        SESSION,
        BedrockFactory,
        needs=BOTO,
    ),
    Case(
        "bedrock-short-term-key-presigned",
        "bedrock",
        SHORT_TERM,
        CONFIG,
        BedrockFactory,
        needs=BOTO,
        presigns=True,
    ),
    Case(
        "bedrock-long-term-key",
        "bedrock",
        {
            "kind": "bedrock",
            "region": "eu-west-2",
            "allow_long_term_api_keys": True,
            "auth": {"mode": "long_term_api_key", "api_key": "secret://key"},
        },
        SESSION,
        BedrockFactory,
        needs=BOTO,
    ),
    Case(
        "bedrock-mantle-short-term-key",
        "bedrock_mantle",
        {
            "kind": "bedrock_mantle",
            "region": "eu-west-2",
            "auth": {"mode": "short_term_api_key", "source": ACCESS_KEYS},
        },
        ("anthropic.lib.bedrock", "AsyncAnthropicBedrockMantle"),
        BedrockMantleFactory,
        needs=MANTLE,
        presigns=True,
    ),
    Case(
        "bedrock-mantle-long-term-key",
        "bedrock_mantle",
        {
            "kind": "bedrock_mantle",
            "region": "eu-west-2",
            "allow_long_term_api_keys": True,
            "auth": {"mode": "long_term_api_key", "api_key": "secret://key"},
        },
        ("anthropic.lib.bedrock", "AsyncAnthropicBedrockMantle"),
        BedrockMantleFactory,
        needs=MANTLE,
    ),
)

EMBEDDING_CASES: tuple[Case, ...] = (
    Case(
        "embed-bedrock-access-keys",
        "bedrock",
        {"kind": "bedrock", "region": "eu-west-2", "auth": ACCESS_KEYS},
        SESSION,
        BedrockEmbeddingFactory,
        embedding=True,
        needs=BOTO,
    ),
    Case(
        "embed-bedrock-short-term-key",
        "bedrock",
        {
            "kind": "bedrock",
            "region": "eu-west-2",
            "auth": {"mode": "short_term_api_key", "source": ACCESS_KEYS},
        },
        SESSION,
        BedrockEmbeddingFactory,
        embedding=True,
        needs=BOTO,
    ),
    Case(
        "embed-bedrock-short-term-key-presigned",
        "bedrock",
        SHORT_TERM,
        CONFIG,
        BedrockEmbeddingFactory,
        embedding=True,
        needs=BOTO,
        presigns=True,
    ),
    Case(
        "embed-openai-key",
        "openai_compat",
        {"kind": "openai_compat", "service": {"name": "openai", "api_key": "secret://key"}},
        ("openai", "AsyncOpenAI"),
        OpenAICompatEmbeddingFactory,
        embedding=True,
        needs=OPENAI,
    ),
    Case(
        "embed-foundry-key",
        "openai_compat",
        {
            "kind": "openai_compat",
            "service": {
                "name": "foundry",
                "endpoint": FOUNDRY_OPENAI,
                "auth": {"mode": "api_key", "api_key": "secret://key"},
            },
        },
        ("openai", "AsyncOpenAI"),
        OpenAICompatEmbeddingFactory,
        embedding=True,
        needs=OPENAI,
    ),
    Case(
        "embed-foundry-entra-id",
        "openai_compat",
        {
            "kind": "openai_compat",
            "service": {
                "name": "foundry",
                "endpoint": FOUNDRY_OPENAI,
                "auth": {"mode": "entra_id"},
            },
        },
        CREDENTIAL,
        OpenAICompatEmbeddingFactory,
        embedding=True,
        needs=AZURE,
    ),
    Case(
        "embed-gemini",
        "gemini",
        {"kind": "gemini", "api_key": "secret://key"},
        ("google.genai", "Client"),
        GeminiEmbeddingFactory,
        embedding=True,
        needs=GENAI,
    ),
    Case(
        "embed-ollama",
        "ollama",
        {"kind": "ollama"},
        ("httpx", "AsyncClient"),
        OllamaEmbeddingFactory,
        embedding=True,
        needs=HTTPX,
    ),
    Case(
        "embed-local-hub-token",
        "sentence_transformers",
        {"kind": "sentence_transformers", "hub_token": "secret://hub"},
        ("conduit.embeddings.local.factory", "LocalModels"),
        LocalEmbeddingFactory,
        embedding=True,
        stand_ins=LOCAL,
    ),
)

ALL_CASES = CHAT_CASES + EMBEDDING_CASES


def ids(cases: tuple[Case, ...]) -> list[str]:
    return [case.id for case in cases]


def key_marks(key: str) -> tuple[str, str]:
    """What betrays ``key``: the key whole, and the end of its body on its own."""
    body = key.removeprefix(auth.SHORT_TERM_PREFIX)
    return key, body[-BODY_TAIL:]


def leaks(text: str, keys: Sequence[str] | None = None) -> bool:
    """Whether ``text`` holds the placeholder, the SDK's own text, or a short-term key:
    any, by its prefix, and each of ``keys`` (by default those presigned in this
    process) whole or by the end of its body alone."""
    if PLACEHOLDER in text or SDK_TEXT in text or auth.SHORT_TERM_PREFIX in text:
        return True
    found = ISSUED if keys is None else keys
    return any(mark in text for key in found for mark in key_marks(key))


def watch_loggers(caplog: pytest.LogCaptureFixture) -> None:
    for name in LOGGERS:
        caplog.set_level(logging.DEBUG, logger=name)


def logged(caplog: pytest.LogCaptureFixture) -> str:
    """Every captured record, message and arguments, as one text."""
    parts: list[str] = []
    for record in caplog.records:
        parts.append(record.getMessage())
        parts.append(repr(record.args))
        parts.append(repr(record.__dict__))
    return "\n".join(parts)


async def failure_of(step: Awaitable[Any]) -> Exception:
    """What ``step`` raised; the checks, not ``pytest.raises``, say what was wrong."""
    try:
        await step
    except Exception as error:
        return error
    raise AssertionError("the construction did not fail")


def rendered(error: BaseException) -> str:
    """The error as a long traceback with every frame's locals, and its chain."""
    found = traceback.TracebackException.from_exception(error, capture_locals=True)
    return "".join(found.format())


def frame_locals(error: BaseException) -> str:
    """repr of every local of every frame the error's traceback holds."""
    parts: list[str] = []
    tb = error.__traceback__
    while tb is not None:
        parts.extend(f"{name}={value!r}" for name, value in tb.tb_frame.f_locals.items())
        tb = tb.tb_next
    return "\n".join(parts)


def check_hidden(case: Case, error: BaseException, caplog: pytest.LogCaptureFixture) -> None:
    """The fixed refusal, and the key nowhere: not in a traceback, a frame or a log."""
    check_refusal(case, error)
    if case.presigns:
        assert ISSUED, "no short-term key was presigned before the constructor raised"
    assert not leaks(rendered(error)), "the key reached a traceback with locals"
    assert not leaks(frame_locals(error)), "a frame the error holds keeps the key"
    assert not leaks(logged(caplog)), "the key reached a log record"


def check_refusal(case: Case, error: BaseException) -> None:
    """The fixed refusal: provider and exception class only, nothing chained."""
    assert type(error) is ConfigurationError
    assert error.__cause__ is None
    assert error.__context__ is None
    assert case.kind in str(error)
    assert SyntheticConstructorFailure.__name__ in str(error)
    for text in (str(error), repr(error), repr(error.args)):
        assert not leaks(text)
