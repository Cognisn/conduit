"""Model provider adapters, each SDK behind its own extra, plus the stub provider."""

# Client construction
# -------------------
# Every adapter's ``create_client`` (and the local embedding adapter's) resolves its
# credential and constructs its SDK or HTTP client in one guarded step, run through
# ``conduit.core.registry.guarded_client``; the registry calls every ``create_client``
# under the same guard, so a product's own factory is covered too.
#
# What the guard promises:
#
# - A constructor (or anything else in the step) that raises surfaces as a fixed
#   ``ConfigurationError`` naming the adapter, and the configured provider when the
#   registry builds it, and the exception's class only. Its cause and context are both
#   ``None``.
# - The credential is resolved only inside the step's own frames, which have ended when
#   the error is raised, so a traceback printed with its locals (``pytest -l``, or a
#   product that logs frame locals) holds no resolved key, token or secret key. This
#   covers ordinary exceptions only: cancellation, ``KeyboardInterrupt`` or another
#   ``BaseException`` arriving during construction propagates unchanged (D0040), still
#   carrying the SDK constructor's frames.
# - A conduit error the step raises on purpose (a missing extra, an unresolved secret, a
#   refused environment variable, a short-term key that cannot be presigned) goes on as
#   it is, with its traceback, cause, context and notes cleared.
# - The guard itself logs nothing. Cancellation and other ``BaseException`` propagate
#   unchanged.
#
# What it cannot promise: anything a library writes before it raises. An SDK that logs
# its own arguments at DEBUG does so before the guard sees the failure; botocore's SigV4
# signer, for one, logs the canonical request (with the access key id and any session
# token) when a short-term Bedrock key is presigned (see
# ``conduit.providers.bedrock.auth``). Products keep SDK loggers above DEBUG in
# production.

from __future__ import annotations

from conduit.core.registry import ProviderFactory
from conduit.providers.anthropic import AnthropicFactory
from conduit.providers.bedrock import BedrockFactory
from conduit.providers.bedrock_mantle import BedrockMantleFactory
from conduit.providers.gemini import GeminiFactory
from conduit.providers.ollama import OllamaFactory
from conduit.providers.openai_compat import OpenAICompatFactory
from conduit.providers.stub import StubFactory

__all__ = ["builtin_factories"]


def builtin_factories() -> list[ProviderFactory]:
    """A factory for every adapter conduit ships.

    Each adapter imports its SDK only inside ``create_client``, so building this list
    never needs an extra installed.
    """
    return [
        StubFactory(),
        BedrockFactory(),
        BedrockMantleFactory(),
        AnthropicFactory(),
        OpenAICompatFactory(),
        GeminiFactory(),
        OllamaFactory(),
    ]
