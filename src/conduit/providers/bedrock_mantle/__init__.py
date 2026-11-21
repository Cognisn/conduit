"""The bedrock-mantle adapter: single-Region Claude through Amazon Bedrock's mantle endpoint.

Claude is called through the Anthropic Messages API at
``https://bedrock-mantle.{region}.api.aws/anthropic/v1/messages``, with Bedrock API keys
sent as bearer tokens. The request and stream mapping is the anthropic adapter's. Install
with the ``bedrock-mantle`` extra (anthropic and boto3); the SDKs are imported only
when a client is built.
"""

from __future__ import annotations

from conduit.providers.bedrock_mantle.factory import BedrockMantleFactory
from conduit.providers.bedrock_mantle.provider import BedrockMantleProvider
from conduit.providers.bedrock_mantle.settings import (
    MANTLE_BASE_URL,
    BedrockMantleSettings,
    MantleAuth,
    mantle_base_url,
)

__all__ = [
    "MANTLE_BASE_URL",
    "BedrockMantleFactory",
    "BedrockMantleProvider",
    "BedrockMantleSettings",
    "MantleAuth",
    "mantle_base_url",
]
