"""The Amazon Bedrock adapter: Converse and ConverseStream over boto3 (the bedrock extra).

boto3 and botocore are imported only inside ``BedrockFactory.create_client``, so
importing this package, and building ``conduit.providers.builtin_factories()``, never
needs the extra installed.
"""

from __future__ import annotations

from conduit.providers.bedrock.factory import BedrockFactory
from conduit.providers.bedrock.provider import BedrockProvider
from conduit.providers.bedrock.settings import (
    AccessKeyAuth,
    BedrockAuth,
    BedrockSettings,
    CredentialChainAuth,
    LongTermApiKeyAuth,
    ShortTermApiKeyAuth,
)

__all__ = [
    "AccessKeyAuth",
    "BedrockAuth",
    "BedrockFactory",
    "BedrockProvider",
    "BedrockSettings",
    "CredentialChainAuth",
    "LongTermApiKeyAuth",
    "ShortTermApiKeyAuth",
]
