"""Errors from boto3 and botocore mapped to ProviderFailure, with no provider text.

The classes here stand in for botocore's by name and shape (a ``response`` with an
``Error.Code`` and an HTTP status), so the mapping is tested without botocore.
"""

from __future__ import annotations

from typing import Any

import pytest

from conduit.core.errors import ProviderFailure
from conduit.providers.bedrock.errors import classify


class ClientError(Exception):
    def __init__(self, code: str, status: int = 400, message: str = "provider text") -> None:
        super().__init__(f"{code}: {message}")
        self.response: dict[str, Any] = {
            "Error": {"Code": code, "Message": message},
            "ResponseMetadata": {"HTTPStatusCode": status},
        }


class EventStreamError(ClientError):
    """botocore raises this, a ClientError, for an exception event inside a stream."""


def named(name: str, *bases: type[BaseException]) -> BaseException:
    error: BaseException = type(name, bases or (Exception,), {})("provider text")
    return error


@pytest.mark.parametrize(
    ("error", "failure"),
    [
        (ClientError("ThrottlingException", 429), ProviderFailure.UNAVAILABLE),
        (ClientError("ServiceUnavailableException", 503), ProviderFailure.UNAVAILABLE),
        (ClientError("InternalServerException", 500), ProviderFailure.UNAVAILABLE),
        (ClientError("ModelNotReadyException", 429), ProviderFailure.UNAVAILABLE),
        (ClientError("ModelErrorException", 424), ProviderFailure.UNAVAILABLE),
        (EventStreamError("throttlingException"), ProviderFailure.UNAVAILABLE),
        (EventStreamError("modelStreamErrorException"), ProviderFailure.UNAVAILABLE),
        (EventStreamError("internalServerException"), ProviderFailure.UNAVAILABLE),
        (ClientError("ModelTimeoutException", 408), ProviderFailure.TIMEOUT),
        (ClientError("ValidationException", 400), ProviderFailure.REJECTED),
        (EventStreamError("validationException"), ProviderFailure.REJECTED),
        (ClientError("AccessDeniedException", 403), ProviderFailure.REJECTED),
        (ClientError("ResourceNotFoundException", 404), ProviderFailure.REJECTED),
        (ClientError("UnrecognizedClientException", 403), ProviderFailure.REJECTED),
        (ClientError("ExpiredTokenException", 403), ProviderFailure.REJECTED),
        (ClientError("ServiceQuotaExceededException", 400), ProviderFailure.REJECTED),
        (ClientError("SomethingNew", 502), ProviderFailure.UNAVAILABLE),
        (ClientError("SomethingNew", 429), ProviderFailure.UNAVAILABLE),
        (ClientError("SomethingNew", 408), ProviderFailure.TIMEOUT),
        (ClientError("SomethingNew", 400), ProviderFailure.REJECTED),
        (named("ReadTimeoutError"), ProviderFailure.TIMEOUT),
        (named("ConnectTimeoutError"), ProviderFailure.TIMEOUT),
        (TimeoutError(), ProviderFailure.TIMEOUT),
        (named("EndpointConnectionError"), ProviderFailure.UNAVAILABLE),
        (named("ConnectionClosedError"), ProviderFailure.UNAVAILABLE),
        (ConnectionResetError(), ProviderFailure.UNAVAILABLE),
        (named("NoCredentialsError"), ProviderFailure.REJECTED),
        (named("PartialCredentialsError"), ProviderFailure.REJECTED),
        (named("NoRegionError"), ProviderFailure.REJECTED),
        (named("ParamValidationError"), ProviderFailure.REJECTED),
        (RuntimeError("anything else"), ProviderFailure.UNAVAILABLE),
    ],
    ids=lambda value: (
        value.value
        if isinstance(value, ProviderFailure)
        else type(value).__name__
        + ":"
        + getattr(value, "response", {}).get("Error", {}).get("Code", "")
    ),
)
def test_classification(error: BaseException, failure: ProviderFailure) -> None:
    assert classify(error) is failure


def test_a_timeout_subclass_wins_over_a_connection_base() -> None:
    # botocore's ConnectTimeoutError derives from its ConnectionError.
    connection = named("ConnectionError")
    timeout = type("ConnectTimeoutError", (type(connection),), {})("provider text")
    assert classify(timeout) is ProviderFailure.TIMEOUT
