"""Errors raised by boto3 and botocore, classified as conduit ``ProviderFailure``s.

The classification reads class names and the error code and HTTP status of a
``ClientError`` response, so it needs no botocore import. Provider text never leaves
this module: the adapter raises ``ProviderError(classify(error)) from None``.

- Throttling, service and model availability errors, connection failures and HTTP 429
  or 5xx: ``UNAVAILABLE``.
- Read, connect and model timeouts and HTTP 408: ``TIMEOUT``.
- Validation, access, missing resources, credentials and quota errors, and any other
  4xx: ``REJECTED``.
- Anything else: ``UNAVAILABLE``.
"""

from __future__ import annotations

from typing import Any

from conduit.core.errors import ProviderFailure

_TIMEOUT_CLASSES = frozenset({"ReadTimeoutError", "ConnectTimeoutError", "TimeoutError"})
_REJECTED_CLASSES = frozenset(
    {
        "NoCredentialsError",
        "PartialCredentialsError",
        "CredentialRetrievalError",
        "NoAuthTokenError",
        "NoRegionError",
        "ParamValidationError",
        "ProfileNotFound",
    }
)

# Error codes, lower case with any trailing "exception" removed: botocore reports
# "ThrottlingException" for a call and "throttlingException" for a stream event.
_UNAVAILABLE_CODES = frozenset(
    {
        "throttling",
        "serviceunavailable",
        "internalserver",
        "modelnotready",
        "modelerror",
        "modelstreamerror",
    }
)
_TIMEOUT_CODES = frozenset({"modeltimeout"})


def _code(error: BaseException) -> tuple[str | None, int | None]:
    response: Any = getattr(error, "response", None)
    if not isinstance(response, dict):
        return None, None
    code = response.get("Error", {}).get("Code")
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    normal = str(code).lower().removesuffix("exception") if code else None
    return normal, status if isinstance(status, int) else None


def classify(error: BaseException) -> ProviderFailure:
    """The failure class of an error from a Bedrock call."""
    names = {cls.__name__ for cls in type(error).__mro__}
    if names & _TIMEOUT_CLASSES:
        return ProviderFailure.TIMEOUT
    if names & _REJECTED_CLASSES:
        return ProviderFailure.REJECTED
    code, status = _code(error)
    if code in _TIMEOUT_CODES:
        return ProviderFailure.TIMEOUT
    if code in _UNAVAILABLE_CODES:
        return ProviderFailure.UNAVAILABLE
    if code is not None or status is not None:
        if status == 408:
            return ProviderFailure.TIMEOUT
        if status is not None and (status == 429 or status >= 500):
            return ProviderFailure.UNAVAILABLE
        return ProviderFailure.REJECTED
    # Connection failures (EndpointConnectionError, ConnectionClosedError) and anything
    # unrecognised.
    return ProviderFailure.UNAVAILABLE
