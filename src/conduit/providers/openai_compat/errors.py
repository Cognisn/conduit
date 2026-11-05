"""Classify SDK errors as provider failures, without importing the SDK.

The openai SDK's errors are recognised by class name (anywhere in the class's
hierarchy), HTTP status and error code; azure-identity's credential errors by class
name. The provider raises ``ProviderError(classify(error)) from None``, so no provider
text reaches a log, an exception or an event.
"""

from __future__ import annotations

from conduit.core.errors import ProviderFailure

_TIMEOUT = frozenset({"APITimeoutError", "TimeoutError", "TimeoutException"})
_UNAVAILABLE = frozenset({"APIConnectionError", "RateLimitError", "InternalServerError"})
_REJECTED = frozenset(
    {
        "BadRequestError",
        "AuthenticationError",
        "PermissionDeniedError",
        "NotFoundError",
        "ConflictError",
        "UnprocessableEntityError",
        "ClientAuthenticationError",
        "CredentialUnavailableError",
    }
)
_REFUSED_CODES = frozenset({"content_filter"})


def _names(error: BaseException) -> set[str]:
    return {cls.__name__ for cls in type(error).__mro__}


def classify(error: BaseException) -> ProviderFailure:
    """The failure class of ``error``. Anything unrecognised counts as unavailable."""
    names = _names(error)
    code = getattr(error, "code", None)
    if isinstance(code, str) and code in _REFUSED_CODES:
        return ProviderFailure.REFUSED
    if names & _TIMEOUT:
        return ProviderFailure.TIMEOUT
    status = getattr(error, "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        if status == 408:
            return ProviderFailure.TIMEOUT
        if status == 429 or status >= 500:
            return ProviderFailure.UNAVAILABLE
        if 400 <= status < 500:
            return ProviderFailure.REJECTED
    if names & _UNAVAILABLE:
        return ProviderFailure.UNAVAILABLE
    if names & _REJECTED:
        return ProviderFailure.REJECTED
    return ProviderFailure.UNAVAILABLE
