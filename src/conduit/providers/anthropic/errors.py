"""Classifying failures from the anthropic SDK without importing it.

The SDK is imported only inside the factory's ``create_client``, so errors are matched
by the names of their classes (and the HTTP status they carry). The SDK's messages can
echo request content, so only the failure class leaves this adapter.
"""

from __future__ import annotations

from conduit.core.errors import ProviderFailure

_TIMEOUT = frozenset({"APITimeoutError", "DeadlineExceededError"})
_UNAVAILABLE = frozenset(
    {
        "APIConnectionError",
        "RateLimitError",
        "InternalServerError",
        "OverloadedError",
        "ServiceUnavailableError",
    }
)
_SDK_BASES = frozenset({"AnthropicError", "APIError", "APIStatusError"})
_RETRYABLE_STATUS = frozenset({408, 409, 429})

#: Error types the Messages API reports inside a stream (``event: error``).
_UNAVAILABLE_STREAM_ERRORS = frozenset(
    {"overloaded_error", "api_error", "rate_limit_error", "timeout_error"}
)


def failure_for(error: BaseException) -> ProviderFailure:
    """The failure class for an exception raised by the SDK or by reading its stream."""
    names = {cls.__name__ for cls in type(error).__mro__}
    # APITimeoutError is a kind of APIConnectionError, so timeouts are checked first.
    if names & _TIMEOUT or isinstance(error, TimeoutError):
        return ProviderFailure.TIMEOUT
    if names & _UNAVAILABLE:
        return ProviderFailure.UNAVAILABLE
    status = getattr(error, "status_code", None)
    if isinstance(status, int) and (status >= 500 or status in _RETRYABLE_STATUS):
        return ProviderFailure.UNAVAILABLE
    if names & _SDK_BASES or isinstance(status, int):
        return ProviderFailure.REJECTED
    # Anything else came from reading a stream conduit could not make sense of.
    return ProviderFailure.PROTOCOL


def failure_for_stream_error(error_type: object) -> ProviderFailure:
    """The failure class for an ``error`` event inside a stream."""
    if error_type in _UNAVAILABLE_STREAM_ERRORS:
        return ProviderFailure.UNAVAILABLE
    return ProviderFailure.REJECTED
