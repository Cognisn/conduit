"""Error classification: SDK errors become provider failures, with no provider text."""

from __future__ import annotations

from typing import Any

import pytest

from conduit.core.errors import ProviderFailure
from conduit.providers.openai_compat.errors import classify


def _error(name: str, base: type[Exception] = Exception, **attributes: Any) -> Exception:
    """An exception named like the SDK's, without importing the SDK."""
    error: Exception = type(name, (base,), {})("provider text that must never be repeated")
    for key, value in attributes.items():
        setattr(error, key, value)
    return error


APIError = type("APIError", (Exception,), {})
APIConnectionError = type("APIConnectionError", (APIError,), {})
APIStatusError = type("APIStatusError", (APIError,), {})


@pytest.mark.parametrize(
    ("error", "failure"),
    [
        (_error("APITimeoutError", APIConnectionError), ProviderFailure.TIMEOUT),
        (_error("APIConnectionError", APIError), ProviderFailure.UNAVAILABLE),
        (_error("RateLimitError", APIStatusError, status_code=429), ProviderFailure.UNAVAILABLE),
        (
            _error("InternalServerError", APIStatusError, status_code=503),
            ProviderFailure.UNAVAILABLE,
        ),
        (_error("BadRequestError", APIStatusError, status_code=400), ProviderFailure.REJECTED),
        (_error("AuthenticationError", APIStatusError, status_code=401), ProviderFailure.REJECTED),
        (
            _error("PermissionDeniedError", APIStatusError, status_code=403),
            ProviderFailure.REJECTED,
        ),
        (_error("NotFoundError", APIStatusError, status_code=404), ProviderFailure.REJECTED),
        (
            _error("UnprocessableEntityError", APIStatusError, status_code=422),
            ProviderFailure.REJECTED,
        ),
        (_error("APIStatusError", APIError, status_code=408), ProviderFailure.TIMEOUT),
        (_error("APIStatusError", APIError, status_code=529), ProviderFailure.UNAVAILABLE),
        (_error("APIStatusError", APIError, status_code=418), ProviderFailure.REJECTED),
        (
            _error("BadRequestError", APIStatusError, status_code=400, code="content_filter"),
            ProviderFailure.REFUSED,
        ),
        (_error("ClientAuthenticationError"), ProviderFailure.REJECTED),
        (_error("CredentialUnavailableError"), ProviderFailure.REJECTED),
        (_error("APIError"), ProviderFailure.UNAVAILABLE),  # an error event mid-stream
        (TimeoutError(), ProviderFailure.TIMEOUT),
        (RuntimeError("anything else"), ProviderFailure.UNAVAILABLE),
    ],
    ids=lambda value: type(value).__name__ if isinstance(value, Exception) else str(value),
)
def test_classify(error: Exception, failure: ProviderFailure) -> None:
    assert classify(error) is failure


def test_a_status_code_that_is_not_an_int_is_ignored() -> None:
    error = _error("APIStatusError", APIError, status_code="500")
    assert classify(error) is ProviderFailure.UNAVAILABLE


def test_classes_from_the_sdk() -> None:
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx2", reason="the openai SDK's HTTP client")
    request = httpx.Request("POST", "https://api.example.com/v1/chat/completions")

    def status(cls: type[Exception], code: int, body: Any = None) -> Exception:
        response = httpx.Response(code, request=request)
        return cls("text", response=response, body=body)  # type: ignore[call-arg]

    assert classify(openai.APITimeoutError(request=request)) is ProviderFailure.TIMEOUT
    assert classify(openai.APIConnectionError(request=request)) is ProviderFailure.UNAVAILABLE
    assert classify(status(openai.RateLimitError, 429)) is ProviderFailure.UNAVAILABLE
    assert classify(status(openai.InternalServerError, 500)) is ProviderFailure.UNAVAILABLE
    assert classify(status(openai.AuthenticationError, 401)) is ProviderFailure.REJECTED
    assert classify(status(openai.NotFoundError, 404)) is ProviderFailure.REJECTED
    filtered = status(openai.BadRequestError, 400, {"code": "content_filter", "message": "x"})
    assert classify(filtered) is ProviderFailure.REFUSED
