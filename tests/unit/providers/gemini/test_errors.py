"""Gemini errors map to provider failures, by status code and by transport error."""

from __future__ import annotations

import pytest

from conduit.core.errors import ProviderFailure
from conduit.providers.gemini.errors import classify, failure_for_status


@pytest.mark.parametrize(
    ("status", "failure"),
    [
        (400, ProviderFailure.REJECTED),
        (401, ProviderFailure.REJECTED),
        (403, ProviderFailure.REJECTED),
        (404, ProviderFailure.REJECTED),
        (408, ProviderFailure.TIMEOUT),
        (429, ProviderFailure.UNAVAILABLE),
        (500, ProviderFailure.UNAVAILABLE),
        (503, ProviderFailure.UNAVAILABLE),
        (504, ProviderFailure.TIMEOUT),
    ],
)
def test_status_codes(status: int, failure: ProviderFailure) -> None:
    assert failure_for_status(status) is failure


def test_a_timeout_is_a_timeout() -> None:
    assert classify(TimeoutError()) is ProviderFailure.TIMEOUT


@pytest.mark.parametrize("error", [RuntimeError("secret detail"), ValueError("secret detail")])
def test_anything_else_is_unavailable(error: Exception) -> None:
    assert classify(error) is ProviderFailure.UNAVAILABLE


class TestWithTheSdk:
    @pytest.fixture(autouse=True)
    def _sdk(self) -> None:
        pytest.importorskip("google.genai", reason="needs the gemini extra")

    @pytest.mark.parametrize(
        ("status", "failure"),
        [
            (400, ProviderFailure.REJECTED),
            (403, ProviderFailure.REJECTED),
            (429, ProviderFailure.UNAVAILABLE),
            (500, ProviderFailure.UNAVAILABLE),
            (504, ProviderFailure.TIMEOUT),
        ],
    )
    def test_api_errors_by_code(self, status: int, failure: ProviderFailure) -> None:
        errors = pytest.importorskip("google.genai.errors")
        kind = errors.ClientError if status < 500 else errors.ServerError
        error = kind(status, {"error": {"code": status, "message": "secret detail"}})
        assert classify(error) is failure

    @pytest.mark.parametrize(
        ("name", "failure"),
        [
            ("ReadTimeout", ProviderFailure.TIMEOUT),
            ("ConnectTimeout", ProviderFailure.TIMEOUT),
            ("ConnectError", ProviderFailure.UNAVAILABLE),
            ("RemoteProtocolError", ProviderFailure.UNAVAILABLE),
        ],
    )
    def test_transport_errors(self, name: str, failure: ProviderFailure) -> None:
        httpx = pytest.importorskip("httpx")
        assert classify(getattr(httpx, name)("secret detail")) is failure
