"""Gemini errors as provider failures.

google-genai raises ``APIError`` (``ClientError`` for 4xx, ``ServerError`` for 5xx)
with the HTTP status in ``code``, and lets its HTTP library's transport errors
through. Neither the SDK nor the HTTP library is imported here: errors are recognised
by their ``code`` and by class name, so this module loads without the gemini extra
(and the gemini layer never imports httpx, which belongs to other layers).
"""

from __future__ import annotations

from conduit.core.errors import ProviderFailure

_SDK_MODULE = "google.genai"
_HTTP_MODULES = ("httpx", "httpx2", "httpcore", "requests", "urllib3", "aiohttp")
_TIMEOUT_NAMES = frozenset({"TimeoutException", "Timeout", "TimeoutError"})


def failure_for_status(status: int) -> ProviderFailure:
    """The failure class for an HTTP error status from the Gemini API."""
    if status in (408, 504):
        return ProviderFailure.TIMEOUT
    if status == 429 or status >= 500:
        return ProviderFailure.UNAVAILABLE
    return ProviderFailure.REJECTED


def _from(error: BaseException, modules: tuple[str, ...]) -> bool:
    return type(error).__module__.split(".")[0] in {m.split(".")[0] for m in modules}


def classify(error: BaseException) -> ProviderFailure:
    """The failure for an error raised by a Gemini call. Never reads the error text."""
    if isinstance(error, TimeoutError):
        return ProviderFailure.TIMEOUT
    code = getattr(error, "code", None)
    if (
        type(error).__module__.startswith(_SDK_MODULE)
        and isinstance(code, int)
        and not isinstance(code, bool)
    ):
        return failure_for_status(code)
    if _from(error, _HTTP_MODULES) and any(
        cls.__name__ in _TIMEOUT_NAMES for cls in type(error).__mro__
    ):
        return ProviderFailure.TIMEOUT
    return ProviderFailure.UNAVAILABLE
