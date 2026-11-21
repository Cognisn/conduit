"""The wave-wide rule for conduit errors: nothing of the failing exception is attached."""

from __future__ import annotations


def assert_nothing_attached(error: BaseException, *secrets: str) -> None:
    """``error`` has no cause and no context, and no secret is in its text or args."""
    assert error.__cause__ is None, "the original exception is the cause"
    assert error.__context__ is None, "the original exception is the context"
    texts = [str(error), repr(error), *(repr(arg) for arg in error.args)]
    for secret in secrets:
        assert not any(secret in text for text in texts), f"{secret!r} is in the error"
