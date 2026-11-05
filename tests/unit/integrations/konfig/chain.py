"""Assertions that an error carries no part of the exception it replaced."""

from __future__ import annotations

import traceback


def exception_chain(error: BaseException) -> list[BaseException]:
    """``error`` and every exception reachable through its cause and context."""
    seen: list[BaseException] = []
    pending: list[BaseException] = [error]
    while pending:
        current = pending.pop()
        if any(current is known for known in seen):
            continue
        seen.append(current)
        pending.extend(e for e in (current.__cause__, current.__context__) if e is not None)
    return seen


def assert_carries_nothing(error: BaseException, secret: str) -> None:
    """``error`` has no cause or context, and ``secret`` appears nowhere in it."""
    assert error.__cause__ is None
    assert error.__context__ is None
    for link in exception_chain(error):
        assert secret not in str(link)
        assert secret not in repr(link)
        assert secret not in repr(link.args)
    rendered = "".join(traceback.format_exception(error))
    assert secret not in rendered
