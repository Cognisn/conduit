"""Errors the reference stores raise."""

from __future__ import annotations

from conduit.core.errors import ConduitError

__all__ = ["StoreError"]


class StoreError(ConduitError):
    """A store could not complete an operation.

    The message names the store, the operation and the class of the underlying error
    only. It never carries a session id, a token key, stored content or token values,
    and the underlying error is not chained, because database drivers print statement
    parameters in their messages.
    """

    def __init__(self, store: str, operation: str, error_class: str | None = None) -> None:
        detail = f" ({error_class})" if error_class else ""
        super().__init__(f"the {store} could not {operation}{detail}")
        self.store = store
        self.operation = operation
        self.error_class = error_class
