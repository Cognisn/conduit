"""Conformance suites for product-supplied backends, and the pytest plugin that runs them.

Every backend interface conduit defines ships a suite here (C11): the transcript
store, token store, skill source, secret resolver, event sink and approval handler. A
product runs one against its own backend by subclassing it in a test module and saying
how to make the backend::

    from conduit.testing.conformance import TranscriptStoreSuite

    class TestOurTranscriptStore(TranscriptStoreSuite):
        async def make_store(self):
            return OurTranscriptStore(...)

Suites drive their scenarios on their own event loop (``loop_factory`` chooses it, for
example a selector loop for psycopg on Windows), so they need no async pytest plugin.

As a pytest plugin (``-p conduit.testing.conformance``, or
``pytest_plugins = ["conduit.testing.conformance"]`` in the root conftest) this module
registers the ``conformance`` marker and applies it to every suite test, so
``pytest -m conformance`` runs exactly the suites.
"""

from __future__ import annotations

import pytest

# Rewrite the suites' assertions so a failing scenario reports its values.
pytest.register_assert_rewrite(
    "conduit.testing.conformance.transcript",
    "conduit.testing.conformance.token",
    "conduit.testing.conformance.skill",
    "conduit.testing.conformance.secret",
    "conduit.testing.conformance.sink",
    "conduit.testing.conformance.approval",
)

from conduit.testing.conformance.approval import ApprovalHandlerSuite  # noqa: E402
from conduit.testing.conformance.base import ConformanceSuite  # noqa: E402
from conduit.testing.conformance.secret import SecretResolverSuite  # noqa: E402
from conduit.testing.conformance.sink import EventSinkSuite  # noqa: E402
from conduit.testing.conformance.skill import SkillSourceSuite  # noqa: E402
from conduit.testing.conformance.token import TokenStoreSuite  # noqa: E402
from conduit.testing.conformance.transcript import TranscriptStoreSuite  # noqa: E402

__all__ = [
    "ApprovalHandlerSuite",
    "ConformanceSuite",
    "EventSinkSuite",
    "SecretResolverSuite",
    "SkillSourceSuite",
    "TokenStoreSuite",
    "TranscriptStoreSuite",
]


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "conformance: a conduit backend conformance suite test (added by the plugin)"
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        owner = getattr(item, "cls", None)
        if isinstance(owner, type) and issubclass(owner, ConformanceSuite):
            item.add_marker(pytest.mark.conformance)
