"""The secret resolver conformance suite.

Each scenario states one thing conduit relies on of a ``SecretResolver``: the provider
registry resolves a provider's references when it builds the provider, and the MCP
client when it connects, so a resolver must give back the exact value as a
``SecretStr``, refuse a reference it cannot resolve with ``SecretUnavailable`` naming
the reference and nothing it holds, keep values out of its ``repr`` and its logs, and
stay correct when several tasks resolve at once.
"""

# Suites are test code shipped for products to run under pytest: assertions are the point.
# ruff: noqa: S101

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from typing import ClassVar

from pydantic import SecretStr

from conduit.core.errors import SecretUnavailable
from conduit.core.secrets import SecretRef, SecretResolver
from conduit.testing.conformance.base import ConformanceSuite


def _chain(error: BaseException) -> Iterator[BaseException]:
    """``error`` and every exception it keeps as a cause or a context."""
    seen: set[int] = set()
    pending: list[BaseException] = [error]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for linked in (current.__cause__, current.__context__):
            if linked is not None:
                pending.append(linked)


def _renderings(error: BaseException) -> str:
    """Every way the error and its chain might be shown: str, repr and each argument."""
    parts: list[str] = []
    for linked in _chain(error):
        parts += [str(linked), repr(linked), *(repr(arg) for arg in linked.args)]
    return "\n".join(parts)


class _Collect(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.NOTSET)
        self.texts: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.texts.append(record.getMessage())
        self.texts.append(repr(record.args))
        if record.exc_info and record.exc_info[1] is not None:
            self.texts.append(_renderings(record.exc_info[1]))


@contextmanager
def _every_log_record() -> Iterator[_Collect]:
    """Every record logged anywhere, at every level, while the block runs."""
    root = logging.getLogger()
    handler = _Collect()
    previous = root.level
    disabled = logging.root.manager.disable
    root.addHandler(handler)
    root.setLevel(logging.NOTSET)
    logging.disable(logging.NOTSET)
    try:
        yield handler
    finally:
        root.removeHandler(handler)
        root.setLevel(previous)
        logging.disable(disabled)


class SecretResolverSuite(ConformanceSuite):
    """Known and unknown references, exact values, errors, reprs, logs and concurrency."""

    interface: ClassVar[str] = "secret_resolver"
    #: References resolved at once in the concurrency scenario.
    concurrent_references: ClassVar[int] = 16

    async def make_resolver(self, secrets: Mapping[str, str]) -> SecretResolver:
        """A resolver holding exactly ``secrets`` (reference name to value), for one
        scenario. Products override this, storing the values the way they hold secrets."""
        raise NotImplementedError("override make_resolver to return your secret resolver")

    async def close_resolver(self, resolver: SecretResolver) -> None:
        """Release a resolver after a scenario. Override if it needs closing."""
        closer = getattr(resolver, "aclose", None)
        if closer is not None:
            await closer()

    def reference_name(self, n: int) -> str:
        """The ``n``th reference name the suite uses. The default suits an environment
        variable as well as a vault path; override it if your resolver needs another form."""
        return f"CONDUIT_SUITE_SECRET_{n:02d}"

    def secret_value(self, n: int) -> str:
        """The ``n``th synthetic value. No value is a substring of another."""
        return f"synthetic-suite-value-{n:04d}-not-real"

    def _held(self, count: int) -> dict[str, str]:
        return {self.reference_name(n): self.secret_value(n) for n in range(count)}

    def _scenario(
        self,
        secrets: Mapping[str, str],
        body: Callable[[SecretResolver], Awaitable[None]],
    ) -> None:
        async def scenario() -> None:
            resolver = await self.make_resolver(secrets)
            try:
                await body(resolver)
            finally:
                await self.close_resolver(resolver)

        self.run(scenario)

    async def _resolve(self, resolver: SecretResolver, name: str) -> SecretStr:
        value = await resolver.resolve(SecretRef(name))
        assert isinstance(value, SecretStr), (
            f"resolve returned a {type(value).__name__}, not a SecretStr"
        )
        return value

    def test_a_known_reference_resolves_to_its_value(self) -> None:
        held = self._held(3)

        async def body(resolver: SecretResolver) -> None:
            for name, value in held.items():
                resolved = await self._resolve(resolver, name)
                assert resolved.get_secret_value() == value
            again = await self._resolve(resolver, self.reference_name(0))
            assert again.get_secret_value() == self.secret_value(0)

        self._scenario(held, body)

    def test_every_character_of_a_value_is_kept(self) -> None:
        values = {
            self.reference_name(0): "  synthetic value with spaces and = and 'quotes'  ",
            self.reference_name(1): "synthetic-ünïcödé-välue-€-not-real",
            self.reference_name(2): "synthetic-" + "x" * 500 + "-long",
        }

        async def body(resolver: SecretResolver) -> None:
            for name, value in values.items():
                resolved = await self._resolve(resolver, name)
                assert resolved.get_secret_value() == value, f"{name} was not kept exactly"

        self._scenario(values, body)

    def test_an_unknown_reference_is_secret_unavailable(self) -> None:
        held = self._held(2)
        unknown = SecretRef(self.reference_name(90))

        async def body(resolver: SecretResolver) -> None:
            raised: BaseException | None = None
            returned: object = None
            try:
                returned = await resolver.resolve(unknown)
            except Exception as error:  # the suite judges whatever was raised
                raised = error
            assert raised is not None, (
                f"an unknown reference resolved to a {type(returned).__name__}"
            )
            assert isinstance(raised, SecretUnavailable), (
                f"an unknown reference raised {type(raised).__name__}, not SecretUnavailable"
            )
            assert str(unknown) in str(raised) or unknown.name in str(raised), (
                "the error does not name the reference"
            )
            shown = _renderings(raised)
            leaked = [name for name, value in held.items() if value in shown]
            assert not leaked, f"the error shows the values of {leaked}"
            assert raised.__cause__ is None, (
                f"the error keeps a {type(raised.__cause__).__name__} as its cause"
            )

        self._scenario(held, body)

    def test_no_value_shows_in_a_repr(self) -> None:
        held = self._held(3)

        async def body(resolver: SecretResolver) -> None:
            resolved = [await self._resolve(resolver, name) for name in held]
            shown = "\n".join(
                [repr(resolver), str(resolver), *(repr(v) for v in resolved), repr(resolved)]
            )
            leaked = [name for name, value in held.items() if value in shown]
            assert not leaked, f"a repr shows the values of {leaked}"

        self._scenario(held, body)

    def test_concurrent_references_resolve_to_their_own_values(self) -> None:
        count = self.concurrent_references
        held = self._held(count)
        # Each reference twice, interleaved, so concurrent calls overlap on every name.
        order = [self.reference_name(n % count) for n in range(2 * count)]

        async def body(resolver: SecretResolver) -> None:
            resolved = await asyncio.gather(*(self._resolve(resolver, name) for name in order))
            got = [value.get_secret_value() for value in resolved]
            assert got == [held[name] for name in order]

        self._scenario(held, body)

    def test_resolving_logs_no_value(self) -> None:
        held = self._held(3)

        async def body(resolver: SecretResolver) -> None:
            with _every_log_record() as logged:
                for name in held:
                    await self._resolve(resolver, name)
                with suppress(SecretUnavailable):
                    await resolver.resolve(SecretRef(self.reference_name(91)))
            text = "\n".join(logged.texts)
            leaked = [name for name, value in held.items() if value in text]
            assert not leaked, f"resolving logged the values of {leaked}"

        self._scenario(held, body)
