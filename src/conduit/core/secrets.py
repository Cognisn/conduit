"""Secret references and the secret resolver protocol.

conduit's settings never hold a credential, only a ``secret://<name>`` reference. The
product supplies a ``SecretResolver`` that turns a reference into the value when a
provider is built (C11, C15). conduit ships two resolvers: ``EnvironmentSecretResolver``
(a variable per reference) and ``LiteralSecretResolver`` (values held in the resolver,
for tests). The konfig extra adds the default resolver for products that use konfig.

Error messages name the reference, never a value.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, GetCoreSchemaHandler, SecretStr
from pydantic_core import core_schema  # installed with pydantic: no extra dependency

from conduit.core.errors import SecretUnavailable

SCHEME = "secret://"
_NAME = re.compile(r"^[A-Za-z0-9_./:-]{1,256}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class SecretRef:
    """A reference to a secret, written ``secret://<name>``. Never the secret itself."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError("a secret reference name must match [A-Za-z0-9_./:-]{1,256}")
        self.name = name

    @classmethod
    def parse(cls, text: str) -> SecretRef:
        """A reference from ``secret://<name>``. The error never echoes the text."""
        if not isinstance(text, str) or not text.startswith(SCHEME):
            raise ValueError("expected a secret reference of the form secret://<name>")
        return cls(text[len(SCHEME) :])

    def __str__(self) -> str:
        return f"{SCHEME}{self.name}"

    def __repr__(self) -> str:
        return f"SecretRef({self.name!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SecretRef) and other.name == self.name

    def __hash__(self) -> int:
        return hash((SecretRef, self.name))

    @classmethod
    def _validate(cls, value: Any) -> SecretRef:
        if isinstance(value, SecretRef):
            return value
        return cls.parse(value)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            json_schema_input_schema=core_schema.str_schema(pattern=r"^secret://"),
            serialization=core_schema.plain_serializer_function_ser_schema(str),
        )


@runtime_checkable
class SecretResolver(Protocol):
    """Turns a secret reference into its value. Supplied by the product."""

    async def resolve(self, ref: SecretRef) -> SecretStr: ...


class EnvironmentSecretResolver:
    """Resolves ``secret://NAME`` from the environment variable ``<prefix>NAME``."""

    def __init__(self, *, prefix: str = "", environ: Mapping[str, str] | None = None) -> None:
        self._prefix = prefix
        self._environ = os.environ if environ is None else environ

    async def resolve(self, ref: SecretRef) -> SecretStr:
        if not _ENV_NAME.fullmatch(ref.name):
            raise SecretUnavailable(f"{ref} is not an environment variable name")
        value = self._environ.get(self._prefix + ref.name, "")
        if not value:
            raise SecretUnavailable(f"{ref} is not set in the environment")
        return SecretStr(value)

    def __repr__(self) -> str:
        return f"EnvironmentSecretResolver(prefix={self._prefix!r})"


class LiteralSecretResolver:
    """Holds values by reference name. For tests; never put real credentials here."""

    def __init__(self, values: Mapping[str, str]) -> None:
        self._values = dict(values)

    async def resolve(self, ref: SecretRef) -> SecretStr:
        try:
            return SecretStr(self._values[ref.name])
        except KeyError:
            raise SecretUnavailable(f"{ref} is not held by this resolver") from None

    def __repr__(self) -> str:
        return f"LiteralSecretResolver(<{len(self._values)} values>)"


async def resolve_secret_fields(
    settings: BaseModel, resolver: SecretResolver, *, _prefix: str = ""
) -> dict[str, SecretStr]:
    """Every ``SecretRef`` field of ``settings`` (nested models too), resolved.

    Keys are dotted field paths (``api_key``, ``auth.token``). Unset optional
    references are skipped.
    """
    resolved: dict[str, SecretStr] = {}
    for name in type(settings).model_fields:
        value = getattr(settings, name)
        path = f"{_prefix}{name}"
        if isinstance(value, SecretRef):
            resolved[path] = await resolver.resolve(value)
        elif isinstance(value, BaseModel):
            resolved.update(await resolve_secret_fields(value, resolver, _prefix=f"{path}."))
    return resolved
