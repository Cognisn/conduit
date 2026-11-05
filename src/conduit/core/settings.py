"""Typed settings: buildable in code or from a mapping, credentials only as references.

Each provider adapter declares a subclass of ``ProviderSettings``. A field whose name
marks it as a credential (``api_key``, ``secret_access_key``, ``session_token``,
``password`` and the like) must be typed ``SecretRef``; a subclass that types one as
anything else is refused when the class is defined. Validation errors never echo input.

The generic ``ProviderSettings`` (what ``ConduitSettings.providers`` holds before the
provider registry re-validates each entry against its adapter's own model) keeps
unknown fields, but no unknown field may hold a plaintext credential, at any depth: a
value under a credential-like name (``api_key``, ``auth.token``, ``bearer``,
``Authorization``) must be a ``secret://`` reference, and a string that reads as an HTTP
authorisation value (``Bearer ...``, ``Basic ...``) is refused under any name. The
refusal names the field, never the value.

A validator's own message (a ``ValueError`` it raises) must not quote its input.
``refusals`` reports it only when it holds no string from the input, and never
pydantic's own wording.
"""

from __future__ import annotations

import re
import types
from collections.abc import Iterator, Mapping
from typing import Any, ClassVar, Self, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError, model_validator

from conduit.core.errors import ConfigurationError
from conduit.core.secrets import SecretRef

#: Name segments that mark a field as a credential.
CREDENTIAL_SEGMENTS = frozenset(
    {"key", "apikey", "secret", "token", "password", "passphrase", "credential", "credentials"}
)

KIND_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"


#: How ``failed_fields`` names a failure that belongs to no single field.
WHOLE = "(the whole object)"


#: Input strings shorter than this are not looked for in a validator's message.
_SHORTEST_ECHO = 3


class FieldsRefused(ValueError):
    """Raised by a settings validator about named fields. The message names them only."""

    def __init__(self, reason: str, fields: list[str]) -> None:
        super().__init__(f"{reason}: {', '.join(fields)}")
        self.reason = reason
        self.fields = tuple(fields)


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():  # keys are field names, not input values
            yield from _strings(item)
    elif isinstance(value, list | tuple | set | frozenset):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, BaseModel):
        yield from _strings(value.model_dump(warnings=False))
    elif value is not None and not isinstance(value, bool | int | float):
        yield str(value)


def _quotes(message: str, value: str) -> bool:
    """Whether ``message`` holds ``value`` as a whole token (not inside a longer name)."""
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(value)}(?![A-Za-z0-9_])"
    return re.search(pattern, message) is not None


def refusals(error: ValidationError, given: Any = None) -> tuple[list[str], list[str]]:
    """Where validation failed, and why, without pydantic's text or any input value.

    The first list holds the dotted paths of the failing fields, in order and without
    repeats; a failure raised for the object as a whole is named by the fields a
    ``FieldsRefused`` lists, or ``WHOLE``. The second holds the messages of the
    ``ValueError``s validators raised, each kept only if it holds no string of
    ``given`` (the input); pydantic's own messages are never kept.
    """
    echoes = [s for s in _strings(given) if len(s) >= _SHORTEST_ECHO]
    fields: list[str] = []
    reasons: list[str] = []
    for detail in error.errors(include_url=False, include_input=False):
        location = ".".join(str(part) for part in detail["loc"])
        cause = (detail.get("ctx") or {}).get("error")
        if isinstance(cause, FieldsRefused):
            names = [f"{location}.{f}" if location else f for f in cause.fields]
            reason = cause.reason
        else:
            names = [location or WHOLE]
            reason = str(cause) if isinstance(cause, ValueError) else ""
        fields += [name for name in names if name not in fields]
        if reason and reason not in reasons and not any(_quotes(reason, e) for e in echoes):
            reasons.append(reason)
    return fields, reasons


def failed_fields(error: ValidationError) -> list[str]:
    """The dotted paths of the fields that failed (``refusals`` without the reasons)."""
    return refusals(error)[0]


#: Further name segments that mark an unknown (extra) field as a credential.
EXTRA_CREDENTIAL_SEGMENTS = CREDENTIAL_SEGMENTS | frozenset(
    {"bearer", "auth", "authorization", "authorisation", "pwd", "passwd", "jwt", "cookie"}
)
#: A value that reads as an HTTP authorisation header value: refused under any name.
_AUTHORISATION_VALUE = re.compile(r"^\s*(bearer|basic|digest|token)\s+\S", re.IGNORECASE)
_SEGMENT = re.compile(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])")


def is_credential_name(name: str) -> bool:
    return any(segment in CREDENTIAL_SEGMENTS for segment in name.lower().split("_"))


def _is_extra_credential_name(name: str) -> bool:
    """A broader test for unknown fields: any separator, camelCase, and more segments."""
    compact = re.sub(r"[^a-z0-9]", "", name.lower())
    segments = {s.lower() for s in _SEGMENT.findall(name)} | {compact}
    return bool(segments & EXTRA_CREDENTIAL_SEGMENTS)


def _is_reference(value: Any) -> bool:
    if isinstance(value, SecretRef):
        return True
    try:
        SecretRef.parse(value)
    except ValueError:
        return False
    return True


def _plaintext_credentials(value: Any, path: str, credential: bool) -> Iterator[str]:
    """The paths under ``value`` holding a credential that is not a reference."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key)
            yield from _plaintext_credentials(
                item, f"{path}.{name}" if path else name, _is_extra_credential_name(name)
            )
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            yield from _plaintext_credentials(item, f"{path}.{index}", credential)
    elif (credential and not _is_reference(value)) or (
        isinstance(value, str) and _AUTHORISATION_VALUE.match(value)
    ):
        yield path


def _is_reference_annotation(annotation: Any) -> bool:
    if annotation is SecretRef:
        return True
    if get_origin(annotation) in (Union, types.UnionType):
        members = [a for a in get_args(annotation) if a is not type(None)]
        return members == [SecretRef]
    return False


class SettingsModel(BaseModel):
    """Base for conduit's settings models: immutable, strict about fields, no echo."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class ProviderSettings(BaseModel):
    """Settings for one configured provider. Adapters subclass it."""

    model_config = ConfigDict(extra="allow", frozen=True, hide_input_in_errors=True)

    #: Only the generic model keeps unknown fields; adapter subclasses refuse them
    #: unless they set this.
    allow_extra_fields: ClassVar[bool] = False

    kind: str = Field(pattern=KIND_PATTERN)
    #: The region the provider is called in, where that applies (residency policy).
    region: str | None = Field(default=None, min_length=1, max_length=64)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        for name, field in cls.model_fields.items():
            if is_credential_name(name) and not _is_reference_annotation(field.annotation):
                raise ConfigurationError(
                    f"{cls.__name__}.{name} looks like a credential and must be typed SecretRef"
                )

    @model_validator(mode="after")
    def _check_extra_fields(self) -> Self:
        extra = self.model_extra or {}
        if extra and type(self) is not ProviderSettings and not self.allow_extra_fields:
            raise FieldsRefused("unknown fields", sorted(extra))
        plaintext = list(_plaintext_credentials(extra, "", False))
        if plaintext:
            raise FieldsRefused("a credential must be a secret:// reference", plaintext)
        return self

    @property
    def catalogue_provider(self) -> str:
        """The provider name catalogue entries use for this provider (default: kind)."""
        return self.kind


class AssignmentSettings(SettingsModel):
    """The provider and model a product assigns to one of its task classes."""

    provider: str = Field(min_length=1)  # a configured provider's name
    model: str = Field(min_length=1)
    max_output_tokens: PositiveInt | None = None
