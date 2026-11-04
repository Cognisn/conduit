"""Typed settings: buildable in code or from a mapping, credentials only as references.

Each provider adapter declares a subclass of ``ProviderSettings``. A field whose name
marks it as a credential (``api_key``, ``secret_access_key``, ``session_token``,
``password`` and the like) must be typed ``SecretRef``; a subclass that types one as
anything else is refused when the class is defined. Validation errors never echo input.

The generic ``ProviderSettings`` (what ``ConduitSettings.providers`` holds before the
provider registry re-validates each entry against its adapter's own model) keeps
unknown fields, but a credential-like unknown field must already be a reference.
"""

from __future__ import annotations

import types
from typing import Any, ClassVar, Self, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from conduit.core.errors import ConfigurationError
from conduit.core.secrets import SecretRef

#: Name segments that mark a field as a credential.
CREDENTIAL_SEGMENTS = frozenset(
    {"key", "apikey", "secret", "token", "password", "passphrase", "credential", "credentials"}
)

KIND_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"


def is_credential_name(name: str) -> bool:
    return any(segment in CREDENTIAL_SEGMENTS for segment in name.lower().split("_"))


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
            raise ValueError(f"unknown fields: {', '.join(sorted(extra))}")
        for name, value in extra.items():
            if is_credential_name(name):
                SecretRef.parse(value)  # raises without echoing the value
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
