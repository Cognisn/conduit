"""Build ``ConduitSettings`` from a konfig configuration.

konfig merges its layers (defaults, system and user files, the AWS settings document,
environment variables and runtime overrides) and the loader validates the merged
``conduit`` section as ``ConduitSettings``. konfig leaves ``secret://`` values as
written, so credentials stay references: the settings models refuse a credential-like
field that holds anything else, and the provider registry resolves each reference
through the product's secret resolver when it builds the provider.

Errors name the section and the fields at fault, never a value.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import ValidationError

from conduit.core.errors import ConfigurationError
from conduit.core.runtime import ConduitSettings

#: The konfig section conduit's settings are read from unless the product names another.
DEFAULT_SECTION = "conduit"


class KonfigSettingsSource(Protocol):
    """The part of ``konfig.Settings`` the loader uses."""

    def get_section(self, prefix: str) -> dict[str, Any]: ...


def load_settings(
    source: KonfigSettingsSource, *, section: str = DEFAULT_SECTION
) -> ConduitSettings:
    """``ConduitSettings`` from the ``section`` of a konfig configuration.

    ``section`` is a konfig dot-notation key (``conduit``, or ``ai.conduit`` for a
    nested section). A missing or empty section is refused rather than read as the
    defaults, so a misnamed section cannot start conduit unconfigured. Raises
    ``ConfigurationError``.
    """
    data = source.get_section(section)
    if not data:
        raise ConfigurationError(f"konfig has no {section!r} section of conduit settings")
    try:
        return ConduitSettings.model_validate(data)
    except ValidationError as exc:
        raise ConfigurationError(_describe(section, exc)) from None


def _describe(section: str, exc: ValidationError) -> str:
    # The settings models hide input in their errors, and include_input=False keeps it
    # out here too: each line is a field path and pydantic's message for it.
    problems = "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or '(section)'}: {error['msg']}"
        for error in exc.errors(include_input=False, include_url=False)
    )
    return f"konfig section {section!r} is not valid conduit settings: {problems}"
