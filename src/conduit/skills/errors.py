"""Errors the skills layer raises.

Skill files are product content and a model's tool arguments are untrusted, so no
error here carries either: each names a fixed reason code, and a skill or resource
only when its name has already passed the format's own pattern. None keeps the
exception it was raised for (D0040).
"""

from __future__ import annotations

from conduit.core.errors import ConduitError

__all__ = [
    "SkillError",
    "SkillInvalid",
    "SkillNotFound",
    "SkillRefused",
    "SkillResourceRefused",
    "SkillUnavailable",
]


class SkillError(ConduitError):
    """Base class for the skills layer's errors."""


class SkillInvalid(SkillError, ValueError):
    """A SKILL.md, or a skill folder, does not meet the format.

    ``reason`` is a fixed code (``name_invalid``, ``body_empty`` and so on), ``line`` the
    front-matter line it concerns, if any, and ``skill`` the skill's name when it is a
    valid skill name. No value from the file is ever included.
    """

    def __init__(self, reason: str, *, skill: str | None = None, line: int | None = None) -> None:
        where = f" in skill {skill!r}" if skill is not None else ""
        at = f" at line {line}" if line is not None else ""
        super().__init__(f"skill refused{where}: {reason}{at}")
        self.reason = reason
        self.skill = skill
        self.line = line


class SkillNotFound(SkillError, LookupError):
    """No skill of that name is available. The name is given only when it is valid."""

    def __init__(self, skill: str | None = None) -> None:
        super().__init__(
            f"no skill named {skill!r} is available" if skill else "no such skill is available"
        )
        self.skill = skill


class SkillResourceRefused(SkillError, LookupError):
    """A resource path is refused: outside the skill, malformed, too large or absent.

    ``reason`` is a fixed code (``path_traversal``, ``path_absolute``, ``path_link``,
    ``resource_too_large``, ``resource_missing`` and so on). The path is never included.
    """

    def __init__(self, reason: str, *, skill: str | None = None) -> None:
        where = f" in skill {skill!r}" if skill is not None else ""
        super().__init__(f"skill resource refused{where}: {reason}")
        self.reason = reason
        self.skill = skill


class SkillRefused(SkillError):
    """Progressive disclosure refused a request: a limit, or a skill not loaded.

    ``reason`` is a fixed code: ``too_many_loaded``, ``loaded_too_large``,
    ``skill_not_loaded`` or ``already_started``. The skill is named only when listed.
    """

    def __init__(self, reason: str, *, skill: str | None = None) -> None:
        where = f" for skill {skill!r}" if skill is not None else ""
        super().__init__(f"skill request refused{where}: {reason}")
        self.reason = reason
        self.skill = skill


class SkillUnavailable(SkillError):
    """A skill could not be read (the folder is unreadable, for example).

    Names the operation and the class of the underlying error only.
    """

    def __init__(self, operation: str, error_class: str | None = None) -> None:
        detail = f" ({error_class})" if error_class else ""
        super().__init__(f"the skill source could not {operation}{detail}")
        self.operation = operation
        self.error_class = error_class
