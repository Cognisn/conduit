"""Skill events: what was loaded, used, unloaded or refused, by name, version and size.

Skill bodies and resources are content, so no event carries them, nor a hash of them:
only the skill's name and version (both validated identifiers), a resource's path
(also an identifier) and sizes in UTF-8 bytes. A refusal names the skill only when it
is one the session lists, so a model's argument never reaches an event.
"""

from __future__ import annotations

from typing import Literal

from pydantic import NonNegativeInt

from conduit.core.events import Event
from conduit.core.identifiers import Identifier

__all__ = [
    "SkillEvent",
    "SkillLoadedEvent",
    "SkillRefusedEvent",
    "SkillUnloadedEvent",
    "SkillUsedEvent",
]


class SkillLoadedEvent(Event):
    """A skill's body was loaded into the session (and so into the request prefix)."""

    type: Literal["skill_loaded"] = "skill_loaded"
    skill: Identifier
    version: Identifier | None = None
    body_size: NonNegativeInt
    resource_count: NonNegativeInt
    loaded_count: NonNegativeInt  # skills loaded in the session afterwards
    loaded_size: NonNegativeInt  # their bodies' total size afterwards


class SkillUsedEvent(Event):
    """A loaded skill was asked for again, or one of its resources was read."""

    type: Literal["skill_used"] = "skill_used"
    skill: Identifier
    version: Identifier | None = None
    resource: Identifier | None = None
    resource_size: NonNegativeInt | None = None


class SkillUnloadedEvent(Event):
    type: Literal["skill_unloaded"] = "skill_unloaded"
    skill: Identifier
    version: Identifier | None = None
    body_size: NonNegativeInt
    loaded_count: NonNegativeInt
    loaded_size: NonNegativeInt


class SkillRefusedEvent(Event):
    """A skill request was refused. ``reason`` is the refusal's fixed code."""

    type: Literal["skill_refused"] = "skill_refused"
    skill: Identifier | None = None
    reason: Identifier


SkillEvent = SkillLoadedEvent | SkillUsedEvent | SkillUnloadedEvent | SkillRefusedEvent
