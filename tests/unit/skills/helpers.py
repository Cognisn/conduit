"""Synthetic skills and small helpers shared by the skills tests."""

from __future__ import annotations

from pathlib import Path

from conduit.core.backends import Skill, SkillSummary

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "skills"
MARKER = "zz-skills-marker-zz"


def skill(
    name: str,
    body: str = "Follow these steps.\n",
    *,
    version: str | None = "1",
    description: str | None = None,
    resources: dict[str, str] | None = None,
) -> Skill:
    return Skill(
        summary=SkillSummary(
            name=name, description=description or f"The {name} skill.", version=version
        ),
        body=body,
        resources=resources or {},
    )


def unchained(error: BaseException) -> None:
    assert error.__cause__ is None
    assert error.__context__ is None


def clean(error: BaseException, *markers: str) -> None:
    unchained(error)
    for shown in (str(error), repr(error), *map(str, error.args)):
        for marker in markers:
            assert marker not in shown
