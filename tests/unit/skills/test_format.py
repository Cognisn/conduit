"""The SKILL.md format: front matter, body, validation and its refusals."""

from __future__ import annotations

import pytest

from conduit.core.backends import SkillSummary
from conduit.skills.errors import SkillInvalid
from conduit.skills.format import (
    MAX_DESCRIPTION_LENGTH,
    MAX_NAME_LENGTH,
    SkillDocument,
    check_resource_path,
    is_skill_name,
    parse_skill_md,
)

MARKER = "zz-format-marker-zz"

VALID = """---
name: release-notes
description: Write release notes from a list of merged changes.
version: 1.2.0
license: MIT
compatibility: Any model with tool use
allowed-tools:
  - read_file
  - search
metadata:
  owner: example-team
  reviewed: "2026-11-01"
---

# Release notes

Group the changes by kind.
"""


def _refused(text: str, **kwargs: object) -> SkillInvalid:
    with pytest.raises(SkillInvalid) as caught:
        parse_skill_md(text, **kwargs)  # type: ignore[arg-type]
    error = caught.value
    assert error.__cause__ is None
    assert error.__context__ is None
    return error


class TestParsing:
    def test_every_recorded_field_is_read(self) -> None:
        doc = parse_skill_md(VALID)
        assert doc == SkillDocument(
            name="release-notes",
            description="Write release notes from a list of merged changes.",
            version="1.2.0",
            license="MIT",
            compatibility="Any model with tool use",
            allowed_tools=("read_file", "search"),
            metadata={"owner": "example-team", "reviewed": "2026-11-01"},
            body="# Release notes\n\nGroup the changes by kind.\n",
        )

    def test_the_summary_is_the_core_model(self) -> None:
        assert parse_skill_md(VALID).summary() == SkillSummary(
            name="release-notes",
            description="Write release notes from a list of merged changes.",
            version="1.2.0",
        )

    def test_only_name_and_description_are_required(self) -> None:
        doc = parse_skill_md("---\nname: a\ndescription: d\n---\nbody\n")
        assert (doc.version, doc.license, doc.allowed_tools, doc.metadata) == (None, None, (), {})

    def test_crlf_and_a_byte_order_mark_are_accepted(self) -> None:
        doc = parse_skill_md("\ufeff" + VALID.replace("\n", "\r\n"))
        assert doc.name == "release-notes"
        assert "\r" not in doc.body

    def test_quoted_values_and_comments(self) -> None:
        doc = parse_skill_md(
            "---\n# a comment\nname: 'quoted-name'\n"
            'description: "Has: a colon"\nversion: 2  # trailing comment\n---\nbody\n'
        )
        assert (doc.name, doc.description, doc.version) == ("quoted-name", "Has: a colon", "2")

    def test_an_inline_list(self) -> None:
        doc = parse_skill_md("---\nname: a\ndescription: d\nallowed-tools: [one, 'two']\n---\nb\n")
        assert doc.allowed_tools == ("one", "two")

    def test_an_empty_inline_list(self) -> None:
        doc = parse_skill_md("---\nname: a\ndescription: d\nallowed-tools: []\n---\nb\n")
        assert doc.allowed_tools == ()

    def test_a_plain_value_keeps_inner_colons(self) -> None:
        doc = parse_skill_md("---\nname: a\ndescription: Use when: notes are due\n---\nb\n")
        assert doc.description == "Use when: notes are due"

    def test_the_expected_name_is_accepted_when_it_matches(self) -> None:
        assert parse_skill_md(VALID, expected_name="release-notes").name == "release-notes"


class TestRefusals:
    @pytest.mark.parametrize(
        ("text", "reason"),
        [
            ("no front matter\n", "front_matter_missing"),
            ("---\nname: a\ndescription: d\n", "front_matter_unclosed"),
            ("---\ndescription: d\n---\nb\n", "name_missing"),
            ("---\nname: a\n---\nb\n", "description_missing"),
            ("---\nname: a\ndescription: d\n---\n\n  \n", "body_empty"),
            ("---\nname: Bad_Name\ndescription: d\n---\nb\n", "name_invalid"),
            ("---\nname: -lead\ndescription: d\n---\nb\n", "name_invalid"),
            ("---\nname: trail-\ndescription: d\n---\nb\n", "name_invalid"),
            ("---\nname: two--hyphens\ndescription: d\n---\nb\n", "name_invalid"),
            ("---\nname: a\ndescription: ''\n---\nb\n", "description_invalid"),
            ("---\nname: a\ndescription: d\nversion: has space\n---\nb\n", "version_invalid"),
            ("---\nname: a\ndescription: d\nunknown: x\n---\nb\n", "field_unknown"),
            ("---\nname: a\nname: b\ndescription: d\n---\nb\n", "field_repeated"),
            ("---\nname: a\ndescription: >\n  folded\n---\nb\n", "block_scalar_unsupported"),
            ("---\nname: a\ndescription: |\n  literal\n---\nb\n", "block_scalar_unsupported"),
            ("---\nname: a\ndescription: d\njust text\n---\nb\n", "line_invalid"),
            ("---\nname: a\ndescription: d\n  - orphan\n---\nb\n", "line_invalid"),
            ("---\nname:\n  - a\ndescription: d\n---\nb\n", "name_invalid"),
            ("---\nname: a\ndescription: d\nallowed-tools: one\n---\nb\n", "allowed_tools_invalid"),
            (
                "---\nname: a\ndescription: d\nallowed-tools:\n  - has space\n---\nb\n",
                "allowed_tools_invalid",
            ),
            ("---\nname: a\ndescription: d\nmetadata: flat\n---\nb\n", "metadata_invalid"),
            ("---\nname: a\ndescription: d\nmetadata:\n  Bad_Key: v\n---\nb\n", "metadata_invalid"),
            ("---\nname: a\ndescription: d\nmetadata:\n  k: ''\n---\nb\n", "metadata_invalid"),
            (
                "---\nname: a\ndescription: d\nmetadata:\n  k: v\n  - item\n---\nb\n",
                "line_invalid",
            ),
            ("---\nname: a\ndescription: d\nlicense: 'unclosed\n---\nb\n", "value_invalid"),
            ("---\nname: a\ndescription: d\nallowed-tools: [a, [b]]\n---\nb\n", "value_invalid"),
        ],
    )
    def test_a_malformed_skill_is_refused_with_its_reason(self, text: str, reason: str) -> None:
        assert _refused(text).reason == reason

    def test_a_name_at_the_limit_is_accepted_and_one_over_is_refused(self) -> None:
        name = "a" * MAX_NAME_LENGTH
        assert parse_skill_md(f"---\nname: {name}\ndescription: d\n---\nb\n").name == name
        assert _refused(f"---\nname: {name}a\ndescription: d\n---\nb\n").reason == "name_invalid"

    def test_a_description_at_the_limit_is_accepted_and_one_over_is_refused(self) -> None:
        text = "d" * MAX_DESCRIPTION_LENGTH
        assert parse_skill_md(f"---\nname: a\ndescription: {text}\n---\nb\n").description == text
        error = _refused(f"---\nname: a\ndescription: {text}d\n---\nb\n")
        assert error.reason == "description_invalid"

    def test_a_name_that_differs_from_its_folder_is_refused(self) -> None:
        assert _refused(VALID, expected_name="other").reason == "name_mismatch"

    def test_the_refusal_carries_the_line_and_never_the_text(self) -> None:
        error = _refused(f"---\nname: a\ndescription: d\n{MARKER} here\n---\nb\n")
        assert error.reason == "line_invalid"
        assert error.line == 4
        for shown in (str(error), repr(error), *map(str, error.args)):
            assert MARKER not in shown

    @pytest.mark.parametrize(
        "text",
        [
            f"---\nname: {MARKER}_X\ndescription: d\n---\nb\n",
            f"---\nname: a\ndescription: d\n{MARKER}: x\n---\nb\n",
            f"---\nname: a\ndescription: d\nversion: '{MARKER} x'\n---\nb\n",
        ],
    )
    def test_no_refusal_echoes_a_value(self, text: str) -> None:
        error = _refused(text)
        for shown in (str(error), repr(error), *map(str, error.args)):
            assert MARKER not in shown

    def test_a_skill_named_by_its_folder_is_named_in_the_refusal(self) -> None:
        error = _refused("---\nname: a\n---\nb\n", expected_name="release-notes")
        assert error.skill == "release-notes"
        assert "release-notes" in str(error)

    def test_an_invalid_folder_name_is_not_named(self) -> None:
        error = _refused("---\nname: a\n---\nb\n", expected_name=f"{MARKER} x")
        assert error.skill is None
        assert MARKER not in str(error)


class TestNames:
    @pytest.mark.parametrize("name", ["a", "a1", "release-notes", "x-2-y"])
    def test_valid_names(self, name: str) -> None:
        assert is_skill_name(name)

    @pytest.mark.parametrize(
        "name", ["", "A", "a_b", "a b", "-a", "a-", "a--b", "../a", "a/b", "a" * 65, "é"]
    )
    def test_invalid_names(self, name: str) -> None:
        assert not is_skill_name(name)


class TestResourcePaths:
    @pytest.mark.parametrize(
        "path", ["notes.md", "references/style.md", "a/b/c.txt", "scripts/run_1.py", "x.y-z"]
    )
    def test_a_relative_path_inside_the_skill_is_accepted(self, path: str) -> None:
        assert check_resource_path(path) == path

    def test_a_path_at_the_length_limit_is_accepted(self) -> None:
        path = "/".join(["abcdefg"] * 32) + "h"
        assert len(path) == 256
        assert check_resource_path(path) == path

    @pytest.mark.parametrize(
        ("path", "reason"),
        [
            ("", "path_invalid"),
            ("../outside.md", "path_traversal"),
            ("references/../../outside.md", "path_traversal"),
            ("references/..", "path_traversal"),
            ("./notes.md", "path_invalid"),
            ("/etc/passwd", "path_absolute"),
            ("C:/Windows/win.ini", "path_absolute"),
            ("C:notes.md", "path_absolute"),
            ("//server/share/x", "path_absolute"),
            ("references\\style.md", "path_invalid"),
            ("a//b", "path_invalid"),
            ("a/", "path_invalid"),
            ("has space.md", "path_invalid"),
            ("nul\x00.md", "path_invalid"),
            ("SKILL.md", "path_reserved"),
            (".hidden", "path_invalid"),
            ("dir/.hidden", "path_invalid"),
            ("a" * 257, "path_invalid"),
            ("/".join(["abcdefgh"] * 29), "path_invalid"),
        ],
    )
    def test_a_path_outside_the_rules_is_refused(self, path: str, reason: str) -> None:
        from conduit.skills.errors import SkillResourceRefused

        with pytest.raises(SkillResourceRefused) as caught:
            check_resource_path(path)
        assert caught.value.reason == reason
        assert caught.value.__context__ is None
        assert caught.value.__cause__ is None
