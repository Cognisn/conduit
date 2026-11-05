"""The SKILL.md format: front matter, body and resource paths, with validation.

A skill is a folder holding ``SKILL.md`` and, optionally, bundled resource files. The
file opens with front matter between two ``---`` lines, then the body (the skill's
instructions, in Markdown)::

    ---
    name: release-notes
    description: Write release notes from a list of merged changes.
    version: 1.2.0
    allowed-tools:
      - read_file
    ---
    # Release notes
    ...

Front matter is read with the standard library, for the subset skills use: one
``key: value`` per line, values plain or quoted (single or double), simple lists (an
inline ``[a, b]`` or indented ``- item`` lines) and, for ``metadata`` only, one level of
indented ``key: value`` lines. Comments (``#``) and blank lines are ignored. Anything
else, such as a block scalar (``|`` or ``>``) or a nested structure, is refused.

Fields:

- ``name`` (required): lower-case letters and digits in hyphen-separated words, at most
  64 characters, and the same as the skill's folder name.
- ``description`` (required): one line, 1 to 1,024 characters. It is what the model
  sees before the skill is loaded, so it should say when to use the skill.
- ``version`` (optional): a short label without spaces (``1.2.0``, ``2026-11``).
- ``license`` and ``compatibility`` (optional): short single-line text, recorded.
- ``allowed-tools`` (optional): a list of tool names, recorded. conduit does not
  enforce it; which tools a task may call is the product's policy.
- ``metadata`` (optional): a flat mapping of short keys to single-line values.

Any other key is refused. Every refusal is a ``SkillInvalid`` with a fixed reason code
and the front-matter line, never the text it found.

Resource paths are relative POSIX paths inside the skill's folder (``references/x.md``):
segments of letters, digits, ``_``, ``.`` and ``-`` that do not start with a dot, no
``..``, no absolute or drive paths and no backslashes. ``SKILL.md`` itself is not a
resource. ``check_resource_path`` applies the rule; a filesystem source also refuses a
path that a link carries outside the folder.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from conduit.core.backends import SkillSummary
from conduit.skills.errors import SkillInvalid, SkillResourceRefused

__all__ = [
    "MAX_DESCRIPTION_LENGTH",
    "MAX_NAME_LENGTH",
    "MAX_RESOURCE_PATH_LENGTH",
    "SKILL_FILE",
    "SkillDocument",
    "check_resource_path",
    "is_skill_name",
    "is_skill_version",
    "parse_skill_md",
    "resource_path_refusal",
]

SKILL_FILE = "SKILL.md"
MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
MAX_RESOURCE_PATH_LENGTH = 256
_MAX_SHORT_TEXT = 256
_MAX_COMPATIBILITY = 500
_MAX_METADATA_VALUE = 1024

_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,63}")
_TOOL = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,127}")
_METADATA_KEY = re.compile(r"[a-z][a-z0-9_-]{0,63}")
_KEY_LINE = re.compile(r"([A-Za-z0-9_-]+):(?:[ \t]+(.*))?")
_SEGMENT = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")
_DRIVE = re.compile(r"[A-Za-z]:")

_FIELDS = ("name", "description", "version", "license", "compatibility", "allowed-tools")
_KNOWN = frozenset((*_FIELDS, "metadata"))

_Value = str | list[str] | dict[str, str]


def is_skill_name(name: object) -> bool:
    """Whether ``name`` is a valid skill name."""
    return isinstance(name, str) and len(name) <= MAX_NAME_LENGTH and bool(_NAME.fullmatch(name))


def is_skill_version(version: object) -> bool:
    """Whether ``version`` is a valid skill version label."""
    return isinstance(version, str) and bool(_VERSION.fullmatch(version))


@dataclass(frozen=True, slots=True)
class SkillDocument:
    """A parsed and validated SKILL.md."""

    name: str
    description: str
    body: str
    version: str | None = None
    license: str | None = None
    compatibility: str | None = None
    allowed_tools: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def summary(self) -> SkillSummary:
        """What progressive disclosure lists for the model before the skill is loaded."""
        return SkillSummary(name=self.name, description=self.description, version=self.version)


class _Refusal(Exception):
    """Internal: a refusal found while parsing, raised again outside the handler."""

    def __init__(self, reason: str, line: int | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.line = line


def _quoted(text: str, line: int) -> tuple[str, str]:
    """A quoted scalar at the start of ``text``: its value and what follows it."""
    quote = text[0]
    out: list[str] = []
    i = 1
    while i < len(text):
        char = text[i]
        if quote == "'" and char == "'":
            if text[i + 1 : i + 2] == "'":
                out.append("'")
                i += 2
                continue
            return "".join(out), text[i + 1 :]
        if quote == '"' and char == "\\" and text[i + 1 : i + 2] in ('"', "\\"):
            out.append(text[i + 1])
            i += 2
            continue
        if quote == '"' and char == '"':
            return "".join(out), text[i + 1 :]
        out.append(char)
        i += 1
    raise _Refusal("value_invalid", line)


def _no_trailing(rest: str, line: int) -> None:
    rest = rest.strip()
    if rest and not rest.startswith("#"):
        raise _Refusal("value_invalid", line)


def _plain(text: str) -> str:
    cut = re.search(r"[ \t]#", text)
    return (text[: cut.start()] if cut else text).strip()


def _scalar(text: str, line: int) -> str:
    text = text.strip()
    if not text:
        return ""
    if text[0] in "'\"":
        value, rest = _quoted(text, line)
        _no_trailing(rest, line)
        return value
    if text[0] in "|>":
        raise _Refusal("block_scalar_unsupported", line)
    if text[0] in "{[]}&*!%@`":
        raise _Refusal("value_invalid", line)
    return _plain(text)


def _inline_list(text: str, line: int) -> list[str]:
    body = _plain(text)
    if not body.endswith("]"):
        raise _Refusal("value_invalid", line)
    inner = body[1:-1].strip()
    if not inner:
        return []
    items: list[str] = []
    for part in inner.split(","):
        part = part.strip()
        if not part:
            raise _Refusal("value_invalid", line)
        items.append(_scalar(part, line))
    return items


def _value(text: str | None, line: int) -> _Value:
    if text is None:
        return ""
    stripped = text.strip()
    if stripped.startswith("["):
        return _inline_list(stripped, line)
    return _scalar(stripped, line)


def _front_matter(lines: list[str], start: int) -> tuple[dict[str, _Value], dict[str, int]]:
    """Parse front-matter ``lines`` (file line ``start`` is the first). Keys keep file order."""
    values: dict[str, _Value] = {}
    where: dict[str, int] = {}
    open_key: str | None = None  # a key whose value is the indented block below it
    for offset, raw in enumerate(lines):
        number = start + offset
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if "\t" in raw[: len(raw) - len(raw.lstrip())]:
            raise _Refusal("line_invalid", number)
        if raw[0] == " ":
            if open_key is None:
                raise _Refusal("line_invalid", number)
            item = raw.strip()
            block = values[open_key]
            if item.startswith("- ") or item == "-":
                if isinstance(block, dict):
                    raise _Refusal("line_invalid", number)
                entries = block if isinstance(block, list) else []
                entries.append(_scalar(item[1:], number))
                values[open_key] = entries
                continue
            pair = _KEY_LINE.fullmatch(item)
            if pair is None or isinstance(block, list):
                raise _Refusal("line_invalid", number)
            mapping = block if isinstance(block, dict) else {}
            if pair.group(1) in mapping:
                raise _Refusal("field_repeated", number)
            mapping[pair.group(1)] = _scalar(pair.group(2) or "", number)
            values[open_key] = mapping
            continue
        match = _KEY_LINE.fullmatch(raw.rstrip())
        if match is None:
            raise _Refusal("line_invalid", number)
        key, text = match.group(1), match.group(2)
        if key not in _KNOWN:
            raise _Refusal("field_unknown", number)
        if key in values:
            raise _Refusal("field_repeated", number)
        values[key] = _value(text, number)
        where[key] = number
        open_key = key if text is None or not text.strip() else None
    return values, where


def _text_field(
    values: dict[str, _Value], where: dict[str, int], key: str, limit: int
) -> str | None:
    if key not in values:
        return None
    value = values[key]
    if not isinstance(value, str) or not value or len(value) > limit:
        raise _Refusal(f"{key.replace('-', '_')}_invalid", where[key])
    return value


def _document(text: str, expected_name: str | None) -> SkillDocument:
    text = text.removeprefix("﻿").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    if not lines or lines[0].rstrip() != "---":
        raise _Refusal("front_matter_missing", 1)
    close = next((i for i in range(1, len(lines)) if lines[i].rstrip() == "---"), None)
    if close is None:
        raise _Refusal("front_matter_unclosed")
    values, where = _front_matter(lines[1:close], 2)

    if "name" not in values:
        raise _Refusal("name_missing")
    name = values["name"]
    if not is_skill_name(name):
        raise _Refusal("name_invalid", where["name"])
    assert isinstance(name, str)  # noqa: S101 - is_skill_name has checked it
    if expected_name is not None and name != expected_name:
        raise _Refusal("name_mismatch", where["name"])
    if "description" not in values:
        raise _Refusal("description_missing")
    description = _text_field(values, where, "description", MAX_DESCRIPTION_LENGTH)
    assert description is not None  # noqa: S101 - present, so checked
    version = _text_field(values, where, "version", 64)
    if version is not None and not is_skill_version(version):
        raise _Refusal("version_invalid", where["version"])
    licence = _text_field(values, where, "license", _MAX_SHORT_TEXT)
    compatibility = _text_field(values, where, "compatibility", _MAX_COMPATIBILITY)

    tools: tuple[str, ...] = ()
    if "allowed-tools" in values:
        listed = values["allowed-tools"]
        if not isinstance(listed, list) or not all(_TOOL.fullmatch(t) for t in listed):
            raise _Refusal("allowed_tools_invalid", where["allowed-tools"])
        tools = tuple(listed)

    metadata: dict[str, str] = {}
    if "metadata" in values:
        given = values["metadata"]
        if not isinstance(given, dict) or not all(
            _METADATA_KEY.fullmatch(k) and v and len(v) <= _MAX_METADATA_VALUE
            for k, v in given.items()
        ):
            raise _Refusal("metadata_invalid", where["metadata"])
        metadata = dict(given)

    body = "\n".join(lines[close + 1 :]).lstrip("\n")
    if not body.strip():
        raise _Refusal("body_empty")
    return SkillDocument(
        name=name,
        description=description,
        body=body,
        version=version,
        license=licence,
        compatibility=compatibility,
        allowed_tools=tools,
        metadata=MappingProxyType(metadata),
    )


def parse_skill_md(text: str, *, expected_name: str | None = None) -> SkillDocument:
    """Parse and validate a SKILL.md, or raise ``SkillInvalid`` with the reason.

    ``expected_name`` is the skill's folder name; the front matter's ``name`` must
    equal it. The refusal names the skill only by a valid folder name.
    """
    skill = expected_name if is_skill_name(expected_name) else None
    refusal: _Refusal | None = None
    try:
        document = _document(text, expected_name)
    except _Refusal as found:
        refusal = found
    if refusal is not None:
        raise SkillInvalid(refusal.reason, skill=skill, line=refusal.line)
    return document


def check_resource_path(path: str) -> str:
    """``path`` if it is a valid resource path, else ``SkillResourceRefused``."""
    reason = resource_path_refusal(path)
    if reason is not None:
        raise SkillResourceRefused(reason)
    return path


def resource_path_refusal(path: object) -> str | None:
    """The reason ``path`` is not a valid resource path, or ``None`` when it is."""
    if not isinstance(path, str) or not path or len(path) > MAX_RESOURCE_PATH_LENGTH:
        return "path_invalid"
    if path.startswith(("/", "\\")) or _DRIVE.match(path):
        return "path_absolute"
    segments = path.split("/")
    if ".." in segments:
        return "path_traversal"
    # Each segment is letters, digits, "_", "." and "-", not starting with a dot, so no
    # backslash, colon, NUL, space, empty segment or "." passes.
    if not all(_SEGMENT.fullmatch(s) for s in segments):
        return "path_invalid"
    if path.lower() == SKILL_FILE.lower():
        return "path_reserved"
    return None
