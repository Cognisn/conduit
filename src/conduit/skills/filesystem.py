"""A skill source over a directory the product gives: one folder per skill.

::

    skills/
      release-notes/
        SKILL.md
        references/style.md
        templates/entry.md
      status-report/
        SKILL.md

A folder is a skill when its name is a valid skill name and it holds ``SKILL.md``;
hidden entries (a leading dot) are ignored. ``list_skills`` leaves out a skill that
fails validation, logging its reason code, unless the source is ``strict``, when the
first refusal is raised. ``load_skill`` refuses an invalid skill with ``SkillInvalid``.

Every file a skill can reach is confined to the skill's own folder:

- a resource path given to ``read_resource`` must meet the format's path rule, so
  traversal (``..``), absolute and drive paths and backslashes are refused before the
  disk is touched;
- every path is then resolved, following links, and refused (``path_link``) unless it
  stays inside the skill's folder; this covers a link in any part of the path, the
  skill folder itself and its ``SKILL.md``;
- a skill folder must itself resolve inside the source's directory.

A skill's resources are its other text files (UTF-8 without NUL bytes), up to
``SourceLimits``. Binary files are not resources and are left out. A resource larger
than the limit, or too many or too large resources in all, or a folder deeper than the
limit, refuses the whole skill, so a model never sees part of one.

File access runs in a worker thread (C3). Filesystem failures are ``SkillUnavailable``,
naming the operation and the error class only; no error carries file content or a
path from outside the format's rules, and none keeps the exception it was raised for.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from conduit.core.backends import Skill, SkillSummary
from conduit.core.errors import ConfigurationError
from conduit.skills.errors import (
    SkillInvalid,
    SkillNotFound,
    SkillResourceRefused,
    SkillUnavailable,
)
from conduit.skills.format import (
    SKILL_FILE,
    SkillDocument,
    is_skill_name,
    parse_skill_md,
    resource_path_refusal,
)

__all__ = ["FilesystemSkillSource", "SourceLimits"]

logger = logging.getLogger("conduit.skills")

_SNIFF_BYTES = 8192


@dataclass(frozen=True, slots=True)
class SourceLimits:
    """How much a filesystem source reads for one skill. Each limit is inclusive."""

    max_skill_file_bytes: int = 64 * 1024
    max_resource_bytes: int = 256 * 1024
    max_total_resource_bytes: int = 1024 * 1024
    max_resources: int = 128
    max_depth: int = 8  # path segments in a resource path, file name included

    def __post_init__(self) -> None:
        for name in self.__slots__:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ConfigurationError(f"skill source limit {name} must be a positive integer")


def _inside(base_real: str, path: Path | str) -> bool:
    """Whether ``path``, with every link followed, stays inside ``base_real``."""
    real = os.path.normcase(os.path.realpath(path))
    try:
        return os.path.commonpath([base_real, real]) == base_real
    except ValueError:  # another drive
        return False


def _real(path: Path) -> str:
    return os.path.normcase(os.path.realpath(path))


def _text(data: bytes) -> str | None:
    """``data`` as text, or ``None`` when it is not UTF-8 text."""
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _guarded[T](operation: str, call: Callable[[], T]) -> T:
    """Run ``call``; a filesystem failure becomes ``SkillUnavailable``, unchained."""
    failure: str | None = None
    try:
        return call()
    except OSError as error:
        failure = type(error).__name__
    raise SkillUnavailable(operation, failure)


class FilesystemSkillSource:
    """Skills from a directory: a ``SkillSource`` with resource access by path."""

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        limits: SourceLimits | None = None,
        strict: bool = False,
    ) -> None:
        root = Path(directory)
        if not root.is_dir():
            raise ConfigurationError("the skill directory does not exist or is not a directory")
        self._root = root
        self._limits = limits or SourceLimits()
        self._strict = strict

    @property
    def limits(self) -> SourceLimits:
        return self._limits

    # The SkillSource protocol.

    async def list_skills(self) -> list[SkillSummary]:
        """Every valid skill's summary, sorted by name."""
        return await asyncio.to_thread(_guarded, "list the skills", self._list)

    async def load_skill(self, name: str) -> Skill:
        """The skill's summary, body and text resources. ``SkillNotFound`` when absent."""
        return await asyncio.to_thread(_guarded, "load the skill", lambda: self._load(name))

    # Beyond the protocol.

    async def load_document(self, name: str) -> SkillDocument:
        """The parsed SKILL.md, with every recorded front-matter field."""
        return await asyncio.to_thread(_guarded, "load the skill", lambda: self._document(name))

    async def list_resources(self, name: str) -> tuple[str, ...]:
        """The skill's resource paths, sorted."""
        return await asyncio.to_thread(
            _guarded,
            "list the skill's resources",
            lambda: tuple(sorted(self._load(name).resources)),
        )

    async def read_resource(self, name: str, path: str) -> str:
        """One resource's text, confined to the skill's folder."""
        return await asyncio.to_thread(
            _guarded, "read the skill's resource", lambda: self._read_resource(name, path)
        )

    # Implementation, run in a worker thread.

    def _list(self) -> list[SkillSummary]:
        summaries: list[SkillSummary] = []
        with os.scandir(self._root) as entries:
            names = sorted(e.name for e in entries if not e.name.startswith("."))
        for name in names:
            if not is_skill_name(name) or not (self._root / name).is_dir():
                continue
            try:
                document = self._document(name)
            except SkillNotFound:
                continue
            except (SkillInvalid, SkillResourceRefused) as refused:
                if self._strict:
                    raise
                logger.warning("skill %s left out: %s", name, refused.reason)
                continue
            summaries.append(document.summary())
        return summaries

    def _folder(self, name: str) -> tuple[Path, str]:
        if not is_skill_name(name):
            raise SkillNotFound()
        folder = self._root / name
        if not folder.is_dir():
            raise SkillNotFound(name)
        if not _inside(_real(self._root), folder):
            raise SkillResourceRefused("path_link", skill=name)
        return folder, _real(folder)

    def _document(self, name: str) -> SkillDocument:
        folder, real = self._folder(name)
        path = folder / SKILL_FILE
        if os.path.lexists(path) and not _inside(real, path):
            raise SkillResourceRefused("path_link", skill=name)
        if not path.is_file():
            raise SkillNotFound(name)
        if path.stat().st_size > self._limits.max_skill_file_bytes:
            raise SkillInvalid("skill_file_too_large", skill=name)
        text = _text(path.read_bytes())
        if text is None:
            raise SkillInvalid("skill_file_not_text", skill=name)
        return parse_skill_md(text, expected_name=name)

    def _files(self, name: str, folder: Path, real: str) -> list[tuple[str, Path]]:
        """Every candidate resource file, confined and within the depth limit."""
        found: list[tuple[str, Path]] = []
        pending: list[tuple[Path, tuple[str, ...]]] = [(folder, ())]
        while pending:
            current, parts = pending.pop()
            with os.scandir(current) as scan:
                entries = sorted(scan, key=lambda e: e.name)
            for entry in entries:
                rel = (*parts, entry.name)
                if (entry.is_symlink() or entry.is_junction()) and not _inside(real, entry.path):
                    raise SkillResourceRefused("path_link", skill=name)
                if resource_path_refusal("/".join(rel)) is not None:
                    continue  # SKILL.md itself, or a name no resource path can address
                if len(rel) > self._limits.max_depth:
                    raise SkillResourceRefused("resource_too_deep", skill=name)
                if entry.is_dir():
                    pending.append((Path(entry.path), rel))
                elif entry.is_file():
                    found.append(("/".join(rel), Path(entry.path)))
        return sorted(found)

    def _load(self, name: str) -> Skill:
        document = self._document(name)
        folder, real = self._folder(name)
        limits = self._limits
        resources: dict[str, str] = {}
        total = 0
        for path, file in self._files(name, folder, real):
            with file.open("rb") as handle:
                data = handle.read(limits.max_resource_bytes + 1)
            if len(data) > limits.max_resource_bytes:
                if b"\x00" in data[:_SNIFF_BYTES]:
                    continue  # a large binary file: not a resource
                raise SkillResourceRefused("resource_too_large", skill=name)
            text = _text(data)
            if text is None:
                continue
            total += len(data)
            if total > limits.max_total_resource_bytes:
                raise SkillResourceRefused("resources_too_large", skill=name)
            resources[path] = text
            if len(resources) > limits.max_resources:
                raise SkillResourceRefused("resources_too_many", skill=name)
        return Skill(summary=document.summary(), body=document.body, resources=resources)

    def _read_resource(self, name: str, path: str) -> str:
        if not is_skill_name(name):
            raise SkillNotFound()
        reason = resource_path_refusal(path)
        if reason is not None:
            raise SkillResourceRefused(reason, skill=name)
        self._document(name)
        folder, real = self._folder(name)
        target = folder.joinpath(*path.split("/"))
        if not _inside(real, target):
            raise SkillResourceRefused("path_link", skill=name)
        if not target.is_file():
            raise SkillResourceRefused("resource_missing", skill=name)
        with target.open("rb") as handle:
            data = handle.read(self._limits.max_resource_bytes + 1)
        if len(data) > self._limits.max_resource_bytes:
            raise SkillResourceRefused("resource_too_large", skill=name)
        text = _text(data)
        if text is None:
            raise SkillResourceRefused("resource_not_text", skill=name)
        return text
