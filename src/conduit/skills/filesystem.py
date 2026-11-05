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
- a skill folder must itself resolve inside the source's directory;
- every file is checked again after it is opened: the resolved path is opened (on POSIX
  with ``O_NOFOLLOW``), then the path is resolved again, checked again and must name
  the same file (device and file number) as the open handle, so a link swapped in
  between the check and the open is refused rather than followed. Reads are bounded
  on the handle (at most one byte past the limit), so a file that grows after it is
  checked cannot be read past its limit.

Trust assumption: the directory's contents are trusted as far as their location goes.
Whoever can write inside the skills directory can change any skill, and a hard link
cannot be told apart from the file it links to, so a hard link to a file outside the
folder is read as if it were inside. Keep the skills directory writable only by those
trusted to author skills, on a volume where untrusted users cannot make hard links
into it.

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
import errno
import logging
import os
import stat
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


def _within(base_real: str, real: str) -> bool:
    """Whether the already resolved ``real`` is inside ``base_real``."""
    real = os.path.normcase(real)
    try:
        return os.path.commonpath([base_real, real]) == base_real
    except ValueError:  # another drive
        return False


#: Open flags: binary on Windows, and on POSIX never through a link in the last part.
_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
_NOFOLLOW = bool(getattr(os, "O_NOFOLLOW", 0))


def _open_fd(path: str) -> int:
    """Open ``path`` for reading (a seam for the tests)."""
    return os.open(path, _OPEN_FLAGS)


def _fstat(fd: int) -> os.stat_result:
    """What the open handle ``fd`` is (a seam for the tests)."""
    return os.fstat(fd)


def _identity(path: str) -> tuple[int, int]:
    """The device and file number of what ``path`` names, links followed."""
    found = os.stat(path)
    return found.st_dev, found.st_ino


def _read_up_to(fd: int, count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining > 0:
        chunk = os.read(fd, remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_confined(path: Path, base_real: str, limit: int, skill: str) -> bytes | None:
    """At most ``limit + 1`` bytes of the regular file at ``path``, or ``None`` if there
    is none, refusing (``path_link``) unless the file opened is inside ``base_real``.

    The path is resolved and checked, the resolved path opened (on POSIX with
    ``O_NOFOLLOW``), and then, on the open handle, the path is resolved and checked
    again and must name the same file (device and file number) as the handle. A link
    swapped in between the check and the open is therefore refused, not followed.
    """
    real = os.path.realpath(path)
    if not _within(base_real, real):
        raise SkillResourceRefused("path_link", skill=skill)
    if not os.path.isfile(real):
        return None
    missing = looped = False
    fd = -1
    try:
        fd = _open_fd(real)
    except FileNotFoundError:
        missing = True
    except OSError as error:
        looped = _NOFOLLOW and error.errno == errno.ELOOP
        if not looped:
            raise
    if missing:
        return None
    if looped:
        raise SkillResourceRefused("path_link", skill=skill)
    try:
        opened = _fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            return None
        again = os.path.realpath(path)
        if not _within(base_real, again) or _identity(again) != (opened.st_dev, opened.st_ino):
            raise SkillResourceRefused("path_link", skill=skill)
        return _read_up_to(fd, limit + 1)
    finally:
        os.close(fd)


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
    def directory(self) -> Path:
        """The directory the skills are read from."""
        return self._root

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
        limit = self._limits.max_skill_file_bytes
        data = _read_confined(folder / SKILL_FILE, real, limit, name)
        if data is None:
            raise SkillNotFound(name)
        if len(data) > limit:
            raise SkillInvalid("skill_file_too_large", skill=name)
        text = _text(data)
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
            data = _read_confined(file, real, limits.max_resource_bytes, name)
            if data is None:
                continue  # gone, or no longer a regular file, since the folder was read
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
        data = _read_confined(target, real, self._limits.max_resource_bytes, name)
        if data is None:
            raise SkillResourceRefused("resource_missing", skill=name)
        if len(data) > self._limits.max_resource_bytes:
            raise SkillResourceRefused("resource_too_large", skill=name)
        text = _text(data)
        if text is None:
            raise SkillResourceRefused("resource_not_text", skill=name)
        return text
