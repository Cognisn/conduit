"""POSIX-only guards of the filesystem source: O_NOFOLLOW's ELOOP and non-blocking opens.

These run on CI's Linux and macOS jobs; Windows has neither O_NOFOLLOW nor FIFOs.
"""

from __future__ import annotations

import errno
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from conduit.skills import filesystem
from conduit.skills.errors import SkillResourceRefused
from conduit.skills.filesystem import FilesystemSkillSource

MARKER = "zz-posix-marker-zz"
#: The POSIX-only names, reached through Any so type checking passes on every platform.
POSIX: Any = os

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only open flags")


def _skill(root: Path, name: str) -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: The {name} skill.\n---\nBody.\n",
        encoding="utf-8",
        newline="\n",
    )
    return folder


async def test_a_link_met_by_o_nofollow_is_refused_with_a_clean_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = _skill(tmp_path, "s")
    (folder / "note.md").write_text("inside", encoding="utf-8")
    original = filesystem._open_fd

    def looping(path: str) -> int:
        if path.endswith("note.md"):
            raise OSError(errno.ELOOP, f"too many levels of symbolic links {MARKER}")
        return original(path)

    monkeypatch.setattr(filesystem, "_open_fd", looping)
    with pytest.raises(SkillResourceRefused) as caught:
        await FilesystemSkillSource(tmp_path).read_resource("s", "note.md")
    error = caught.value
    assert error.reason == "path_link"
    assert error.__cause__ is None
    assert error.__context__ is None
    for shown in (str(error), repr(error), *map(str, error.args)):
        assert MARKER not in shown


def test_opens_never_block() -> None:
    assert filesystem._OPEN_FLAGS & POSIX.O_NONBLOCK
    assert filesystem._OPEN_FLAGS & POSIX.O_NOFOLLOW


async def test_a_fifo_swapped_in_is_refused_without_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = _skill(tmp_path, "s")
    fifo = folder / "note.md"
    POSIX.mkfifo(fifo)
    real_isfile = os.path.isfile

    # The FIFO arrives between the regular-file check and the open.
    def isfile(path: object) -> bool:
        return True if str(path).endswith("note.md") else real_isfile(path)  # type: ignore[arg-type]

    monkeypatch.setattr(os.path, "isfile", isfile)

    # A safety valve: if the open ever blocks, a writer unblocks it after a while, so a
    # broken guard fails this test on time instead of hanging the run.
    done = threading.Event()

    def valve() -> None:
        if not done.wait(5):
            with open(fifo, "wb"):
                pass

    threading.Thread(target=valve, daemon=True).start()
    started = time.monotonic()
    try:
        with pytest.raises(SkillResourceRefused) as caught:
            await FilesystemSkillSource(tmp_path).read_resource("s", "note.md")
    finally:
        done.set()
    assert time.monotonic() - started < 3
    assert caught.value.reason == "resource_missing"
