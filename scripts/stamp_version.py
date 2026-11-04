"""Stamp conduit's version.

Used only by the integration session, on a release branch:

    python scripts/stamp_version.py 0.1.0a1

The version lives in one place, ``src/conduit/_version.txt``; hatchling reads it
for the distribution metadata and ``conduit.__version__`` reads it at import.
The script refuses to stamp anywhere but ``release/vX.Y.Z``, and only a version
of that same release: the final version or an alpha, beta or release candidate
(``a0`` is allowed, for the opening stamp of a release branch).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

VERSION_PATH = Path(__file__).resolve().parents[1] / "src" / "conduit" / "_version.txt"

_RELEASE_BRANCH = re.compile(r"release/v(\d+\.\d+\.\d+)")
_VERSION = re.compile(r"(\d+\.\d+\.\d+)(?:(?:a|b|rc)(?:0|[1-9]\d*))?")


class StampRefused(Exception):
    """The requested stamp is not allowed."""


def stamp(version: str, *, branch: str, version_path: Path = VERSION_PATH) -> None:
    """Write ``version`` into ``version_path``, or raise ``StampRefused``."""
    branch_match = _RELEASE_BRANCH.fullmatch(branch)
    if branch_match is None:
        raise StampRefused(f"refusing to stamp on {branch!r}: only release/vX.Y.Z branches")
    version_match = _VERSION.fullmatch(version)
    if version_match is None:
        raise StampRefused(f"refusing malformed version {version!r}")
    if version_match.group(1) != branch_match.group(1):
        raise StampRefused(f"version {version} does not belong to {branch}")
    if not version_path.is_file():
        raise StampRefused(f"{version_path} does not exist")
    # newline="\n": the repository is eol=lf, and a text-mode write on Windows emits CRLF.
    version_path.write_text(f"{version}\n", encoding="utf-8", newline="\n")


def _current_branch() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: stamp_version.py VERSION", file=sys.stderr)
        return 2
    try:
        stamp(argv[0], branch=_current_branch())
    except StampRefused as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"stamped {argv[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
