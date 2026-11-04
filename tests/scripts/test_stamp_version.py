"""scripts/stamp_version.py stamps only a version of its own release branch."""

from __future__ import annotations

from pathlib import Path

import pytest

from stamp_version import StampRefused, main, stamp


@pytest.fixture
def version_file(tmp_path: Path) -> Path:
    path = tmp_path / "_version.txt"
    path.write_bytes(b"0.1.0.dev0\n")
    return path


@pytest.mark.parametrize("version", ["0.1.0a0", "0.1.0a1", "0.1.0b12", "0.1.0rc1", "0.1.0"])
def test_a_version_of_the_release_is_stamped_with_lf(version_file: Path, version: str) -> None:
    stamp(version, branch="release/v0.1.0", version_path=version_file)
    assert version_file.read_bytes() == f"{version}\n".encode()


@pytest.mark.parametrize(
    "branch",
    [
        "main",
        "release/0.1.0",
        "feat/0.1.0-core",
        "release/v0.1",
        "release/v0.1.0-x",
        "xrelease/v0.1.0",
    ],
)
def test_only_a_v_style_release_branch_is_stamped(version_file: Path, branch: str) -> None:
    with pytest.raises(StampRefused, match=r"only release/vX\.Y\.Z"):
        stamp("0.1.0a1", branch=branch, version_path=version_file)
    assert version_file.read_bytes() == b"0.1.0.dev0\n"


@pytest.mark.parametrize(
    "version", ["0.1.0.dev0", "v0.1.0a1", "0.1.0a01", "0.1.0-a1", "0.1.0a1\n", "0.1.0c1", "0.1"]
)
def test_a_malformed_version_is_refused(version_file: Path, version: str) -> None:
    with pytest.raises(StampRefused, match="malformed"):
        stamp(version, branch="release/v0.1.0", version_path=version_file)
    assert version_file.read_bytes() == b"0.1.0.dev0\n"


def test_a_version_of_another_release_is_refused(version_file: Path) -> None:
    with pytest.raises(StampRefused, match="does not belong"):
        stamp("0.2.0a1", branch="release/v0.1.0", version_path=version_file)
    assert version_file.read_bytes() == b"0.1.0.dev0\n"


def test_a_missing_version_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(StampRefused, match="does not exist"):
        stamp("0.1.0a1", branch="release/v0.1.0", version_path=tmp_path / "absent.txt")


def test_main_needs_exactly_one_argument() -> None:
    assert main([]) == 2
    assert main(["0.1.0a1", "0.1.0a2"]) == 2
