"""The filesystem skill source: discovery, loading and resource confinement."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conduit.core.backends import SkillSource, SkillSummary
from conduit.core.errors import ConfigurationError
from conduit.skills import filesystem
from conduit.skills.errors import (
    SkillInvalid,
    SkillNotFound,
    SkillResourceRefused,
    SkillUnavailable,
)
from conduit.skills.filesystem import FilesystemSkillSource, SourceLimits

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "skills"
MARKER = "zz-fs-marker-zz"


def _skill(root: Path, name: str, body: str = "Do the thing.\n", **front: str) -> Path:
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    fields = {"name": name, "description": f"The {name} skill.", **front}
    head = "".join(f"{k}: {v}\n" for k, v in fields.items())
    (folder / "SKILL.md").write_text(f"---\n{head}---\n{body}", encoding="utf-8", newline="\n")
    return folder


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _link(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        os.symlink(target, link, target_is_directory=directory)
    except OSError:
        pytest.skip("this platform does not let the test create symbolic links")


def _unchained(error: BaseException) -> None:
    assert error.__cause__ is None
    assert error.__context__ is None


def _clean(error: BaseException, *markers: str) -> None:
    _unchained(error)
    for shown in (str(error), repr(error), *map(str, error.args)):
        for marker in markers:
            assert marker not in shown


class TestFixtures:
    async def test_it_is_a_skill_source(self) -> None:
        assert isinstance(FilesystemSkillSource(FIXTURES), SkillSource)

    async def test_skills_are_listed_by_name(self) -> None:
        assert await FilesystemSkillSource(FIXTURES).list_skills() == [
            SkillSummary(
                name="release-notes",
                description=(
                    "Write release notes from a list of merged changes for example.com products."
                ),
                version="1.2.0",
            ),
            SkillSummary(
                name="status-report",
                description="Summarise a week's work for the example.org status page.",
            ),
        ]

    async def test_a_skill_loads_with_its_body_and_resources(self) -> None:
        skill = await FilesystemSkillSource(FIXTURES).load_skill("release-notes")
        assert skill.summary.version == "1.2.0"
        assert skill.body.startswith("# Release notes\n")
        assert sorted(skill.resources) == ["references/style.md", "templates/entry.md"]
        assert skill.resources["references/style.md"].startswith("# Style\n")

    async def test_the_document_keeps_the_recorded_fields(self) -> None:
        document = await FilesystemSkillSource(FIXTURES).load_document("release-notes")
        assert document.allowed_tools == ("read_file",)
        assert dict(document.metadata) == {"owner": "example-team"}
        assert document.license == "MIT"

    async def test_resources_are_listed_and_read_one_at_a_time(self) -> None:
        source = FilesystemSkillSource(FIXTURES)
        assert await source.list_resources("release-notes") == (
            "references/style.md",
            "templates/entry.md",
        )
        assert (await source.read_resource("release-notes", "templates/entry.md")).startswith(
            "## <version>"
        )

    async def test_a_skill_without_resources(self) -> None:
        skill = await FilesystemSkillSource(FIXTURES).load_skill("status-report")
        assert skill.resources == {}


class TestLookups:
    async def test_an_absent_skill_is_a_lookup_error(self, tmp_path: Path) -> None:
        with pytest.raises(LookupError) as caught:
            await FilesystemSkillSource(tmp_path).load_skill("absent")
        assert isinstance(caught.value, SkillNotFound)
        _unchained(caught.value)

    @pytest.mark.parametrize("name", ["../outside", "a/b", "C:/x", f"{MARKER} x", "", "A"])
    async def test_an_invalid_name_is_not_found_and_never_echoed(
        self, tmp_path: Path, name: str
    ) -> None:
        _skill(tmp_path / "inner", "outside")
        source = FilesystemSkillSource(tmp_path / "inner")
        with pytest.raises(SkillNotFound) as caught:
            await source.load_skill(name)
        assert caught.value.skill is None
        _clean(caught.value, MARKER)

    async def test_a_folder_without_skill_md_is_not_a_skill(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        source = FilesystemSkillSource(tmp_path)
        assert await source.list_skills() == []
        with pytest.raises(SkillNotFound):
            await source.load_skill("empty")

    async def test_hidden_and_plain_files_at_the_root_are_ignored(self, tmp_path: Path) -> None:
        _skill(tmp_path, "real")
        _write(tmp_path / ".hidden" / "SKILL.md", "---\nname: hidden\ndescription: d\n---\nb\n")
        _write(tmp_path / "notes.txt", "not a skill")
        names = [s.name for s in await FilesystemSkillSource(tmp_path).list_skills()]
        assert names == ["real"]

    def test_the_directory_must_exist(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError) as caught:
            FilesystemSkillSource(tmp_path / MARKER)
        _clean(caught.value, MARKER)


class TestValidation:
    async def test_an_invalid_skill_is_left_out_of_the_listing(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _skill(tmp_path, "good")
        _write(tmp_path / "bad" / "SKILL.md", f"---\nname: bad\n{MARKER}: x\n---\nb\n")
        assert [s.name for s in await FilesystemSkillSource(tmp_path).list_skills()] == ["good"]
        assert "field_unknown" in caplog.text
        assert MARKER not in caplog.text

    async def test_strict_listing_refuses_an_invalid_skill(self, tmp_path: Path) -> None:
        _skill(tmp_path, "good")
        _write(tmp_path / "bad" / "SKILL.md", "---\nname: bad\n---\nb\n")
        with pytest.raises(SkillInvalid) as caught:
            await FilesystemSkillSource(tmp_path, strict=True).list_skills()
        assert (caught.value.reason, caught.value.skill) == ("description_missing", "bad")

    async def test_loading_an_invalid_skill_is_refused(self, tmp_path: Path) -> None:
        _write(tmp_path / "bad" / "SKILL.md", "---\nname: other\ndescription: d\n---\nb\n")
        with pytest.raises(SkillInvalid) as caught:
            await FilesystemSkillSource(tmp_path).load_skill("bad")
        assert caught.value.reason == "name_mismatch"
        _unchained(caught.value)

    async def test_a_folder_with_an_invalid_name_is_not_listed(self, tmp_path: Path) -> None:
        _write(tmp_path / "Bad_Name" / "SKILL.md", "---\nname: bad\ndescription: d\n---\nb\n")
        assert await FilesystemSkillSource(tmp_path).list_skills() == []

    async def test_an_oversized_skill_file_is_refused(self, tmp_path: Path) -> None:
        _skill(tmp_path, "big", body="x" * 200)
        source = FilesystemSkillSource(tmp_path, limits=SourceLimits(max_skill_file_bytes=100))
        with pytest.raises(SkillInvalid) as caught:
            await source.load_skill("big")
        assert caught.value.reason == "skill_file_too_large"

    async def test_a_skill_file_at_the_size_limit_is_accepted(self, tmp_path: Path) -> None:
        folder = _skill(tmp_path, "edge")
        size = (folder / "SKILL.md").stat().st_size
        source = FilesystemSkillSource(tmp_path, limits=SourceLimits(max_skill_file_bytes=size))
        assert (await source.load_skill("edge")).summary.name == "edge"

    async def test_a_skill_file_that_is_not_utf8_is_refused(self, tmp_path: Path) -> None:
        folder = tmp_path / "latin"
        folder.mkdir()
        (folder / "SKILL.md").write_bytes(b"---\nname: latin\ndescription: caf\xe9\n---\nb\n")
        with pytest.raises(SkillInvalid) as caught:
            await FilesystemSkillSource(tmp_path).load_skill("latin")
        assert caught.value.reason == "skill_file_not_text"
        _unchained(caught.value)


class TestResourceLimits:
    async def test_an_oversized_resource_is_refused(self, tmp_path: Path) -> None:
        folder = _skill(tmp_path, "s")
        _write(folder / "big.md", "x" * 101)
        source = FilesystemSkillSource(tmp_path, limits=SourceLimits(max_resource_bytes=100))
        with pytest.raises(SkillResourceRefused) as caught:
            await source.load_skill("s")
        assert caught.value.reason == "resource_too_large"
        with pytest.raises(SkillResourceRefused) as caught:
            await source.read_resource("s", "big.md")
        assert caught.value.reason == "resource_too_large"

    async def test_a_resource_at_the_limit_is_accepted(self, tmp_path: Path) -> None:
        folder = _skill(tmp_path, "s")
        _write(folder / "edge.md", "x" * 100)
        source = FilesystemSkillSource(tmp_path, limits=SourceLimits(max_resource_bytes=100))
        assert (await source.load_skill("s")).resources["edge.md"] == "x" * 100

    async def test_the_total_resource_size_is_bounded(self, tmp_path: Path) -> None:
        folder = _skill(tmp_path, "s")
        _write(folder / "a.md", "x" * 60)
        _write(folder / "b.md", "x" * 60)
        source = FilesystemSkillSource(tmp_path, limits=SourceLimits(max_total_resource_bytes=100))
        with pytest.raises(SkillResourceRefused) as caught:
            await source.load_skill("s")
        assert caught.value.reason == "resources_too_large"
        exact = FilesystemSkillSource(tmp_path, limits=SourceLimits(max_total_resource_bytes=120))
        assert len((await exact.load_skill("s")).resources) == 2

    async def test_the_resource_count_is_bounded(self, tmp_path: Path) -> None:
        folder = _skill(tmp_path, "s")
        for i in range(3):
            _write(folder / f"r{i}.md", "x")
        source = FilesystemSkillSource(tmp_path, limits=SourceLimits(max_resources=2))
        with pytest.raises(SkillResourceRefused) as caught:
            await source.load_skill("s")
        assert caught.value.reason == "resources_too_many"
        exact = FilesystemSkillSource(tmp_path, limits=SourceLimits(max_resources=3))
        assert len((await exact.load_skill("s")).resources) == 3

    async def test_the_folder_depth_is_bounded(self, tmp_path: Path) -> None:
        folder = _skill(tmp_path, "s")
        _write(folder / "a" / "b" / "c" / "deep.md", "x")
        source = FilesystemSkillSource(tmp_path, limits=SourceLimits(max_depth=3))
        with pytest.raises(SkillResourceRefused) as caught:
            await source.load_skill("s")
        assert caught.value.reason == "resource_too_deep"
        exact = FilesystemSkillSource(tmp_path, limits=SourceLimits(max_depth=4))
        assert list((await exact.load_skill("s")).resources) == ["a/b/c/deep.md"]

    async def test_binary_files_are_not_resources(self, tmp_path: Path) -> None:
        folder = _skill(tmp_path, "s")
        (folder / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00")
        (folder / "latin.txt").write_bytes(b"caf\xe9")
        (folder / "nul.txt").write_bytes(b"text\x00more")
        _write(folder / "notes.md", "text")
        source = FilesystemSkillSource(tmp_path)
        assert list((await source.load_skill("s")).resources) == ["notes.md"]
        for path in ("logo.png", "latin.txt", "nul.txt"):
            with pytest.raises(SkillResourceRefused) as caught:
                await source.read_resource("s", path)
            assert caught.value.reason == "resource_not_text"
            _unchained(caught.value)

    async def test_files_outside_the_path_rules_are_not_resources(self, tmp_path: Path) -> None:
        folder = _skill(tmp_path, "s")
        _write(folder / "has space.md", "x")
        _write(folder / ".git" / "config", "x")
        _write(folder / "ok.md", "x")
        assert list((await FilesystemSkillSource(tmp_path).load_skill("s")).resources) == ["ok.md"]


class TestConfinement:
    @pytest.mark.parametrize(
        ("path", "reason"),
        [
            ("../outside.md", "path_traversal"),
            ("references/../../outside.md", "path_traversal"),
            ("/etc/hosts", "path_absolute"),
            ("C:/Windows/win.ini", "path_absolute"),
            ("\\\\server\\share\\x", "path_absolute"),
            ("references\\style.md", "path_invalid"),
            ("SKILL.md", "path_reserved"),
            ("SKILL.md.", "path_invalid"),
            ("references/style.md.", "path_invalid"),
        ],
    )
    async def test_a_path_outside_the_skill_is_refused(
        self, tmp_path: Path, path: str, reason: str
    ) -> None:
        root = tmp_path / "root"
        folder = _skill(root, "s")
        _write(folder / "references" / "style.md", "inside")
        _write(root / "outside.md", MARKER)
        _write(tmp_path / "outside.md", MARKER)
        with pytest.raises(SkillResourceRefused) as caught:
            await FilesystemSkillSource(root).read_resource("s", path)
        assert caught.value.reason == reason
        assert caught.value.skill == "s"
        _clean(caught.value, MARKER, path)

    async def test_an_absolute_path_to_a_real_file_inside_is_still_refused(
        self, tmp_path: Path
    ) -> None:
        folder = _skill(tmp_path, "s")
        _write(folder / "inside.md", "x")
        with pytest.raises(SkillResourceRefused) as caught:
            await FilesystemSkillSource(tmp_path).read_resource("s", str(folder / "inside.md"))
        assert caught.value.reason == "path_absolute"

    async def test_a_missing_resource_is_refused(self, tmp_path: Path) -> None:
        _skill(tmp_path, "s")
        with pytest.raises(SkillResourceRefused) as caught:
            await FilesystemSkillSource(tmp_path).read_resource("s", "absent.md")
        assert caught.value.reason == "resource_missing"

    async def test_a_directory_is_not_a_resource(self, tmp_path: Path) -> None:
        folder = _skill(tmp_path, "s")
        (folder / "references").mkdir()
        with pytest.raises(SkillResourceRefused) as caught:
            await FilesystemSkillSource(tmp_path).read_resource("s", "references")
        assert caught.value.reason == "resource_missing"

    async def test_a_resource_in_another_skill_is_not_reachable(self, tmp_path: Path) -> None:
        _skill(tmp_path, "a")
        other = _skill(tmp_path, "b")
        _write(other / "secret.md", MARKER)
        with pytest.raises(SkillResourceRefused) as caught:
            await FilesystemSkillSource(tmp_path).read_resource("a", "../b/secret.md")
        assert caught.value.reason == "path_traversal"

    async def test_a_link_to_a_file_outside_is_refused(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        folder = _skill(root, "s")
        _write(tmp_path / "outside.md", MARKER)
        _link(folder / "escape.md", tmp_path / "outside.md")
        source = FilesystemSkillSource(root)
        with pytest.raises(SkillResourceRefused) as caught:
            await source.read_resource("s", "escape.md")
        assert caught.value.reason == "path_link"
        _clean(caught.value, MARKER)
        with pytest.raises(SkillResourceRefused) as caught:
            await source.load_skill("s")
        assert caught.value.reason == "path_link"
        _clean(caught.value, MARKER)

    async def test_a_link_to_a_folder_outside_is_refused(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        folder = _skill(root, "s")
        _write(tmp_path / "elsewhere" / "data.md", MARKER)
        _link(folder / "linked", tmp_path / "elsewhere", directory=True)
        source = FilesystemSkillSource(root)
        with pytest.raises(SkillResourceRefused) as caught:
            await source.read_resource("s", "linked/data.md")
        assert caught.value.reason == "path_link"
        with pytest.raises(SkillResourceRefused) as caught:
            await source.load_skill("s")
        assert caught.value.reason == "path_link"

    async def test_a_hidden_link_to_outside_is_still_refused(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        folder = _skill(root, "s")
        _write(tmp_path / "outside.md", MARKER)
        _link(folder / ".hidden", tmp_path / "outside.md")
        with pytest.raises(SkillResourceRefused) as caught:
            await FilesystemSkillSource(root).load_skill("s")
        assert caught.value.reason == "path_link"

    async def test_a_link_inside_the_skill_is_allowed(self, tmp_path: Path) -> None:
        folder = _skill(tmp_path, "s")
        _write(folder / "real.md", "inside")
        _link(folder / "alias.md", folder / "real.md")
        source = FilesystemSkillSource(tmp_path)
        assert await source.read_resource("s", "alias.md") == "inside"
        assert (await source.load_skill("s")).resources["alias.md"] == "inside"

    async def test_a_skill_md_linked_from_outside_is_refused(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        (root / "s").mkdir(parents=True)
        _write(tmp_path / "SKILL.md", "---\nname: s\ndescription: d\n---\nb\n")
        _link(root / "s" / "SKILL.md", tmp_path / "SKILL.md")
        source = FilesystemSkillSource(root)
        assert await source.list_skills() == []
        with pytest.raises(SkillResourceRefused) as caught:
            await source.load_skill("s")
        assert caught.value.reason == "path_link"

    async def test_a_skill_folder_linked_from_outside_is_refused(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        _skill(tmp_path / "elsewhere", "s")
        _link(root / "s", tmp_path / "elsewhere" / "s", directory=True)
        source = FilesystemSkillSource(root)
        assert await source.list_skills() == []
        with pytest.raises(SkillResourceRefused) as caught:
            await source.load_skill("s")
        assert caught.value.reason == "path_link"
        with pytest.raises(SkillResourceRefused):
            await source.read_resource("s", "SKILL.md".lower() + "x")


def _junction(link: Path, target: Path) -> None:
    if sys.platform != "win32":
        pytest.skip("junctions are a Windows feature")
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("this machine does not let the test create a junction")


class TestJunctions:
    async def test_a_junction_to_a_folder_outside_is_refused(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        folder = _skill(root, "s")
        _write(tmp_path / "elsewhere" / "data.md", MARKER)
        _junction(folder / "linked", tmp_path / "elsewhere")
        source = FilesystemSkillSource(root)
        with pytest.raises(SkillResourceRefused) as caught:
            await source.read_resource("s", "linked/data.md")
        assert caught.value.reason == "path_link"
        with pytest.raises(SkillResourceRefused) as caught:
            await source.load_skill("s")
        assert caught.value.reason == "path_link"

    async def test_a_skill_folder_that_is_a_junction_outside_is_refused(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "root"
        root.mkdir()
        _skill(tmp_path / "elsewhere", "s")
        _junction(root / "s", tmp_path / "elsewhere" / "s")
        source = FilesystemSkillSource(root)
        assert await source.list_skills() == []
        with pytest.raises(SkillResourceRefused) as caught:
            await source.load_skill("s")
        assert caught.value.reason == "path_link"


class TestOpenAfterCheck:
    """The file read is the file checked: confinement is checked again on the open handle."""

    def _swap_on_open(
        self, monkeypatch: pytest.MonkeyPatch, victim: Path, outside: Path
    ) -> list[str]:
        """Replace ``victim`` with a link to ``outside`` just before it is opened, once.
        Returns the paths swapped (one, once the swap has happened)."""
        swapped: list[str] = []
        original = filesystem._open_fd
        expected = os.path.normcase(str(victim.resolve()))

        def swapping(path: str) -> int:
            if not swapped and os.path.normcase(path) == expected:
                victim.unlink()
                _link(victim, outside)
                swapped.append(path)
            return original(path)

        monkeypatch.setattr(filesystem, "_open_fd", swapping)
        return swapped

    async def test_a_resource_swapped_for_a_link_after_the_check_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "root"
        folder = _skill(root, "s")
        _write(folder / "note.md", "inside")
        _write(tmp_path / "outside.md", MARKER)
        _link(tmp_path / "probe", tmp_path / "outside.md")  # skip early without links
        swapped = self._swap_on_open(monkeypatch, folder / "note.md", tmp_path / "outside.md")
        with pytest.raises(SkillResourceRefused) as caught:
            await FilesystemSkillSource(root).read_resource("s", "note.md")
        assert swapped
        assert caught.value.reason == "path_link"
        _clean(caught.value, MARKER)

    async def test_a_resource_swapped_during_load_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "root"
        folder = _skill(root, "s")
        _write(folder / "note.md", "inside")
        _write(tmp_path / "outside.md", MARKER)
        _link(tmp_path / "probe", tmp_path / "outside.md")
        swapped = self._swap_on_open(monkeypatch, folder / "note.md", tmp_path / "outside.md")
        with pytest.raises(SkillResourceRefused) as caught:
            await FilesystemSkillSource(root).load_skill("s")
        assert swapped
        assert caught.value.reason == "path_link"

    async def test_a_skill_file_swapped_after_the_check_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "root"
        folder = _skill(root, "s")
        _write(tmp_path / "SKILL.md", f"---\nname: s\ndescription: {MARKER}\n---\nb\n")
        _link(tmp_path / "probe", tmp_path / "SKILL.md")
        swapped = self._swap_on_open(monkeypatch, folder / "SKILL.md", tmp_path / "SKILL.md")
        with pytest.raises(SkillResourceRefused) as caught:
            await FilesystemSkillSource(root).load_document("s")
        assert swapped
        assert caught.value.reason == "path_link"
        _clean(caught.value, MARKER)

    @pytest.mark.parametrize("operation", ["read", "load", "document"])
    async def test_a_different_file_behind_the_handle_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
    ) -> None:
        folder = _skill(tmp_path, "s")
        _write(folder / "note.md", "inside")
        monkeypatch.setattr(filesystem, "_identity", lambda path: (-1, -1))
        source = FilesystemSkillSource(tmp_path)
        with pytest.raises(SkillResourceRefused) as caught:
            if operation == "read":
                await source.read_resource("s", "note.md")
            elif operation == "load":
                await source.load_skill("s")
            else:
                await source.load_document("s")
        assert caught.value.reason == "path_link"

    async def test_something_other_than_a_regular_file_behind_the_handle_is_not_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import stat as stat_module

        folder = _skill(tmp_path, "s")
        _write(folder / "note.md", MARKER)
        original = filesystem._fstat

        def as_directory(fd: int) -> os.stat_result:
            fields = list(original(fd))
            fields[0] = stat_module.S_IFDIR | 0o755
            return os.stat_result(fields)

        monkeypatch.setattr(filesystem, "_fstat", as_directory)
        with pytest.raises(SkillResourceRefused) as caught:
            await FilesystemSkillSource(tmp_path).read_resource("s", "note.md")
        assert caught.value.reason == "resource_missing"

    async def test_the_skill_file_is_read_no_further_than_its_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        folder = _skill(tmp_path, "s", body="x" * 1_000_000)
        requested: list[int] = []
        original = os.read

        def counting(fd: int, n: int) -> bytes:
            requested.append(n)
            return original(fd, n)

        monkeypatch.setattr(os, "read", counting)
        source = FilesystemSkillSource(tmp_path, limits=SourceLimits(max_skill_file_bytes=100))
        with pytest.raises(SkillInvalid) as caught:
            await source.load_document("s")
        assert caught.value.reason == "skill_file_too_large"
        assert 0 < sum(requested) <= 101
        assert (folder / "SKILL.md").stat().st_size > 1_000_000


class TestFailures:
    async def test_an_unreadable_folder_is_unavailable_without_its_cause(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _skill(tmp_path, "s")

        def broken(path: object) -> object:
            raise PermissionError(f"denied {MARKER}")

        monkeypatch.setattr(os, "scandir", broken)
        with pytest.raises(SkillUnavailable) as caught:
            await FilesystemSkillSource(tmp_path).list_skills()
        assert caught.value.error_class == "PermissionError"
        _clean(caught.value, MARKER)

    async def test_cancellation_propagates_unchanged(self, tmp_path: Path) -> None:
        _skill(tmp_path, "s")
        task = asyncio.ensure_future(FilesystemSkillSource(tmp_path).load_skill("s"))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
