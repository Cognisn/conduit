"""The transcript store conformance suite, against the in-memory store."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

from conduit.core.transcript import TranscriptStore
from conduit.stores.memory import MemoryTranscriptStore
from conduit.testing.conformance import TranscriptStoreSuite


class TestMemoryTranscriptStore(TranscriptStoreSuite):
    async def make_store(self) -> TranscriptStore:
        return MemoryTranscriptStore()


class _Broken(MemoryTranscriptStore):
    """Numbers each append's entries from 1, the kind of defect the suite must catch."""

    async def append(self, session_id, entries):  # type: ignore[no-untyped-def]
        stored = await super().append(session_id, entries)
        return tuple(e.model_copy(update={"seq": i}) for i, e in enumerate(stored, start=1))


def test_the_plugin_marks_suites_and_catches_a_broken_store(tmp_path: Path) -> None:
    # Run the suite as a product would: in its own project, with the plugin loaded.
    (tmp_path / "test_product.py").write_text(
        textwrap.dedent(
            """
            from conduit.stores.memory import MemoryTranscriptStore
            from conduit.testing.conformance import TranscriptStoreSuite

            class TestGood(TranscriptStoreSuite):
                async def make_store(self):
                    return MemoryTranscriptStore()

            class TestBroken(TranscriptStoreSuite):
                async def make_store(self):
                    from tests.conformance.test_memory_transcript_store import _Broken
                    return _Broken()

            def test_not_a_suite():
                pass
            """
        ),
        encoding="utf-8",
        newline="\n",
    )
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "conduit.testing.conformance",
            "-m",
            "conformance",
            "-q",
            "-p",
            "no:cacheprovider",
            "--rootdir",
            str(tmp_path),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(repo)},
        timeout=120,
        check=False,
    )
    summary = result.stdout.strip().splitlines()[-1]
    # Eight scenarios per suite. The broken store returns wrong sequence numbers, which
    # the ordering scenario and the refused-append scenario (checking the numbers of
    # the batch it then accepts) both catch; the plain test is deselected by the marker.
    assert summary.startswith("2 failed, 14 passed, 1 deselected"), result.stdout
    assert "TestBroken::test_sequence_numbers_are_assigned_in_append_order" in result.stdout
    assert "TestBroken::test_an_append_that_would_not_load_is_refused_whole" in result.stdout
    assert "FAILED test_product.py::TestGood" not in result.stdout
