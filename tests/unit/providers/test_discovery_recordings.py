"""The committed discovery recordings are the ones their builders make, and scrubbed."""

from __future__ import annotations

import pytest

from conduit.testing.replay import load_recording
from tests.unit.providers import discovery_recordings as recordings


@pytest.mark.parametrize("filename", sorted(recordings.RECORDINGS))
def test_committed_recordings_match_their_builders(filename: str) -> None:
    assert load_recording(recordings.ROOT / filename) == recordings.RECORDINGS[filename]()
