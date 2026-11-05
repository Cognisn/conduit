"""The transcript and token store conformance suites, against the JSON file stores."""

from __future__ import annotations

from pathlib import Path

import pytest

from conduit.core.backends import TokenStore
from conduit.core.transcript import TranscriptStore
from conduit.stores.json import JsonTokenStore, JsonTranscriptStore
from conduit.testing.conformance import TokenStoreSuite, TranscriptStoreSuite


class TestJsonTranscriptStore(TranscriptStoreSuite):
    @pytest.fixture(autouse=True)
    def _directory(self, tmp_path: Path) -> None:
        self.directory = tmp_path / "transcripts"

    async def make_store(self) -> TranscriptStore:
        return JsonTranscriptStore(self.directory)

    async def reopen_store(self, store: TranscriptStore) -> TranscriptStore:
        return JsonTranscriptStore(self.directory)


class TestJsonTokenStore(TokenStoreSuite):
    @pytest.fixture(autouse=True)
    def _directory(self, tmp_path: Path) -> None:
        self.directory = tmp_path / "tokens"

    async def make_store(self) -> TokenStore:
        return JsonTokenStore(self.directory)

    async def reopen_store(self, store: TokenStore) -> TokenStore:
        return JsonTokenStore(self.directory)
