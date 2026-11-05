"""The default catalogue's Ollama entries, as the Ollama library pages give them.

Sources (checked 2026-11-04): ollama.com/library/llama3.1/tags (8b: 128K context,
text input, tools), ollama.com/library/qwen2.5/tags (72b: 32K context, text input,
tools) and ollama.com/library/nomic-embed-text (latest: 2K context). Every entry runs
on the deployment's own hosts.
"""

from __future__ import annotations

import pytest

from conduit.core.catalogue import default_catalogue
from conduit.core.residency import ResidencyScope


def test_every_ollama_entry_is_local() -> None:
    entries = [e for e in default_catalogue() if e.provider == "ollama"]
    assert {e.model_id for e in entries} == {"llama3.1:8b", "qwen2.5:72b", "nomic-embed-text"}
    assert all(e.residency_scope is ResidencyScope.LOCAL for e in entries)


@pytest.mark.parametrize(("model", "window"), [("llama3.1:8b", 131072), ("qwen2.5:72b", 32768)])
def test_a_chat_entry(model: str, window: int) -> None:
    entry = default_catalogue().require("ollama", model)
    assert entry.context_window == window
    assert entry.capabilities.tools
    assert not entry.capabilities.vision
    assert not entry.capabilities.prompt_caching


def test_the_embedding_entry() -> None:
    entry = default_catalogue().require("ollama", "nomic-embed-text")
    assert entry.context_window == 2048
    assert entry.embedding_dimensions == 768
