"""Live: the Gemini API, and a local Ollama server.

Gemini caches repeated prefixes implicitly and Ollama reuses its prompt cache when the
server version reports ``prompt_eval_cached_count``; both are best effort, so a second
call that reads nothing from the cache is reported as unobserved, not failed.
"""

from __future__ import annotations

import pytest

from tests.live import needs
from tests.live.harness import check_caching, check_chat, check_tools, started
from tests.live.needs import live_env, value
from tests.live.report import REPORT, Verdict
from tests.live.targets import Target, gemini_settings, ollama_settings

pytestmark = pytest.mark.integration

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"


def _gemini() -> Target:
    model = value(live_env(), "GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    assert model is not None
    return Target("gemini", gemini_settings(), model)


def _ollama() -> Target:
    model = value(live_env(), "OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    assert model is not None
    return Target("ollama", ollama_settings(live_env()), model)


@pytest.mark.live_needs(needs.GEMINI)
async def test_gemini_streaming_chat() -> None:
    target = _gemini()
    async with started(target) as binding:
        await check_chat(binding, target.subject)


@pytest.mark.live_needs(needs.GEMINI)
async def test_gemini_tool_loop_with_thought_signatures() -> None:
    target = _gemini()
    async with started(target) as binding:
        await check_tools(binding, target.subject)


@pytest.mark.live_needs(needs.GEMINI)
async def test_gemini_cached_second_call() -> None:
    target = _gemini()
    async with started(target) as binding:
        await check_caching(binding, target.subject, explicit=False, strict=False)


@pytest.mark.live_needs(needs.OLLAMA)
async def test_ollama_streaming_chat() -> None:
    target = _ollama()
    async with started(target) as binding:
        await check_chat(binding, target.subject)


@pytest.mark.live_needs(needs.OLLAMA)
async def test_ollama_tool_loop() -> None:
    target = _ollama()
    async with started(target) as binding:
        await check_tools(binding, target.subject)


@pytest.mark.live_needs(needs.OLLAMA)
async def test_ollama_prompt_eval_cached_count() -> None:
    """Whether this Ollama server reports cached prompt tokens on a repeated prefix."""
    target = _ollama()
    async with started(target) as binding:
        _, second = await check_caching(binding, target.subject, explicit=False, strict=False)
    REPORT.record(
        "usage",
        target.subject,
        "prompt_eval_cached_count reported",
        Verdict.FINDING,
        "yes" if second.cache_read_tokens > 0 else "no cached count on the repeated prefix",
    )
