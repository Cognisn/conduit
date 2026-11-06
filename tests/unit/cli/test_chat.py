"""``conduit chat``: streaming, tools under budgets, usage and cache, compaction, skills."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.unit.cli.helpers import SKILLS, cli, lines, stub_conduit, write_config


class TestStreamingChat:
    async def test_the_stub_s_answer_streams_back(self) -> None:
        result = await cli("chat", stdin=lines("hello from example.com"))
        assert result.code == 0
        assert "conduit chat on stub stub-echo (local)" in result.output
        assert "hello from example.com" in result.output
        assert "[run] completed" in result.output

    async def test_end_of_input_ends_the_chat(self) -> None:
        result = await cli("chat", stdin="first message\n")
        assert result.code == 0
        assert "first message" in result.output

    async def test_an_unknown_command_is_named(self) -> None:
        result = await cli("chat", stdin=lines("/nonsense"))
        assert "No command /nonsense" in result.output

    async def test_help_lists_the_commands(self) -> None:
        result = await cli("chat", stdin=lines("/help"))
        for command in ("/tools", "/skills", "/load", "/compact", "/usage", "/quit"):
            assert command in result.output


class TestToolsAndBudgets:
    async def test_the_model_calls_a_tool_and_answers_from_its_result(self) -> None:
        result = await cli("chat", stdin=lines('call add {"a": 2, "b": 3}'))
        assert "[tool] add called" in result.output
        assert "[tool] add: ok" in result.output
        assert "Tool results:" in result.output
        assert re.search(r"\b5\b", result.output.split("Tool results:", 1)[1])
        assert "2 model calls, 1 tool calls" in result.output

    async def test_the_tools_command_lists_the_interface_s_tools(self) -> None:
        result = await cli("chat", stdin=lines("/tools"))
        assert "Tools: add, word_count" in result.output

    async def test_a_tool_call_budget_stops_the_run(self) -> None:
        result = await cli(
            "chat", "--max-tool-calls", "0", stdin=lines('call word_count {"text": "a b c"}')
        )
        assert "[budget] the tool_calls budget is exhausted: 0 used of 0" in result.output
        assert "[run] failed: budget_exhausted (tool_calls)" in result.output
        assert "[tool] word_count: ok" not in result.output

    async def test_a_model_call_budget_stops_the_run(self) -> None:
        result = await cli(
            "chat", "--max-model-calls", "1", stdin=lines('call add {"a": 1, "b": 1}')
        )
        assert "[budget] the model_calls budget is exhausted: 1 used of 1" in result.output
        assert "[run] failed: budget_exhausted (model_calls)" in result.output

    async def test_the_configuration_s_budget_applies(self, tmp_path: Path) -> None:
        path = write_config(tmp_path, {"budget": {"max_model_calls": 1}})
        result = await cli("chat", "--config", path, stdin=lines('call add {"a": 1, "b": 1}'))
        assert "the model_calls budget is exhausted" in result.output


class TestUsageAndCache:
    async def test_every_model_call_shows_its_four_usage_counts(self) -> None:
        result = await cli("chat", stdin=lines("one"))
        assert re.search(
            r"\[usage\] stub stub-echo: input \d+, output \d+, cache read 0, cache write \d+",
            result.output,
        )

    async def test_the_cached_prefix_is_written_once_then_read(self) -> None:
        result = await cli("chat", stdin=lines("one", "two"))
        cache = re.findall(r"\[cache\] read (\d+), written (\d+)", result.output)
        assert len(cache) == 2
        (read1, written1), (read2, written2) = cache
        assert read1 == "0" and int(written1) > 0
        assert int(read2) == int(written1) and written2 == "0"

    async def test_usage_so_far_is_totalled(self) -> None:
        result = await cli("chat", stdin=lines("one", "/usage"))
        assert re.search(
            r"Usage so far: input \d+, output \d+, cache read 0, cache write \d+ "
            r"over 1 model calls",
            result.output,
        )
        assert "Conversation: 2 messages" in result.output


class TestCompaction:
    async def test_a_forced_compaction_shows_its_size_before_and_after(self) -> None:
        long = "a long synthetic message about example.com " * 40
        result = await cli(
            "chat",
            stdin=lines(long, long, "three", "/compact keep_last_turns", "four"),
        )
        assert "Compaction forced (keep_last_turns). Before: 6 messages" in result.output
        found = re.search(
            r"\[compaction\] forced keep_last_turns: compacted; before (\d+) tokens in (\d+) "
            r"messages, after (\d+) tokens in (\d+) messages",
            result.output,
        )
        assert found is not None
        before_tokens, before_messages, after_tokens, after_messages = map(int, found.groups())
        assert before_messages == 7 and after_messages < before_messages
        assert after_tokens < before_tokens
        assert "[compaction] the conversation went from 7 to" in result.output

    async def test_with_nothing_to_replace_it_says_so(self) -> None:
        result = await cli("chat", stdin=lines("/compact", "one"))
        assert "[compaction] forced keep_last_turns: skipped (nothing_to_compact)" in result.output

    async def test_an_unknown_strategy_is_refused_with_the_names(self) -> None:
        result = await cli("chat", stdin=lines("/compact shrink"))
        assert "no strategy named 'shrink'" in result.output
        assert "Strategies: drop_tool_results, keep_last_turns" in result.output


class TestSkills:
    async def test_the_skills_are_listed(self) -> None:
        result = await cli("chat", "--skills", str(SKILLS), stdin=lines("/skills"))
        assert "release-notes" in result.output and "status-report" in result.output

    async def test_the_model_loads_a_skill_and_uses_it(self) -> None:
        result = await cli(
            "chat",
            "--skills",
            str(SKILLS),
            stdin=lines('call use_skill {"name": "release-notes"}', "/skills"),
        )
        assert "[skill] loaded release-notes" in result.output
        assert "[tool] use_skill: ok" in result.output
        assert "Skill release-notes (version 1.2.0) is loaded" in result.output
        assert re.search(r"release-notes\s+1\.2\.0\s+loaded", result.output)

    async def test_a_skill_is_loaded_on_request(self) -> None:
        result = await cli("chat", "--skills", str(SKILLS), stdin=lines("/load status-report"))
        assert "[skill] loaded status-report" in result.output
        assert "Skill status-report is loaded" in result.output

    async def test_an_unknown_skill_is_refused(self) -> None:
        result = await cli("chat", "--skills", str(SKILLS), stdin=lines("/load nothing-here"))
        assert "The skill was not loaded: SkillNotFound" in result.output

    async def test_without_a_directory_skills_say_so(self) -> None:
        result = await cli("chat", stdin=lines("/skills"))
        assert "No skills directory was given" in result.output


class TestNothingIsKept:
    @pytest.fixture
    def quiet_home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        home = tmp_path / "home"
        work = tmp_path / "work"
        home.mkdir()
        work.mkdir()
        for name in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "XDG_DATA_HOME"):
            monkeypatch.setenv(name, str(home))
        monkeypatch.chdir(work)
        return tmp_path

    async def test_a_chat_writes_nothing_anywhere(self, quiet_home: Path) -> None:
        result = await cli(
            "chat",
            "--skills",
            str(SKILLS),
            stdin=lines(
                "a secret plan for example.com", 'call add {"a": 1, "b": 2}', "/compact", "again"
            ),
        )
        assert result.code == 0
        assert "kept in memory only" in result.output
        written = [p for p in quiet_home.rglob("*") if p.is_file()]
        assert written == []

    async def test_transcripts_are_kept_only_where_asked(self, quiet_home: Path) -> None:
        kept = quiet_home / "kept"
        result = await cli("chat", "--transcripts", str(kept), stdin=lines("keep this one"))
        assert result.code == 0
        assert "is written to the transcript cli-" in result.output
        files = [p for p in quiet_home.rglob("*") if p.is_file()]
        assert files and all(kept in p.parents for p in files)
        assert any("keep this one" in p.read_text(encoding="utf-8") for p in files)

    async def test_compaction_with_a_transcript_writes_its_point(self, quiet_home: Path) -> None:
        kept = quiet_home / "kept"
        result = await cli(
            "chat",
            "--transcripts",
            str(kept),
            stdin=lines("one", "two", "three", "/compact keep_last_turns", "four"),
        )
        assert "forced keep_last_turns: compacted" in result.output
        (stored,) = [p for p in kept.rglob("*") if p.is_file()]
        assert '"type":"compaction"' in stored.read_text(encoding="utf-8").replace(" ", "")


class TestShownSafely:
    async def test_control_characters_never_reach_the_terminal(self) -> None:
        result = await cli("chat", stdin=lines("\x1b[2Jcleared\x07 and \x9b31m"))
        assert "\x1b" not in result.output and "\x07" not in result.output
        assert "\x9b" not in result.output
        assert "cleared" in result.output

    @pytest.mark.parametrize("chunk_size", [4, 4096])
    async def test_brackets_are_shown_as_typed(self, tmp_path: Path, chunk_size: int) -> None:
        # One chunk carries a whole tag, as a real model's delta can.
        conduit = stub_conduit()
        conduit["providers"]["stub"]["chunk_size"] = chunk_size
        path = write_config(tmp_path, {"conduit": conduit})
        result = await cli("chat", "--config", path, stdin=lines("[bold]not markup[/bold]"))
        assert "[bold]not markup[/bold]" in result.output.split("Type /help", 1)[1]
