"""LLM_DEBUG is a learning aid: silent by default, verbose on stderr when switched on.

These tests run fully offline via FakeJudge — no API key, no network. They prove four things:
the kit is silent when LLM_DEBUG is unset (or falsy), it prints request/response blocks to
stderr when set, it never floods the terminal (long fields are truncated), and a project
``.env`` file can switch it on — with the real environment variable always taking precedence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_eval_kit.debuglog import _dotenv_llm_debug, debug_enabled, log_block
from llm_eval_kit.judge import FakeJudge
from llm_eval_kit.models import EvalCase

_CASE = EvalCase(id="t", input="Capital of France?", reference="Paris")


def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run the test from an empty temp dir so the developer's real .env can't leak in."""
    monkeypatch.chdir(tmp_path)
    _dotenv_llm_debug.cache_clear()


# --- the on/off switch ------------------------------------------------------------------


def test_debug_disabled_when_unset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _isolate(monkeypatch, tmp_path)  # no .env here — and no env var below
    monkeypatch.delenv("LLM_DEBUG", raising=False)
    assert not debug_enabled()


@pytest.mark.parametrize("value", ["0", "false", "False", "FALSE", ""])
def test_debug_disabled_for_falsy_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("LLM_DEBUG", value)
    assert not debug_enabled()


@pytest.mark.parametrize("value", ["1", "true", "yes", "anything"])
def test_debug_enabled_for_truthy_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("LLM_DEBUG", value)
    assert debug_enabled()


# --- the .env fallback ------------------------------------------------------------------


def test_dotenv_file_enables_debug(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("LLM_DEBUG=1\n", encoding="utf-8")
    _isolate(monkeypatch, tmp_path)
    monkeypatch.delenv("LLM_DEBUG", raising=False)  # no env var → .env decides
    assert debug_enabled()


def test_env_var_beats_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("LLM_DEBUG=1\n", encoding="utf-8")
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("LLM_DEBUG", "0")  # env var is set → it wins, .env is ignored
    assert not debug_enabled()


def test_env_var_beats_dotenv_in_the_other_direction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".env").write_text("LLM_DEBUG=0\n", encoding="utf-8")
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("LLM_DEBUG", "1")
    assert debug_enabled()


def test_falsy_dotenv_value_stays_off(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("LLM_DEBUG=false\n", encoding="utf-8")
    _isolate(monkeypatch, tmp_path)
    monkeypatch.delenv("LLM_DEBUG", raising=False)
    assert not debug_enabled()


def test_neither_env_var_nor_dotenv_means_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate(monkeypatch, tmp_path)  # empty dir: no .env at all
    monkeypatch.delenv("LLM_DEBUG", raising=False)
    assert not debug_enabled()


# --- silence by default -----------------------------------------------------------------


def test_judge_is_silent_when_debug_unset(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _isolate(monkeypatch, tmp_path)
    monkeypatch.delenv("LLM_DEBUG", raising=False)
    FakeJudge().score("Paris.", _CASE)
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_log_block_is_silent_when_debug_is_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LLM_DEBUG", "0")
    log_block("AI REQUEST (judge: offline fake judge)", judge_prompt="hello")
    assert capsys.readouterr().err == ""


# --- verbose when switched on -----------------------------------------------------------


def test_judge_logs_request_and_response_to_stderr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LLM_DEBUG", "1")
    FakeJudge().score("Paris.", _CASE)
    captured = capsys.readouterr()
    err = captured.err

    # Request block: labelled as the offline judge, showing the full judge prompt.
    assert "=== AI REQUEST (judge: offline fake judge) ===" in err
    assert "judge prompt:" in err
    assert "Capital of France?" in err  # the question
    assert "Paris." in err  # the candidate answer under judgment
    assert "Rubric: rate the answer on" in err  # the criteria

    # Response block: the parsed verdict.
    assert "=== AI RESPONSE (judge) ===" in err
    assert "verdict:" in err
    assert '"passed":true' in err

    # Debug output goes to stderr only — stdout stays clean for real program output.
    assert captured.out == ""


def test_log_block_truncates_long_fields(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LLM_DEBUG", "1")
    log_block("AI REQUEST (judge: offline fake judge)", judge_prompt="x" * 5000)
    err = capsys.readouterr().err
    assert "... [truncated]" in err
    assert "x" * 5000 not in err
