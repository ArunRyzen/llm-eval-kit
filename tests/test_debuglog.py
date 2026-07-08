"""LLM_DEBUG is a learning aid: silent by default, verbose on stderr when switched on.

These tests run fully offline via FakeJudge — no API key, no network. They prove three things:
the kit is silent when LLM_DEBUG is unset (or falsy), it prints request/response blocks to
stderr when set, and it never floods the terminal (long fields are truncated).
"""

from __future__ import annotations

import pytest

from llm_eval_kit.debuglog import debug_enabled, log_block
from llm_eval_kit.judge import FakeJudge
from llm_eval_kit.models import EvalCase

_CASE = EvalCase(id="t", input="Capital of France?", reference="Paris")

# --- the on/off switch ------------------------------------------------------------------


def test_debug_disabled_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
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


# --- silence by default -----------------------------------------------------------------


def test_judge_is_silent_when_debug_unset(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
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
