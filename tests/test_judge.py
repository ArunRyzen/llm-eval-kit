"""Judges are testable offline: FakeJudge is deterministic; GeminiJudge is tested with a mock.

Why no real API calls here? Tests must be fast, free, and reproducible. So we test the real
judges' *plumbing* (prompt building, verdict parsing, error wrapping) against a mocked SDK client,
and use FakeJudge everywhere a test just needs "some judge".
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from llm_eval_kit.config import Settings
from llm_eval_kit.judge import AnthropicJudge, FakeJudge, GeminiJudge, make_judge
from llm_eval_kit.models import EvalCase

# --- FakeJudge (the offline stand-in) -------------------------------------------------


def test_fake_judge_passes_nonempty() -> None:
    s = FakeJudge().score("a real answer", EvalCase(id="t", input="q"))
    assert s.passed and s.value == 1.0
    assert s.scorer == "llm_judge"


def test_fake_judge_fails_empty() -> None:
    assert not FakeJudge().score("   ", EvalCase(id="t", input="q")).passed


def test_fake_judge_fixed_score() -> None:
    s = FakeJudge(fixed=0.4).score("anything", EvalCase(id="t", input="q"))
    assert s.value == 0.4 and not s.passed


# --- GeminiJudge (mocked client — no network, no key) ----------------------------------


def _gemini_judge() -> GeminiJudge:
    return GeminiJudge(model="gemini-2.5-flash", max_tokens=512, api_key="test-key")


def _mock_gemini_response(parsed: object, text: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(parsed=parsed, text=text)


def test_gemini_judge_parses_structured_verdict() -> None:
    from llm_eval_kit.judge import _Verdict

    verdict = _Verdict(score=0.9, passed=True, reasoning="Accurate and concise.")
    client = MagicMock()
    client.models.generate_content.return_value = _mock_gemini_response(verdict)

    with patch("google.genai.Client", return_value=client) as client_cls:
        s = _gemini_judge().score("Paris.", EvalCase(id="t", input="Capital of France?"))

    assert s.passed and s.value == 0.9
    assert s.scorer == "llm_judge"
    assert s.detail == "Accurate and concise."
    client_cls.assert_called_once_with(api_key="test-key")
    # The request must force robust JSON output via mime type + schema.
    config = client.models.generate_content.call_args.kwargs["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_schema is _Verdict


def test_gemini_judge_prompt_includes_case_and_reference() -> None:
    from llm_eval_kit.judge import _Verdict

    client = MagicMock()
    client.models.generate_content.return_value = _mock_gemini_response(
        _Verdict(score=1.0, passed=True, reasoning="ok")
    )
    with patch("google.genai.Client", return_value=client):
        _gemini_judge().score("Paris.", EvalCase(id="t", input="Capital?", reference="Paris"))

    prompt = client.models.generate_content.call_args.kwargs["contents"]
    assert "Capital?" in prompt and "Paris." in prompt and "Reference answer" in prompt


def test_gemini_judge_clamps_out_of_range_scores() -> None:
    from llm_eval_kit.judge import _Verdict

    client = MagicMock()
    client.models.generate_content.return_value = _mock_gemini_response(
        _Verdict(score=1.7, passed=True, reasoning="over-enthusiastic")
    )
    with patch("google.genai.Client", return_value=client):
        s = _gemini_judge().score("x", EvalCase(id="t", input="q"))
    assert s.value == 1.0  # clamped into [0, 1]


def test_gemini_judge_falls_back_to_json_text() -> None:
    client = MagicMock()
    client.models.generate_content.return_value = _mock_gemini_response(
        parsed=None, text='{"score": 0.3, "passed": false, "reasoning": "weak"}'
    )
    with patch("google.genai.Client", return_value=client):
        s = _gemini_judge().score("x", EvalCase(id="t", input="q"))
    assert s.value == 0.3 and not s.passed


def test_gemini_judge_wraps_api_errors() -> None:
    from google.genai import errors

    from llm_eval_kit.errors import JudgeError

    client = MagicMock()
    client.models.generate_content.side_effect = errors.APIError(
        500, {"error": {"message": "boom"}}
    )
    with patch("google.genai.Client", return_value=client):
        with pytest.raises(JudgeError):
            _gemini_judge().score("x", EvalCase(id="t", input="q"))


def test_gemini_judge_raises_on_empty_verdict() -> None:
    from llm_eval_kit.errors import JudgeError

    client = MagicMock()
    client.models.generate_content.return_value = _mock_gemini_response(parsed=None, text=None)
    with patch("google.genai.Client", return_value=client):
        with pytest.raises(JudgeError):
            _gemini_judge().score("x", EvalCase(id="t", input="q"))


# --- make_judge (the selection mechanism) ----------------------------------------------


def _settings(**overrides: object) -> Settings:
    # _env_file=None -> ignore any local .env so these tests are hermetic.
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type, call-arg]


def test_make_judge_prefers_gemini_when_key_set() -> None:
    judge = make_judge(_settings(gemini_api_key="g-key", anthropic_api_key="a-key"))
    assert isinstance(judge, GeminiJudge)


def test_make_judge_uses_anthropic_without_gemini_key() -> None:
    judge = make_judge(_settings(anthropic_api_key="a-key"))
    assert isinstance(judge, AnthropicJudge)


def test_make_judge_falls_back_to_fake_offline() -> None:
    judge = make_judge(_settings())
    assert isinstance(judge, FakeJudge)
