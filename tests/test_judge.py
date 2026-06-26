"""The FakeJudge gives deterministic, interface-compatible scores."""

from __future__ import annotations

from llm_eval_kit.judge import FakeJudge
from llm_eval_kit.models import EvalCase


def test_fake_judge_passes_nonempty() -> None:
    s = FakeJudge().score("a real answer", EvalCase(id="t", input="q"))
    assert s.passed and s.value == 1.0
    assert s.scorer == "llm_judge"


def test_fake_judge_fails_empty() -> None:
    assert not FakeJudge().score("   ", EvalCase(id="t", input="q")).passed


def test_fake_judge_fixed_score() -> None:
    s = FakeJudge(fixed=0.4).score("anything", EvalCase(id="t", input="q"))
    assert s.value == 0.4 and not s.passed
