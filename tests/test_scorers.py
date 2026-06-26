"""Scorers produce correct, bounded scores."""

from __future__ import annotations

from llm_eval_kit.models import EvalCase
from llm_eval_kit.scorers import (
    ContainsReference,
    ExactMatch,
    JaccardSimilarity,
    JsonValid,
    RegexMatch,
)


def _case(reference: str | None = None) -> EvalCase:
    return EvalCase(id="t", input="q", reference=reference)


def test_exact_match() -> None:
    assert ExactMatch().score("Paris", _case("Paris ")).passed
    assert not ExactMatch().score("Lyon", _case("Paris")).passed


def test_contains_reference() -> None:
    s = ContainsReference().score("The capital is Paris.", _case("paris"))
    assert s.passed and s.value == 1.0


def test_jaccard_similarity_in_range() -> None:
    s = JaccardSimilarity(0.3).score("a step budget caps loops", _case("a step budget"))
    assert 0.0 <= s.value <= 1.0
    assert s.passed  # high overlap


def test_regex_match() -> None:
    assert RegexMatch(r"\d{3}-\d{4}").score("call 555-1234", _case()).passed


def test_json_valid() -> None:
    assert JsonValid().score('{"a": 1}', _case()).passed
    assert not JsonValid().score("not json", _case()).passed
