"""Deterministic scorers.

A `Scorer` turns one (output, case) pair into a `Score` in [0, 1] with a pass/fail. They receive the
whole `EvalCase` (so a scorer can use the input and/or the reference). These are cheap, fast, and
reproducible — run them on every case. The LLM-as-judge (`judge.py`) is the expensive, nuanced
complement for what rules can't capture (faithfulness, helpfulness); it shares this interface.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

from llm_eval_kit.models import EvalCase, Score


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class Scorer(Protocol):
    name: str

    def score(self, output: str, case: EvalCase) -> Score: ...


class ExactMatch:
    """Exact (whitespace-normalized) string equality with the reference."""

    name = "exact_match"

    def score(self, output: str, case: EvalCase) -> Score:
        if case.reference is None:
            return Score(scorer=self.name, value=0.0, passed=False, detail="no reference")
        ok = output.strip() == case.reference.strip()
        return Score(scorer=self.name, value=1.0 if ok else 0.0, passed=ok)


class ContainsReference:
    """Passes if the (case-insensitive) reference text appears in the output."""

    name = "contains_reference"

    def score(self, output: str, case: EvalCase) -> Score:
        if case.reference is None:
            return Score(scorer=self.name, value=0.0, passed=False, detail="no reference")
        ok = case.reference.strip().lower() in output.lower()
        return Score(scorer=self.name, value=1.0 if ok else 0.0, passed=ok)


class RegexMatch:
    """Passes if the output matches a regex (reference-independent)."""

    def __init__(self, pattern: str, *, name: str = "regex_match") -> None:
        self.name = name
        self._re = re.compile(pattern)

    def score(self, output: str, case: EvalCase) -> Score:
        ok = self._re.search(output) is not None
        return Score(scorer=self.name, value=1.0 if ok else 0.0, passed=ok)


class JaccardSimilarity:
    """Token-overlap similarity vs the reference — a cheap offline proxy for semantic similarity."""  # noqa: E501

    def __init__(self, threshold: float = 0.5) -> None:
        self.name = "jaccard_similarity"
        self._threshold = threshold

    def score(self, output: str, case: EvalCase) -> Score:
        if case.reference is None:
            return Score(scorer=self.name, value=0.0, passed=False, detail="no reference")
        a, b = _tokens(output), _tokens(case.reference)
        union = a | b
        value = len(a & b) / len(union) if union else 0.0
        return Score(
            scorer=self.name,
            value=value,
            passed=value >= self._threshold,
            detail=f"threshold={self._threshold}",
        )


class JsonValid:
    """Passes if the output parses as JSON — a guard for structured-output systems."""

    name = "json_valid"

    def score(self, output: str, case: EvalCase) -> Score:
        try:
            json.loads(output)
            return Score(scorer=self.name, value=1.0, passed=True)
        except (json.JSONDecodeError, TypeError) as exc:
            return Score(scorer=self.name, value=0.0, passed=False, detail=str(exc))
