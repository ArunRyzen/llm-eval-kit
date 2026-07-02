"""Deterministic scorers.

A `Scorer` turns one (output, case) pair into a `Score` in [0, 1] with a pass/fail. They receive the
whole `EvalCase` (so a scorer can use the input and/or the reference). These are cheap, fast, and
reproducible — run them on every case. The LLM-as-judge (`judge.py`) is the expensive, nuanced
complement for what rules can't capture (faithfulness, helpfulness); it shares this interface.

In plain words: a scorer is a tiny grader. You hand it the model's answer and the test case, and
it hands back a mark between 0 and 1 plus a pass/fail. Each class below grades in a different way.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

from llm_eval_kit.models import EvalCase, Score


def _tokens(text: str) -> set[str]:
    # Chop text into lowercase words/numbers, e.g. "The Cat!" -> {"the", "cat"}.
    # Used by JaccardSimilarity so word order and punctuation don't matter.
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class Scorer(Protocol):
    """The one interface every grader implements — deterministic scorers AND LLM judges.

    Why a Protocol? Any object with a `name` and a `score()` method counts as a Scorer, no
    inheritance required. That's what lets `EvalRunner` treat a regex check and a live LLM judge
    exactly the same.
    """

    name: str

    def score(self, output: str, case: EvalCase) -> Score: ...


class ExactMatch:
    """Exact (whitespace-normalized) string equality with the reference.

    The strictest grader: the answer must be *identical* to the reference (ignoring leading and
    trailing spaces). Great for closed-form answers like "4" or "Paris"; too harsh for prose.
    """

    name = "exact_match"

    def score(self, output: str, case: EvalCase) -> Score:
        if case.reference is None:
            # No answer key -> we can't grade. Fail loudly rather than guess.
            return Score(scorer=self.name, value=0.0, passed=False, detail="no reference")
        ok = output.strip() == case.reference.strip()
        return Score(scorer=self.name, value=1.0 if ok else 0.0, passed=ok)


class ContainsReference:
    """Passes if the (case-insensitive) reference text appears in the output.

    A gentler grader: the answer just has to *mention* the reference somewhere. "The capital is
    Paris, of course" passes against reference "Paris".
    """

    name = "contains_reference"

    def score(self, output: str, case: EvalCase) -> Score:
        if case.reference is None:
            return Score(scorer=self.name, value=0.0, passed=False, detail="no reference")
        ok = case.reference.strip().lower() in output.lower()
        return Score(scorer=self.name, value=1.0 if ok else 0.0, passed=ok)


class RegexMatch:
    """Passes if the output matches a regex (reference-independent).

    For checking *shape*, not content — e.g. "does the output contain a date like 2026-01-01?".
    It ignores the reference entirely; the pattern you construct it with is the whole rule.
    """

    def __init__(self, pattern: str, *, name: str = "regex_match") -> None:
        self.name = name
        self._re = re.compile(pattern)  # compile once here, reuse for every case (faster)

    def score(self, output: str, case: EvalCase) -> Score:
        ok = self._re.search(output) is not None
        return Score(scorer=self.name, value=1.0 if ok else 0.0, passed=ok)


class JaccardSimilarity:
    """Token-overlap similarity vs the reference — a cheap offline proxy for semantic similarity.

    How it grades: turn both texts into sets of words, then score = (words in common) / (all words
    in either). 1.0 = same words, 0.0 = nothing shared. Crude but free, instant, and deterministic.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.name = "jaccard_similarity"
        self._threshold = threshold  # pass mark: how much overlap counts as "close enough"

    def score(self, output: str, case: EvalCase) -> Score:
        if case.reference is None:
            return Score(scorer=self.name, value=0.0, passed=False, detail="no reference")
        a, b = _tokens(output), _tokens(case.reference)
        union = a | b
        # intersection / union — the Jaccard index. Guard against dividing by zero.
        value = len(a & b) / len(union) if union else 0.0
        return Score(
            scorer=self.name,
            value=value,
            passed=value >= self._threshold,
            detail=f"threshold={self._threshold}",
        )


class JsonValid:
    """Passes if the output parses as JSON — a guard for structured-output systems.

    When your system promises machine-readable JSON, "almost JSON" is a failure. This scorer
    doesn't care what the JSON *says*, only that `json.loads` accepts it.
    """

    name = "json_valid"

    def score(self, output: str, case: EvalCase) -> Score:
        try:
            json.loads(output)
            return Score(scorer=self.name, value=1.0, passed=True)
        except (json.JSONDecodeError, TypeError) as exc:
            # Put the parser's complaint in `detail` so a failing report tells you *why*.
            return Score(scorer=self.name, value=0.0, passed=False, detail=str(exc))
