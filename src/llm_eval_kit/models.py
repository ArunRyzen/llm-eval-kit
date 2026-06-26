"""Core evaluation types.

An eval is: run a system over a `Dataset` of `EvalCase`s, apply `Score`rs to each output, and
aggregate `CaseResult`s into an `EvalReport` that passes or fails a threshold (the ship gate).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    """One labelled example: an input, an optional reference answer, and metadata."""

    id: str
    input: str
    reference: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class Dataset(BaseModel):
    name: str
    cases: list[EvalCase] = Field(default_factory=list)


class Score(BaseModel):
    """The output of one scorer on one case."""

    scorer: str
    value: float = Field(ge=0.0, le=1.0, description="Normalized score in [0, 1].")
    passed: bool
    detail: str = ""


class CaseResult(BaseModel):
    case_id: str
    output: str
    scores: list[Score] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        """A case passes only if every scorer passed it."""
        return all(s.passed for s in self.scores)


class EvalReport(BaseModel):
    """Aggregated results for a run, plus the gate verdict."""

    dataset: str
    threshold: float
    results: list[CaseResult] = Field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def pass_rate(self) -> float:
        return sum(r.passed for r in self.results) / self.n if self.n else 0.0

    @property
    def passed(self) -> bool:
        """The ship gate: does the pass rate meet the threshold?"""
        return self.pass_rate >= self.threshold

    def mean_by_scorer(self) -> dict[str, float]:
        """Mean value per scorer across all cases — useful for dashboards."""
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for result in self.results:
            for score in result.scores:
                sums[score.scorer] = sums.get(score.scorer, 0.0) + score.value
                counts[score.scorer] = counts.get(score.scorer, 0) + 1
        return {name: sums[name] / counts[name] for name in sums}

    def summary(self) -> str:
        means = ", ".join(f"{k}={v:.2f}" for k, v in self.mean_by_scorer().items())
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"[{verdict}] {self.dataset}: pass_rate={self.pass_rate:.2f} "
            f"(threshold {self.threshold:.2f}, n={self.n}) | {means}"
        )
