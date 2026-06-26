"""The eval runner and the CI gate.

`EvalRunner` runs a *system under test* (any `input -> output` callable) over a `Dataset`, applies a
set of scorers to each output, and aggregates an `EvalReport`. `gate()` turns that report into a
build verdict: meet the threshold or fail CI. This is the "ship gate" — a versioned dataset, a
numeric score, and a regression alarm — that 2026 interviews ask you to design.
"""

from __future__ import annotations

from collections.abc import Callable

from llm_eval_kit.errors import GateFailure
from llm_eval_kit.models import CaseResult, Dataset, EvalReport
from llm_eval_kit.scorers import Scorer

# A system under test: maps a case input to the model's output text.
System = Callable[[str], str]


class EvalRunner:
    def __init__(self, scorers: list[Scorer], *, threshold: float = 0.8) -> None:
        if not scorers:
            raise ValueError("EvalRunner needs at least one scorer.")
        self._scorers = scorers
        self._threshold = threshold

    def run(self, dataset: Dataset, system: System) -> EvalReport:
        results: list[CaseResult] = []
        for case in dataset.cases:
            output = system(case.input)
            scores = [scorer.score(output, case) for scorer in self._scorers]
            results.append(CaseResult(case_id=case.id, output=output, scores=scores))
        return EvalReport(dataset=dataset.name, threshold=self._threshold, results=results)


def gate(report: EvalReport, *, raise_on_fail: bool = True) -> bool:
    """Return whether the report passed; optionally raise `GateFailure` to fail a CI build."""
    if not report.passed and raise_on_fail:
        raise GateFailure(report.summary())
    return report.passed
