"""The eval runner and the CI gate.

`EvalRunner` runs a *system under test* (any `input -> output` callable) over a `Dataset`, applies a
set of scorers to each output, and aggregates an `EvalReport`. `gate()` turns that report into a
build verdict: meet the threshold or fail CI. This is the "ship gate" — a versioned dataset, a
numeric score, and a regression alarm — that 2026 interviews ask you to design.

In plain words: the runner is the exam invigilator. It asks your system every question in the
dataset, hands each answer to every grader (scorer), and collects the marks into one report card.
`gate()` then reads the report card and decides: good enough to ship, or fail the build.
"""

from __future__ import annotations

from collections.abc import Callable

from llm_eval_kit.errors import GateFailure
from llm_eval_kit.models import CaseResult, Dataset, EvalReport
from llm_eval_kit.scorers import Scorer

# A system under test: maps a case input to the model's output text.
# It's just "a function from question to answer" — which means ANYTHING (a chatbot, a RAG
# pipeline, a canned dict lookup) can be evaluated, as long as it fits this shape.
System = Callable[[str], str]


class EvalRunner:
    def __init__(self, scorers: list[Scorer], *, threshold: float = 0.8) -> None:
        if not scorers:
            # An eval with no graders would silently pass everything — refuse early.
            raise ValueError("EvalRunner needs at least one scorer.")
        self._scorers = scorers
        # The pass mark for the WHOLE run: the fraction of cases that must pass (default 80%).
        # This number is what ultimately decides ship / no-ship (see EvalReport.passed).
        self._threshold = threshold

    def run(self, dataset: Dataset, system: System) -> EvalReport:
        results: list[CaseResult] = []
        for case in dataset.cases:
            # Step 1: ask the system under test the question.
            output = system(case.input)
            # Step 2: let every scorer grade that one answer.
            scores = [scorer.score(output, case) for scorer in self._scorers]
            # Step 3: keep the answer + its marks together so failures are debuggable later.
            results.append(CaseResult(case_id=case.id, output=output, scores=scores))
        # The report computes pass_rate and compares it to the threshold (models.py).
        return EvalReport(dataset=dataset.name, threshold=self._threshold, results=results)


def gate(report: EvalReport, *, raise_on_fail: bool = True) -> bool:
    """Return whether the report passed; optionally raise `GateFailure` to fail a CI build.

    This is the ship gate itself. `report.passed` is True only when
    `pass_rate >= threshold` (see `EvalReport.passed` in models.py). Raising an exception is
    deliberate: in CI, an uncaught exception = non-zero exit code = red build. Quality
    regressions become impossible to merge quietly.
    """
    if not report.passed and raise_on_fail:
        raise GateFailure(report.summary())
    return report.passed
