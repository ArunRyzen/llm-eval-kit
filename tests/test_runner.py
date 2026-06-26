"""The runner aggregates scores and the gate enforces the threshold."""

from __future__ import annotations

import pytest

from llm_eval_kit.datasets import SAMPLE_DATASET, demo_system
from llm_eval_kit.errors import GateFailure
from llm_eval_kit.runner import EvalRunner, gate
from llm_eval_kit.scorers import ContainsReference, JaccardSimilarity


def _runner(threshold: float) -> EvalRunner:
    return EvalRunner([ContainsReference(), JaccardSimilarity(0.2)], threshold=threshold)


def test_demo_has_one_failing_case() -> None:
    report = _runner(0.8).run(SAMPLE_DATASET, demo_system)
    # 3 of 4 cases pass (the 'capital' answer is wrong on purpose).
    assert report.n == 4
    assert report.pass_rate == 0.75
    failing = [r.case_id for r in report.results if not r.passed]
    assert failing == ["capital"]


def test_gate_fails_below_threshold() -> None:
    report = _runner(0.8).run(SAMPLE_DATASET, demo_system)  # 0.75 < 0.8
    assert not report.passed
    with pytest.raises(GateFailure):
        gate(report)


def test_gate_passes_at_lower_threshold() -> None:
    report = _runner(0.7).run(SAMPLE_DATASET, demo_system)  # 0.75 >= 0.7
    assert report.passed
    assert gate(report) is True


def test_mean_by_scorer_reported() -> None:
    report = _runner(0.7).run(SAMPLE_DATASET, demo_system)
    means = report.mean_by_scorer()
    assert set(means) == {"contains_reference", "jaccard_similarity"}


def test_runner_requires_a_scorer() -> None:
    with pytest.raises(ValueError):
        EvalRunner([])
