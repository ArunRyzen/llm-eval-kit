"""FastAPI service exposing guardrails and a demo eval over HTTP."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from llm_eval_kit.datasets import SAMPLE_DATASET, demo_system
from llm_eval_kit.guardrails import default_pipeline
from llm_eval_kit.runner import EvalRunner
from llm_eval_kit.scorers import ContainsReference, JaccardSimilarity

app = FastAPI(title="llm-eval-kit", version="0.1.0")
_pipeline = default_pipeline()


class GuardRequest(BaseModel):
    text: str = Field(min_length=1)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/guard/input")
def guard_input(request: GuardRequest) -> dict:
    result = _pipeline.guard_input(request.text, raise_on_block=False)
    return {"blocked": result.blocked, "violations": result.violations, "sanitized": result.text}


@app.get("/eval/demo")
def eval_demo(threshold: float = 0.8) -> dict:
    runner = EvalRunner([ContainsReference(), JaccardSimilarity(0.2)], threshold=threshold)
    report = runner.run(SAMPLE_DATASET, demo_system)
    return {
        "dataset": report.dataset,
        "pass_rate": report.pass_rate,
        "passed": report.passed,
        "threshold": report.threshold,
        "means": report.mean_by_scorer(),
        "cases": [{"id": r.case_id, "passed": r.passed} for r in report.results],
    }
