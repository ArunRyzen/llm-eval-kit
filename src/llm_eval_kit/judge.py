"""LLM-as-judge — scoring what rules can't.

Deterministic scorers can't tell if an answer is *faithful*, *helpful*, or *correct in substance*.
An LLM judge can: it reads the input, the output, and (optionally) a reference, and rates against
a rubric. It implements the same `Scorer` interface, so it drops into the same runner. `FakeJudge`
makes judge-based evals testable offline; `AnthropicJudge` uses Claude with structured output.

Caveats interviewers probe: judges are biased (verbosity, position, self-preference) and need
calibration — keep the rubric explicit and pair the judge with cheap deterministic scorers.
"""

from __future__ import annotations

from pydantic import BaseModel

from llm_eval_kit.errors import JudgeError
from llm_eval_kit.models import EvalCase, Score

_JUDGE_SYSTEM = (
    "You are a strict, fair evaluator. Judge the assistant's answer against the rubric and, when "
    "provided, the reference answer. Score from 0.0 (poor) to 1.0 (excellent). Be specific about "
    "why. Do not reward verbosity or confident tone — only correctness and helpfulness."
)


class _Verdict(BaseModel):
    score: float
    passed: bool
    reasoning: str


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


class FakeJudge:
    """A deterministic judge for tests/offline demos. Optionally returns a fixed score."""

    name = "llm_judge"

    def __init__(self, fixed: float | None = None) -> None:
        self._fixed = fixed

    def score(self, output: str, case: EvalCase) -> Score:
        value = self._fixed if self._fixed is not None else (1.0 if output.strip() else 0.0)
        return Score(scorer=self.name, value=value, passed=value >= 0.5, detail="fake judge")


class AnthropicJudge:
    """LLM-as-judge via Claude with schema-constrained output."""

    name = "llm_judge"

    def __init__(
        self,
        *,
        model: str,
        max_tokens: int,
        api_key: str | None,
        criteria: str = "overall correctness, faithfulness, and helpfulness",
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._api_key = api_key
        self._criteria = criteria

    def _prompt(self, output: str, case: EvalCase) -> str:
        parts = [
            f"Rubric: rate the answer on {self._criteria}.",
            f"\nUser input:\n{case.input}",
            f"\nAssistant answer:\n{output}",
        ]
        if case.reference:
            parts.append(f"\nReference answer:\n{case.reference}")
        parts.append("\nReturn a score in [0,1], a pass/fail, and one-sentence reasoning.")
        return "\n".join(parts)

    def score(self, output: str, case: EvalCase) -> Score:
        import anthropic

        try:
            response = anthropic.Anthropic(api_key=self._api_key).messages.parse(
                model=self._model,
                max_tokens=self._max_tokens,
                system=_JUDGE_SYSTEM,
                messages=[{"role": "user", "content": self._prompt(output, case)}],
                output_format=_Verdict,
            )
        except anthropic.APIError as exc:
            raise JudgeError(f"Judge request failed: {exc}") from exc

        verdict = response.parsed_output
        if verdict is None:
            raise JudgeError("Judge returned no structured verdict.")
        return Score(
            scorer=self.name,
            value=_clamp(verdict.score),
            passed=verdict.passed,
            detail=verdict.reasoning[:200],
        )
