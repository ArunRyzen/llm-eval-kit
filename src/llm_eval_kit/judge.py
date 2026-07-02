"""LLM-as-judge — scoring what rules can't.

Deterministic scorers can't tell if an answer is *faithful*, *helpful*, or *correct in substance*.
An LLM judge can: it reads the input, the output, and (optionally) a reference, and rates against
a rubric. It implements the same `Scorer` interface, so it drops into the same runner. `FakeJudge`
makes judge-based evals testable offline; `GeminiJudge` (Google) and `AnthropicJudge` (Claude) use
real models with structured output. `make_judge()` picks one based on which API key you have.

Caveats interviewers probe: judges are biased (verbosity, position, self-preference) and need
calibration — keep the rubric explicit and pair the judge with cheap deterministic scorers.
"""

from __future__ import annotations

from pydantic import BaseModel

from llm_eval_kit.config import Settings, load_settings
from llm_eval_kit.errors import JudgeError
from llm_eval_kit.models import EvalCase, Score
from llm_eval_kit.scorers import Scorer

# The "job description" we give the judge model. It applies to every judged case, which is why
# it is a module-level constant and not rebuilt per call.
_JUDGE_SYSTEM = (
    "You are a strict, fair evaluator. Judge the assistant's answer against the rubric and, when "
    "provided, the reference answer. Score from 0.0 (poor) to 1.0 (excellent). Be specific about "
    "why. Do not reward verbosity or confident tone — only correctness and helpfulness."
)


class _Verdict(BaseModel):
    """The exact shape we force the judge model to reply in (a score, a pass/fail, and why).

    Asking for structured output instead of free text means we never have to "hope" the model's
    answer is parseable — the SDK validates it against this schema for us.
    """

    score: float
    passed: bool
    reasoning: str


def _clamp(value: float) -> float:
    # Models occasionally return 1.2 or -0.1; keep scores inside the promised [0, 1] range.
    return max(0.0, min(1.0, value))


class FakeJudge:
    """A deterministic judge for tests/offline demos. Optionally returns a fixed score.

    Why fake it? Real judges cost money, need network + API keys, and can answer differently
    each run — all poison for a test suite. FakeJudge has the *same interface and shape of
    output* as the real judges, so tests prove the plumbing works without ever calling an API.
    """

    name = "llm_judge"

    def __init__(self, fixed: float | None = None) -> None:
        self._fixed = fixed

    def score(self, output: str, case: EvalCase) -> Score:
        # Rule: use the fixed score if one was given; otherwise pass any non-empty answer.
        value = self._fixed if self._fixed is not None else (1.0 if output.strip() else 0.0)
        return Score(scorer=self.name, value=value, passed=value >= 0.5, detail="fake judge")


class _PromptMixin:
    """Shared prompt building: every real judge shows the model the same rubric + evidence."""

    _criteria: str

    def _prompt(self, output: str, case: EvalCase) -> str:
        # Lay out everything the judge needs: the rubric, the question, the answer under test,
        # and (when we have one) the known-good reference answer to compare against.
        parts = [
            f"Rubric: rate the answer on {self._criteria}.",
            f"\nUser input:\n{case.input}",
            f"\nAssistant answer:\n{output}",
        ]
        if case.reference:
            parts.append(f"\nReference answer:\n{case.reference}")
        parts.append("\nReturn a score in [0,1], a pass/fail, and one-sentence reasoning.")
        return "\n".join(parts)


class GeminiJudge(_PromptMixin):
    """LLM-as-judge via Google Gemini with schema-constrained JSON output.

    `response_mime_type="application/json"` + `response_schema=_Verdict` tell Gemini it MUST
    reply with JSON matching our verdict model, so parsing is robust instead of hopeful.
    """

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

    def score(self, output: str, case: EvalCase) -> Score:
        # Imported here (not at the top) so the SDK is only touched when a real judge runs —
        # offline users never pay the import.
        from google import genai
        from google.genai import errors, types

        try:
            response = genai.Client(api_key=self._api_key).models.generate_content(
                model=self._model,
                contents=self._prompt(output, case),
                config=types.GenerateContentConfig(
                    system_instruction=_JUDGE_SYSTEM,
                    max_output_tokens=self._max_tokens,
                    # These two lines are the robustness trick: force JSON in our exact shape.
                    response_mime_type="application/json",
                    response_schema=_Verdict,
                ),
            )
        except errors.APIError as exc:
            # Wrap SDK errors in our own domain error so callers only catch one type.
            raise JudgeError(f"Judge request failed: {exc}") from exc

        # The SDK parses the JSON into a _Verdict for us; fall back to raw text if needed.
        verdict = response.parsed
        if not isinstance(verdict, _Verdict):
            if not response.text:
                raise JudgeError("Judge returned no structured verdict.")
            verdict = _Verdict.model_validate_json(response.text)
        return Score(
            scorer=self.name,
            value=_clamp(verdict.score),
            passed=verdict.passed,
            detail=verdict.reasoning[:200],
        )


class AnthropicJudge(_PromptMixin):
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

    def score(self, output: str, case: EvalCase) -> Score:
        # Imported here (not at the top) for the same reason as GeminiJudge: offline users
        # never touch the SDK.
        import anthropic

        try:
            response = anthropic.Anthropic(api_key=self._api_key).messages.parse(
                model=self._model,
                max_tokens=self._max_tokens,
                system=_JUDGE_SYSTEM,
                messages=[{"role": "user", "content": self._prompt(output, case)}],
                output_format=_Verdict,  # same trick: the reply must match our verdict schema
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


def make_judge(settings: Settings | None = None) -> Scorer:
    """Pick the best judge available from your environment (the judge *selection mechanism*).

    Order of preference:
    1. `GEMINI_API_KEY` set  -> GeminiJudge (free-tier friendly, the default for this repo)
    2. `ANTHROPIC_API_KEY` set -> AnthropicJudge
    3. no key at all -> FakeJudge, so everything still works fully offline
    """
    settings = settings or load_settings()
    if settings.gemini_api_key:
        return GeminiJudge(
            model=settings.gemini_model,
            max_tokens=settings.max_tokens,
            api_key=settings.gemini_api_key,
        )
    if settings.anthropic_api_key:
        return AnthropicJudge(
            model=settings.anthropic_model,
            max_tokens=settings.max_tokens,
            api_key=settings.anthropic_api_key,
        )
    return FakeJudge()
