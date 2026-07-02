"""llm-eval-kit: measure LLM output quality and guard LLM I/O.

Three capabilities, each usable on its own:
- **Evaluation** — score a system's outputs against a dataset (deterministic scorers + LLM-judge),
  aggregate into a report, and gate CI on a threshold.
- **Tracing** — a lightweight in-memory tracer so an LLM call tree is observable (spans + cost).
- **Guardrails** — input/output filters for PII and prompt injection, plus output validation.
"""

from llm_eval_kit.judge import AnthropicJudge, FakeJudge, GeminiJudge, make_judge
from llm_eval_kit.models import CaseResult, Dataset, EvalCase, EvalReport, Score
from llm_eval_kit.runner import EvalRunner, gate

__version__ = "0.1.0"

__all__ = [
    "Dataset",
    "EvalCase",
    "Score",
    "CaseResult",
    "EvalReport",
    "EvalRunner",
    "gate",
    "FakeJudge",
    "GeminiJudge",
    "AnthropicJudge",
    "make_judge",
    "__version__",
]
