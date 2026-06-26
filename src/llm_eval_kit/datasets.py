"""A built-in sample dataset and demo system, so the CLI works out of the box.

The demo system answers most cases correctly and one incorrectly — so the default run *fails* the
gate, demonstrating exactly what a regression alarm looks like. Point the CLI at your own JSON
dataset to evaluate real systems.
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_eval_kit.models import Dataset, EvalCase

SAMPLE_DATASET = Dataset(
    name="sample-qa",
    cases=[
        EvalCase(id="bm25", input="What does BM25 reward?", reference="exact keyword matches"),
        EvalCase(
            id="mcp", input="What is MCP?", reference="a standard for exposing tools to models"
        ),
        EvalCase(id="agent", input="What stops runaway agent loops?", reference="a step budget"),
        EvalCase(id="capital", input="What is the capital of France?", reference="Paris"),
    ],
)

# A toy system under test. Note the deliberately wrong answer for 'capital'.
_DEMO_ANSWERS: dict[str, str] = {
    "What does BM25 reward?": "BM25 rewards exact keyword matches, weighting rarer terms more.",
    "What is MCP?": "MCP is a standard for exposing tools to models across hosts.",
    "What stops runaway agent loops?": "A step budget caps the loop so it cannot run forever.",
    "What is the capital of France?": "The capital of France is Lyon.",  # wrong on purpose
}


def demo_system(question: str) -> str:
    """The system under test for the demo — a canned lookup."""
    return _DEMO_ANSWERS.get(question, "I don't know.")


def load_dataset(path: Path) -> Dataset:
    """Load a dataset from JSON: {"name": ..., "cases": [{"id","input","reference"}, ...]}."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return Dataset.model_validate(data)
