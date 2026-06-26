"""Command-line interface: run an eval (and gate), test guardrails, or print a trace."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from llm_eval_kit.datasets import SAMPLE_DATASET, demo_system, load_dataset
from llm_eval_kit.guardrails import default_pipeline
from llm_eval_kit.runner import EvalRunner, gate
from llm_eval_kit.scorers import ContainsReference, JaccardSimilarity
from llm_eval_kit.tracing import Tracer

app = typer.Typer(help="LLM evaluation + guardrails toolkit.", no_args_is_help=True)


@app.command()
def run(
    data: Annotated[Path | None, typer.Option(help="Dataset JSON; defaults to the sample.")] = None,
    threshold: Annotated[float, typer.Option(help="Ship-gate pass-rate threshold.")] = 0.8,
) -> None:
    """Evaluate the demo system on a dataset and apply the ship gate (exit 1 on failure)."""
    dataset = load_dataset(data) if data else SAMPLE_DATASET
    runner = EvalRunner([ContainsReference(), JaccardSimilarity(0.2)], threshold=threshold)
    report = runner.run(dataset, demo_system)

    typer.echo(report.summary())
    for result in report.results:
        mark = "✓" if result.passed else "✗"
        typer.echo(f"  {mark} {result.case_id}: {result.output}", err=True)

    if not gate(report, raise_on_fail=False):
        typer.echo("Gate FAILED — a case regressed.", err=True)
        raise typer.Exit(code=1)


@app.command()
def guard(
    text: Annotated[str, typer.Argument(help="Text to run through input guardrails.")],
) -> None:
    """Run text through the default guardrail pipeline (prompt-injection + PII)."""
    result = default_pipeline().guard_input(text, raise_on_block=False)
    typer.echo(f"blocked: {result.blocked}")
    typer.echo(f"violations: {result.violations}")
    typer.echo(f"sanitized: {result.text}")


@app.command()
def trace() -> None:
    """Run a small traced pipeline and print the span tree (observability demo)."""
    tracer = Tracer()
    with tracer.span("rag_answer", question="What is MCP?") as root:
        with tracer.span("retrieve") as s:
            s.set(k=3)
        with tracer.span("generate") as s:
            s.record_usage(tokens=180, cost_usd=0.0021)
        root.set(answer="MCP is a standard for exposing tools to models.")
    typer.echo(tracer.render())
    typer.echo(f"\ntotal: {tracer.total_tokens()} tokens, ${tracer.total_cost():.4f}", err=True)


if __name__ == "__main__":
    app()
