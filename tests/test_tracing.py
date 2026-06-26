"""The tracer nests spans and aggregates tokens/cost."""

from __future__ import annotations

from llm_eval_kit.tracing import Tracer


def test_nested_spans_and_aggregation() -> None:
    tracer = Tracer()
    with tracer.span("root") as root:
        with tracer.span("child") as child:
            child.record_usage(tokens=100, cost_usd=0.001)
        root.record_usage(tokens=10, cost_usd=0.0001)

    assert len(tracer.roots) == 1
    assert tracer.roots[0].name == "root"
    assert len(tracer.roots[0].children) == 1
    # Totals roll up across the tree.
    assert tracer.total_tokens() == 110
    assert abs(tracer.total_cost() - 0.0011) < 1e-9


def test_to_dict_is_serializable() -> None:
    tracer = Tracer()
    with tracer.span("a"):
        pass
    d = tracer.roots[0].to_dict()
    assert d["name"] == "a"
    assert "duration_ms" in d and "children" in d
