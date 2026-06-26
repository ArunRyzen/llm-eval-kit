"""A lightweight, in-memory tracer for LLM call trees.

Observability is half of "if you can't measure it, you can't ship it." This tracer records nested
**spans** — name, attributes, duration, tokens, cost — so an LLM pipeline (retrieve → judge →
answer) becomes an inspectable tree. It has zero external dependencies on purpose: in production you
would export the same spans to Langfuse / Arize Phoenix / OpenTelemetry; the *shape* is identical.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any


@dataclass
class Span:
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)
    start: float = 0.0
    end: float | None = None
    tokens: int = 0
    cost_usd: float = 0.0
    children: list[Span] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return ((self.end or self.start) - self.start) * 1000.0

    def set(self, **attributes: Any) -> None:
        self.attributes.update(attributes)

    def record_usage(self, *, tokens: int = 0, cost_usd: float = 0.0) -> None:
        self.tokens += tokens
        self.cost_usd += cost_usd

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "duration_ms": round(self.duration_ms, 2),
            "tokens": self.tokens,
            "cost_usd": round(self.cost_usd, 6),
            "attributes": self.attributes,
            "children": [c.to_dict() for c in self.children],
        }


class Tracer:
    """Collects spans into a forest. Use `with tracer.span(name) as span:` to nest."""

    def __init__(self) -> None:
        self._roots: list[Span] = []
        self._stack: list[Span] = []

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        span = Span(name=name, attributes=dict(attributes), start=perf_counter())
        parent = self._stack[-1] if self._stack else None
        (parent.children if parent else self._roots).append(span)
        self._stack.append(span)
        try:
            yield span
        finally:
            span.end = perf_counter()
            self._stack.pop()

    @property
    def roots(self) -> list[Span]:
        return self._roots

    def total_tokens(self) -> int:
        return int(sum((_aggregate(s, "tokens") for s in self._roots), 0.0))

    def total_cost(self) -> float:
        return sum((_aggregate(s, "cost_usd") for s in self._roots), 0.0)

    def render(self) -> str:
        lines: list[str] = []
        for root in self._roots:
            _render(root, 0, lines)
        return "\n".join(lines)


def _aggregate(span: Span, attr: str) -> float:
    return getattr(span, attr) + sum(_aggregate(c, attr) for c in span.children)


def _render(span: Span, depth: int, lines: list[str]) -> None:
    pad = "  " * depth
    extra = f" [{span.tokens} tok, ${span.cost_usd:.4f}]" if span.tokens or span.cost_usd else ""
    lines.append(f"{pad}{span.name} ({span.duration_ms:.1f} ms){extra}")
    for child in span.children:
        _render(child, depth + 1, lines)
