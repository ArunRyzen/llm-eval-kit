"""A lightweight, in-memory tracer for LLM call trees.

Observability is half of "if you can't measure it, you can't ship it." This tracer records nested
**spans** — name, attributes, duration, tokens, cost — so an LLM pipeline (retrieve → judge →
answer) becomes an inspectable tree. It has zero external dependencies on purpose: in production you
would export the same spans to Langfuse / Arize Phoenix / OpenTelemetry; the *shape* is identical.

In plain words: a span is a stopwatch with a notebook. You "open" one around a step of your
pipeline, jot notes on it (attributes, token counts, cost), and it "closes" itself when the step
ends. Nesting spans inside each other produces a tree that shows where the time and money went.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any


@dataclass
class Span:
    """One timed step of a pipeline: what it was, how long it took, what it cost."""

    name: str
    attributes: dict[str, Any] = field(default_factory=dict)  # free-form notes ("k=3", the answer)
    start: float = 0.0  # clock reading when the span opened
    end: float | None = None  # clock reading when it closed (None = still running)
    tokens: int = 0
    cost_usd: float = 0.0
    children: list[Span] = field(default_factory=list)  # sub-steps nested inside this one

    @property
    def duration_ms(self) -> float:
        # If the span never closed, report 0 instead of a nonsense negative number.
        return ((self.end or self.start) - self.start) * 1000.0

    def set(self, **attributes: Any) -> None:
        """Attach extra notes to the span, e.g. `span.set(k=3, model="gemini-2.5-flash")`."""
        self.attributes.update(attributes)

    def record_usage(self, *, tokens: int = 0, cost_usd: float = 0.0) -> None:
        """Add token/cost usage to this span (accumulates across multiple calls)."""
        self.tokens += tokens
        self.cost_usd += cost_usd

    def to_dict(self) -> dict[str, Any]:
        # A plain-dict view — this is the shape you'd ship to Langfuse / OTel in production.
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
        self._roots: list[Span] = []  # top-level spans (no parent)
        self._stack: list[Span] = []  # currently-open spans; the top is the "current parent"

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        """THE span lifecycle: everything before `yield` runs at `with` entry (span OPENS);
        everything in `finally` runs at `with` exit (span CLOSES) — even if the body raised."""
        # OPEN: stamp the start time...
        span = Span(name=name, attributes=dict(attributes), start=perf_counter())
        # ...and attach to the innermost open span (nesting), or as a new root if none is open.
        parent = self._stack[-1] if self._stack else None
        (parent.children if parent else self._roots).append(span)
        self._stack.append(span)
        try:
            yield span  # the caller's `with`-block body runs here
        finally:
            # CLOSE: stamp the end time and step back out to the parent.
            span.end = perf_counter()
            self._stack.pop()

    @property
    def roots(self) -> list[Span]:
        return self._roots

    def total_tokens(self) -> int:
        # Sum tokens over every span in every tree (children included).
        return int(sum((_aggregate(s, "tokens") for s in self._roots), 0.0))

    def total_cost(self) -> float:
        return sum((_aggregate(s, "cost_usd") for s in self._roots), 0.0)

    def render(self) -> str:
        """Draw the span forest as an indented text tree (what `evalkit trace` prints)."""
        lines: list[str] = []
        for root in self._roots:
            _render(root, 0, lines)
        return "\n".join(lines)


def _aggregate(span: Span, attr: str) -> float:
    # A span's total = its own number + the totals of all its children (recursive).
    return getattr(span, attr) + sum(_aggregate(c, attr) for c in span.children)


def _render(span: Span, depth: int, lines: list[str]) -> None:
    # Two spaces of indent per nesting level, plus tokens/cost when there are any.
    pad = "  " * depth
    extra = f" [{span.tokens} tok, ${span.cost_usd:.4f}]" if span.tokens or span.cost_usd else ""
    lines.append(f"{pad}{span.name} ({span.duration_ms:.1f} ms){extra}")
    for child in span.children:
        _render(child, depth + 1, lines)
