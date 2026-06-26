"""Guardrails for LLM input and output.

Indirect prompt injection (a malicious instruction hidden in retrieved content) is "the new XSS":
the defense is layered — filter the input, constrain and validate the output, and limit blast
radius. These guards are pattern-based and deterministic (fast, testable, no model call); in
production you'd pair them with a model-based classifier (Llama Guard / NeMo Guardrails) as a second
layer. A `Guard` either **redacts** (transforms text) or **blocks** (flags a violation).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from llm_eval_kit.errors import GuardrailViolation


@dataclass
class GuardResult:
    text: str  # possibly redacted
    violations: list[str] = field(default_factory=list)
    blocked: bool = False


class Guard(Protocol):
    name: str

    def apply(self, text: str) -> GuardResult: ...


# --- PII redaction -------------------------------------------------------------------

_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "PHONE": re.compile(r"\b(?:\+?\d{1,2}[ -]?)?(?:\(?\d{3}\)?[ -]?)\d{3}[ -]?\d{4}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
}


class PiiRedactionGuard:
    """Redacts emails, phone numbers, SSNs, and card-like numbers. Redacts, never blocks."""

    name = "pii_redaction"

    def apply(self, text: str) -> GuardResult:
        redacted = text
        found: list[str] = []
        for label, pattern in _PII_PATTERNS.items():
            if pattern.search(redacted):
                found.append(label)
                redacted = pattern.sub(f"[REDACTED_{label}]", redacted)
        return GuardResult(text=redacted, violations=found, blocked=False)


# --- Prompt-injection detection ------------------------------------------------------

_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+|the\s+)?(previous|above|prior)\s+(instructions|prompts)", re.I),
    re.compile(r"disregard\s+(the\s+)?(previous|above|system)", re.I),
    re.compile(r"\b(system\s+prompt|your\s+instructions)\b", re.I),
    re.compile(r"you\s+are\s+now\b", re.I),
    re.compile(r"reveal\s+(your\s+)?(instructions|system\s+prompt|prompt)", re.I),
    re.compile(r"forget\s+(everything|all|your\s+instructions)", re.I),
    re.compile(r"new\s+instructions\s*:", re.I),
]


class PromptInjectionGuard:
    """Flags (blocks) text containing common prompt-injection phrasings."""

    name = "prompt_injection"

    def apply(self, text: str) -> GuardResult:
        hits = [p.pattern for p in _INJECTION_PATTERNS if p.search(text)]
        return GuardResult(text=text, violations=hits, blocked=bool(hits))


class LengthGuard:
    """Blocks text over a character limit (a crude cost/abuse guard)."""

    def __init__(self, max_chars: int) -> None:
        self.name = "length"
        self._max = max_chars

    def apply(self, text: str) -> GuardResult:
        over = len(text) > self._max
        violations = [f"len {len(text)} > {self._max}"] if over else []
        return GuardResult(text=text, violations=violations, blocked=over)


class BannedContentGuard:
    """Blocks output containing any banned term (case-insensitive)."""

    def __init__(self, terms: list[str]) -> None:
        self.name = "banned_content"
        self._terms = [t.lower() for t in terms]

    def apply(self, text: str) -> GuardResult:
        lowered = text.lower()
        hits = [t for t in self._terms if t in lowered]
        return GuardResult(text=text, violations=hits, blocked=bool(hits))


# --- Pipeline ------------------------------------------------------------------------


class GuardrailPipeline:
    """Runs input guards before the model and output guards after, carrying redactions forward."""

    def __init__(
        self, input_guards: list[Guard] | None = None, output_guards: list[Guard] | None = None
    ) -> None:
        self._input = input_guards or []
        self._output = output_guards or []

    @staticmethod
    def _run(guards: list[Guard], text: str) -> GuardResult:
        violations: list[str] = []
        blocked = False
        for guard in guards:
            result = guard.apply(text)
            text = result.text  # redactions carry forward to later guards
            violations.extend(f"{guard.name}: {v}" for v in result.violations)
            blocked = blocked or result.blocked
        return GuardResult(text=text, violations=violations, blocked=blocked)

    def guard_input(self, text: str, *, raise_on_block: bool = True) -> GuardResult:
        result = self._run(self._input, text)
        if result.blocked and raise_on_block:
            raise GuardrailViolation(f"Input blocked: {result.violations}", guard="input")
        return result

    def guard_output(self, text: str, *, raise_on_block: bool = True) -> GuardResult:
        result = self._run(self._output, text)
        if result.blocked and raise_on_block:
            raise GuardrailViolation(f"Output blocked: {result.violations}", guard="output")
        return result


def default_pipeline() -> GuardrailPipeline:
    """A sensible default: detect injection + redact PII in, redact PII out."""
    return GuardrailPipeline(
        input_guards=[PromptInjectionGuard(), PiiRedactionGuard(), LengthGuard(20_000)],
        output_guards=[PiiRedactionGuard()],
    )
