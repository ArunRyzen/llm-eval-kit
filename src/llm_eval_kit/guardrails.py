"""Guardrails for LLM input and output.

Indirect prompt injection (a malicious instruction hidden in retrieved content) is "the new XSS":
the defense is layered — filter the input, constrain and validate the output, and limit blast
radius. These guards are pattern-based and deterministic (fast, testable, no model call); in
production you'd pair them with a model-based classifier (Llama Guard / NeMo Guardrails) as a second
layer. A `Guard` either **redacts** (transforms text) or **blocks** (flags a violation).

In plain words: guards are checkpoints that text passes through on its way in to the model and on
its way out. Some checkpoints clean the text (redact emails/phone numbers); others stop it entirely
(a prompt-injection attempt, an over-long input, a banned word).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from llm_eval_kit.errors import GuardrailViolation


@dataclass
class GuardResult:
    """What a guard hands back: the (possibly cleaned) text and what, if anything, it found."""

    text: str  # possibly redacted
    violations: list[str] = field(default_factory=list)  # human-readable "what was found"
    blocked: bool = False  # True = do NOT let this text through


class Guard(Protocol):
    """The one interface every checkpoint implements: take text in, return a GuardResult."""

    name: str

    def apply(self, text: str) -> GuardResult: ...


# --- PII redaction -------------------------------------------------------------------

# THE PII PATTERN LIST. Each entry maps a label (used in the "[REDACTED_...]" placeholder)
# to a regex that recognizes that kind of personal data. To redact a NEW kind of PII
# (e.g. IP addresses or passport numbers), add one entry here — nothing else changes.
_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "PHONE": re.compile(r"\b(?:\+?\d{1,2}[ -]?)?(?:\(?\d{3}\)?[ -]?)\d{3}[ -]?\d{4}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
}


class PiiRedactionGuard:
    """Redacts emails, phone numbers, SSNs, and card-like numbers. Redacts, never blocks.

    Why redact instead of block? A message containing an email address is usually still a
    legitimate message — we just don't want the PII reaching (or leaving) the model. So we
    swap it for a placeholder and let the conversation continue.
    """

    name = "pii_redaction"

    def apply(self, text: str) -> GuardResult:
        redacted = text
        found: list[str] = []
        for label, pattern in _PII_PATTERNS.items():
            if pattern.search(redacted):
                found.append(label)
                # Replace every match with a placeholder like "[REDACTED_EMAIL]".
                redacted = pattern.sub(f"[REDACTED_{label}]", redacted)
        return GuardResult(text=redacted, violations=found, blocked=False)


# --- Prompt-injection detection ------------------------------------------------------

# THE PROMPT-INJECTION PATTERN LIST. Each regex is a known attack phrasing ("ignore all
# previous instructions", "you are now...", "reveal your system prompt"). To catch a NEW
# attack phrasing, append one `re.compile(...)` line here — the guard picks it up
# automatically. `re.I` makes matching case-insensitive; `\s+` tolerates odd spacing.
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
    """Flags (blocks) text containing common prompt-injection phrasings.

    Unlike PII (which we clean), injection attempts are hostile — we don't try to "fix" the
    text, we refuse it. `violations` lists which patterns matched so you can see why.
    """

    name = "prompt_injection"

    def apply(self, text: str) -> GuardResult:
        # Collect every pattern that matches (not just the first) — better error messages.
        hits = [p.pattern for p in _INJECTION_PATTERNS if p.search(text)]
        return GuardResult(text=text, violations=hits, blocked=bool(hits))


class LengthGuard:
    """Blocks text over a character limit (a crude cost/abuse guard).

    Long inputs cost real money (tokens) and are a classic way to smuggle attacks or blow up
    latency — so past the limit we simply say no.
    """

    def __init__(self, max_chars: int) -> None:
        self.name = "length"
        self._max = max_chars

    def apply(self, text: str) -> GuardResult:
        over = len(text) > self._max
        violations = [f"len {len(text)} > {self._max}"] if over else []
        return GuardResult(text=text, violations=violations, blocked=over)


class BannedContentGuard:
    """Blocks output containing any banned term (case-insensitive).

    A last-line output filter: whatever the model says, these words never reach the user.
    """

    def __init__(self, terms: list[str]) -> None:
        self.name = "banned_content"
        self._terms = [t.lower() for t in terms]

    def apply(self, text: str) -> GuardResult:
        lowered = text.lower()
        hits = [t for t in self._terms if t in lowered]
        return GuardResult(text=text, violations=hits, blocked=bool(hits))


# --- Pipeline ------------------------------------------------------------------------


class GuardrailPipeline:
    """Runs input guards before the model and output guards after, carrying redactions forward.

    Think of it as two rows of checkpoints: one row between the user and the model
    (`guard_input`), one row between the model and the user (`guard_output`).
    """

    def __init__(
        self, input_guards: list[Guard] | None = None, output_guards: list[Guard] | None = None
    ) -> None:
        self._input = input_guards or []
        self._output = output_guards or []

    @staticmethod
    def _run(guards: list[Guard], text: str) -> GuardResult:
        # Run the guards IN ORDER, feeding each one the previous guard's (possibly cleaned)
        # text. One combined result comes out: all violations, blocked if ANY guard blocked.
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
        # By default a block raises — callers can't accidentally ignore a hostile input.
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
