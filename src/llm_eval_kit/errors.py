"""Domain exceptions."""

from __future__ import annotations


class EvalKitError(Exception):
    """Base class."""


class JudgeError(EvalKitError):
    """The LLM judge failed."""


class GateFailure(EvalKitError):  # noqa: N818 - "Failure" reads better than "Error" here
    """An evaluation did not meet its threshold — used to fail a CI build."""


class GuardrailViolation(EvalKitError):  # noqa: N818 - "Violation" is the clearer domain term
    """Input or output content violated a guardrail (e.g. prompt injection, PII)."""

    def __init__(self, message: str, *, guard: str) -> None:
        super().__init__(message)
        self.guard = guard
