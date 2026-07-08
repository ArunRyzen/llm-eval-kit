"""LLM_DEBUG — a learning aid that prints every judge request/response to stderr.

Set the environment variable ``LLM_DEBUG=1`` and every judge call — the offline ``FakeJudge``
as well as the real ``GeminiJudge``/``AnthropicJudge`` — prints the full prompt it sends and the
verdict it gets back, as plain-ASCII blocks on **stderr** (so it never mixes with normal program
output on stdout). Unset it, or set it to ``0``/``false`` (any casing), and the kit stays silent.

Example::

    === AI REQUEST (judge: gemini/gemini-2.5-flash) ===
    judge prompt: Rubric: rate the answer on ...
    ===================================================
    === AI RESPONSE (judge) ===
    verdict: {"score": 0.9, "passed": true, "reasoning": "..."}
    ===========================

Two safety rules, enforced by convention at every call site:

1. **Never log secrets.** Call sites pass prompts and verdicts — never API keys.
2. **Never flood the terminal.** Any field longer than ~2000 characters is cut off with an
   explicit ``... [truncated]`` marker.
"""

from __future__ import annotations

import os
import sys

# Fields longer than this are truncated so a giant prompt can't drown your terminal.
_MAX_FIELD_CHARS = 2000


def debug_enabled() -> bool:
    """Is debug logging on? True when LLM_DEBUG is set to anything but ""/"0"/"false"."""
    return os.environ.get("LLM_DEBUG", "").strip().lower() not in {"", "0", "false"}


def _truncate(text: str) -> str:
    if len(text) <= _MAX_FIELD_CHARS:
        return text
    return text[:_MAX_FIELD_CHARS] + "... [truncated]"


def log_block(title: str, **fields: object) -> None:
    """Print one framed block to stderr — a silent no-op unless LLM_DEBUG is on.

    Field names arrive as keyword arguments; underscores become spaces in the output, so
    ``log_block("AI RESPONSE (judge)", judge_prompt=p)`` prints a ``judge prompt:`` line.
    """
    if not debug_enabled():
        return
    header = f"=== {title} ==="
    lines = [header]
    for name, value in fields.items():
        lines.append(f"{name.replace('_', ' ')}: {_truncate(str(value))}")
    lines.append("=" * len(header))
    print("\n".join(lines), file=sys.stderr)
