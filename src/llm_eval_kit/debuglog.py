"""LLM_DEBUG — a learning aid that prints every judge request/response to stderr.

Set the environment variable ``LLM_DEBUG=1`` — or put ``LLM_DEBUG=1`` in the project's ``.env``
file — and every judge call — the offline ``FakeJudge`` as well as the real
``GeminiJudge``/``AnthropicJudge`` — prints the full prompt it sends and the verdict it gets
back, as plain-ASCII blocks on **stderr** (so it never mixes with normal program output on
stdout). Unset it, or set it to ``0``/``false`` (any casing), and the kit stays silent.

Precedence: the real environment variable always wins when it is set (even ``LLM_DEBUG=0``);
only when it is *absent* do we fall back to ``LLM_DEBUG`` in ``.env`` in the current working
directory. That way a one-off ``LLM_DEBUG=0 evalkit run`` can silence a project whose ``.env``
says ``LLM_DEBUG=1``.

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

import functools
import os
import sys

from dotenv import dotenv_values

# Fields longer than this are truncated so a giant prompt can't drown your terminal.
_MAX_FIELD_CHARS = 2000


def _is_truthy(value: str) -> bool:
    """Shared truthiness rule: anything except ""/"0"/"false" (any casing) means "on"."""
    return value.strip().lower() not in {"", "0", "false"}


@functools.lru_cache(maxsize=1)
def _dotenv_llm_debug() -> str:
    """Read LLM_DEBUG from the ``.env`` file in the current working directory.

    Returns "" when there is no ``.env`` or it has no LLM_DEBUG entry. Cached with
    ``lru_cache`` so the file is read at most once per process — debug logging sits on the
    hot path of every judge call, and the setting is not expected to change mid-run.
    (Tests call ``_dotenv_llm_debug.cache_clear()`` when they swap directories.)
    """
    # python-dotenv is already here as a pydantic-settings dependency — no new install needed.
    # dotenv_values quietly returns {} when the file doesn't exist.
    return dotenv_values(".env", encoding="utf-8-sig").get("LLM_DEBUG") or ""


def debug_enabled() -> bool:
    """Is debug logging on?

    Precedence, beginner version:

    1. If the real environment variable ``LLM_DEBUG`` is set, it decides — truthy turns
       logging on, and an explicit ``0``/``false`` turns it off *even if* ``.env`` says 1.
    2. Only when the environment variable is completely absent do we fall back to the
       ``LLM_DEBUG`` line in the project's ``.env`` file (if any).
    """
    from_environ = os.environ.get("LLM_DEBUG")
    if from_environ is not None:
        return _is_truthy(from_environ)
    return _is_truthy(_dotenv_llm_debug())


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
