"""Shared test fixtures — keep the suite hermetic.

``debug_enabled()`` can now be switched on by a ``.env`` file in the current working
directory, and a developer's real ``.env`` may well contain ``LLM_DEBUG=1``. Without
protection, that file would flip debug output on for every test and break the
"silent by default" assertions. The autouse fixture below pins the real environment
variable to ``"0"`` for every test — the env var always beats ``.env``, so the suite
passes no matter what the developer's ``.env`` says. Tests that want debug output
simply ``monkeypatch.setenv("LLM_DEBUG", "1")``, and tests that exercise the ``.env``
fallback ``delenv`` the variable inside their own temporary directory.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from llm_eval_kit import debuglog


@pytest.fixture(autouse=True)
def _pin_llm_debug(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pin LLM_DEBUG=0 and reset the cached .env lookup around every test."""
    monkeypatch.setenv("LLM_DEBUG", "0")
    # The .env lookup is lru_cached; clear it so no test sees a value cached by another
    # test (or by a stray import) from a different working directory.
    debuglog._dotenv_llm_debug.cache_clear()
    yield
    debuglog._dotenv_llm_debug.cache_clear()
