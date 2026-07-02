"""Configuration from the environment / `.env`.

In plain words: this file reads settings (API keys, model names, the gate threshold) from
environment variables or a local `.env` file, so the code never hard-codes secrets.
`pydantic-settings` does the reading and type-checking for us.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM-as-judge (optional). Without any key, evals use deterministic scorers / FakeJudge.
    # If GEMINI_API_KEY is set it is preferred (see `make_judge` in judge.py).
    gemini_api_key: str | None = Field(default=None)
    gemini_model: str = Field(default="gemini-2.5-flash")

    anthropic_api_key: str | None = Field(default=None)
    anthropic_model: str = Field(default="claude-opus-4-8")

    max_tokens: int = Field(default=512)

    # The default ship-gate threshold (fraction of cases that must pass).
    threshold: float = Field(default=0.8)


def load_settings() -> Settings:
    return Settings()
