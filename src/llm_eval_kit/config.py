"""Configuration from the environment / `.env`."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM-as-judge (optional). Without a key, evals use deterministic scorers / FakeJudge.
    anthropic_api_key: str | None = Field(default=None)
    anthropic_model: str = Field(default="claude-opus-4-8")
    max_tokens: int = Field(default=512)

    # The default ship-gate threshold (fraction of cases that must pass).
    threshold: float = Field(default=0.8)


def load_settings() -> Settings:
    return Settings()
