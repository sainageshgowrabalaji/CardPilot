"""Settings, read once from environment variables (and a local .env file).

Secrets never live in code. Keys come from the environment only, and nothing here is logged.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CARDPILOT_", env_file=".env", extra="ignore")

    # Which model answers. "auto" uses Groq when its key is set, then Claude, then the offline engine.
    engine: Literal["auto", "groq", "claude", "offline"] = "auto"
    groq_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("GROQ_API_KEY", "CARDPILOT_GROQ_API_KEY")
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("ANTHROPIC_API_KEY", "CARDPILOT_ANTHROPIC_API_KEY")
    )
    groq_model: str = "openai/gpt-oss-120b"
    groq_fallback_model: str = "openai/gpt-oss-20b"
    claude_model: str = "claude-haiku-4-5"
    model_timeout_s: float = 25.0

    # Where the search index lives. SQLite needs no setup. Postgres with pgvector is the production store.
    store: Literal["sqlite", "postgres"] = "sqlite"
    database_url: str | None = None
    index_dir: Path = ROOT / "data" / "index"
    cards_dir: Path = ROOT / "data" / "cards"
    web_dir: Path = ROOT / "web"

    # "glove" ships inside the package and needs no download. "bge-small" is stronger and downloads once.
    embedder: Literal["glove", "bge-small"] = "glove"

    # Limits that keep one user from using up the free model quota for everyone.
    max_tool_calls: int = 6
    max_question_chars: int = 800
    rate_limit_per_minute: int = 20
    session_ttl_hours: int = 12
    # Read the client address from X-Forwarded-For. Turn on only behind a proxy you run.
    trust_proxy_headers: bool = False

    # Optional tracing to Langfuse. Off unless both keys are set.
    langfuse_public_key: str | None = Field(default=None, validation_alias=AliasChoices("LANGFUSE_PUBLIC_KEY"))
    langfuse_secret_key: SecretStr | None = Field(default=None, validation_alias=AliasChoices("LANGFUSE_SECRET_KEY"))

    def chosen_engine(self) -> Literal["groq", "claude", "offline"]:
        if self.engine != "auto":
            return self.engine
        if self.groq_api_key:
            return "groq"
        if self.anthropic_api_key:
            return "claude"
        return "offline"


@lru_cache
def get_settings() -> Settings:
    return Settings()
