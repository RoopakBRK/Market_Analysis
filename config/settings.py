import os
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── LLM / Groq ──────────────────────────────────────────────────────────
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_FALLBACK_API_KEY: str = os.getenv("GROQ_FALLBACK_API_KEY", "")

    PRIMARY_MODEL: str = "openai/gpt-oss-20b"
    FALLBACK_MODEL: str = "openai/gpt-oss-20b"

    # ── App metadata ─────────────────────────────────────────────────────────
    APP_NAME: str = "market-analysis"
    ENVIRONMENT: str = "development"

    DATABASE_URL: str | None = None

    # ── Tavily ───────────────────────────────────────────────────────────────
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    # ── Financial Agent API ──────────────────────────────────────────────────
    # NOTE: The Financial Agent API provider is not yet confirmed.
    # The key is read here; the adapter stub will use it when the real
    # provider contract is documented.
    FINANCIAL_AGENT_API_KEY: str = os.getenv("FINANCIAL_AGENT_API_KEY", "")
    FINANCIAL_AGENT_BASE_URL: str = os.getenv(
        "FINANCIAL_AGENT_BASE_URL", ""
    )

    # ── Reddit ───────────────────────────────────────────────────────────────
    REDDIT_CLIENT_ID: str = os.getenv("REDDIT_CLIENT_ID", "")
    REDDIT_CLIENT_SECRET: str = os.getenv("REDDIT_CLIENT_SECRET", "")
    REDDIT_USER_AGENT: str = os.getenv(
        "REDDIT_USER_AGENT", "MarketAnalysisBot/1.0"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

