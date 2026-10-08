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
    # A different model from the primary, called with the fallback API key, so
    # the fallback covers a model outage as well as a rate-limited key.
    FALLBACK_MODEL: str = "openai/gpt-oss-120b"

    # Output token cap for all LLM calls. The ReportAgent's JSON payload grows
    # with watchlist size and can exceed provider defaults, truncating the
    # response mid-string and breaking JSON parsing. Override via env if needed.
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "8000"))

    # ── Anthropic (last-resort LLM fallback) ─────────────────────────────────
    # Used only when both Groq models fail, e.g. the Groq quota has run out.
    # Leave the key blank to run on Groq alone.
    ANTHROPIC_FALLBACK_API_KEY: str = os.getenv("ANTHROPIC_FALLBACK_API_KEY", "")
    # A current-generation Claude model id, e.g. claude-opus-5-5 (most
    # capable), claude-sonnet-5-5 or claude-haiku-5-5 (cheapest).
    ANTHROPIC_FALLBACK_MODEL: str = os.getenv("ANTHROPIC_FALLBACK_MODEL", "claude-opus-5-5")
    # Covers Claude's thinking as well as its answer, hence higher than LLM_MAX_TOKENS.
    ANTHROPIC_MAX_TOKENS: int = int(os.getenv("ANTHROPIC_MAX_TOKENS", "16000"))

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

    # ── Qdrant (price-history RAG store) ─────────────────────────────────────
    # Leave QDRANT_URL blank to run without the store; the pipeline then
    # skips historical context, the same way it skips Reddit.
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "nifty_price_history")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

