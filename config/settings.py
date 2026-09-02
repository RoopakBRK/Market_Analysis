import os
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_FALLBACK_API_KEY: str = os.getenv("GROQ_FALLBACK_API_KEY", "")

    PRIMARY_MODEL: str = "qwen/qwen3.8-27b"
    FALLBACK_MODEL: str = "llama-3.3-70b-versatile"

    APP_NAME: str = "market-analysis"
    ENVIRONMENT: str = "development"

    DATABASE_URL: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
