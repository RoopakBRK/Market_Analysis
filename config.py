import os
import requests
from functools import lru_cache
# pyrefly: ignore [missing-import]
from python_dotenv import load_dotenv
load_dotenv()
from pydantic import BaseSettings, BaseConfig, SettingsConfigDict 


groqapikey= os.getenv('GROQ_API_KEY')
groqfallbackapikey=os.getenv('GROQ_FALLBACK_API_KEY')

class Settings(BaseSettings):
    GROQ_API_KEY: str
    GROQ_FALLBACK_API_KEY: str

    PRIMARY_MODEL: str = "openai/gpt-oss-20b"
    FALLBACK_MODEL: str = "openai/gpt-oss-20b"

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


