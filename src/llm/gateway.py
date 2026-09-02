from langchain.chat_models import init_chat_model

from config.settings import settings


class LLMGateway:
    """
    Centralized LLM Gateway.

    Responsibilities:
    - Initialize models
    - Use primary/fallback API keys
    - Hide provider-specific logic
    """

    def __init__(self):
        self.primary_llm = init_chat_model(
            model=settings.PRIMARY_MODEL,
            model_provider="groq",
            api_key=settings.GROQ_API_KEY,
            temperature=0,
        )

        self.fallback_llm = init_chat_model(
            model=settings.FALLBACK_MODEL,
            model_provider="groq",
            api_key=settings.GROQ_FALLBACK_API_KEY,
            temperature=0,
        )

    def get_llm(self):
        return self.primary_llm

    def get_fallback_llm(self):
        return self.fallback_llm


gateway = LLMGateway()


def get_llm():
    return gateway.get_llm()


def get_fallback_llm():
    return gateway.get_fallback_llm()