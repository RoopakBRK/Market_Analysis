from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from config.settings import settings


import time
import random

llm_usage_stats = {
    "MacroAgent": 0,
    "CompanyNewsAgent": 0,
    "MarketDataAgent": 0,
    "FinancialDataAgent": 0,
    "RedditSentimentAgent": 0,
    "SentimentAgent": 0,
    "ReportAgent": 0,
    "Unknown": 0
}

def print_usage_stats():
    print("\n## LLM USAGE")
    print("-" * 27)
    total = 0
    for agent, count in llm_usage_stats.items():
        if count > 0 or agent != "Unknown":
            print(f"{agent:<20} {count} calls")
            total += count
    print("-" * 27)
    print(f"{'Total:':<20} {total} calls\n")

class RateLimitedLLM:
    """
    Wraps a chat model with usage counting, rate-limit retries and an
    optional fallback model.

    Failures are raised, not swallowed: callers (the graph nodes) already
    catch exceptions and substitute a placeholder result, and need the real
    error to report. Returning an empty message instead made every failure
    look like "the model returned an empty response".
    """

    def __init__(self, llm, agent_name: str, fallback_llm=None):
        self.llm = llm
        self.agent_name = agent_name
        self.fallback_llm = fallback_llm

    def _invoke_with_retry(self, llm, *args, **kwargs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return llm.invoke(*args, **kwargs)
            except Exception as e:
                error_msg = str(e).lower()
                retryable = "429" in error_msg or "rate limit" in error_msg or "503" in error_msg
                if not retryable or attempt == max_retries - 1:
                    raise

                sleep_time = (2 ** attempt) + random.uniform(1, 3)
                print(f"\n[{self.agent_name}] Rate limit hit. Retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)

    def invoke(self, *args, **kwargs):
        global llm_usage_stats
        llm_usage_stats[self.agent_name] = llm_usage_stats.get(self.agent_name, 0) + 1

        try:
            return self._invoke_with_retry(self.llm, *args, **kwargs)
        except Exception as e:
            if self.fallback_llm is None:
                raise
            print(f"\n[{self.agent_name}] Primary LLM failed ({type(e).__name__}: {e}). Trying fallback model...")

        llm_usage_stats[self.agent_name] += 1
        return self._invoke_with_retry(self.fallback_llm, *args, **kwargs)

    def bind_tools(self, *args, **kwargs):
        # Wrap the bound LLM so it continues to track usage and retry
        bound_llm = self.llm.bind_tools(*args, **kwargs)
        bound_fallback = self.fallback_llm.bind_tools(*args, **kwargs) if self.fallback_llm else None
        return RateLimitedLLM(bound_llm, self.agent_name, bound_fallback)

class LLMGateway:
    """
    Centralized LLM Gateway.
    """

    def __init__(self):
        self.primary_llm = init_chat_model(
            model=settings.PRIMARY_MODEL,
            model_provider="groq",
            api_key=settings.GROQ_API_KEY,
            temperature=0,
            # ReportAgent/MacroAgent emit large structured JSON (per-ticker
            # breakdowns); the provider default cap truncates mid-string on
            # larger watchlists, which then fails JSON parsing downstream.
            max_tokens=settings.LLM_MAX_TOKENS,
        )

        self.fallback_llm = init_chat_model(
            model=settings.FALLBACK_MODEL,
            model_provider="groq",
            api_key=settings.GROQ_FALLBACK_API_KEY,
            temperature=0,
            max_tokens=settings.LLM_MAX_TOKENS,
        )

    def get_llm(self, agent_name: str = "Unknown"):
        return RateLimitedLLM(self.primary_llm, agent_name, fallback_llm=self.fallback_llm)

    def get_fallback_llm(self, agent_name: str = "Unknown"):
        return RateLimitedLLM(self.fallback_llm, agent_name)


gateway = LLMGateway()

def get_llm(agent_name: str = "Unknown"):
    return gateway.get_llm(agent_name)

def get_fallback_llm(agent_name: str = "Unknown"):
    return gateway.get_fallback_llm(agent_name)