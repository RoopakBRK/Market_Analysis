from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from config.settings import settings


import time
import random
from langchain_core.messages import AIMessage

llm_usage_stats = {
    "MacroAgent": 0,
    "CompanyNewsAgent": 0,
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
    def __init__(self, llm, agent_name: str):
        self.llm = llm
        self.agent_name = agent_name

    def invoke(self, *args, **kwargs):
        global llm_usage_stats
        llm_usage_stats[self.agent_name] = llm_usage_stats.get(self.agent_name, 0) + 1
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return self.llm.invoke(*args, **kwargs)
            except Exception as e:
                error_msg = str(e).lower()
                if "429" in error_msg or "rate limit" in error_msg or "503" in error_msg:
                    if attempt == max_retries - 1:
                        print(f"\n[{self.agent_name}] Permanent RateLimitError after {max_retries} attempts.")
                        return AIMessage(content="")
                    
                    sleep_time = (2 ** attempt) + random.uniform(1, 3)
                    print(f"\n[{self.agent_name}] Rate limit hit. Retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                else:
                    print(f"\n[{self.agent_name}] Unexpected LLM Error: {e}")
                    return AIMessage(content="")

    def bind_tools(self, *args, **kwargs):
        # Wrap the bound LLM so it continues to track usage and retry
        bound_llm = self.llm.bind_tools(*args, **kwargs)
        return RateLimitedLLM(bound_llm, self.agent_name)

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
        )

        self.fallback_llm = init_chat_model(
            model=settings.FALLBACK_MODEL,
            model_provider="groq",
            api_key=settings.GROQ_FALLBACK_API_KEY,
            temperature=0,
        )

    def get_llm(self, agent_name: str = "Unknown"):
        return RateLimitedLLM(self.primary_llm, agent_name)

    def get_fallback_llm(self, agent_name: str = "Unknown"):
        return RateLimitedLLM(self.fallback_llm, agent_name)


gateway = LLMGateway()

def get_llm(agent_name: str = "Unknown"):
    return gateway.get_llm(agent_name)

def get_fallback_llm(agent_name: str = "Unknown"):
    return gateway.get_fallback_llm(agent_name)