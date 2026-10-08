from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from config.settings import settings


import sys
import time
import random
from dataclasses import dataclass

llm_usage_stats = {
    "MacroAgent": 0,
    "CompanyNewsAgent": 0,
    "MarketDataAgent": 0,
    "FinancialDataAgent": 0,
    "RedditSentimentAgent": 0,
    "SentimentAgent": 0,
    "ReportAgent": 0,
    "FactCheckAgent": 0,
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


# After a model fails, it is skipped for this long. When a provider is out of
# quota, every call would otherwise sit through that model's retries and
# backoff again before reaching the next one.
_COOLDOWN_SECONDS = 60


@dataclass
class _Model:
    """One model in a fallback chain, with when it may next be tried."""
    name: str
    llm: object
    unavailable_until: float = 0.0


def _as_model(llm) -> _Model:
    if isinstance(llm, _Model):
        return llm
    return _Model(name=getattr(llm, "model", None) or type(llm).__name__, llm=llm)


class RateLimitedLLM:
    """
    Wraps a chain of chat models — the model to use, then its fallbacks in
    order — with usage counting and rate-limit retries.

    Each call goes to the first model that is not cooling down after a recent
    failure. If it fails the next one is tried, and so on.

    Failures are raised, not swallowed: callers (the graph nodes) already
    catch exceptions and substitute a placeholder result, and need the real
    error to report. Returning an empty message instead made every failure
    look like "the model returned an empty response".
    """

    def __init__(self, llm, agent_name: str, *fallback_llms):
        self.llm = llm
        self.agent_name = agent_name
        self.models = [_as_model(m) for m in (llm, *fallback_llms) if m is not None]

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

    def _call(self, model: _Model, *args, **kwargs):
        # A client that retries by itself is called once; retrying it again
        # here would multiply the waiting.
        if getattr(model.llm, "retries_itself", False):
            return model.llm.invoke(*args, **kwargs)
        return self._invoke_with_retry(model.llm, *args, **kwargs)

    def invoke(self, *args, **kwargs):
        global llm_usage_stats

        now = time.monotonic()
        # If every model is cooling down, try them all rather than fail outright.
        candidates = [m for m in self.models if m.unavailable_until <= now] or self.models

        last_error = None
        for position, model in enumerate(candidates):
            llm_usage_stats[self.agent_name] = llm_usage_stats.get(self.agent_name, 0) + 1
            try:
                return self._call(model, *args, **kwargs)
            except Exception as e:
                last_error = e
                model.unavailable_until = time.monotonic() + _COOLDOWN_SECONDS
                if position + 1 < len(candidates):
                    print(
                        f"\n[{self.agent_name}] {model.name} failed "
                        f"({type(e).__name__}: {str(e)[:200]}). Trying {candidates[position + 1].name}..."
                    )
        raise last_error

    def bind_tools(self, *args, **kwargs):
        # Wrap the bound LLMs so they continue to track usage and retry.
        # Models without tool binding (the Anthropic adapter) drop out of the chain.
        bound = [
            _Model(name=m.name, llm=m.llm.bind_tools(*args, **kwargs))
            for m in self.models if hasattr(m.llm, "bind_tools")
        ]
        return RateLimitedLLM(bound[0], self.agent_name, *bound[1:])

class LLMGateway:
    """
    Centralized LLM Gateway.

    Fallback order for ordinary calls:
      1. Groq primary model
      2. Groq fallback model (a different model, on its own API key)
      3. Claude, via ANTHROPIC_FALLBACK_API_KEY — only if that key is set
    """

    def __init__(self):
        self.primary_llm = _Model(settings.PRIMARY_MODEL, init_chat_model(
            model=settings.PRIMARY_MODEL,
            model_provider="groq",
            api_key=settings.GROQ_API_KEY,
            temperature=0,
            # ReportAgent/MacroAgent emit large structured JSON (per-ticker
            # breakdowns); the provider default cap truncates mid-string on
            # larger watchlists, which then fails JSON parsing downstream.
            max_tokens=settings.LLM_MAX_TOKENS,
        ))

        self.fallback_llm = _Model(settings.FALLBACK_MODEL, init_chat_model(
            model=settings.FALLBACK_MODEL,
            model_provider="groq",
            api_key=settings.GROQ_FALLBACK_API_KEY,
            temperature=0,
            max_tokens=settings.LLM_MAX_TOKENS,
        ))

        self.anthropic_llm = self._build_anthropic_fallback()

    @staticmethod
    def _build_anthropic_fallback() -> _Model | None:
        """Claude as the last resort when both Groq models fail; None if no key is set."""
        if not settings.ANTHROPIC_FALLBACK_API_KEY:
            return None
        try:
            from src.llm.anthropic_fallback import AnthropicFallbackLLM
        except ImportError:
            print("[LLM] anthropic not installed; the Claude fallback is off. Install with: pip install anthropic", file=sys.stderr)
            return None

        return _Model(settings.ANTHROPIC_FALLBACK_MODEL, AnthropicFallbackLLM(
            api_key=settings.ANTHROPIC_FALLBACK_API_KEY,
            model=settings.ANTHROPIC_FALLBACK_MODEL,
            max_tokens=settings.ANTHROPIC_MAX_TOKENS,
        ))

    def get_llm(self, agent_name: str = "Unknown"):
        return RateLimitedLLM(self.primary_llm, agent_name, self.fallback_llm, self.anthropic_llm)

    def get_fallback_llm(self, agent_name: str = "Unknown"):
        return RateLimitedLLM(self.fallback_llm, agent_name, self.anthropic_llm)

    def get_checker_llm(self, agent_name: str = "Unknown"):
        """
        An LLM for reviewing another agent's output. The two Groq models swap
        roles (the fallback model leads, backed by the primary) so that the
        review is made by a different model from the one that wrote the text.
        """
        return RateLimitedLLM(self.fallback_llm, agent_name, self.primary_llm, self.anthropic_llm)


gateway = LLMGateway()

def get_llm(agent_name: str = "Unknown"):
    return gateway.get_llm(agent_name)

def get_fallback_llm(agent_name: str = "Unknown"):
    return gateway.get_fallback_llm(agent_name)

def get_checker_llm(agent_name: str = "Unknown"):
    return gateway.get_checker_llm(agent_name)
