"""Offline tests for the LLM gateway's retry / fallback wrapper."""

import pytest

from src.llm.gateway import RateLimitedLLM


class FakeLLM:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = 0

    def invoke(self, *args, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


def test_primary_result_is_returned_without_touching_fallback():
    primary, fallback = FakeLLM(result="primary"), FakeLLM(result="fallback")

    assert RateLimitedLLM(primary, "TestAgent", fallback).invoke("hi") == "primary"
    assert fallback.calls == 0


def test_fallback_is_used_when_primary_fails():
    primary = FakeLLM(error=RuntimeError("model decommissioned"))
    fallback = FakeLLM(result="fallback")

    assert RateLimitedLLM(primary, "TestAgent", fallback).invoke("hi") == "fallback"
    assert primary.calls == 1 and fallback.calls == 1


def test_error_is_raised_when_every_model_fails():
    primary = FakeLLM(error=RuntimeError("primary down"))
    fallback = FakeLLM(error=RuntimeError("fallback down"))

    with pytest.raises(RuntimeError, match="fallback down"):
        RateLimitedLLM(primary, "TestAgent", fallback).invoke("hi")


def test_error_is_raised_without_a_fallback():
    with pytest.raises(RuntimeError, match="primary down"):
        RateLimitedLLM(FakeLLM(error=RuntimeError("primary down")), "TestAgent").invoke("hi")


# ── Longer chains and cooldown ───────────────────────────────────────────────

def test_chain_falls_through_to_the_last_model():
    groq_primary = FakeLLM(error=RuntimeError("quota exhausted"))
    groq_fallback = FakeLLM(error=RuntimeError("quota exhausted"))
    claude = FakeLLM(result="claude")

    assert RateLimitedLLM(groq_primary, "TestAgent", groq_fallback, claude).invoke("hi") == "claude"
    assert (groq_primary.calls, groq_fallback.calls, claude.calls) == (1, 1, 1)


def test_missing_fallbacks_are_left_out_of_the_chain():
    # The gateway passes None for Claude when no Anthropic key is configured.
    llm = RateLimitedLLM(FakeLLM(error=RuntimeError("down")), "TestAgent", FakeLLM(result="fallback"), None)

    assert len(llm.models) == 2
    assert llm.invoke("hi") == "fallback"


def test_failed_model_is_skipped_while_it_cools_down():
    primary = FakeLLM(error=RuntimeError("quota exhausted"))
    fallback = FakeLLM(result="fallback")
    llm = RateLimitedLLM(primary, "TestAgent", fallback)

    assert llm.invoke("first") == "fallback"
    assert llm.invoke("second") == "fallback"
    # The primary was tried once, then left alone.
    assert (primary.calls, fallback.calls) == (1, 2)

    # Once the cooldown has passed it is tried again.
    llm.models[0].unavailable_until = 0.0
    llm.invoke("third")
    assert primary.calls == 2


def test_all_models_are_tried_when_every_one_is_cooling_down():
    primary = FakeLLM(error=RuntimeError("down"))
    llm = RateLimitedLLM(primary, "TestAgent")

    with pytest.raises(RuntimeError):
        llm.invoke("first")
    with pytest.raises(RuntimeError):
        llm.invoke("second")
    assert primary.calls == 2


def test_rate_limits_are_retried_unless_the_client_retries_itself(monkeypatch):
    import src.llm.gateway as gateway_module

    monkeypatch.setattr(gateway_module.time, "sleep", lambda seconds: None)

    plain = FakeLLM(error=RuntimeError("Error code: 429 rate limit exceeded"))
    with pytest.raises(RuntimeError):
        RateLimitedLLM(plain, "TestAgent").invoke("hi")
    assert plain.calls == 3

    self_retrying = FakeLLM(error=RuntimeError("Error code: 429 rate limit exceeded"))
    self_retrying.retries_itself = True
    with pytest.raises(RuntimeError):
        RateLimitedLLM(self_retrying, "TestAgent").invoke("hi")
    assert self_retrying.calls == 1
