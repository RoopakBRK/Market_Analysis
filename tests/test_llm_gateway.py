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
