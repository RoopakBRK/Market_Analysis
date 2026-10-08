"""Offline tests for the Claude fallback adapter: the Anthropic client is faked."""

from types import SimpleNamespace

import anthropic
import httpx2
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from src.llm.anthropic_fallback import AnthropicFallbackLLM, to_anthropic_messages


def _response(text="ok", stop_reason="end_turn", category=None):
    return SimpleNamespace(
        model="claude-opus-5-5",
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category=category) if stop_reason == "refusal" else None,
        # Thinking and fallback blocks come back alongside the answer text.
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
    )


class FakeEndpoint:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeClient:
    def __init__(self, result=None):
        result = result or _response()
        self.messages = FakeEndpoint(result)
        self.beta = SimpleNamespace(messages=FakeEndpoint(result))


def _llm(client, model="claude-opus-5-5") -> AnthropicFallbackLLM:
    return AnthropicFallbackLLM(api_key="unused", model=model, max_tokens=16000, client=client)


# ── Message conversion ───────────────────────────────────────────────────────

def test_prompt_forms_convert_to_system_and_messages():
    expected = ("Be brief.", [{"role": "user", "content": "Hello {braces}"}])

    template = ChatPromptTemplate.from_messages([("system", "Be brief."), ("human", "{input}")])
    assert to_anthropic_messages(template.invoke({"input": "Hello {braces}"})) == expected
    assert to_anthropic_messages([SystemMessage(content="Be brief."), HumanMessage(content="Hello {braces}")]) == expected
    assert to_anthropic_messages("Hello") == ("", [{"role": "user", "content": "Hello"}])


def test_assistant_turns_and_block_content_are_converted():
    system, messages = to_anthropic_messages([
        HumanMessage(content=[{"type": "text", "text": "First "}, {"type": "text", "text": "question"}]),
        AIMessage(content="An answer"),
        HumanMessage(content="A follow-up"),
    ])

    assert system == ""
    assert messages == [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "An answer"},
        {"role": "user", "content": "A follow-up"},
    ]


def test_prompt_without_a_user_message_is_rejected():
    with pytest.raises(ValueError, match="no user message"):
        to_anthropic_messages([SystemMessage(content="Only a system prompt")])


# ── Request shape ────────────────────────────────────────────────────────────

def test_request_for_the_default_model():
    client = FakeClient(_response('{"sentiment": "Bullish"}'))
    result = _llm(client).invoke([SystemMessage(content="Be brief."), HumanMessage(content="Hello")])

    assert isinstance(result, AIMessage)
    assert result.content == '{"sentiment": "Bullish"}'   # thinking block left out
    assert result.response_metadata == {"model": "claude-opus-5-5", "stop_reason": "end_turn"}

    assert client.messages.calls == []
    request = client.beta.messages.calls[0]
    assert request == {
        "model": "claude-opus-5-5",
        "max_tokens": 16000,
        "system": "Be brief.",
        "messages": [{"role": "user", "content": "Hello"}],
        "output_config": {"effort": "medium"},
        # Server-side refusal fallback.
        "betas": ["server-side-fallback-2026-07-01"],
        "fallbacks": "default",
    }
    # Sampling parameters and explicit thinking config are rejected by this model.
    assert "temperature" not in request and "thinking" not in request


def test_models_without_server_side_fallback_use_the_plain_endpoint():
    client = FakeClient()
    _llm(client, model="claude-haiku-5-5").invoke("Hello")

    assert client.beta.messages.calls == []
    request = client.messages.calls[0]
    assert request["model"] == "claude-haiku-5-5"
    assert "fallbacks" not in request and "betas" not in request and "system" not in request


# ── Stop reasons and errors ──────────────────────────────────────────────────

def test_refusal_and_truncation_are_errors_not_answers():
    with pytest.raises(RuntimeError, match=r"declined the request \(category: cyber\)"):
        _llm(FakeClient(_response(stop_reason="refusal", category="cyber"))).invoke("Hello")

    with pytest.raises(RuntimeError, match="cut off at max_tokens=16000"):
        _llm(FakeClient(_response(text='{"partial": ', stop_reason="max_tokens"))).invoke("Hello")


def _api_error(cls, status):
    response = httpx2.Response(status, request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    return cls("upstream message", response=response, body=None)


def test_key_and_model_errors_name_the_setting_to_fix():
    with pytest.raises(RuntimeError, match="ANTHROPIC_FALLBACK_API_KEY"):
        _llm(FakeClient(_api_error(anthropic.AuthenticationError, 401))).invoke("Hello")

    with pytest.raises(RuntimeError, match="ANTHROPIC_FALLBACK_MODEL"):
        _llm(FakeClient(_api_error(anthropic.NotFoundError, 404))).invoke("Hello")


def test_other_api_errors_keep_their_type():
    with pytest.raises(anthropic.RateLimitError):
        _llm(FakeClient(_api_error(anthropic.RateLimitError, 429))).invoke("Hello")


# ── Gateway wiring ───────────────────────────────────────────────────────────

def test_claude_joins_the_chain_only_when_a_key_is_set(monkeypatch):
    from config.settings import settings
    from src.llm.gateway import LLMGateway

    monkeypatch.setattr(settings, "ANTHROPIC_FALLBACK_API_KEY", "")
    assert LLMGateway._build_anthropic_fallback() is None

    monkeypatch.setattr(settings, "ANTHROPIC_FALLBACK_API_KEY", "sk-ant-test")
    model = LLMGateway._build_anthropic_fallback()
    assert model.name == settings.ANTHROPIC_FALLBACK_MODEL == "claude-opus-5-5"
    assert isinstance(model.llm, AnthropicFallbackLLM) and model.llm.max_tokens == 16000
