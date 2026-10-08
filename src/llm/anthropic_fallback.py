"""
Claude as the last-resort LLM, behind the two Groq models.

The rest of the project talks to models through LangChain's `.invoke()`, so
this adapter offers the same call and returns an `AIMessage`, while making the
request with the official Anthropic SDK.

Request shape, for current-generation Claude models:
- No `temperature`: sampling parameters are rejected by these models.
- No `thinking` field: thinking is adaptive by default and cannot be switched
  off; `effort` is the control for how much of it happens.
- `max_tokens` covers thinking plus the answer, so it is set generously.
"""

import anthropic
from langchain_core.messages import AIMessage

# Effort trades thoroughness for speed and cost. The pipeline's calls are
# short structured-output tasks, which "medium" handles well.
_EFFORT = "medium"

# Models that accept the server-side refusal fallback: if the model's safety
# classifiers decline a request, Anthropic re-runs it on the model it
# recommends for that refusal category, inside the same call.
_REFUSAL_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
_REFUSAL_FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _text(content) -> str:
    """Plain text of a LangChain message's content (a string or a list of blocks)."""
    if isinstance(content, str):
        return content
    return "".join(
        block if isinstance(block, str) else block.get("text", "")
        for block in content
        if isinstance(block, (str, dict))
    )


def to_anthropic_messages(prompt) -> tuple[str, list[dict]]:
    """
    Convert what the agents pass to `.invoke()` — a string, a list of LangChain
    messages, or a prompt value — into Anthropic's (system, messages) pair.
    """
    if isinstance(prompt, str):
        return "", [{"role": "user", "content": prompt}]

    source = prompt.to_messages() if hasattr(prompt, "to_messages") else list(prompt)

    system_parts, messages = [], []
    for message in source:
        text = _text(message.content)
        if message.type == "system":
            system_parts.append(text)
        else:
            messages.append({"role": "assistant" if message.type == "ai" else "user", "content": text})

    if not messages:
        raise ValueError("The prompt has no user message to send to Claude.")
    return "\n\n".join(system_parts), messages


class AnthropicFallbackLLM:
    # The SDK already retries 429, 5xx and connection errors with backoff, so
    # the gateway must not wrap this in its own retry loop.
    retries_itself = True

    def __init__(self, api_key: str, model: str, max_tokens: int, client=None):
        self.model = model
        self.max_tokens = max_tokens
        self._client = client or anthropic.Anthropic(api_key=api_key)

    def invoke(self, prompt, *args, **kwargs) -> AIMessage:
        system, messages = to_anthropic_messages(prompt)

        request = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "output_config": {"effort": _EFFORT},
        }
        if system:
            request["system"] = system

        try:
            if self.model in _REFUSAL_FALLBACK_MODELS:
                response = self._client.beta.messages.create(
                    **request, betas=[_REFUSAL_FALLBACK_BETA], fallbacks="default"
                )
            else:
                response = self._client.messages.create(**request)
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
            raise RuntimeError(f"Anthropic rejected the key in ANTHROPIC_FALLBACK_API_KEY: {e.message}") from e
        except anthropic.NotFoundError as e:
            raise RuntimeError(
                f"Anthropic does not recognise the model '{self.model}' (ANTHROPIC_FALLBACK_MODEL): {e.message}"
            ) from e
        # RateLimitError, other APIStatusError and APIConnectionError propagate
        # as they are: the SDK has already retried them, and their type names
        # say what happened.

        # Check why the model stopped before trusting the content.
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) or "unspecified"
            raise RuntimeError(f"Claude declined the request (category: {category}).")
        if response.stop_reason == "max_tokens":
            raise RuntimeError(
                f"Claude's answer was cut off at max_tokens={self.max_tokens}; raise ANTHROPIC_MAX_TOKENS."
            )

        # Content is a list of blocks; thinking and fallback blocks carry no answer text.
        text = "".join(block.text for block in response.content if block.type == "text")
        return AIMessage(
            content=text,
            response_metadata={"model": response.model, "stop_reason": response.stop_reason},
        )
