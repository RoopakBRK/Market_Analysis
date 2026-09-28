"""
LLM Structured Output utilities.

Smaller/open-weight models (e.g. the Groq gpt-oss models used across this
project) are unreliable at "return JSON matching this JSON Schema" prompts:
they frequently echo the schema's own shape back — wrapping every field in
{"type": ..., "title": ..., "value": ...} — instead of emitting the flat
object the schema describes. This produces malformed / unparseable JSON
(observed in SentimentAgent, MacroAgent, ReportAgent).

A flat *example* object, with placeholder values that hint at the expected
type, is far more reliable for these models than the raw schema. This module
builds that example from a Pydantic model so each agent doesn't hand-roll
schema formatting.
"""

from typing import get_args, get_origin
from pydantic import BaseModel


def _placeholder_for_annotation(annotation, field) -> object:
    """Return a representative example value for a Pydantic field's type."""
    origin = get_origin(annotation)
    args = get_args(annotation)

    # Optional[X] / X | None -> unwrap to X
    if origin is not None and type(None) in args:
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            return _placeholder_for_annotation(non_none[0], field)
        return None

    # Literal["A", "B", ...] -> show the choices, use the first as example
    if origin is not None and getattr(origin, "__name__", "") == "Literal":
        return f"<one of: {' | '.join(str(a) for a in args)}>"
    if hasattr(annotation, "__members__"):  # Enum
        return f"<one of: {' | '.join(annotation.__members__.keys())}>"

    if origin in (list, tuple, set):
        inner = args[0] if args else str
        return [_placeholder_for_annotation(inner, field)]

    if origin is dict or annotation is dict:
        return {}

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _example_object(annotation)

    if annotation is int:
        desc = getattr(field, "description", None)
        return f"<integer{': ' + desc if desc else ''}>"
    if annotation is float:
        return 0.0
    if annotation is bool:
        return False
    if annotation is str:
        desc = getattr(field, "description", None)
        return f"<{desc}>" if desc else "<string>"

    return None


def _example_object(model: type[BaseModel], exclude: frozenset[str] = frozenset()) -> dict:
    example = {}
    for name, field in model.model_fields.items():
        if name in exclude:
            continue
        example[name] = _placeholder_for_annotation(field.annotation, field)
    return example


def build_json_instruction(
    model: type[BaseModel],
    *,
    extra_rules: str = "",
    exclude_fields: tuple[str, ...] = (),
    escape_braces: bool = True,
) -> str:
    """
    Build a JSON-output instruction block for a Pydantic model using a flat
    example object (not a raw JSON Schema dump) — this is what smaller LLMs
    reliably imitate.

    escape_braces=True (default) pre-escapes braces ({{ / }}) so the result
    is safe to embed inside a LangChain ChatPromptTemplate string, which
    treats single braces as format placeholders. Pass escape_braces=False
    when embedding directly in an f-string / plain message content (e.g. a
    SystemMessage built without a ChatPromptTemplate) — escaping there would
    inject literal double braces into the prompt sent to the model.
    """
    import json

    example = _example_object(model, exclude=frozenset(exclude_fields))
    example_str = json.dumps(example, indent=2)
    braces = ("{{", "}}") if escape_braces else ("{", "}")
    open_b, close_b = braces
    shown = example_str.replace("{", open_b).replace("}", close_b)

    return f"""
Return ONLY a single valid JSON object — the object itself, not a schema.
Do NOT wrap fields as {open_b}"type": ..., "value": ...{close_b} — write the actual values directly.
Do NOT wrap JSON inside markdown code blocks.
Do NOT explain or add any text before or after the JSON.
{extra_rules}
Match this exact shape (values below are placeholders showing the expected type/format):
{shown}"""


def unwrap_schema_echo(data: dict) -> dict:
    """
    Safety net for models that ignore the instruction and echo a JSON-Schema
    shape anyway. Handles two observed patterns:

    1. The whole object wrapped under a top-level "properties" key.
    2. Each field individually echoed as {"type": ..., "title": ...,
       "value": <actual value>} instead of the bare value.

    Fields that are already plain values pass through untouched.
    """
    if not isinstance(data, dict):
        return data

    if "properties" in data and isinstance(data["properties"], dict):
        data = data["properties"]

    unwrapped = {}
    for key, value in data.items():
        if (
            isinstance(value, dict)
            and "value" in value
            and ({"type", "title", "enum", "description", "anyOf"} & value.keys())
        ):
            unwrapped[key] = value["value"]
        else:
            unwrapped[key] = value
    return unwrapped
