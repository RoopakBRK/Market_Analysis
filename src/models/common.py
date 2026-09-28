"""
Shared field types for LLM-produced structured output.

LLMs (especially smaller open-weight models) are inconsistent about the
"confidence: 0-100" instruction: they frequently emit a 0-1 fractional score
instead (e.g. 0.7 for "70% confident"), and Pydantic v2's strict `int` type
rejects any float with a fractional part outright (`int_from_float`),
turning a cosmetic scale mismatch into a hard validation failure that drops
the whole agent output.

`ConfidenceScore` normalizes this before validation: floats on a 0-1 scale
are rescaled to 0-100, other floats are rounded, and the result is clamped
to the valid range.
"""

from typing import Annotated
from pydantic import BeforeValidator


def _coerce_confidence(value: object) -> object:
    if isinstance(value, bool):  # bool is an int subclass; reject explicitly
        return value
    if isinstance(value, float):
        # A model answering "0-100" with a 0-1 fraction (e.g. 0.7 meaning
        # 70%) is the dominant failure mode observed in practice.
        if 0.0 <= value <= 1.0:
            value = value * 100
        value = round(value)
    if isinstance(value, int):
        return max(0, min(100, value))
    return value


ConfidenceScore = Annotated[int, BeforeValidator(_coerce_confidence)]


def _coerce_str_list(value: object) -> object:
    """
    Some list[str] fields (e.g. important_news, key_positive_signals) are
    occasionally returned by the model as a single paragraph string instead
    of a list of individual items — observed for ReportAgent's
    `important_news`. Wrap a bare string into a single-item list rather than
    failing validation and dropping the whole report.
    """
    if isinstance(value, str):
        return [value] if value.strip() else []
    return value


StringList = Annotated[list[str], BeforeValidator(_coerce_str_list)]
