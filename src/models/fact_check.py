import re
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, Field


def _coerce_id(value: Any) -> Any:
    """Accept "3", "[3]" or "S3" for 3: models do not always return a bare integer."""
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        if match:
            return int(match.group())
    return value


class UnsupportedClaim(BaseModel):
    """One sentence the fact-checker could not support from the facts."""

    id: Annotated[int, BeforeValidator(_coerce_id)] = Field(
        description="Number of the unsupported sentence"
    )

    # Why it is unsupported. "not_in_facts" is the one kind the code can
    # double-check, by looking for the sentence's figures in the facts.
    kind: str = Field(
        default="",
        description="one of: contradicts | not_in_facts | wrong_company",
    )

    reason: str = Field(
        default="",
        description="One line saying what the facts do not support",
    )


class FactCheckResult(BaseModel):
    """Structured output from the Fact Check Agent."""

    unsupported: list[UnsupportedClaim] = Field(default_factory=list)
