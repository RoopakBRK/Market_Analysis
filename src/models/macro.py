from typing import Literal

from pydantic import BaseModel, Field


class MacroSummary(BaseModel):
    """
    Structured output from the Macro Agent.
    """

    overall_sentiment: Literal[
        "Bullish",
        "Bearish",
        "Neutral",
    ]

    confidence: int = Field(
        ge=0,
        le=100,
        description="Overall confidence score (0-100).",
    )

    summary: str = Field(
        description="Concise summary of the macro environment."
    )

    key_drivers: list[str] = Field(
        description="Primary reasons behind the macro sentiment."
    )

    market_events: list[str] = Field(
        default_factory=list,
        description="Important market events to watch today.",
    )

    last_updated: str = Field(
        description="Timestamp when the summary was generated."
    )