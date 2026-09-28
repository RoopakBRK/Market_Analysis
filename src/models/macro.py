from typing import Literal

from pydantic import BaseModel, Field

from src.models.common import ConfidenceScore, StringList


class MacroSummary(BaseModel):
    """
    Structured output from the Macro Agent.
    """

    overall_sentiment: Literal[
        "Bullish",
        "Bearish",
        "Neutral",
    ]

    confidence: ConfidenceScore = Field(
        ge=0,
        le=100,
        description="Overall confidence score (0-100).",
    )

    summary: str = Field(
        description="Concise summary of the macro environment."
    )

    key_drivers: StringList = Field(
        description="Primary reasons behind the macro sentiment."
    )

    market_events: StringList = Field(
        default_factory=list,
        description="Important market events to watch today.",
    )

    last_updated: str = Field(
        description="Timestamp when the summary was generated."
    )

    market_data: dict = Field(
        default_factory=dict,
        description="Deterministic raw market data from tools.",
    )