from typing import Literal

from pydantic import BaseModel, Field


class SentimentResult(BaseModel):
    """
    Structured output from the Sentiment Agent.
    """

    ticker: str

    company_name: str

    sentiment: Literal[
        "Bullish",
        "Bearish",
        "Neutral",
        "Unknown",
    ]

    confidence: int = Field(
        ge=0,
        le=100,
        description="Confidence score between 0 and 100."
    )

    impact: Literal[
        "High",
        "Medium",
        "Low",
        "Unknown",
    ]

    summary: str = Field(
        description="Concise summary of what happened and why it matters."
    )

    positive_drivers: list[str] = Field(
        default_factory=list,
        description="Specific positive factors from the supplied articles."
    )

    negative_drivers: list[str] = Field(
        default_factory=list,
        description="Specific negative factors from the supplied articles."
    )

    articles_analyzed: int = Field(
        description="Number of articles used to determine this sentiment."
    )