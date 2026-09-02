from typing import Literal

from pydantic import BaseModel, Field


class SentimentResult(BaseModel):
    """
    Structured output from the Sentiment Agent.
    """

    ticker: str

    company_name: str

    overall_sentiment: Literal[
        "Positive",
        "Neutral",
        "Negative",
    ]

    impact: Literal[
        "Low",
        "Medium",
        "High",
    ]

    expected_duration: str = Field(
        description="Expected duration of the impact (e.g. '1-3 trading sessions')."
    )

    confidence: int = Field(
        ge=0,
        le=100,
        description="Confidence score between 0 and 100."
    )

    reasons: list[str] = Field(
        default_factory=list,
        description="Key reasons supporting the sentiment."
    )

    citations: list[str] = Field(
        default_factory=list,
        description="Sources used to generate the sentiment."
    )

    last_updated: str