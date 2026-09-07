from typing import Literal, Optional

from pydantic import BaseModel, Field


class SentimentResult(BaseModel):
    """
    Structured output from the Sentiment Agent.

    Distinguishes verified news, financial data, and Reddit/community signals
    so that downstream components can reason about source reliability.
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
        description="Confidence score between 0 and 100.",
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
        description="Specific positive factors from the supplied articles.",
    )

    negative_drivers: list[str] = Field(
        default_factory=list,
        description="Specific negative factors from the supplied articles.",
    )

    articles_analyzed: int = Field(
        description="Number of articles used to determine this sentiment."
    )

    # ── Multi-signal source breakdown ────────────────────────────────────────

    # Sentiment derived only from verified Tier-1/Tier-2 news articles.
    # None if no verified news was available.
    verified_news_sentiment: Optional[str] = Field(
        default=None,
        description="Sentiment from verified news sources only (Tier 1/2).",
    )

    # Signal from structured financial data (PE, revenue, events, etc.).
    # None if financial data was unavailable.
    financial_data_signal: Optional[str] = Field(
        default=None,
        description="Qualitative signal from financial metrics (positive/neutral/negative).",
    )

    # Aggregated Reddit/community sentiment.
    # This is community/alternative evidence — NOT verified information.
    # None if Reddit data was unavailable or empty.
    reddit_sentiment: Optional[str] = Field(
        default=None,
        description="Community sentiment from Reddit (community signal, not verified news).",
    )

    # Structured breakdown of how each signal source contributed.
    # Keys: "verified_news", "financial_data", "reddit"
    source_breakdown: dict = Field(
        default_factory=dict,
        description="Per-source signal summary for explainability.",
    )