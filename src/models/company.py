from typing import Literal, Optional
from pydantic import BaseModel, Field


class NewsArticle(BaseModel):
    """
    Represents a single news article or filing...
    """

    title: str

    summary: str

    source: str

    url: str

    published_at: str

    is_official: bool = False

    # Source type for provenance / reliability tier classification.
    # "official"  → NSE, RBI, regulatory, company IR
    # "tier1"     → Reuters, Bloomberg, established wires
    # "tier2"     → ET, Moneycontrol, Mint, regional financial press
    # "tavily"    → retrieved via Tavily (underlying publisher may vary)
    source_type: Literal["official", "tier1", "tier2", "tavily"] = "tier2"

    # Relevance score assigned deterministically by ranking logic (0–20).
    # Higher = more relevant to the company/event being tracked.
    relevance_score: int = Field(default=0, ge=0)


class CompanyNews(BaseModel):
    """
    Structured output from the Company News Agent.
    """

    ticker: str

    company_name: str

    articles: list[NewsArticle] = Field(
        default_factory=list,
        description="Deduplicated, ranked news articles for the company.",
    )

    total_articles: int = 0