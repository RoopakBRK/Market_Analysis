from typing import Literal, Optional
from pydantic import BaseModel, Field


class RedditPost(BaseModel):
    """
    A single post retrieved from Reddit.
    Treated as community/alternative sentiment — NOT verified financial news.
    """

    title: str

    subreddit: str

    url: str

    score: int = Field(
        default=0,
        description="Reddit post score (upvotes minus downvotes)."
    )

    published_at: str = Field(
        description="UTC ISO-8601 timestamp of post creation."
    )

    body: str = Field(
        default="",
        description="Post body/selftext, truncated to avoid token bloat."
    )

    # Fixed label: all Reddit content is community-sourced.
    source_type: str = "reddit"


class RedditSignal(BaseModel):
    """
    Aggregated community/alternative sentiment signal from Reddit.

    This is an ALTERNATIVE evidence source, not equivalent to verified news.
    It should be clearly labeled as such when passed to the Sentiment Agent.
    """

    ticker: str

    company_name: str

    posts: list[RedditPost] = Field(default_factory=list)

    # Deterministically computed from post scores + keyword heuristics.
    # "Unknown" when there are no usable posts.
    overall_sentiment: Literal["Bullish", "Bearish", "Neutral", "Unknown"] = "Unknown"

    post_count: int = 0

    avg_score: float = 0.0

    retrieved_at: str = Field(
        description="UTC ISO-8601 timestamp of retrieval."
    )
