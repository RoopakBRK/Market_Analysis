from typing import Annotated

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """
    Shared state passed between all LangGraph nodes.
    """

    # LangGraph conversation history
    messages: Annotated[list, add_messages] = Field(default_factory=list)

    # Macro analysis
    macro_summary: dict | None = None

    # Company news
    company_news: dict = Field(default_factory=dict)

    # Market data
    market_data: dict = Field(default_factory=dict)

    # Sentiment analysis
    sentiments: dict = Field(default_factory=dict)

    # Confidence scores
    confidence_scores: dict = Field(default_factory=dict)

    # Final report
    daily_report: str | None = None

    # Execution metadata
    execution_metadata: dict = Field(default_factory=dict)