from pydantic import BaseModel, Field


class NewsArticle(BaseModel):
    """
    Represents a single news article or filing.
    """

    title: str

    summary: str

    source: str

    url: str

    published_at: str

    is_official: bool = False


class CompanyNews(BaseModel):
    """
    Structured output from the Company News Agent.
    """

    ticker: str

    company_name: str

    articles: list[NewsArticle] = Field(
        default_factory=list,
        description="Deduplicated news articles for the company.",
    )

    total_articles: int = 0