import requests
from langchain_core.tools import tool
from src.models.company import NewsArticle
from src.tools.common.google_news import search_google_news
from src.tools.common.scraper_utils import deduplicate_articles


@tool
def search_reuters_news(company: str) -> dict:
    """
    Search Reuters for company-specific news.
    Goes through Google News: reuters.com's own search blocks scrapers.
    """
    try:
        articles = [
            NewsArticle(
                title=item["title"],
                summary="",
                url=item["url"],
                published_at=item["published_at"],
                source="Reuters",
                is_official=False,
            ).model_dump()
            for item in search_google_news(company.strip(), site="reuters.com")
        ]

        return {
            "company": company,
            "articles": deduplicate_articles(articles),
        }

    except requests.RequestException as e:
        import sys
        print(f"Error in search_reuters_news: {e}", file=sys.stderr)
        # Graceful degradation on network failures
        return {
            "company": company,
            "articles": [],
        }
