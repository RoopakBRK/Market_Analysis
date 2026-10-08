import requests
from langchain_core.tools import tool
from src.models.company import NewsArticle
from src.tools.common.google_news import search_google_news
from src.tools.common.scraper_utils import deduplicate_articles


@tool
def search_mint_news(company: str) -> dict:
    """
    Search Mint for company-specific news.
    Goes through Google News: Livemint's own search endpoint no longer exists.
    """
    try:
        articles = [
            NewsArticle(
                title=item["title"],
                summary="",
                url=item["url"],
                published_at=item["published_at"],
                source="Mint",
                is_official=False,
            ).model_dump()
            for item in search_google_news(company.strip(), site="livemint.com")
        ]

        return {
            "company": company,
            "articles": deduplicate_articles(articles),
        }

    except requests.RequestException as e:
        import sys
        print(f"Error in search_mint_news: {e}", file=sys.stderr)
        # Graceful degradation on network failures
        return {
            "company": company,
            "articles": [],
        }
