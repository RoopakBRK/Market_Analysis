import requests
import urllib.parse
from langchain_core.tools import tool
from src.models.company import NewsArticle
from src.tools.common.scraper_utils import (
    fetch_html, clean_text, absolute_url, deduplicate_articles
)

ARTICLE_SELECTORS = [".story-box", ".flx"]


@tool
def search_economic_times_news(company: str) -> dict:
    """
    Search Economic Times for company-specific news.
    """
    try:
        formatted_company = urllib.parse.quote(company.strip().lower().replace(" ", "-"))
        url = f"https://economictimes.indiatimes.com/topic/{formatted_company}"
        
        parser = fetch_html(url)
        if not parser:
            raise requests.RequestException("Failed to fetch or parse HTML")
            
        articles = []
        
        nodes = parser.css("a")
        for node in nodes:
            href = node.attributes.get("href") or ""
            if "/articleshow/" in href and "/topic/" not in href:
                title = clean_text(node.text())
                if len(title) > 20 and not title.lower().startswith("read more"):
                    link = absolute_url("https://economictimes.indiatimes.com", href)
                    article = NewsArticle(
                        title=title,
                        summary="",
                        url=link,
                        published_at="",
                        source="Economic Times",
                        is_official=False,
                    ).model_dump()
                    articles.append(article)
                    if len(articles) >= 5:
                        break
                
        return {
            "company": company,
            "articles": deduplicate_articles(articles),
        }

    except requests.RequestException as e:
        import sys
        print(f"Error in search_economic_times_news: {e}", file=sys.stderr)
        # Graceful degradation on network failures
        return {
            "company": company,
            "articles": [],
        }