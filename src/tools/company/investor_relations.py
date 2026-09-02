import requests
import urllib.parse
from langchain_core.tools import tool
from src.models.company import NewsArticle
from src.tools.common.scraper_utils import (
    fetch_html, clean_text, absolute_url, deduplicate_articles
)

ARTICLE_SELECTORS = [".result__snippet", ".result__url"]


@tool
def get_investor_relations(company: str) -> dict:
    """
    Fetch latest Investor Relations updates.
    """
    try:
        query = f"{company} investor relations press releases"
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        
        parser = fetch_html(url)
        if not parser:
            raise requests.RequestException("Failed to fetch or parse HTML")
            
        articles = []
        
        nodes = parser.css("a.result__url")[:5]
        for node in nodes:
            link = absolute_url("", node.attributes.get("href") or "")
            
            # Find the closest previous title snippet
            # duckduckgo layout uses a class result__snippet for titles sometimes
            # We'll just use a generic approach for the title here since it's a search result
            title_node = node.parent.css_first("a.result__snippet") if node.parent else None
            title = clean_text(title_node.text() if title_node else "Investor Relations Link")
            
            if link:
                article = NewsArticle(
                    title=title,
                    summary="",
                    url=link,
                    published_at="",
                    source="Investor Relations",
                    is_official=True,
                ).model_dump()
                articles.append(article)
                
        return {
            "company": company,
            "articles": deduplicate_articles(articles),
        }

    except requests.RequestException:
        # Graceful degradation on network failures
        return {
            "company": company,
            "articles": [],
        }