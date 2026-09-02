import requests
import urllib.parse
from langchain_core.tools import tool
from src.models.company import NewsArticle
from src.tools.common.scraper_utils import (
    fetch_html, clean_text, absolute_url, deduplicate_articles
)

ARTICLE_SELECTORS = [".search-results__item__2oqiX a"]


@tool
def search_reuters_news(company: str) -> dict:
    """
    Search Reuters for company-specific news.
    """
    try:
        formatted_company = urllib.parse.quote(company.strip())
        url = f"https://www.reuters.com/site-search/?query={formatted_company}"
        
        parser = fetch_html(url)
        if not parser:
            raise requests.RequestException("Failed to fetch or parse HTML")
            
        articles = []
        
        for selector in ARTICLE_SELECTORS:
            nodes = parser.css(selector)
            if nodes:
                for link_node in nodes[:5]:
                    title = clean_text(link_node.text())
                    link = absolute_url("https://www.reuters.com", link_node.attributes.get("href", ""))
                    
                    if title and link:
                        article = NewsArticle(
                            title=title,
                            summary="",
                            url=link,
                            published_at="",
                            source="Reuters",
                            is_official=False,
                        ).model_dump()
                        articles.append(article)
                break

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