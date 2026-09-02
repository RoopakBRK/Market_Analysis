import requests
import urllib.parse
from langchain_core.tools import tool
from src.models.company import NewsArticle
from src.tools.common.scraper_utils import (
    fetch_html, clean_text, absolute_url, deduplicate_articles
)

ARTICLE_SELECTORS = [".headline"]


@tool
def search_mint_news(company: str) -> dict:
    """
    Search Mint for company-specific news.
    """
    try:
        formatted_company = urllib.parse.quote(company.strip().lower().replace(" ", "-"))
        url = f"https://www.livemint.com/search/news?keyword={formatted_company}"
        
        parser = fetch_html(url)
        if not parser:
            raise requests.RequestException("Failed to fetch or parse HTML")
            
        articles = []
        
        for selector in ARTICLE_SELECTORS:
            nodes = parser.css(selector)
            if nodes:
                for node in nodes[:5]:
                    link_node = node.css_first("a")
                    if link_node:
                        title = clean_text(link_node.text())
                        link = absolute_url("https://www.livemint.com", link_node.attributes.get("href") or "")
                        
                        if title:
                            article = NewsArticle(
                                title=title,
                                summary="",
                                url=link,
                                published_at="",
                                source="Mint",
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