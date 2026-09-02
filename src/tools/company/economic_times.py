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
        
        for selector in ARTICLE_SELECTORS:
            nodes = parser.css(selector)
            if nodes:
                for node in nodes[:5]:
                    title_node = node.css_first("h2, h3, h4, a")
                    link_node = node.css_first("a")
                    
                    if title_node:
                        title = clean_text(title_node.text())
                        link = absolute_url("https://economictimes.indiatimes.com", link_node.attributes.get("href") if link_node else "")
                        
                        if title:
                            article = NewsArticle(
                                title=title,
                                summary="",
                                url=link,
                                published_at="",
                                source="Economic Times",
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