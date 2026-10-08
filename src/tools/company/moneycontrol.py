import requests
import urllib.parse
from langchain_core.tools import tool
from src.models.company import NewsArticle
from selectolax.parser import HTMLParser
from src.tools.common.scraper_utils import (
    make_headers, clean_text, absolute_url, deduplicate_articles
)

ARTICLE_SELECTORS = [".clearfix h2 a", ".news_list li a"]


@tool
def search_moneycontrol_news(company: str) -> dict:
    """
    Search Moneycontrol for company-specific news.
    """
    try:
        # Moneycontrol tag pages are lowercase hyphenated slugs; a space in the
        # URL is rejected with 403.
        formatted_company = urllib.parse.quote(company.strip().lower().replace(" ", "-"))
        url = f"https://www.moneycontrol.com/news/tags/{formatted_company}.html"
        
        response = requests.get(url, headers=make_headers(), timeout=10)
        response.raise_for_status()
        # An unknown tag redirects to the generic /news/ front page. Those
        # headlines are not about this company, so treat it as no results.
        if "/news/tags/" not in response.url:
            return {"company": company, "articles": []}

        parser = HTMLParser(response.content)
        articles = []
        
        nodes = parser.css(".clearfix a")
        if not nodes:
            nodes = parser.css("li.clearfix a")
            
        for link_node in nodes:
            href = link_node.attributes.get("href") or ""
            if "/news/" in href and href.endswith(".html") and "/tags/" not in href and "/category/" not in href:
                title = clean_text(link_node.attributes.get("title") or link_node.text())
                if len(title) > 20 and not title.lower().startswith("read more"):
                    # Attempt to find summary in parent list item's <p> tag
                    summary = ""
                    parent = link_node.parent
                    if parent:
                        p_node = parent.css_first("p")
                        if p_node:
                            summary = clean_text(p_node.text())

                    link = absolute_url("https://www.moneycontrol.com", href)
                    article = NewsArticle(
                        title=title,
                        summary=summary,
                        url=link,
                        published_at="",
                        source="Moneycontrol",
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
        print(f"Error in search_moneycontrol_news: {e}", file=sys.stderr)
        # Graceful degradation on network failures
        return {
            "company": company,
            "articles": [],
        }