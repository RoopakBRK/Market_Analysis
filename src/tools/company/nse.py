import requests
import urllib.parse
from langchain_core.tools import tool
from src.models.company import NewsArticle
from src.tools.common.scraper_utils import deduplicate_articles, make_headers


@tool
def get_nse_announcements(company: str) -> dict:
    """
    Fetch latest NSE announcements for a company.
    """
    try:
        session = requests.Session()
        
        # Hit main page first to get cookies
        session.get("https://www.nseindia.com", headers=make_headers(), timeout=10)
        
        symbol = urllib.parse.quote(company.strip().upper())
        url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}&section=corp_info"
        
        response = session.get(url, headers=make_headers(), timeout=10)
        response.raise_for_status()
        data = response.json()
        
        articles = []
        corp_info = data.get("corporateInfo", {})
        recent_announcements = corp_info.get("announcements", [])[:5]
        
        for item in recent_announcements:
            title = item.get("desc", "")
            if title:
                date = item.get("anndate", "")
                attachment = item.get("attchmntText", "")
                link = f"https://www.nseindia.com{attachment}" if attachment else ""
                
                article = NewsArticle(
                    title=title,
                    summary="",
                    url=link,
                    published_at=date,
                    source="NSE",
                    is_official=True,
                ).model_dump()
                articles.append(article)

        return {
            "company": company,
            "articles": deduplicate_articles(articles),
        }

    except requests.RequestException as e:
        import sys
        print(f"Error in get_nse_announcements: {e}", file=sys.stderr)
        # Graceful degradation on network failures
        return {
            "company": company,
            "articles": [],
        }