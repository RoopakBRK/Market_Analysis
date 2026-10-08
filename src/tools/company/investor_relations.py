import requests
from langchain_core.tools import tool
from src.tools.common.scraper_utils import deduplicate_articles
from src.tools.company.nse import announcement_to_article, fetch_announcements

# Exchange filing categories that are investor communications, as opposed to
# general or compliance filings.
_INVESTOR_CATEGORIES = (
    "press release",
    "investor presentation",
    "analysts/institutional investor meet",
    "financial result",
    "outcome of board meeting",
    "earnings call",
    "transcript",
)


@tool
def get_investor_relations(company: str) -> dict:
    """
    Fetch a company's recent investor communications (press releases, results,
    investor presentations and analyst-meet notices) as filed with NSE.
    `company` must be the NSE trading symbol (e.g. "ADANIPORTS").
    """
    try:
        articles = []
        # Results and presentations are infrequent, so look back a month.
        for item in fetch_announcements(company, days=30):
            category = (item.get("desc") or "").lower()
            if not any(wanted in category for wanted in _INVESTOR_CATEGORIES):
                continue
            article = announcement_to_article(item, source="Investor Relations")
            if article:
                articles.append(article)
            if len(articles) >= 5:
                break

        return {
            "company": company,
            "articles": deduplicate_articles(articles),
        }

    except requests.RequestException as e:
        import sys
        print(f"Error in get_investor_relations: {e}", file=sys.stderr)
        # Graceful degradation on network failures
        return {
            "company": company,
            "articles": [],
        }
