import requests
from datetime import date, timedelta
from langchain_core.tools import tool
from src.models.company import NewsArticle
from src.tools.common.normalization import trim_content
from src.tools.common.scraper_utils import deduplicate_articles, make_headers

_ANNOUNCEMENTS_URL = "https://www.nseindia.com/api/corporate-announcements"
_REFERER = "https://www.nseindia.com/companies-listing/corporate-filings-announcements"

# Filing categories that are compliance paperwork with no news value.
_ROUTINE_CATEGORIES = (
    "certificate under sebi",
    "trading window",
    "copy of newspaper publication",
    "loss of share certificate",
    "duplicate share certificate",
)


def fetch_announcements(symbol: str, days: int = 7) -> list[dict]:
    """
    Fetch a company's NSE corporate announcements from the last `days` days,
    newest first, as raw NSE records.

    `symbol` must be the NSE trading symbol (e.g. "ADANIPORTS"), not the
    company name. Raises requests.RequestException on failure.
    """
    session = requests.Session()
    # The API rejects requests that don't look like they came from the site:
    # it wants the site's cookies plus a JSON Accept header and a Referer.
    session.get("https://www.nseindia.com", headers=make_headers(), timeout=10)

    today = date.today()
    response = session.get(
        _ANNOUNCEMENTS_URL,
        params={
            "index": "equities",
            "symbol": symbol.strip().upper(),
            "from_date": (today - timedelta(days=days)).strftime("%d-%m-%Y"),
            "to_date": today.strftime("%d-%m-%Y"),
        },
        headers={
            **make_headers(),
            "Accept": "application/json, text/plain, */*",
            "Referer": _REFERER,
        },
        timeout=15,
    )
    response.raise_for_status()

    try:
        data = response.json()
    except ValueError as exc:
        raise requests.RequestException(f"NSE returned a non-JSON response: {exc}") from exc
    return data if isinstance(data, list) else []


def announcement_to_article(item: dict, source: str) -> dict | None:
    """Convert one raw NSE announcement into a serialized NewsArticle."""
    category = (item.get("desc") or "").strip()
    detail = (item.get("attchmntText") or "").strip()
    if not category and not detail:
        return None

    title = f"{category}: {detail}" if category and detail else (category or detail)
    return NewsArticle(
        title=trim_content(title, max_chars=200),
        summary=trim_content(detail),
        url=item.get("attchmntFile") or "",
        published_at=item.get("an_dt") or "",
        source=source,
        is_official=True,
    ).model_dump()


@tool
def get_nse_announcements(company: str) -> dict:
    """
    Fetch a company's latest NSE corporate announcements.
    `company` must be the NSE trading symbol (e.g. "ADANIPORTS").
    """
    try:
        articles = []
        for item in fetch_announcements(company, days=7):
            category = (item.get("desc") or "").lower()
            if any(routine in category for routine in _ROUTINE_CATEGORIES):
                continue
            article = announcement_to_article(item, source="NSE")
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
        print(f"Error in get_nse_announcements: {e}", file=sys.stderr)
        # Graceful degradation on network failures
        return {
            "company": company,
            "articles": [],
        }
