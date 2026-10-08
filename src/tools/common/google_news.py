"""
Google News RSS search.

Used for publishers whose own site search cannot be scraped (Reuters sits
behind bot protection, Livemint's search endpoint is gone). The feed carries
a headline, publisher and publication date per item.
"""

import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import requests

from src.tools.common.scraper_utils import make_headers

_FEED_URL = "https://news.google.com/rss/search"


def search_google_news(query: str, site: str = "", days: int = 7, limit: int = 5) -> list[dict]:
    """
    Search Google News, optionally restricted to one publisher's domain.

    Returns dicts with 'title', 'url', 'published_at' (ISO-8601) and 'source'.
    Raises requests.RequestException on network failure or a malformed feed.
    """
    q = f"{query} site:{site}" if site else query
    response = requests.get(
        _FEED_URL,
        params={"q": f"{q} when:{days}d", "hl": "en-IN", "gl": "IN", "ceid": "IN:en"},
        headers=make_headers(),
        timeout=10,
    )
    response.raise_for_status()

    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as exc:
        raise requests.RequestException(f"Malformed Google News feed: {exc}") from exc

    items = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        source = (item.findtext("source") or "").strip()
        # Google appends the publisher to every headline: "Headline - Reuters".
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3].strip()

        published_at = ""
        try:
            published_at = parsedate_to_datetime(item.findtext("pubDate") or "").isoformat()
        except (TypeError, ValueError):
            pass

        if title:
            items.append({
                "title": title,
                "url": (item.findtext("link") or "").strip(),
                "published_at": published_at,
                "source": source,
            })
        if len(items) >= limit:
            break

    return items
