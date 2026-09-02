import requests
from typing import Optional
from urllib.parse import urljoin
from selectolax.parser import HTMLParser
from src.models.company import NewsArticle


def make_headers() -> dict:
    """Returns a standard robust User-Agent dictionary for scraping."""
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }


def fetch_html(url: str, timeout: int = 10) -> Optional[HTMLParser]:
    """
    Fetches HTML from a URL and returns a Selectolax HTMLParser.
    Raises requests.RequestException on network failure.
    """
    response = requests.get(url, headers=make_headers(), timeout=timeout)
    response.raise_for_status()
    return HTMLParser(response.content)


def clean_text(text: Optional[str]) -> str:
    """Cleans up text parsed from HTML elements."""
    if not text:
        return ""
    return text.strip().replace("\n", " ")


def absolute_url(base_url: str, partial_url: str) -> str:
    """Safely constructs an absolute URL."""
    if not partial_url:
        return ""
    return urljoin(base_url, partial_url)


def deduplicate_articles(articles: list[dict]) -> list[dict]:
    """Deduplicates a list of serialized NewsArticle dictionaries by URL."""
    seen_urls = set()
    deduped = []
    for article in articles:
        url = article.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            deduped.append(article)
        elif not url:
            deduped.append(article)
    return deduped
