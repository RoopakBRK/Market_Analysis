"""
Low-level Firecrawl client.

Responsibilities:
- API key management (read from settings, never logged)
- News search, normalised into internal article dicts
- Article scraping, reduced to the plain prose of the article body
- Graceful failure: search returns [] and scrape returns None on any error
"""

import re
import sys
from datetime import datetime, timezone
from typing import Any

from config.settings import settings
from src.tools.common.date_utils import is_within_days, parse_published_date
from src.tools.common.normalization import normalize_url, trim_content
from src.tools.common.source_classifier import classify_source, publisher_from_url

_DEFAULT_MAX_RESULTS = 5
_DEFAULT_DAYS_BACK = 3
# Search is billed per ten results, so asking for ten costs the same as asking
# for five and leaves spare results for the date window to drop.
_MIN_FETCH = 10
# Maximum characters to preserve from a search snippet.
_MAX_CONTENT_CHARS = 500
_SEARCH_TIMEOUT_MS = 30_000
_SCRAPE_TIMEOUT_MS = 30_000
# Seconds one HTTP call may take. The SDK sets no limit of its own.
_HTTP_TIMEOUT = 45

# A paragraph with fewer words than this is a byline, caption, label or
# button rather than prose.
_MIN_PARAGRAPH_WORDS = 12
# Less prose than this and the page was a stub, a paywall or a block notice.
_MIN_ARTICLE_CHARS = 200
# Links to other stories that publishers drop between an article's paragraphs.
_FURNITURE_PREFIXES = ("also read", "read more", "read also", "also watch", "follow us", "subscribe", "disclaimer")
# Photo captions and credits, which sit above the first paragraph.
_CAPTION = re.compile(
    r"^(?:file photo|photo|image|representational image|representative image)\b|\((?:photo|image)\s*:|REUTERS/",
    re.IGNORECASE,
)
# Where a page states when it was published, in order of preference.
_PUBLISHED_KEYS = ("published_time", "article:published_time", "og:article:published_time")

_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_LIST_MARKER = re.compile(r"^(?:[-*+>]|\d+\.)\s+")

_warned_unconfigured = False


def is_configured() -> bool:
    """True if a Firecrawl API key is set."""
    return bool(settings.FIRECRAWL_API_KEY)


def _build_client():
    """
    Return a Firecrawl client instance, or None if the key is missing.
    We import lazily to avoid import errors when firecrawl-py is not installed.
    """
    global _warned_unconfigured
    api_key = settings.FIRECRAWL_API_KEY
    if not api_key:
        # Said once, not per call: every news tool comes through here.
        if not _warned_unconfigured:
            _warned_unconfigured = True
            print("[Firecrawl] Skipped — no API key set.", file=sys.stderr)
        return None
    try:
        from firecrawl import Firecrawl  # type: ignore
        return Firecrawl(api_key=api_key, timeout=_HTTP_TIMEOUT, max_retries=2)
    except ImportError:
        print("[Firecrawl] firecrawl-py not installed. Install with: pip install firecrawl-py", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"[Firecrawl] Client initialisation failed: {type(exc).__name__}", file=sys.stderr)
        return None


def _normalize_result(title: str, url: str, snippet: str, raw_date: str, published: datetime | None) -> dict[str, Any]:
    """
    Normalise a single Firecrawl news result into the project's internal
    article representation.
    """
    source = publisher_from_url(url)
    source_type = classify_source(source)
    return {
        "title": title,
        "url": url,
        "source": source,
        # Results are dated relative to now ("2 days ago"), which is good to
        # the day at best. Anything unparseable is passed on as it came.
        "published_at": published.date().isoformat() if published else raw_date,
        "summary": trim_content(snippet, max_chars=_MAX_CONTENT_CHARS),
        "source_type": source_type,
        "is_official": source_type == "official",
    }


def search(
    query: str,
    max_results: int = _DEFAULT_MAX_RESULTS,
    days_back: int = _DEFAULT_DAYS_BACK,
) -> list[dict[str, Any]]:
    """
    Execute a Firecrawl news search and return a list of normalised article
    dicts, in the order Firecrawl ranked them.

    Firecrawl's time filter does not apply to news results, so the date
    window is enforced here. Undated results are kept.

    There is deliberately no option to restrict the search to one site:
    combined with a domain filter, the news search ignores the query and
    returns the site's section and policy pages.

    Returns [] gracefully on:
    - missing API key
    - missing firecrawl-py library
    - network timeout
    - API error (including a rate limit or exhausted credits)
    - malformed response
    """
    client = _build_client()
    if client is None:
        return []

    label = f"'{query[:60]}'"
    try:
        response = client.search(
            query,
            sources=["news"],
            limit=max(max_results, _MIN_FETCH),
            location="India",
            country="IN",
            timeout=_SEARCH_TIMEOUT_MS,
        )

        now = datetime.now(timezone.utc)
        articles = []
        stale = 0
        for item in getattr(response, "news", None) or []:
            title = (getattr(item, "title", "") or "").strip()
            url = normalize_url(getattr(item, "url", "") or "")
            if not title or not url:
                continue
            raw_date = (getattr(item, "date", "") or "").strip()
            published = parse_published_date(raw_date, now=now)
            if not is_within_days(published, days_back, now=now):
                stale += 1
                continue
            articles.append(_normalize_result(title, url, getattr(item, "snippet", "") or "", raw_date, published))

        print(
            f"[Firecrawl] Query {label} → {min(len(articles), max_results)} results"
            + (f" ({stale} older than {days_back} days dropped)" if stale else ""),
            file=sys.stderr,
        )
        return articles[:max_results]

    except Exception as exc:
        print(f"[Firecrawl] Search failed for {label}: {type(exc).__name__}: {str(exc)[:200]}", file=sys.stderr)
        return []


def _article_text(markdown: str) -> str:
    """
    Reduce an article's markdown to its prose: the paragraphs, without images,
    link targets, headings, tables or one-line page furniture.
    """
    paragraphs = []
    for line in markdown.splitlines():
        line = _LINK.sub(r"\1", _IMAGE.sub("", line)).strip()
        if not line or line.startswith(("#", "|", "```")):
            continue
        line = _LIST_MARKER.sub("", line)
        line = re.sub(r"\s+", " ", re.sub(r"[*_`]+", "", line)).strip()
        if len(line.split()) < _MIN_PARAGRAPH_WORDS or line.lower().startswith(_FURNITURE_PREFIXES):
            continue
        if _CAPTION.search(line):
            continue
        paragraphs.append(line)
    return " ".join(paragraphs)


def scrape(url: str, wait_ms: int = 0) -> dict[str, Any] | None:
    """
    Fetch one article through Firecrawl and return its body as plain text:
    {"text": ..., "published_at": ..., "url": ...}. published_at is the page's
    own publication timestamp, and url the address the request ended up at
    (which differs from the one asked for when that was a redirect link);
    either may be "".

    `wait_ms` holds the page open that long before reading it. A link that
    redirects in JavaScript needs it: read at once, the page is still the
    empty redirect stub about half the time.

    Returns None gracefully if Firecrawl is unavailable, the page could not be
    fetched, or it came back without an article's worth of text.
    """
    client = _build_client()
    if client is None:
        return None

    try:
        document = client.scrape(
            url,
            formats=["markdown"],
            only_main_content=True,
            # No document parsing: a PDF is billed per page when parsed, and
            # one flat credit when not. Only article pages are wanted here.
            parsers=[],
            wait_for=wait_ms or None,
            # A stub read earlier is cached as an empty page; ask for a fresh one.
            max_age=0 if wait_ms else None,
            timeout=_SCRAPE_TIMEOUT_MS,
        )
    except Exception as exc:
        print(f"[Firecrawl] Scrape failed for {url[:80]}: {type(exc).__name__}: {str(exc)[:200]}", file=sys.stderr)
        return None

    metadata = getattr(document, "metadata", None)
    status = getattr(metadata, "status_code", None)
    # A block or error page still comes back as a document.
    if isinstance(status, int) and not 200 <= status < 300:
        print(f"[Firecrawl] Scrape of {url[:80]} returned HTTP {status}.", file=sys.stderr)
        return None

    text = _article_text(getattr(document, "markdown", "") or "")
    if len(text) < _MIN_ARTICLE_CHARS:
        return None

    fields = metadata.model_dump() if metadata is not None else {}
    published = next((fields[key] for key in _PUBLISHED_KEYS if fields.get(key)), "")
    # A meta tag that appears twice on the page arrives as a list.
    if isinstance(published, list):
        published = published[0] if published else ""

    return {
        "text": text,
        "published_at": str(published),
        "url": normalize_url(fields.get("url") or ""),
    }
