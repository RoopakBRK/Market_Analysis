"""
Full-text enrichment: reads an article through Firecrawl and uses the opening
of its body as the article's summary.

Most sources hand back a headline alone, or a headline with a one-line search
snippet. The opening paragraphs say what actually happened, and the summary is
what the SentimentAgent reasons over.
"""

import re
import sys
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from src.models.company import NewsArticle
from src.tools.common.normalization import MAX_BODY_CHARS, trim_content
from src.tools.firecrawl import client as firecrawl_client

# A summary this long already says what happened; reading the article would
# spend a credit to learn little more.
_ENOUGH_SUMMARY_CHARS = 200
# Firecrawl's free plan runs two scrapes at a time.
_MAX_PARALLEL_SCRAPES = 2
# Hosts whose links are not article pages: NSE links are filing attachments.
_UNREADABLE_HOSTS = ("nseindia.com",)
# Google News links redirect to the publisher. Firecrawl follows them, so
# reading one also reveals the article's real address.
_REDIRECT_HOSTS = ("news.google.com",)
# How long to let a redirect link settle on the publisher's page.
_REDIRECT_WAIT_MS = 3000


def _is_readable(article: NewsArticle) -> bool:
    """True if the article's URL points at a page worth fetching."""
    parsed = urlparse(article.url)
    if parsed.scheme not in ("http", "https") or parsed.path.lower().endswith(".pdf"):
        return False
    return not _on_host(article.url, _UNREADABLE_HOSTS)


def _on_host(url: str, hosts: tuple[str, ...]) -> bool:
    host = urlparse(url).netloc.lower()
    return any(host == known or host.endswith("." + known) for known in hosts)


def _is_about(text: str, title: str) -> bool:
    """
    True if the page text shares enough of the headline's words to be that
    article, rather than a home page or notice the URL redirected to.
    """
    keywords = {word for word in re.findall(r"[a-z0-9]+", title.lower()) if len(word) >= 4}
    if not keywords:
        return True
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    return len(keywords & words) * 2 >= len(keywords)


def enrich_articles(articles: list[NewsArticle], limit: int) -> list[NewsArticle]:
    """
    Return the articles with a missing or thin summary replaced by the opening
    of the article itself, for at most `limit` of them. The list is taken to
    be in rank order, so the highest-ranked articles are read first. A missing
    publication date is filled in from the page as well, and a Google News
    redirect link is replaced by the publisher's own URL.

    Exchange filings are left alone: their links are PDF attachments, and the
    filing's own text is already the summary. Articles that cannot be read
    are returned as they were, as is the whole list when Firecrawl is not
    configured.
    """
    if limit <= 0 or not firecrawl_client.is_configured():
        return articles

    targets = [
        i for i, article in enumerate(articles)
        if len(article.summary) < _ENOUGH_SUMMARY_CHARS and _is_readable(article)
    ][:limit]
    if not targets:
        return articles

    def fetch(i: int):
        url = articles[i].url
        return firecrawl_client.scrape(url, wait_ms=_REDIRECT_WAIT_MS if _on_host(url, _REDIRECT_HOSTS) else 0)

    with ThreadPoolExecutor(max_workers=min(len(targets), _MAX_PARALLEL_SCRAPES)) as executor:
        pages = list(executor.map(fetch, targets))

    enriched = list(articles)
    read = 0
    for i, page in zip(targets, pages):
        article = articles[i]
        if page is None or not _is_about(page["text"], article.title):
            continue
        resolved = page["url"] if _on_host(article.url, _REDIRECT_HOSTS) else ""
        enriched[i] = article.model_copy(update={
            "summary": trim_content(page["text"], max_chars=MAX_BODY_CHARS),
            "published_at": article.published_at or page["published_at"],
            "url": resolved or article.url,
        })
        read += 1

    print(f"[Firecrawl] Read {read} of {len(targets)} articles in full.", file=sys.stderr)
    return enriched
