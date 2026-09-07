"""
Shared normalization utilities for news articles and retrieved content.

These are deterministic Python functions — no LLM involvement.
"""

import re
import urllib.parse
from datetime import datetime, timezone


# ── URL normalization ─────────────────────────────────────────────────────────

def normalize_url(url: str) -> str:
    """
    Return a canonical form of a URL for deduplication.
    Strips UTM parameters, fragments, and trailing slashes.
    Returns "" if the URL is empty or unparseable.
    """
    if not url:
        return ""
    try:
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        # Remove tracking parameters
        clean_query = {k: v for k, v in query.items() if not k.startswith("utm_")}
        new_query = urllib.parse.urlencode(clean_query, doseq=True)
        clean = urllib.parse.urlunparse((
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/"),
            parsed.params,
            new_query,
            "",  # drop fragment
        ))
        return clean
    except Exception:
        return url


# ── Title normalization ───────────────────────────────────────────────────────

def normalize_title(title: str) -> str:
    """
    Return a normalised title fingerprint for secondary deduplication.
    Lowercase, strips all non-alphanumeric characters.
    """
    if not title:
        return ""
    return re.sub(r"[^a-z0-9]", "", title.lower())


# ── Content trimming ──────────────────────────────────────────────────────────

MAX_SUMMARY_CHARS = 400
MAX_BODY_CHARS = 600


def trim_content(text: str, max_chars: int = MAX_SUMMARY_CHARS) -> str:
    """
    Trim content to a maximum character length, appending '…' if truncated.
    Prevents token bloat when passing content to the LLM.
    """
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


# ── Source normalization ──────────────────────────────────────────────────────

def normalize_source(source: str) -> str:
    """Return a canonical source name."""
    if not source:
        return "Unknown"
    return source.strip()


# ── Timestamp normalization ───────────────────────────────────────────────────

def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()


# ── Deduplication ─────────────────────────────────────────────────────────────

def deduplicate_article_dicts(articles: list[dict]) -> list[dict]:
    """
    Deduplicate a list of raw article dicts by normalised URL.
    Falls back to normalised title + source fingerprint when URL is missing.
    Preserves order (first occurrence wins).
    """
    seen_urls: set[str] = set()
    seen_fingerprints: set[str] = set()
    result: list[dict] = []

    for article in articles:
        url = normalize_url(article.get("url", ""))
        if url:
            if url in seen_urls:
                continue
            seen_urls.add(url)
        else:
            title_fp = normalize_title(article.get("title", ""))
            source_fp = normalize_source(article.get("source", "")).lower()
            fp = f"{title_fp}|{source_fp}"
            if fp in seen_fingerprints:
                continue
            seen_fingerprints.add(fp)

        result.append(article)

    return result
