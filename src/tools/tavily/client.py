"""
Low-level Tavily client.

Responsibilities:
- API key management (read from settings, never logged)
- Request construction and timeout
- Response normalization into internal article dicts
- Graceful failure: returns [] on any error
"""

import sys
from datetime import datetime, timezone
from typing import Any

from config.settings import settings
from src.tools.common.normalization import normalize_url, trim_content, utc_now_iso
from src.tools.common.source_classifier import classify_source

# Maximum characters to preserve from Tavily content snippets.
_MAX_CONTENT_CHARS = 500
_DEFAULT_MAX_RESULTS = 5
_DEFAULT_DAYS_BACK = 3
_REQUEST_TIMEOUT = 15  # seconds


def _build_client():
    """
    Return a Tavily client instance, or None if the key is missing.
    We import lazily to avoid import errors when Tavily is not installed.
    """
    api_key = settings.TAVILY_API_KEY
    if not api_key:
        return None
    try:
        from tavily import TavilyClient  # type: ignore
        return TavilyClient(api_key=api_key)
    except ImportError:
        print("[Tavily] tavily-python not installed. Install with: pip install tavily-python", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"[Tavily] Client initialisation failed: {type(exc).__name__}", file=sys.stderr)
        return None


def _normalize_result(item: dict[str, Any], query: str) -> dict[str, Any]:
    """
    Normalise a single Tavily result dict into the project's internal
    article representation.
    """
    raw_url = item.get("url", "")
    raw_source = item.get("source", "") or ""

    # Try to infer source domain if 'source' not given.
    if not raw_source and raw_url:
        try:
            from urllib.parse import urlparse
            raw_source = urlparse(raw_url).netloc.replace("www.", "")
        except Exception:
            pass

    content = trim_content(
        item.get("content", "") or item.get("snippet", ""),
        max_chars=_MAX_CONTENT_CHARS,
    )

    return {
        "title": (item.get("title") or "").strip(),
        "url": normalize_url(raw_url),
        "source": raw_source,
        "published_at": item.get("published_date", "") or "",
        "summary": content,
        "query": query,
        "retrieved_at": utc_now_iso(),
        # provenance: retrieved via Tavily
        "source_type": "tavily",
        # classify underlying publisher for display/LLM labelling
        "underlying_source_type": classify_source(raw_source),
        "relevance_score": int(item.get("score", 0) * 10) if item.get("score") else 0,
        "is_official": classify_source(raw_source) == "official",
    }


def search(
    query: str,
    max_results: int = _DEFAULT_MAX_RESULTS,
    days_back: int = _DEFAULT_DAYS_BACK,
    search_depth: str = "basic",
) -> list[dict[str, Any]]:
    """
    Execute a Tavily search and return a list of normalised article dicts.

    Returns [] gracefully on:
    - missing API key
    - missing tavily-python library
    - network timeout
    - API error
    - malformed response
    """
    client = _build_client()
    if client is None:
        print("[Tavily] Skipped — no API key or client unavailable.", file=sys.stderr)
        return []

    try:
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            days=days_back,
        )

        raw_results = response.get("results", []) if isinstance(response, dict) else []

        if not raw_results:
            print(f"[Tavily] Query '{query[:60]}' returned 0 results.", file=sys.stderr)
            return []

        normalized = [_normalize_result(item, query) for item in raw_results if isinstance(item, dict)]
        print(f"[Tavily] Query '{query[:60]}' → {len(normalized)} results", file=sys.stderr)
        return normalized

    except Exception as exc:
        print(f"[Tavily] Search failed for '{query[:60]}': {type(exc).__name__}: {exc}", file=sys.stderr)
        return []
