"""
Source classification: assigns a source_type tier to news articles.

Tier definitions:
  official  → NSE, RBI, SEBI, government, company investor relations
  tier1     → Reuters, Bloomberg, AP, major global wire services
  tier2     → Economic Times, Moneycontrol, Mint, Business Standard, NDTV Profit
  tavily    → Retrieved via Tavily (underlying publisher determines real tier)

These are deterministic rules — no LLM.
"""

from typing import Literal
from urllib.parse import urlparse

SourceType = Literal["official", "tier1", "tier2", "tavily"]

# Canonical source name → tier mapping.
# Keys are lowercased for matching.
_OFFICIAL_SOURCES = {
    "nse",
    "national stock exchange",
    "rbi",
    "reserve bank of india",
    "sebi",
    "ministry of finance",
    "investor relations",
    "company announcement",
}

_TIER1_SOURCES = {
    "reuters",
    "bloomberg",
    "associated press",
    "ap news",
    "apnews",
    "financial times",
    "wsj",
    "wall street journal",
}

_TIER2_SOURCES = {
    "economic times",
    "economictimes",
    "et markets",
    "moneycontrol",
    "mint",
    "livemint",
    "business standard",
    "ndtv profit",
    "cnbc tv18",
    "the hindu businessline",
    "businessline",
    "financial express",
}


# Publisher domain → the name the dedicated tools give that publisher, so an
# article found through a search API is labelled, tiered and capped per
# source the same way as one found by the publisher's own tool.
_PUBLISHER_DOMAINS = {
    "reuters.com": "Reuters",
    "bloomberg.com": "Bloomberg",
    "apnews.com": "Associated Press",
    "ft.com": "Financial Times",
    "wsj.com": "Wall Street Journal",
    "economictimes.indiatimes.com": "Economic Times",
    "economictimes.com": "Economic Times",
    "moneycontrol.com": "Moneycontrol",
    "livemint.com": "Mint",
    "business-standard.com": "Business Standard",
    "ndtvprofit.com": "NDTV Profit",
    "cnbctv18.com": "CNBC TV18",
    "thehindubusinessline.com": "The Hindu BusinessLine",
    "financialexpress.com": "Financial Express",
}


def publisher_from_url(url: str) -> str:
    """
    Return the publisher's name for an article URL: the canonical name for a
    known publisher, otherwise the bare domain. Returns "" if the URL has no
    host.
    """
    host = urlparse(url or "").netloc.lower().removeprefix("www.")
    for domain, name in _PUBLISHER_DOMAINS.items():
        if host == domain or host.endswith("." + domain):
            return name
    return host


def classify_source(source: str) -> SourceType:
    """
    Return the reliability tier for a given source name.

    Falls back to "tier2" for unknown sources — conservative default.
    """
    s = (source or "").lower().strip()

    if any(off in s for off in _OFFICIAL_SOURCES):
        return "official"
    if any(t1 in s for t1 in _TIER1_SOURCES):
        return "tier1"
    if any(t2 in s for t2 in _TIER2_SOURCES):
        return "tier2"

    # Unknown source — conservative fallback
    return "tier2"


def is_official(source_type: SourceType) -> bool:
    return source_type == "official"


def is_verified(source_type: SourceType) -> bool:
    """True for official, tier1, tier2 — false for tavily (until underlying is classified)."""
    return source_type in ("official", "tier1", "tier2")
