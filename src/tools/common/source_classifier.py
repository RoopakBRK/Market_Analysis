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
    "ap",
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
