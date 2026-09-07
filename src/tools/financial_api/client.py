"""
Financial Agent API — low-level client adapter.

STATUS: STUB / ADAPTER ONLY
=============================================================================
The Financial Agent API provider has not been confirmed and no API
documentation or endpoint specification exists in this repository.

DO NOT fabricate:
  - endpoint paths
  - request parameters
  - authentication header names
  - response field names
  - any financial values

This adapter provides:
  1. A clean provider abstraction.
  2. Settings-driven API key + base URL configuration.
  3. A stable internal request interface.
  4. Graceful empty/None results when the API is unconfigured or unavailable.
  5. A clearly marked placeholder for the real provider implementation.

HOW TO IMPLEMENT THE REAL PROVIDER:
  1. Obtain API documentation from the provider.
  2. Replace the body of `_call_endpoint()` with the real HTTP logic.
  3. Update `_parse_profile()`, `_parse_metrics()`, `_parse_events()`.
  4. All agent/tool interfaces remain unchanged.
=============================================================================
"""

import sys
from typing import Any, Optional

from config.settings import settings


def _is_configured() -> bool:
    """Return True only if the API key and base URL are both set."""
    return bool(settings.FINANCIAL_AGENT_API_KEY and settings.FINANCIAL_AGENT_BASE_URL)


def _call_endpoint(path: str, params: dict[str, Any]) -> Optional[dict[str, Any]]:
    """
    Make a request to the Financial Agent API.

    STUB: Returns None because the provider contract is unknown.

    Real implementation should:
      - Construct the full URL from FINANCIAL_AGENT_BASE_URL + path
      - Add the auth header (e.g. Authorization: Bearer <key> or X-API-Key: <key>)
        — the exact header name must come from the provider docs
      - Issue a GET/POST request with `params`
      - Parse and return the JSON response body
      - Handle HTTP errors, timeouts, rate limits

    Never log the API key.
    """
    if not _is_configured():
        # Silently skip — key/URL not set, provider not configured.
        return None

    # ── STUB: insert real HTTP call here ──────────────────────────────────────
    # Example skeleton (do NOT use until endpoint is confirmed):
    #
    #   import requests
    #   url = f"{settings.FINANCIAL_AGENT_BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    #   headers = {"Authorization": f"Bearer {settings.FINANCIAL_AGENT_API_KEY}"}
    #   response = requests.get(url, params=params, headers=headers, timeout=10)
    #   response.raise_for_status()
    #   return response.json()
    #
    print(
        f"[FinancialAPI] STUB — provider not configured. "
        f"Set FINANCIAL_AGENT_API_KEY and FINANCIAL_AGENT_BASE_URL.",
        file=sys.stderr,
    )
    return None


# ── Public interface ──────────────────────────────────────────────────────────
# These functions define the stable contract used by the tools layer.
# They must not change even when the real provider is implemented.

def fetch_company_profile(ticker: str) -> Optional[dict[str, Any]]:
    """
    Fetch basic company profile data.

    Expected response shape (once real provider is known):
      {
        "company_name": str,
        "sector": str,
        "industry": str,
        "description": str,
        ...
      }

    Returns None if unavailable.
    """
    return _call_endpoint("company/profile", {"ticker": ticker})


def fetch_financial_metrics(ticker: str) -> Optional[dict[str, Any]]:
    """
    Fetch financial metrics.

    Expected response shape (once real provider is known):
      {
        "market_cap": float,      # in provider units
        "pe_ratio": float,
        "eps": float,
        "revenue": float,
        "debt_to_equity": float,
        ...
      }

    Returns None if unavailable.
    Never returns fabricated zeros for missing values.
    """
    return _call_endpoint("company/metrics", {"ticker": ticker})


def fetch_company_events(ticker: str) -> Optional[dict[str, Any]]:
    """
    Fetch upcoming corporate events.

    Expected response shape (once real provider is known):
      {
        "events": [
          {"type": "earnings", "date": "2025-10-15", "description": "Q2 results"},
          ...
        ]
      }

    Returns None if unavailable.
    """
    return _call_endpoint("company/events", {"ticker": ticker})
