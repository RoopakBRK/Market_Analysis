"""Which companies the price-history store covers."""

import csv
import io
from dataclasses import dataclass

import requests

from src.tools.common.scraper_utils import make_headers
from src.utils.constants import WATCHLIST

# NSE's published list of current NIFTY 50 constituents.
_CONSTITUENTS_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"


@dataclass(frozen=True)
class Company:
    symbol: str      # NSE trading symbol, e.g. "ADANIPORTS"
    name: str
    industry: str = ""


def get_nifty50() -> list[Company]:
    """
    Fetch the current NIFTY 50 constituents from NSE.

    The list is fetched rather than hardcoded because the index is
    rebalanced twice a year. Raises requests.RequestException on failure.
    """
    response = requests.get(_CONSTITUENTS_URL, headers=make_headers(), timeout=15)
    response.raise_for_status()

    companies = [
        Company(
            symbol=row["Symbol"].strip(),
            name=row["Company Name"].strip(),
            industry=(row.get("Industry") or "").strip(),
        )
        for row in csv.DictReader(io.StringIO(response.text))
        if row.get("Symbol") and row.get("Company Name")
    ]
    if not companies:
        raise requests.RequestException("NSE constituent list came back empty")
    return companies


def resolve_universe(tickers: list[str] | None = None) -> list[Company]:
    """
    The companies to ingest: the NIFTY 50 plus any watchlist company outside
    it, or just `tickers` when given.
    """
    if tickers:
        try:
            known = {c.symbol: c for c in get_nifty50()}
        except requests.RequestException:
            known = {}
        return [
            known.get(t) or Company(symbol=t, name=WATCHLIST.get(t, t))
            for t in (t.strip().upper() for t in tickers)
            if t
        ]

    companies = get_nifty50()
    covered = {c.symbol for c in companies}
    companies += [Company(symbol=t, name=name) for t, name in WATCHLIST.items() if t not in covered]
    return companies
