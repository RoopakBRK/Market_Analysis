import requests
from typing import Optional, Dict
from src.tools.common.scraper_utils import make_headers

_session = requests.Session()
_session.headers.update(make_headers())


def nse_symbol(ticker: str) -> str:
    """
    Map a watchlist ticker to its Yahoo Finance symbol.

    Yahoo lists NSE stocks as <SYMBOL>.NS — a bare "TMPV" is a 404. Symbols
    that already carry an exchange suffix, or are indices / FX / futures
    (^NSEI, INR=X, BZ=F), pass through unchanged.
    """
    symbol = ticker.strip().upper()
    if "." in symbol or "=" in symbol or symbol.startswith("^"):
        return symbol
    return f"{symbol}.NS"


def get_latest_quote(symbol: str) -> Optional[Dict[str, float]]:
    """
    Fetch the latest quote data for a symbol from Yahoo Finance.
    Returns a dict with 'price', 'previous_close', 'change', and 'change_percent'.
    Returns None gracefully on network failure, empty response, or malformed JSON.
    """
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=5d"

    try:
        response = _session.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        result = data.get("chart", {}).get("result", [])
        if not result:
            return None

        result_data = result[0]
        indicators = result_data.get("indicators", {}).get("quote", [])
        if not indicators:
            return None

        close_prices = indicators[0].get("close", [])

        # Filter out None values that Yahoo sometimes injects
        valid_closes = [p for p in close_prices if p is not None]
        if not valid_closes:
            return None

        current_price = valid_closes[-1]

        # Get previous close from meta if it exists, otherwise use the previous array value
        meta = result_data.get("meta", {})
        previous_close = meta.get("previousClose")
        if previous_close is None and len(valid_closes) > 1:
            previous_close = valid_closes[-2]
        elif previous_close is None:
            # Yahoo keeps a single daily bar for some indices (e.g. Nifty
            # Auto); chartPreviousClose is then the prior session's close.
            previous_close = meta.get("chartPreviousClose") or current_price

        change = current_price - previous_close
        change_percent = (change / previous_close) * 100 if previous_close > 0 else 0.0

        return {
            "price": float(current_price),
            "previous_close": float(previous_close),
            "change": float(change),
            "change_percent": float(change_percent)
        }
    except Exception:
        return None


def get_price_history(symbol: str, range_: str = "6mo") -> list[dict]:
    """
    Fetch daily bars for a symbol, oldest first, as dicts with 'high', 'low',
    'close' and 'volume'. Bars with any missing value are dropped whole so the
    series stay aligned. Returns [] gracefully on any failure.
    """
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range={range_}"

    try:
        response = _session.get(url, timeout=10)
        response.raise_for_status()
        result = response.json().get("chart", {}).get("result", [])
        if not result:
            return []

        quote = result[0].get("indicators", {}).get("quote", [{}])[0]
        rows = zip(
            quote.get("high", []),
            quote.get("low", []),
            quote.get("close", []),
            quote.get("volume", []),
        )
        return [
            {"high": float(h), "low": float(l), "close": float(c), "volume": float(v)}
            for h, l, c, v in rows
            if None not in (h, l, c, v)
        ]
    except Exception:
        return []


def get_sector_industry(symbol: str) -> tuple[str, str]:
    """
    Look up a symbol's sector and industry (Yahoo's classification), e.g.
    ("Consumer Cyclical", "Auto Manufacturers"). Returns ("", "") if unknown.

    Uses the search endpoint: quoteSummary/assetProfile now requires a
    session crumb and answers 401 without one.
    """
    url = "https://query2.finance.yahoo.com/v1/finance/search"

    try:
        response = _session.get(
            url, params={"q": symbol, "quotesCount": 5, "newsCount": 0}, timeout=10
        )
        response.raise_for_status()
        for quote in response.json().get("quotes", []):
            if quote.get("symbol") == symbol:
                return quote.get("sector") or "", quote.get("industry") or ""
    except Exception:
        pass
    return "", ""
