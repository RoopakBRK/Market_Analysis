"""Daily price history from Yahoo Finance, via yfinance."""

from datetime import date, timedelta

import pandas as pd

from src.tools.common.yahoo_finance import nse_symbol

NIFTY50_SYMBOL = "^NSEI"

_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def load_price_history(symbol: str, years: int = 20) -> pd.DataFrame:
    """
    Load up to `years` years of daily bars for an NSE symbol (or a Yahoo
    index symbol such as "^NSEI"), oldest first, indexed by session date.

    Prices are adjusted for splits and dividends, so that returns across
    two decades are comparable; they therefore differ from the prices
    quoted at the time. Returns an empty frame if Yahoo has no data.
    """
    import yfinance as yf

    start = date.today() - timedelta(days=round(365.25 * years))
    frame = yf.Ticker(nse_symbol(symbol)).history(
        start=start.isoformat(), interval="1d", auto_adjust=True
    )
    if frame.empty:
        return pd.DataFrame(columns=_COLUMNS)

    frame = frame[_COLUMNS].dropna()
    # Session timestamps come back as midnight Asia/Kolkata; keep the date.
    frame.index = pd.DatetimeIndex(frame.index.tz_localize(None).normalize(), name="Date")
    return frame[~frame.index.duplicated(keep="last")].sort_index()
