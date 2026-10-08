from langchain_core.tools import tool
from src.tools.common.yahoo_finance import get_price_history, nse_symbol

RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
VWAP_SESSIONS = 20


def compute_rsi(closes: list[float], period: int = RSI_PERIOD) -> float | None:
    """
    Wilder's RSI over the full series, reported for the latest close.
    Returns None when there are too few closes to seed the average.
    """
    if len(closes) <= period:
        return None

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    avg_gain = sum(max(d, 0.0) for d in deltas[:period]) / period
    avg_loss = sum(max(-d, 0.0) for d in deltas[:period]) / period

    for d in deltas[period:]:
        avg_gain = (avg_gain * (period - 1) + max(d, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-d, 0.0)) / period

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    return 100 - 100 / (1 + avg_gain / avg_loss)


def _ema(values: list[float], period: int) -> list[float]:
    """EMA seeded with an SMA; element i lines up with values[period - 1 + i]."""
    k = 2 / (period + 1)
    ema = [sum(values[:period]) / period]
    for v in values[period:]:
        ema.append(v * k + ema[-1] * (1 - k))
    return ema


def compute_macd(
    closes: list[float],
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> tuple[float, float] | None:
    """
    Latest MACD line and signal line. Returns None when the series is too
    short for the slow EMA plus the signal EMA.
    """
    if len(closes) < slow + signal:
        return None

    # Drop the fast EMA's head so both series start at the same close.
    fast_ema = _ema(closes, fast)[slow - fast:]
    slow_ema = _ema(closes, slow)
    macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]
    return macd_line[-1], _ema(macd_line, signal)[-1]


def compute_vwap(bars: list[dict]) -> float | None:
    """Volume-weighted average of the typical price across the given bars."""
    volume = sum(b["volume"] for b in bars)
    if volume <= 0:
        return None
    return sum((b["high"] + b["low"] + b["close"]) / 3 * b["volume"] for b in bars) / volume


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 2)


@tool
def get_technical_indicators(ticker: str) -> dict:
    """
    Compute RSI(14), MACD(12,26,9) and the 20-session VWAP for a ticker from
    daily Yahoo Finance bars. An indicator is None when it cannot be computed.
    """
    bars = get_price_history(nse_symbol(ticker), range_="6mo")
    if not bars:
        # No data is reported as an error, never as zeros: an RSI of 0 would
        # read as "extremely oversold" downstream.
        return {"ticker": ticker, "error": "No price history available"}

    closes = [b["close"] for b in bars]
    macd = compute_macd(closes)

    return {
        "ticker": ticker,
        "rsi": _rounded(compute_rsi(closes)),
        "macd": _rounded(macd[0]) if macd else None,
        "macd_signal": _rounded(macd[1]) if macd else None,
        "vwap": _rounded(compute_vwap(bars[-VWAP_SESSIONS:])),
        "source": "Yahoo Finance",
    }
