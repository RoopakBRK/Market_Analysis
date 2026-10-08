"""
Turns one stock's daily price history into text chunks for the vector store.

A row of prices means nothing to an embedding model, and 5,000 rows per stock
would drown retrieval in near-identical points. So the history is split along
the calendar instead, into three kinds of self-contained summary:

    year   one chunk per calendar year      (the parent of its months)
    month  one chunk per calendar month     (the parent of its events)
    event  one chunk per session that moved at least EVENT_THRESHOLD_PCT

Each chunk is a short paragraph, generated from a template with the figures
computed here in code, that names the company, ticker and period — so it can
be matched semantically ("worst months") and lexically ("March 2020"). Chunks
do not overlap: the year/month/event hierarchy, linked by `parent_id`, gives
the surrounding context that overlapping windows would otherwise provide.

The raw daily bars are kept too, in each month chunk's `sessions` payload, so
the full 20-year series is in the store for exact lookups.
"""

import math
import uuid
from dataclasses import dataclass

import pandas as pd

from src.rag.universe import Company

# A single-session move of at least this size (percent) becomes an event chunk.
EVENT_THRESHOLD_PCT = 4.0

_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


@dataclass
class Chunk:
    id: str        # deterministic, so re-ingesting a period overwrites it
    text: str      # the paragraph that gets embedded
    payload: dict  # stored alongside the vectors


def chunk_id(ticker: str, granularity: str, period_start: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"market-analysis/{ticker}/{granularity}/{period_start}"))


# ── Formatting helpers ───────────────────────────────────────────────────────

def _day(ts) -> str:
    return ts.strftime("%d %b %Y")


def _iso(ts) -> str:
    return ts.strftime("%Y-%m-%d")


def _moved(pct: float) -> str:
    """ "rose 4.2%" / "fell 4.2%" / "was unchanged" """
    if round(pct, 1) == 0:
        return "was unchanged"
    return f"{'rose' if pct > 0 else 'fell'} {abs(pct):.1f}%"


def _num(value, digits: int = 2):
    """Round for the payload; NaN / missing become None (NaN is not valid JSON)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(float(value), digits)


def _versus_index(stock_pct: float, index_pct: float | None, period: str) -> str:
    if index_pct is None:
        return ""
    gap = stock_pct - index_pct
    if round(gap, 1) == 0:
        verdict = "in line with the index"
    else:
        verdict = f"{'ahead of' if gap > 0 else 'behind'} the index by {abs(gap):.1f} percentage points"
    return f" The NIFTY 50 moved {index_pct:+.1f}% in the same {period}, leaving the stock {verdict}."


# ── Period returns ───────────────────────────────────────────────────────────

def period_returns(prices: pd.DataFrame) -> tuple[dict, dict]:
    """
    Percentage return of each calendar month and year, measured from the
    previous period's last close (the first period, from its first open).
    Returns ({(year, month): pct}, {year: pct}).
    """
    monthly, yearly = {}, {}
    for returns, keys in ((monthly, [prices.index.year, prices.index.month]), (yearly, [prices.index.year])):
        base = None
        for key, group in prices.groupby(keys):
            start = base if base is not None else group["Open"].iloc[0]
            base = group["Close"].iloc[-1]
            key = key[0] if isinstance(key, tuple) and len(key) == 1 else key
            returns[key] = (base / start - 1) * 100
    return monthly, yearly


# ── Chunk builders ───────────────────────────────────────────────────────────

def build_chunks(
    company: Company,
    prices: pd.DataFrame,
    benchmark: pd.DataFrame | None = None,
    as_of: pd.Timestamp | None = None,
) -> list[Chunk]:
    """
    Build every year, month and event chunk for one company.

    prices     daily bars (Open/High/Low/Close/Volume), indexed by date
    benchmark  NIFTY 50 daily bars, for the comparison sentences (optional)
    as_of      "today"; the period containing it is labelled as in progress
    """
    if prices.empty:
        return []

    prices = prices.sort_index()
    as_of = as_of or pd.Timestamp.today().normalize()
    close = prices["Close"]

    monthly, yearly = period_returns(prices)
    index_monthly, index_yearly = period_returns(benchmark.sort_index()) if benchmark is not None and not benchmark.empty else ({}, {})
    index_daily = benchmark["Close"].sort_index().pct_change() * 100 if benchmark is not None and not benchmark.empty else None

    series = {
        "daily": close.pct_change() * 100,
        "high_52w": close.rolling(252, min_periods=60).max(),
        "low_52w": close.rolling(252, min_periods=60).min(),
        # Average of the 20 sessions before each day, excluding the day itself.
        "volume_avg": prices["Volume"].rolling(20, min_periods=5).mean().shift(1),
        "fwd_5": (close.shift(-5) / close - 1) * 100,
        "fwd_20": (close.shift(-20) / close - 1) * 100,
    }

    base = {
        "ticker": company.symbol,
        "company": company.name,
        "industry": company.industry,
        "exchange": "NSE",
        "source": "Yahoo Finance (yfinance)",
        "price_basis": "adjusted for splits and dividends",
    }
    label = f"{company.name} ({company.symbol}) on NSE"

    chunks = []
    for year, group in prices.groupby(prices.index.year):
        chunks.append(_year_chunk(label, base, year, group, yearly, monthly, index_yearly, as_of))
    for (year, month), group in prices.groupby([prices.index.year, prices.index.month]):
        chunks.append(_month_chunk(label, base, year, month, group, monthly, index_monthly, series, as_of))

    daily = series["daily"]
    for day in daily.index[daily.abs() >= EVENT_THRESHOLD_PCT]:
        chunks.append(_event_chunk(label, base, day, prices, series, index_daily))

    return chunks


def _year_chunk(label, base, year, group, yearly, monthly, index_yearly, as_of) -> Chunk:
    ret = yearly[year]
    close = group["Close"]
    in_progress = year == as_of.year
    period = f"calendar year {year}" + (f" (year to date, through {_day(group.index[-1])})" if in_progress else "")

    months = {m: pct for (y, m), pct in monthly.items() if y == year}
    best, worst = max(months, key=months.get), min(months, key=months.get)
    drawdown = (close / close.cummax() - 1).min() * 100

    text = (
        f"{label}, {period}. The stock {_moved(ret)} over the year, closing at Rs {close.iloc[-1]:,.2f}. "
        f"It traded between a low of Rs {group['Low'].min():,.2f} ({_day(group['Low'].idxmin())}) "
        f"and a high of Rs {group['High'].max():,.2f} ({_day(group['High'].idxmax())}). "
        f"The largest peak-to-trough decline within the year was {abs(drawdown):.1f}%. "
        f"It rose in {sum(pct > 0 for pct in months.values())} of {len(months)} months; "
        f"the best month was {_MONTHS[best - 1]} ({months[best]:+.1f}%) "
        f"and the worst was {_MONTHS[worst - 1]} ({months[worst]:+.1f}%). "
        f"Average volume was {group['Volume'].mean():,.0f} shares a day."
        + _versus_index(ret, index_yearly.get(year), "year")
    )

    start = _iso(group.index[0])
    return Chunk(
        id=chunk_id(base["ticker"], "year", str(year)),
        text=text,
        payload={
            **base,
            "granularity": "year",
            "year": int(year),
            "month": None,
            "period_start": start,
            "period_end": _iso(group.index[-1]),
            "in_progress": in_progress,
            "return_pct": _num(ret),
            "close": _num(close.iloc[-1]),
            "high": _num(group["High"].max()),
            "low": _num(group["Low"].min()),
            "max_drawdown_pct": _num(drawdown),
            "index_return_pct": _num(index_yearly.get(year)),
            "parent_id": None,
            "text": text,
        },
    )


def _month_chunk(label, base, year, month, group, monthly, index_monthly, series, as_of) -> Chunk:
    ret = monthly[(year, month)]
    last = group.index[-1]
    in_progress = (year, month) == (as_of.year, as_of.month)
    period = f"{_MONTHS[month - 1]} {year}" + (f" (month to date, through {_day(last)})" if in_progress else "")

    text = (
        f"{label}, {period}. The stock {_moved(ret)} over the month, "
        f"from an opening price of Rs {group['Open'].iloc[0]:,.2f} to a close of Rs {group['Close'].iloc[-1]:,.2f}, "
        f"trading between a low of Rs {group['Low'].min():,.2f} ({_day(group['Low'].idxmin())}) "
        f"and a high of Rs {group['High'].max():,.2f} ({_day(group['High'].idxmax())}) "
        f"across {len(group)} sessions."
    )

    daily = series["daily"].loc[group.index].dropna()
    if len(daily) > 1:
        text += (
            f" Its best session was {_day(daily.idxmax())} ({daily.max():+.1f}%) "
            f"and its worst was {_day(daily.idxmin())} ({daily.min():+.1f}%); "
            f"daily moves averaged {daily.abs().mean():.1f}% in size."
        )
    text += f" Average volume was {group['Volume'].mean():,.0f} shares a day."

    high_52w, low_52w = series["high_52w"].loc[last], series["low_52w"].loc[last]
    month_close = group["Close"].iloc[-1]
    if not math.isnan(high_52w) and not math.isnan(low_52w) and low_52w > 0:
        text += (
            f" The month ended {abs(month_close / high_52w - 1) * 100:.1f}% below the 52-week high "
            f"and {(month_close / low_52w - 1) * 100:.1f}% above the 52-week low."
        )
    text += _versus_index(ret, index_monthly.get((year, month)), "month")

    return Chunk(
        id=chunk_id(base["ticker"], "month", f"{year}-{month:02d}"),
        text=text,
        payload={
            **base,
            "granularity": "month",
            "year": int(year),
            "month": int(month),
            "period_start": _iso(group.index[0]),
            "period_end": _iso(last),
            "in_progress": in_progress,
            "return_pct": _num(ret),
            "open": _num(group["Open"].iloc[0]),
            "close": _num(month_close),
            "high": _num(group["High"].max()),
            "low": _num(group["Low"].min()),
            "index_return_pct": _num(index_monthly.get((year, month))),
            "parent_id": chunk_id(base["ticker"], "year", str(year)),
            # The raw daily bars for this month.
            "sessions": [
                {
                    "date": _iso(day),
                    "open": _num(row.Open),
                    "high": _num(row.High),
                    "low": _num(row.Low),
                    "close": _num(row.Close),
                    "volume": int(row.Volume),
                }
                for day, row in group.iterrows()
            ],
            "text": text,
        },
    )


def _event_chunk(label, base, day, prices, series, index_daily) -> Chunk:
    ret = series["daily"].loc[day]
    close = prices["Close"]
    position = close.index.get_loc(day)
    day_close, prev_close = close.iloc[position], close.iloc[position - 1]

    text = (
        f"{label}, {_day(day)} ({day.strftime('%A')}). The stock {_moved(ret)} in a single session, "
        f"closing at Rs {day_close:,.2f} against Rs {prev_close:,.2f} the session before."
    )

    volume_avg = series["volume_avg"].loc[day]
    volume_ratio = prices["Volume"].loc[day] / volume_avg if volume_avg and not math.isnan(volume_avg) else None
    if volume_ratio is not None:
        text += f" Volume was {volume_ratio:.1f} times its 20-session average."

    fwd_5, fwd_20 = series["fwd_5"].loc[day], series["fwd_20"].loc[day]
    if not math.isnan(fwd_5):
        text += f" Over the following 5 sessions it moved {fwd_5:+.1f}%"
        text += f", and over the following 20 sessions {fwd_20:+.1f}%." if not math.isnan(fwd_20) else "."

    index_move = index_daily.get(day) if index_daily is not None else None
    if index_move is not None and not math.isnan(index_move):
        text += f" The NIFTY 50 moved {index_move:+.1f}% on the same day."
    else:
        index_move = None

    return Chunk(
        id=chunk_id(base["ticker"], "event", _iso(day)),
        text=text,
        payload={
            **base,
            "granularity": "event",
            "year": int(day.year),
            "month": int(day.month),
            "period_start": _iso(day),
            "period_end": _iso(day),
            "in_progress": False,
            "return_pct": _num(ret),
            "close": _num(day_close),
            "prev_close": _num(prev_close),
            "volume_ratio": _num(volume_ratio),
            "fwd_5d_pct": _num(fwd_5),
            "fwd_20d_pct": _num(fwd_20),
            "index_return_pct": _num(index_move),
            "parent_id": chunk_id(base["ticker"], "month", f"{day.year}-{day.month:02d}"),
            "text": text,
        },
    )
