"""
Historical statistics computed from the price-history store.

Every figure here is arithmetic on stored prices and returns, done in code, so
a number in the report can be traced to the data and cannot be misquoted by a
model. The functions take plain lists of Qdrant payloads (see src/rag/chunking)
and return short text lines for the report's "Historical Context" section.

Four kinds of line are produced:

    Comparable sessions   how the stock behaved after past sessions of today's size
    Precedent             the closest individual past sessions, with what followed
    Seasonality           how the stock did in the same calendar month in past years
    Year context          this year so far, against the stock's own history and the index
"""

import statistics
from dataclasses import dataclass
from datetime import date

import pandas as pd

# Fewer comparable sessions than this is an anecdote, not a statistic.
MIN_COMPARABLE = 15
# Seasonality and year statistics need at least this many past years.
MIN_YEARS_SEASONALITY = 5
MIN_YEARS_CONTEXT = 3
# "Comparable" means a move within this fraction of today's move (3.5% -> 2.6% to 4.4%).
BAND = 0.25
# Precedents are kept this many sessions apart, so three of them do not all come
# from the same crash.
PRECEDENT_SPACING = 20
FORWARD_SHORT, FORWARD_LONG = 5, 20


# ── Series ───────────────────────────────────────────────────────────────────

def closes_from_months(month_payloads: list[dict]) -> pd.Series:
    """
    Rebuild the daily close series from the raw bars stored in the month
    chunks, oldest first, indexed by session date.
    """
    closes = {}
    for payload in month_payloads:
        for session in payload.get("sessions") or []:
            if session.get("close") is not None:
                closes[session["date"]] = float(session["close"])
    if not closes:
        return pd.Series(dtype=float)
    series = pd.Series(closes)
    series.index = pd.DatetimeIndex(series.index)
    return series.sort_index()


def _frame(closes: pd.Series) -> pd.DataFrame:
    """Daily return and the 5- and 20-session forward returns, all in percent."""
    return pd.DataFrame({
        "close": closes,
        "ret": closes.pct_change() * 100,
        "fwd5": (closes.shift(-FORWARD_SHORT) / closes - 1) * 100,
        "fwd20": (closes.shift(-FORWARD_LONG) / closes - 1) * 100,
    })


def _signed(value: float) -> str:
    return f"{value:+.1f}%"


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%d %b %Y")


# ── Comparable sessions ──────────────────────────────────────────────────────

@dataclass
class Comparable:
    low: float            # band edges, percent
    high: float
    count: int            # comparable sessions with a 5-session outcome
    years: int            # how many different calendar years they fall in
    first_year: int
    up5: float            # share (0-1) higher 5 sessions later
    median5: float        # median 5-session return, percent
    up20: float | None    # same for 20 sessions; None if too few have a 20-session outcome
    median20: float | None
    base_up5: float       # the same share across ALL sessions, for comparison
    base_up20: float | None


def comparable_sessions(closes: pd.Series, move_pct: float) -> Comparable | None:
    """
    Past sessions whose move was within BAND of today's move, in the same
    direction, and what the stock did over the next 5 and 20 sessions.
    Returns None when today's move is zero or there is no history.
    """
    frame = _frame(closes)
    if move_pct == 0 or frame.empty:
        return None

    low, high = sorted((move_pct * (1 - BAND), move_pct * (1 + BAND)))
    sample = frame[frame["ret"].between(low, high) & frame["fwd5"].notna()]

    all5, all20 = frame["fwd5"].dropna(), frame["fwd20"].dropna()
    sample20 = sample["fwd20"].dropna()
    enough20 = len(sample20) >= MIN_COMPARABLE

    return Comparable(
        low=low,
        high=high,
        count=len(sample),
        years=sample.index.year.nunique() if len(sample) else 0,
        first_year=int(closes.index[0].year),
        up5=float((sample["fwd5"] > 0).mean()) if len(sample) else 0.0,
        median5=float(sample["fwd5"].median()) if len(sample) else 0.0,
        up20=float((sample20 > 0).mean()) if enough20 else None,
        median20=float(sample20.median()) if enough20 else None,
        base_up5=float((all5 > 0).mean()),
        base_up20=float((all20 > 0).mean()) if len(all20) else None,
    )


def format_comparable(ticker: str, move_pct: float, result: Comparable) -> str:
    verb = "a fall" if move_pct < 0 else "a rise"
    # The band edges are signed; show them as sizes, smaller first.
    band = f"{min(abs(result.low), abs(result.high)):.1f}% to {max(abs(result.low), abs(result.high)):.1f}%"

    if result.count < MIN_COMPARABLE:
        return (
            f"Comparable sessions: only {result.count} past sessions had {verb} of {band} "
            f"(today: {_signed(move_pct)}), too few to draw statistics from."
        )

    text = (
        f"Comparable sessions: since {result.first_year}, {ticker} had {result.count} sessions with {verb} of {band} "
        f"(today: {_signed(move_pct)}), spread over {result.years} different years. "
        f"5 sessions later it was higher in {result.up5:.0%} of them (median {_signed(result.median5)}), "
        f"against {result.base_up5:.0%} of all sessions."
    )
    if result.up20 is not None and result.base_up20 is not None:
        text += (
            f" 20 sessions later it was higher in {result.up20:.0%} (median {_signed(result.median20)}), "
            f"against {result.base_up20:.0%} of all sessions."
        )
    return text


# ── Precedents ───────────────────────────────────────────────────────────────

@dataclass
class Precedent:
    day: pd.Timestamp
    ret: float
    close: float
    fwd5: float
    fwd20: float | None


def precedents(closes: pd.Series, move_pct: float, count: int = 3) -> list[Precedent]:
    """
    The past sessions whose move is closest to today's (same direction), most
    recent first among equals. At most one per PRECEDENT_SPACING sessions, so
    they are distinct episodes rather than consecutive days of one crash.
    """
    frame = _frame(closes)
    if move_pct == 0 or frame.empty:
        return []

    low, high = sorted((move_pct * (1 - BAND), move_pct * (1 + BAND)))
    pool = frame[frame["ret"].between(low, high) & frame["fwd5"].notna()].copy()
    pool["distance"] = (pool["ret"] - move_pct).abs()
    pool["position"] = [frame.index.get_loc(day) for day in pool.index]
    # Closest first; where two are equally close, the more recent one.
    pool = pool.sort_index(ascending=False).sort_values("distance", kind="stable")

    chosen: list[tuple[pd.Timestamp, pd.Series]] = []
    for day, row in pool.iterrows():
        if all(abs(row["position"] - other["position"]) >= PRECEDENT_SPACING for _, other in chosen):
            chosen.append((day, row))
        if len(chosen) == count:
            break

    return [
        Precedent(
            day=day,
            ret=float(row["ret"]),
            close=float(row["close"]),
            fwd5=float(row["fwd5"]),
            fwd20=None if pd.isna(row["fwd20"]) else float(row["fwd20"]),
        )
        for day, row in sorted(chosen, key=lambda item: item[0], reverse=True)
    ]


def format_precedent(ticker: str, precedent: Precedent, event: dict | None = None) -> str:
    """
    One line for a precedent. `event` is the stored chunk payload for that
    session when it was large enough to have one (4% or more); it adds the
    index move and volume.
    """
    verb = "fell" if precedent.ret < 0 else "rose"
    text = (
        f"Precedent: on {_day(precedent.day)} {ticker} {verb} {abs(precedent.ret):.1f}% to Rs {precedent.close:,.2f}; "
        f"5 sessions later it was {_signed(precedent.fwd5)}"
    )
    text += f", 20 sessions later {_signed(precedent.fwd20)}" if precedent.fwd20 is not None else ""

    extras = []
    if event:
        if event.get("index_return_pct") is not None:
            extras.append(f"the NIFTY 50 moved {_signed(event['index_return_pct'])} that day")
        if event.get("volume_ratio") is not None:
            extras.append(f"volume was {event['volume_ratio']:.1f} times its 20-session average")
    return text + (f" ({'; '.join(extras)})." if extras else ".")


# ── Seasonality ──────────────────────────────────────────────────────────────

_MONTHS = ["January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December"]


def seasonality(ticker: str, month_payloads: list[dict], month: int) -> str | None:
    """
    How the stock did in this calendar month in past years. The month in
    progress is excluded. Returns None if there are too few past years.
    """
    # The first month on record is usually part of a month (the stock listed
    # mid-month), so it is left out along with the month in progress.
    first_start = min((p["period_start"] for p in month_payloads if p.get("period_start")), default=None)
    past = [
        p for p in month_payloads
        if p.get("month") == month
        and not p.get("in_progress")
        and p.get("return_pct") is not None
        and p.get("period_start") != first_start
    ]
    if len(past) < MIN_YEARS_SEASONALITY:
        return None

    returns = [p["return_pct"] for p in past]
    best = max(past, key=lambda p: p["return_pct"])
    worst = min(past, key=lambda p: p["return_pct"])
    up = sum(r > 0 for r in returns)

    return (
        f"Seasonality: {ticker} rose in {up} of {len(past)} past {_MONTHS[month - 1]}s "
        f"(median {_signed(statistics.median(returns))}; best {_signed(best['return_pct'])} in {best['year']}, "
        f"worst {_signed(worst['return_pct'])} in {worst['year']})."
    )


# ── Year context ─────────────────────────────────────────────────────────────

def year_context(ticker: str, year_payloads: list[dict]) -> str | None:
    """
    The current year so far, against the stock's own completed years and the
    NIFTY 50. Returns None if there is no year in progress or too few past years.
    """
    current = next((p for p in year_payloads if p.get("in_progress") and p.get("return_pct") is not None), None)
    # A year counts as completed only if it ran the whole way: a stock that
    # listed in October has a part-year as its first, which is not comparable.
    done = [
        p for p in year_payloads
        if not p.get("in_progress")
        and p.get("return_pct") is not None
        and str(p.get("period_start", ""))[5:7] == "01"
    ]
    if current is None or len(done) < MIN_YEARS_CONTEXT:
        return None

    returns = [p["return_pct"] for p in done]
    best = max(done, key=lambda p: p["return_pct"])
    worst = min(done, key=lambda p: p["return_pct"])
    with_index = [p for p in done if p.get("index_return_pct") is not None]
    beat = sum(p["return_pct"] > p["index_return_pct"] for p in with_index)

    text = f"Year context: {current['year']} year to date (through {_day(current['period_end'])}): {_signed(current['return_pct'])}"
    if current.get("index_return_pct") is not None:
        text += f", against the NIFTY 50's {_signed(current['index_return_pct'])}"
    text += (
        f". Over its {len(done)} completed calendar years the stock's return had a median of "
        f"{_signed(statistics.median(returns))}, from {_signed(worst['return_pct'])} ({worst['year']}) "
        f"to {_signed(best['return_pct'])} ({best['year']})"
    )
    if with_index:
        text += f"; it beat the NIFTY 50 in {beat} of {len(with_index)} of them"
    return text + "."


def staleness_note(closes: pd.Series, as_of: date, max_gap_days: int = 4) -> str | None:
    """A warning when the store's last session is well behind today."""
    if closes.empty:
        return None
    last = closes.index[-1].date()
    if (as_of - last).days <= max_gap_days:
        return None
    return (
        f"Note: the price-history store runs only through {_day(last)}; "
        f"re-run `python -m src.rag.ingest` to bring it up to date."
    )
