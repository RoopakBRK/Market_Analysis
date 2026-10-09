"""
Offline tests for the historical statistics and the HistoricalContextAgent that
uses them. Qdrant runs in-process; nothing touches the network.
"""

from datetime import date

import pandas as pd
import pytest
from qdrant_client import QdrantClient

from src.agents.historical_context_agent import HistoricalContextAgent
from src.models.market import MarketData
from src.rag import history_stats as hs
from src.rag.chunking import build_chunks
from src.rag.reader import HistoryReader
from src.rag.store import ensure_collection, upsert_chunks
from src.rag.universe import Company

from tests.test_rag import FakeEmbedder

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")

ACME = Company(symbol="ACME", name="Acme Industries", industry="Capital Goods")


def _series(returns: list[float], start="2024-01-01", base=100.0) -> pd.Series:
    """A close series whose session-to-session returns (percent) are `returns`."""
    closes = [base]
    for r in returns:
        closes.append(closes[-1] * (1 + r / 100))
    return pd.Series(closes, index=pd.bdate_range(start, periods=len(closes)))


def _bars(closes: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "Open": closes - 1, "High": closes + 1, "Low": closes - 2, "Close": closes, "Volume": 1000.0,
    })


# ── Comparable sessions ──────────────────────────────────────────────────────

def _falls_then(after: float, count: int) -> list[float]:
    """`count` repetitions of: a 3% fall, then five sessions of `after` percent in total."""
    block = [-3.0] + [after / 5] * 5 + [0.0] * 20
    return block * count


def test_comparable_sessions_measure_what_followed_a_similar_move():
    # 20 falls of 3%, each followed by a 5% rise over the next 5 sessions
    # (600 sessions, so spread over two calendar years).
    closes = _series(_falls_then(5.0, 20))
    result = hs.comparable_sessions(closes, -3.2)

    assert (result.count, result.years) == (20, 2)
    assert result.low == pytest.approx(-4.0) and result.high == pytest.approx(-2.4)
    assert result.up5 == 1.0 and result.median5 == pytest.approx(5.1, abs=0.1)   # five +1% sessions compound
    assert 0 < result.base_up5 < 1


def test_only_same_direction_moves_within_the_band_count():
    closes = _series([-3.0, 0, 0, 0, 0, 0] * 16 + [+3.0, 0, 0, 0, 0, 0] * 16 + [-8.0, 0, 0, 0, 0, 0] * 16)

    assert hs.comparable_sessions(closes, -3.0).count == 16      # not the rises, not the -8% falls
    assert hs.comparable_sessions(closes, +3.0).count == 16
    assert hs.comparable_sessions(closes, 0.0) is None


def test_sessions_without_a_five_session_outcome_are_left_out():
    # The last fall is too recent to have a 5-session result.
    closes = _series([-3.0, 0, 0, 0, 0, 0] * 15 + [-3.0, 0, 0])
    assert hs.comparable_sessions(closes, -3.0).count == 15


def test_twenty_session_statistics_need_enough_outcomes():
    # 16 falls, but only the first has a 20-session outcome... build so that fewer than 15 do.
    closes = _series([-3.0, 0, 0, 0, 0, 0] * 16)           # 6 sessions per block: 20-session outcomes are rare
    result = hs.comparable_sessions(closes, -3.0)

    assert result.count == 16
    assert result.up20 is None and result.median20 is None


def test_comparable_line_wording():
    closes = _series(_falls_then(5.0, 20))
    line = hs.format_comparable("ACME", -3.2, hs.comparable_sessions(closes, -3.2))

    assert line.startswith("Comparable sessions: since 2024, ACME had 20 sessions with a fall of 2.4% to 4.0% (today: -3.2%)")
    assert "5 sessions later it was higher in 100% of them (median +5.1%), against" in line
    assert "of all sessions." in line


def test_too_few_comparable_sessions_are_not_turned_into_statistics():
    closes = _series(_falls_then(5.0, 4))
    line = hs.format_comparable("ACME", -3.2, hs.comparable_sessions(closes, -3.2))

    assert line == "Comparable sessions: only 4 past sessions had a fall of 2.4% to 4.0% (today: -3.2%), too few to draw statistics from."


# ── Precedents ───────────────────────────────────────────────────────────────

def test_precedents_are_the_closest_moves_one_per_episode():
    # Falls of 2.0, 2.9, 3.0, 3.1, 4.0 percent, separated by plenty of flat sessions.
    flat = [0.0] * 30
    closes = _series(flat + [-2.0] + flat + [-2.9] + flat + [-3.0] + flat + [-3.1] + flat + [-4.0] + flat)
    found = hs.precedents(closes, -3.0, count=3)

    assert sorted(round(p.ret, 1) for p in found) == [-3.1, -3.0, -2.9]
    assert [p.day for p in found] == sorted((p.day for p in found), reverse=True)   # most recent first


def test_consecutive_days_of_one_episode_count_once():
    closes = _series([0.0] * 30 + [-3.0, -3.0, -3.0, -3.0] + [0.0] * 30)

    assert len(hs.precedents(closes, -3.0, count=3)) == 1


def test_precedent_line_with_and_without_a_stored_event():
    precedent = hs.Precedent(day=pd.Timestamp("2023-06-23"), ret=-4.2, close=700.13, fwd5=1.2, fwd20=-3.0)
    assert hs.format_precedent("ACME", precedent) == (
        "Precedent: on 23 Jun 2023 ACME fell 4.2% to Rs 700.13; 5 sessions later it was +1.2%, 20 sessions later -3.0%."
    )

    event = {"index_return_pct": -0.8, "volume_ratio": 2.8}
    assert hs.format_precedent("ACME", precedent, event).endswith(
        "(the NIFTY 50 moved -0.8% that day; volume was 2.8 times its 20-session average)."
    )

    recent = hs.Precedent(day=pd.Timestamp("2026-09-30"), ret=3.1, close=100.0, fwd5=-1.0, fwd20=None)
    assert hs.format_precedent("ACME", recent) == (
        "Precedent: on 30 Sep 2026 ACME rose 3.1% to Rs 100.00; 5 sessions later it was -1.0%."
    )


# ── Seasonality and year context ─────────────────────────────────────────────

def _month(year, month, ret, **extra):
    return {"year": year, "month": month, "return_pct": ret, "in_progress": False,
            "period_start": f"{year}-{month:02d}-01", **extra}


def test_seasonality_uses_past_completed_months_only():
    payloads = [_month(y, 10, r) for y, r in zip(range(2008, 2016), [-10, 4, 2, -1, 6, 3, -2, 8])]
    # An earlier month exists, so 2008-10 is not the first on record; the month in progress is left out.
    payloads += [_month(2007, 9, 1.0), _month(2026, 10, -5, in_progress=True), _month(2020, 11, 99)]

    assert hs.seasonality("ACME", payloads, 10) == (
        "Seasonality: ACME rose in 5 of 8 past Octobers (median +2.5%; best +8.0% in 2015, worst -10.0% in 2008)."
    )


def test_seasonality_skips_the_part_month_at_listing_and_needs_enough_years():
    payloads = [_month(2008, 10, -10, period_start="2008-10-16")] + [_month(y, 10, 1.0) for y in range(2009, 2012)]
    payloads.append(_month(2007, 9, 5.0, period_start="2007-09-04"))   # the earliest month overall

    # 2008-10 is not the earliest month, so it counts; only 4 Octobers: too few.
    assert hs.seasonality("ACME", payloads, 10) is None
    assert hs.seasonality("ACME", payloads[:1] + [_month(2007, 10, 9.0, period_start="2007-10-16")], 10) is None

    many = [_month(2007, 10, -50.0, period_start="2007-10-16")] + [_month(y, 10, 1.0) for y in range(2008, 2014)]
    line = hs.seasonality("ACME", many, 10)
    assert "rose in 6 of 6 past Octobers" in line and "2007" not in line   # the part-month listing is excluded


def _year(year, ret, index=None, start=None, **extra):
    return {"year": year, "return_pct": ret, "index_return_pct": index, "in_progress": False,
            "period_start": start or f"{year}-01-02", "period_end": f"{year}-12-31", **extra}


def test_year_context():
    payloads = [
        _year(2022, 10.0, 5.0), _year(2023, -20.0, 8.0), _year(2024, 30.0, 12.0),
        _year(2026, 16.7, -14.9, in_progress=True, period_end="2026-10-08"),
    ]
    assert hs.year_context("ACME", payloads) == (
        "Year context: 2026 year to date (through 08 Oct 2026): +16.7%, against the NIFTY 50's -14.9%. "
        "Over its 3 completed calendar years the stock's return had a median of +10.0%, from -20.0% (2023) to +30.0% (2024); "
        "it beat the NIFTY 50 in 2 of 3 of them."
    )


def test_year_context_ignores_the_part_year_at_listing_and_needs_history():
    payloads = [
        _year(2007, 80.0, None, start="2007-11-27"),     # listed in November: not a full year
        _year(2008, -74.0, -50.0), _year(2009, 100.0, 60.0),
        _year(2026, 5.0, 1.0, in_progress=True, period_end="2026-10-08"),
    ]
    assert hs.year_context("ACME", payloads) is None   # only 2 full years

    payloads.insert(1, _year(2010, 10.0, 17.0))
    line = hs.year_context("ACME", payloads)
    assert "Over its 3 completed calendar years" in line and "2007" not in line

    assert hs.year_context("ACME", payloads[:-1]) is None        # no year in progress


# ── Stale store ──────────────────────────────────────────────────────────────

def test_stale_store_is_flagged():
    closes = _series([0.0] * 4, start="2026-09-28")   # five sessions, the last on Fri 2026-10-02

    assert hs.staleness_note(closes, date(2026, 10, 5)) is None
    assert hs.staleness_note(closes, date(2026, 10, 9)).startswith("Note: the price-history store runs only through 02 Oct 2026")
    assert hs.staleness_note(pd.Series(dtype=float), date(2026, 10, 9)) is None


# ── Agent, against an in-memory store ────────────────────────────────────────

@pytest.fixture
def reader():
    # 20 falls of 4.5% (large enough to have their own event chunks), each followed by a rise.
    returns = [-4.5, 1.0, 1.0, 1.0, 1.0, 1.0] + [0.2] * 24
    closes = _series(returns * 20, start="2024-01-01")
    client, embedder = QdrantClient(":memory:"), FakeEmbedder()
    ensure_collection(client, "test", embedder.dense_size)
    upsert_chunks(client, "test", build_chunks(ACME, _bars(closes), as_of=pd.Timestamp("2026-10-08")), embedder)
    return HistoryReader(client, collection="test")


def test_reader_scans_by_ticker_and_granularity(reader):
    months = reader.scan("acme", "month", fields=["year", "month", "sessions"])
    years = reader.scan("ACME", "year")

    assert len(months) == len({(p["year"], p["month"]) for p in months}) > 12
    assert all(set(p) == {"year", "month", "sessions"} for p in months)
    assert [p["year"] for p in sorted(years, key=lambda p: p["year"])] == [2024, 2025, 2026]
    assert reader.scan("NOSUCH", "month") == []

    assert set(reader.events("ACME", ["2024-01-02", "1999-01-01"])) <= {"2024-01-02"}
    assert reader.events("ACME", []) == {}


def test_agent_builds_every_kind_of_line(reader):
    lines = HistoricalContextAgent(reader).run(
        "ACME", "Acme Industries", MarketData(ticker="ACME", day_change_percent=-4.4), as_of=date(2026, 10, 8)
    )
    labels = [line.split(":")[0] for line in lines]

    # The synthetic history ends in spring 2026, months before as_of, so the store is flagged as stale;
    # there are too few completed years or past Octobers for the other two lines.
    assert labels == ["Comparable sessions", "Precedent", "Precedent", "Precedent", "Note"]
    assert "20 sessions with a fall of 3.3% to 5.5%" in lines[0]
    # The 4.5% falls have stored event chunks, which add the volume.
    assert "volume was" in lines[1]


def test_quiet_day_gets_no_session_analysis(reader):
    lines = HistoricalContextAgent(reader).run(
        "ACME", "Acme Industries", MarketData(ticker="ACME", day_change_percent=0.4), as_of=date(2025, 6, 10)
    )

    assert not any(line.startswith(("Comparable", "Precedent")) for line in lines)
    assert HistoricalContextAgent(reader).run("ACME", "Acme", None, as_of=date(2025, 6, 10)) == lines


def test_agent_is_a_no_op_without_a_store_or_data(reader, monkeypatch):
    import src.rag.reader as reader_module

    monkeypatch.setattr(reader_module, "get_reader", lambda: None)
    assert HistoricalContextAgent().run("ACME", "Acme Industries") == []

    assert HistoricalContextAgent(reader).run("NOSUCH", "No Such Co", MarketData(day_change_percent=-5.0)) == []
