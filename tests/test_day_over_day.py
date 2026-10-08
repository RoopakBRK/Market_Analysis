"""Offline tests for the day-over-day comparison and the run-history file."""

import json

from src.models.macro import MacroSummary
from src.models.market import MarketData
from src.models.sentiment import SentimentResult
from src.services.change_detection import ChangeDetectionService
from src.services.historical_service import build_snapshot, load_previous_snapshot, save_snapshot


def _macro(sentiment="Bearish", confidence=85, fii=-6121.37, flows_date="07-Oct-2026", repo=5.5) -> MacroSummary:
    return MacroSummary(
        overall_sentiment=sentiment,
        confidence=confidence,
        summary="s",
        key_drivers=[],
        last_updated="2026-10-08T11:00:00+00:00",
        market_data={
            "india_market": {
                "nifty50": {"current": 22231.8, "previous_close": 22603.05, "change": -371.25, "change_percent": -1.64},
                "sensex": {"current": 71593.24, "previous_close": 72638.7, "change": -1045.46, "change_percent": -1.44},
            },
            "gold": {"price": 4146.9, "previous_close": 4140.7},
            "crude": {"price": 104.97, "previous_close": 100.2},
            "usd_inr": {"exchange_rate": 96.78, "previous_close": 96.37},
            # A source that failed: no US market data this run.
            "us_market": None,
            "fii_dii_flows": {"date": flows_date, "fii_net": fii, "dii_net": 4596.57},
            "rbi": {"policy_repo_rate": repo},
        },
    )


def _sentiment(ticker, label, confidence) -> SentimentResult:
    return SentimentResult(
        ticker=ticker, company_name=ticker, sentiment=label, confidence=confidence,
        impact="Medium", summary="s", articles_analyzed=1,
    )


def _rows(rows) -> dict:
    return {r.item: r for r in rows}


# ── Markets versus the previous close ────────────────────────────────────────

def test_market_changes_cover_indices_commodities_and_stocks():
    market_data = {
        "TMPV": MarketData(ticker="TMPV", current_price=273.0, previous_close=283.0),
        # No quote retrieved for this one.
        "ADANIPORTS": MarketData(ticker="ADANIPORTS"),
    }
    rows = _rows(ChangeDetectionService().market_changes(_macro(), market_data))

    assert list(rows) == ["Nifty 50", "Sensex", "Gold (USD/oz)", "Brent crude (USD/barrel)", "USD/INR", "TMPV (Rs)"]

    nifty = rows["Nifty 50"]
    assert (nifty.previous, nifty.current, nifty.change) == ("22,603.05", "22,231.80", "-371.25 (-1.64%)")
    assert rows["Brent crude (USD/barrel)"].change == "+4.77 (+4.76%)"
    assert rows["TMPV (Rs)"].change == "-10.00 (-3.53%)"


def test_market_changes_without_macro_data():
    assert ChangeDetectionService().market_changes(None, {}) == []


# ── Signals versus the previous run ──────────────────────────────────────────

def test_run_changes_against_previous_snapshot():
    previous = build_snapshot(
        "2026-10-07",
        _macro(sentiment="Bearish", confidence=70, fii=-2500.0, flows_date="06-Oct-2026", repo=5.25),
        {"TMPV": _sentiment("TMPV", "Bearish", 70), "ADANIPORTS": _sentiment("ADANIPORTS", "Bullish", 90)},
    )
    sentiments = {
        "TMPV": _sentiment("TMPV", "Bullish", 90),
        "ADANIPORTS": _sentiment("ADANIPORTS", "Bullish", 80),
        "RELIANCE": _sentiment("RELIANCE", "Neutral", 50),
    }
    rows = _rows(ChangeDetectionService().run_changes(_macro(), sentiments, previous))

    macro = rows["Macro sentiment"]
    assert (macro.previous, macro.current, macro.change) == ("Bearish (70%)", "Bearish (85%)", "confidence +15")

    assert rows["FII net flow (INR crore)"].change == "-3,621.37"
    assert rows["DII net flow (INR crore)"].change == "+0.00"

    repo = rows["RBI repo rate"]
    assert (repo.previous, repo.current, repo.change) == ("5.25%", "5.50%", "+0.25 pp")

    assert rows["TMPV sentiment"].change == "Bearish to Bullish"
    assert rows["ADANIPORTS sentiment"].change == "confidence -10"
    # Not in the previous run's watchlist.
    assert (rows["RELIANCE sentiment"].previous, rows["RELIANCE sentiment"].change) == ("not covered", "new")


def test_unchanged_values_and_stale_flow_figures():
    macro = _macro()
    sentiments = {"TMPV": _sentiment("TMPV", "Bullish", 90)}
    previous = build_snapshot("2026-10-07", macro, sentiments)

    rows = _rows(ChangeDetectionService().run_changes(macro, sentiments, previous))

    assert rows["Macro sentiment"].change == "unchanged"
    assert rows["RBI repo rate"].change == "unchanged"
    assert rows["TMPV sentiment"].change == "unchanged"
    # Same publication date as last run: NSE has not published a new figure.
    assert rows["FII net flow (INR crore)"].change == "no new figure since 07-Oct-2026"


def test_no_previous_run_means_no_run_changes():
    assert ChangeDetectionService().run_changes(_macro(), {"TMPV": _sentiment("TMPV", "Bullish", 90)}, None) == []


def test_missing_figures_are_left_out():
    macro = _macro()
    macro.market_data["fii_dii_flows"] = None
    macro.market_data["rbi"] = None
    previous = build_snapshot("2026-10-07", _macro(), {})

    rows = _rows(ChangeDetectionService().run_changes(macro, {}, previous))
    assert list(rows) == ["Macro sentiment"]


# ── History file ─────────────────────────────────────────────────────────────

def test_snapshot_round_trip(tmp_path):
    snapshot = build_snapshot("2026-10-08", _macro(), {"TMPV": _sentiment("TMPV", "Bullish", 90)})
    path = save_snapshot(snapshot, history_dir=tmp_path)

    assert path.name == "2026-10-08.json"
    assert json.loads(path.read_text())["macro"] == {
        "sentiment": "Bearish", "confidence": 85, "fii_net": -6121.37, "dii_net": 4596.57,
        "flows_date": "07-Oct-2026", "repo_rate": 5.5,
    }
    assert load_previous_snapshot("2026-10-09", history_dir=tmp_path)["companies"] == {
        "TMPV": {"sentiment": "Bullish", "confidence": 90}
    }


def test_previous_snapshot_is_the_latest_earlier_day(tmp_path):
    for day in ("2026-10-05", "2026-10-07", "2026-10-08"):
        save_snapshot(build_snapshot(day, _macro(), {}), history_dir=tmp_path)

    # Today's own snapshot is never the "previous" one, even on a re-run.
    assert load_previous_snapshot("2026-10-08", history_dir=tmp_path)["date"] == "2026-10-07"
    assert load_previous_snapshot("2026-10-06", history_dir=tmp_path)["date"] == "2026-10-05"
    assert load_previous_snapshot("2026-10-05", history_dir=tmp_path) is None


def test_same_day_rerun_overwrites_the_snapshot(tmp_path):
    save_snapshot(build_snapshot("2026-10-08", _macro(confidence=70), {}), history_dir=tmp_path)
    save_snapshot(build_snapshot("2026-10-08", _macro(confidence=85), {}), history_dir=tmp_path)

    assert len(list(tmp_path.glob("*.json"))) == 1
    assert load_previous_snapshot("2026-10-09", history_dir=tmp_path)["macro"]["confidence"] == 85


def test_unreadable_snapshot_is_skipped(tmp_path):
    save_snapshot(build_snapshot("2026-10-06", _macro(), {}), history_dir=tmp_path)
    (tmp_path / "2026-10-07.json").write_text("{not json")

    assert load_previous_snapshot("2026-10-08", history_dir=tmp_path)["date"] == "2026-10-06"
    assert load_previous_snapshot("2026-10-08", history_dir=tmp_path / "missing") is None
