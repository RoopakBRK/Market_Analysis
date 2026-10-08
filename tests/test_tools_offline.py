"""Offline tests for the data tools: no network, no API keys."""

import pytest

from src.tools.common.yahoo_finance import nse_symbol
from src.tools.market.indicators import compute_macd, compute_rsi, compute_vwap
from src.tools.market.sector import get_sector_performance


# ── Yahoo symbol mapping ─────────────────────────────────────────────────────

def test_nse_symbol_adds_exchange_suffix():
    assert nse_symbol("TMPV") == "TMPV.NS"
    assert nse_symbol(" adaniports ") == "ADANIPORTS.NS"


def test_nse_symbol_leaves_qualified_symbols_alone():
    assert nse_symbol("TMPV.BO") == "TMPV.BO"
    assert nse_symbol("^NSEI") == "^NSEI"
    assert nse_symbol("INR=X") == "INR=X"


# ── Technical indicators ─────────────────────────────────────────────────────

# Wilder's worked RSI example: 15 closes give a first RSI of ~70.46.
_WILDER_CLOSES = [
    44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
    45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28,
]


def test_rsi_matches_reference_value():
    assert compute_rsi(_WILDER_CLOSES) == pytest.approx(70.46, abs=0.05)


def test_rsi_extremes_and_short_series():
    assert compute_rsi([float(i) for i in range(1, 30)]) == 100.0
    assert compute_rsi([float(i) for i in range(30, 1, -1)]) == pytest.approx(0.0)
    assert compute_rsi(_WILDER_CLOSES[:14]) is None


def test_macd_direction_and_short_series():
    assert compute_macd([100.0] * 60) == pytest.approx((0.0, 0.0))

    macd, signal = compute_macd([100.0 + i for i in range(60)])
    assert macd > 0 and signal > 0

    macd, _ = compute_macd([100.0 - i for i in range(60)])
    assert macd < 0

    assert compute_macd([100.0] * 34) is None


def test_vwap_weights_by_volume():
    bars = [
        {"high": 10.0, "low": 10.0, "close": 10.0, "volume": 100.0},
        {"high": 20.0, "low": 20.0, "close": 20.0, "volume": 300.0},
    ]
    assert compute_vwap(bars) == pytest.approx(17.5)
    assert compute_vwap([{"high": 1.0, "low": 1.0, "close": 1.0, "volume": 0.0}]) is None


# ── Sector benchmark ─────────────────────────────────────────────────────────

def test_unknown_sector_has_no_performance():
    result = get_sector_performance.invoke({"sector": "", "industry": ""})
    assert result["index"] is None
    assert result["performance"] is None


def test_sector_maps_to_nse_index(monkeypatch):
    import src.tools.market.sector as sector_module

    requested = []

    def fake_quote(symbol):
        requested.append(symbol)
        return {"price": 99.0, "previous_close": 100.0, "change": -1.0, "change_percent": -1.0}

    monkeypatch.setattr(sector_module, "get_latest_quote", fake_quote)

    auto = get_sector_performance.invoke({"sector": "Consumer Cyclical", "industry": "Auto Manufacturers"})
    ports = get_sector_performance.invoke({"sector": "Industrials", "industry": "Marine Shipping"})

    assert requested == ["^CNXAUTO", "^CNXINFRA"]
    assert auto["index"] == "Nifty Auto" and auto["performance"] == -1.0
    assert ports["index"] == "Nifty Infrastructure"


# ── Crude oil label ──────────────────────────────────────────────────────────

def test_rising_crude_is_negative_for_india(monkeypatch):
    import src.tools.macro.crude as crude_module

    monkeypatch.setattr(
        crude_module,
        "get_latest_quote",
        lambda symbol: {"price": 105.0, "previous_close": 100.0, "change": 5.0, "change_percent": 5.0},
    )
    result = crude_module.get_crude_price.invoke({})

    assert result["trend"] == "Rising"
    assert result["impact_on_india"] == "Negative"


# ── Moneycontrol redirect ────────────────────────────────────────────────────

def test_moneycontrol_unknown_tag_returns_no_articles(monkeypatch):
    import src.tools.company.moneycontrol as mc_module

    class FrontPage:
        # What an unknown tag resolves to: the generic news front page.
        url = "https://www.moneycontrol.com/news/"
        content = (
            b'<ul><li class="clearfix"><a href="/news/business/unrelated-story-1.html" '
            b'title="A headline that has nothing to do with the company">x</a></li></ul>'
        )

        def raise_for_status(self):
            pass

    monkeypatch.setattr(mc_module.requests, "get", lambda *args, **kwargs: FrontPage())

    result = mc_module.search_moneycontrol_news.invoke({"company": "No Such Company"})
    assert result["articles"] == []


# ── Tavily date window ───────────────────────────────────────────────────────

def test_tavily_drops_results_older_than_the_window(monkeypatch):
    from datetime import datetime, timedelta, timezone
    from email.utils import format_datetime

    import src.tools.tavily.client as tavily_module

    now = datetime.now(timezone.utc)

    class FakeClient:
        def search(self, **kwargs):
            return {"results": [
                {"title": "Fresh", "url": "https://example.com/1", "published_date": format_datetime(now - timedelta(days=1))},
                {"title": "Stale", "url": "https://example.com/2", "published_date": format_datetime(now - timedelta(days=200))},
                {"title": "Undated", "url": "https://example.com/3"},
            ]}

    monkeypatch.setattr(tavily_module, "_build_client", lambda: FakeClient())

    titles = [a["title"] for a in tavily_module.search("anything", days_back=3)]
    assert titles == ["Fresh", "Undated"]


# ── India benchmark indices ──────────────────────────────────────────────────

def test_india_market_summary_reports_nifty_and_sensex(monkeypatch):
    import src.tools.macro.india_market as india_module

    quotes = {
        "^NSEI": {"price": 22231.8, "previous_close": 22603.05, "change": -371.25, "change_percent": -1.6425},
        "^BSESN": {"price": 71593.24, "previous_close": 72638.7, "change": -1045.46, "change_percent": -1.4393},
    }
    monkeypatch.setattr(india_module, "get_latest_quote", quotes.get)
    result = india_module.get_india_market_summary.invoke({})

    assert result["nifty50"] == {"current": 22231.8, "previous_close": 22603.05, "change": -371.25, "change_percent": -1.64}
    assert result["sensex"]["change_percent"] == -1.44
    assert result["trend"] == "Falling"


def test_india_market_summary_is_none_without_quotes(monkeypatch):
    import src.tools.macro.india_market as india_module

    monkeypatch.setattr(india_module, "get_latest_quote", lambda symbol: None)
    assert india_module.get_india_market_summary.invoke({}) is None
