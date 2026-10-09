"""
Offline tests for the price-history RAG: chunking, the Qdrant store and
hybrid retrieval. Qdrant runs in-process (":memory:") and the embedding and
reranking models are replaced with small deterministic stand-ins, so nothing
is downloaded and no network is used.
"""

import math
import re
import zlib

import pandas as pd
import pytest
from qdrant_client import QdrantClient

from src.rag.chunking import EVENT_THRESHOLD_PCT, build_chunks, chunk_id, period_returns
from src.rag.retriever import HistoryRetriever
from src.rag.store import ensure_collection, upsert_chunks
from src.rag.universe import Company

# Qdrant's in-process mode accepts payload indexes but notes they do nothing.
pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")

ACME = Company(symbol="ACME", name="Acme Industries", industry="Capital Goods")
ZETA = Company(symbol="ZETA", name="Zeta Motors", industry="Automobile")
AS_OF = pd.Timestamp("2024-03-15")


def _prices(closes: dict[str, float]) -> pd.DataFrame:
    """Daily bars where open/high/low sit a fixed distance from each close."""
    index = pd.DatetimeIndex(list(closes), name="Date")
    close = pd.Series(list(closes.values()), index=index, dtype=float)
    return pd.DataFrame({
        "Open": close - 1, "High": close + 2, "Low": close - 2, "Close": close,
        "Volume": [1000.0] * len(close),
    })


def _history() -> pd.DataFrame:
    """Dec 2023 to mid-Mar 2024: steady, apart from a +10% and a -10% session."""
    days = pd.bdate_range("2023-12-01", "2024-03-15")
    closes, price = {}, 100.0
    for day in days:
        if day == pd.Timestamp("2024-01-10"):
            price *= 1.10
        elif day == pd.Timestamp("2024-02-14"):
            price *= 0.90
        else:
            price *= 1.001
        closes[day.strftime("%Y-%m-%d")] = price
    return _prices(closes)


def _by(chunks, granularity):
    return [c for c in chunks if c.payload["granularity"] == granularity]


# ── Chunking ─────────────────────────────────────────────────────────────────

def test_history_is_split_by_year_month_and_event():
    chunks = build_chunks(ACME, _history(), as_of=AS_OF)

    assert [c.payload["year"] for c in _by(chunks, "year")] == [2023, 2024]
    assert [(c.payload["year"], c.payload["month"]) for c in _by(chunks, "month")] == [
        (2023, 12), (2024, 1), (2024, 2), (2024, 3),
    ]
    # Only the two sessions that moved at least the threshold become events.
    assert [c.payload["period_start"] for c in _by(chunks, "event")] == ["2024-01-10", "2024-02-14"]
    assert all(abs(c.payload["return_pct"]) >= EVENT_THRESHOLD_PCT for c in _by(chunks, "event"))


def test_chunk_text_names_the_company_period_and_move():
    chunks = build_chunks(ACME, _history(), as_of=AS_OF)
    january = next(c for c in _by(chunks, "month") if c.payload["month"] == 1)
    rally, selloff = _by(chunks, "event")

    assert january.text.startswith("Acme Industries (ACME) on NSE, January 2024. The stock rose ")
    assert "Its best session was 10 Jan 2024 (+10.0%)" in january.text
    assert "10 Jan 2024 (Wednesday). The stock rose 10.0% in a single session" in rally.text
    assert "The stock fell 10.0% in a single session" in selloff.text
    assert "Over the following 5 sessions it moved +0.5%" in rally.text


def test_period_in_progress_is_labelled():
    chunks = build_chunks(ACME, _history(), as_of=AS_OF)
    by_period = {(c.payload["granularity"], c.payload["year"], c.payload["month"]): c for c in chunks}

    march, year = by_period[("month", 2024, 3)], by_period[("year", 2024, None)]
    assert march.payload["in_progress"] and "(month to date, through 15 Mar 2024)" in march.text
    assert year.payload["in_progress"] and "(year to date, through 15 Mar 2024)" in year.text
    assert not by_period[("month", 2024, 2)].payload["in_progress"]
    assert not by_period[("year", 2023, None)].payload["in_progress"]


def test_chunks_link_to_their_parent_and_keep_the_daily_bars():
    history = _history()
    chunks = build_chunks(ACME, history, as_of=AS_OF)
    january = next(c for c in _by(chunks, "month") if c.payload["month"] == 1)
    rally = _by(chunks, "event")[0]

    assert january.payload["parent_id"] == chunk_id("ACME", "year", "2024")
    assert rally.payload["parent_id"] == january.id

    # Every session is stored exactly once, in its month's payload.
    stored = [s for c in _by(chunks, "month") for s in c.payload["sessions"]]
    assert [s["date"] for s in stored] == [d.strftime("%Y-%m-%d") for d in history.index]
    assert stored[0] == {"date": "2023-12-01", "open": 99.1, "high": 102.1, "low": 98.1, "close": 100.1, "volume": 1000}


def test_chunk_ids_are_stable_and_payloads_are_json_safe():
    first = build_chunks(ACME, _history(), as_of=AS_OF)
    second = build_chunks(ACME, _history(), as_of=AS_OF)

    assert [c.id for c in first] == [c.id for c in second]
    assert len({c.id for c in first}) == len(first)

    def has_nan(value):
        if isinstance(value, dict):
            return any(has_nan(v) for v in value.values())
        if isinstance(value, list):
            return any(has_nan(v) for v in value)
        return isinstance(value, float) and math.isnan(value)

    assert not any(has_nan(c.payload) for c in first)


def test_returns_are_measured_from_the_previous_period_close():
    prices = _prices({"2024-01-30": 100.0, "2024-01-31": 110.0, "2024-02-01": 121.0, "2024-02-02": 99.0})
    monthly, yearly = period_returns(prices)

    assert monthly[(2024, 1)] == pytest.approx((110 / 99 - 1) * 100)   # first period: from its first open
    assert monthly[(2024, 2)] == pytest.approx(-10.0)                  # 110 -> 99
    assert yearly[2024] == pytest.approx(0.0)


def test_index_comparison_is_added_when_a_benchmark_is_given():
    history = _history()
    flat_index = _prices({d.strftime("%Y-%m-%d"): 20000.0 for d in history.index})
    with_index = build_chunks(ACME, history, benchmark=flat_index, as_of=AS_OF)
    without = build_chunks(ACME, history, as_of=AS_OF)

    january = next(c for c in _by(with_index, "month") if c.payload["month"] == 1)
    assert "The NIFTY 50 moved +0.0% in the same month, leaving the stock ahead of the index by" in january.text
    assert "The NIFTY 50 moved +0.0% on the same day." in _by(with_index, "event")[0].text
    assert not any("NIFTY 50" in c.text for c in without)


def test_empty_history_gives_no_chunks():
    assert build_chunks(ACME, _prices({})) == []


# ── Store and retrieval ──────────────────────────────────────────────────────

def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class FakeEmbedder:
    """Bag-of-words vectors: enough for Qdrant to rank by word overlap."""

    dense_size = 32

    def _embed(self, text):
        dense = [0.0] * self.dense_size
        counts = {}
        for token in _tokens(text):
            bucket = zlib.crc32(token.encode())
            dense[bucket % self.dense_size] += 1.0
            counts[bucket] = counts.get(bucket, 0.0) + 1.0
        norm = math.sqrt(sum(v * v for v in dense)) or 1.0
        return [v / norm for v in dense], (list(counts), list(counts.values()))

    def embed_documents(self, texts):
        pairs = [self._embed(t) for t in texts]
        return [p[0] for p in pairs], [p[1] for p in pairs]

    def embed_query(self, text):
        return self._embed(text)


class KeywordReranker:
    """Scores a passage by how many times it contains a chosen word."""

    def __init__(self, word):
        self.word = word
        self.calls = 0

    def score(self, query, passages):
        self.calls += 1
        return [float(_tokens(p).count(self.word)) for p in passages]


@pytest.fixture
def store():
    client = QdrantClient(":memory:")
    embedder = FakeEmbedder()
    ensure_collection(client, "test", embedder.dense_size)
    chunks = build_chunks(ACME, _history(), as_of=AS_OF) + build_chunks(ZETA, _history(), as_of=AS_OF)
    upsert_chunks(client, "test", chunks, embedder, batch_size=4)
    return client, embedder, chunks


def test_every_chunk_is_stored_and_reingesting_does_not_duplicate(store):
    client, embedder, chunks = store
    assert client.count("test", exact=True).count == len(chunks)

    ensure_collection(client, "test", embedder.dense_size)
    upsert_chunks(client, "test", chunks, embedder)
    assert client.count("test", exact=True).count == len(chunks)

    ensure_collection(client, "test", embedder.dense_size, recreate=True)
    assert client.count("test", exact=True).count == 0


def test_search_finds_the_matching_period(store):
    client, embedder, _ = store
    hits = HistoryRetriever(client, embedder, collection="test").search("Acme Industries ACME January 2024", limit=3)

    assert hits[0].payload["ticker"] == "ACME"
    assert (hits[0].payload["granularity"], hits[0].payload["month"]) == ("month", 1)
    # The bulky daily rows are left out of search results.
    assert all("sessions" not in h.payload for h in hits)


def test_search_filters_by_ticker_and_granularity(store):
    client, embedder, _ = store
    retriever = HistoryRetriever(client, embedder, collection="test")

    by_ticker = retriever.search("stock fell in a single session", ticker="zeta", limit=20)
    assert by_ticker and {h.payload["ticker"] for h in by_ticker} == {"ZETA"}

    events = retriever.search("stock fell in a single session", ticker="ACME", granularity="event", limit=20)
    assert [h.payload["granularity"] for h in events] == ["event", "event"]

    assert retriever.search("anything", ticker="NOSUCH") == []


def test_reranker_reorders_the_fused_candidates(store):
    client, embedder, _ = store
    reranker = KeywordReranker("fell")
    retriever = HistoryRetriever(client, embedder, reranker, collection="test")

    hits = retriever.search("Acme Industries single session", ticker="ACME", granularity="event", limit=2)

    assert reranker.calls == 1
    assert "fell 10.0% in a single session" in hits[0].text
    assert hits[0].score > hits[1].score
