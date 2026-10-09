"""
Exact reads from the price-history store: everything for a ticker at one
granularity, or specific chunks by id. Unlike the retriever (src/rag/retriever),
this does no embedding or ranking, so it needs no models and returns the data
as stored.
"""

from functools import lru_cache

from config.settings import settings
from src.rag.chunking import chunk_id
from src.rag.store import get_client

_PAGE = 100


class HistoryReader:
    def __init__(self, client, collection: str | None = None):
        self.client = client
        self.collection = collection or settings.QDRANT_COLLECTION

    def scan(self, ticker: str, granularity: str, fields: list[str] | None = None) -> list[dict]:
        """
        Every chunk payload for a ticker at one granularity ("year", "month"
        or "event"), in the order Qdrant returns them. `fields` limits which
        payload keys come back; leaving out `sessions` and `text` keeps a
        month scan small.
        """
        from qdrant_client import models

        query_filter = models.Filter(must=[
            models.FieldCondition(key="ticker", match=models.MatchValue(value=ticker.upper())),
            models.FieldCondition(key="granularity", match=models.MatchValue(value=granularity)),
        ])

        payloads, offset = [], None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=query_filter,
                limit=_PAGE,
                offset=offset,
                with_payload=fields if fields else True,
                with_vectors=False,
            )
            payloads += [point.payload for point in points]
            if offset is None:
                return payloads

    def events(self, ticker: str, days: list[str]) -> dict[str, dict]:
        """
        The single-session chunks for the given dates (YYYY-MM-DD), keyed by
        date. Only sessions that moved 4% or more have one; others are absent.
        """
        if not days:
            return {}
        points = self.client.retrieve(
            collection_name=self.collection,
            ids=[chunk_id(ticker.upper(), "event", day) for day in days],
            with_payload=True,
            with_vectors=False,
        )
        return {point.payload["period_start"]: point.payload for point in points}


@lru_cache
def get_reader() -> HistoryReader | None:
    """The reader for the configured Qdrant store, or None if it isn't configured."""
    client = get_client()
    return HistoryReader(client) if client is not None else None
