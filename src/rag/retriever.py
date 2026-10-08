"""Hybrid retrieval with cross-encoder reranking over the price-history store."""

from dataclasses import dataclass
from functools import lru_cache

from config.settings import settings
from src.rag.store import DENSE_VECTOR, SPARSE_VECTOR, get_client


@dataclass
class RetrievedChunk:
    text: str
    payload: dict
    score: float


class HistoryRetriever:
    """
    Two-stage retrieval:

    1. Qdrant runs a dense search and a BM25 sparse search (both restricted
       by the same payload filter) and merges the two ranked lists with
       Reciprocal Rank Fusion.
    2. A cross-encoder rescores the fused candidates against the query and
       the best `limit` are returned.

    Pass reranker=None to return the fused order unchanged.
    """

    def __init__(self, client, embedder, reranker=None, collection: str | None = None):
        self.client = client
        self.embedder = embedder
        self.reranker = reranker
        self.collection = collection or settings.QDRANT_COLLECTION

    def search(
        self,
        query: str,
        ticker: str | None = None,
        granularity: str | None = None,
        limit: int = 5,
        candidates: int = 30,
    ) -> list[RetrievedChunk]:
        from qdrant_client import models

        conditions = []
        if ticker:
            conditions.append(models.FieldCondition(key="ticker", match=models.MatchValue(value=ticker.upper())))
        if granularity:
            conditions.append(models.FieldCondition(key="granularity", match=models.MatchValue(value=granularity)))
        query_filter = models.Filter(must=conditions) if conditions else None

        dense, (indices, values) = self.embedder.embed_query(query)

        response = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                models.Prefetch(query=dense, using=DENSE_VECTOR, filter=query_filter, limit=candidates),
                models.Prefetch(
                    query=models.SparseVector(indices=indices, values=values),
                    using=SPARSE_VECTOR,
                    filter=query_filter,
                    limit=candidates,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=candidates,
            # The raw daily rows are only needed for exact lookups.
            with_payload=models.PayloadSelectorExclude(exclude=["sessions"]),
        )

        hits = [
            RetrievedChunk(text=p.payload.get("text", ""), payload=p.payload, score=float(p.score))
            for p in response.points
        ]

        if self.reranker and hits:
            scores = self.reranker.score(query, [h.text for h in hits])
            for hit, score in zip(hits, scores):
                hit.score = score
            hits.sort(key=lambda h: h.score, reverse=True)

        return hits[:limit]


@lru_cache
def get_retriever() -> HistoryRetriever | None:
    """
    The retriever for the configured Qdrant store, or None if it isn't
    configured. Built once: loading the embedding models takes a few seconds.
    """
    client = get_client()
    if client is None:
        return None

    from src.rag.embeddings import CrossEncoderReranker, FastEmbedder

    return HistoryRetriever(client, FastEmbedder(), CrossEncoderReranker())
