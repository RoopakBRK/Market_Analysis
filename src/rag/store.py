"""Qdrant collection management and writes."""

import sys

from config.settings import settings
from src.rag.chunking import Chunk

# Named vectors in the collection.
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"

# Payload fields used as filters, and the index type each needs.
_PAYLOAD_INDEXES = {"ticker": "keyword", "granularity": "keyword", "year": "integer", "month": "integer"}


def get_client():
    """
    Return a Qdrant client built from settings, or None if QDRANT_URL is not
    configured. Never logs the API key.
    """
    if not settings.QDRANT_URL:
        print("[RAG] Skipped — QDRANT_URL not configured.", file=sys.stderr)
        return None

    from qdrant_client import QdrantClient

    return QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY or None,
        timeout=60,
    )


def ensure_collection(client, collection: str, dense_size: int, recreate: bool = False) -> None:
    """Create the collection and its payload indexes if they don't exist."""
    from qdrant_client import models

    if recreate and client.collection_exists(collection):
        client.delete_collection(collection)

    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(size=dense_size, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                # IDF is applied by Qdrant at query time, completing BM25.
                SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )

    for field, schema in _PAYLOAD_INDEXES.items():
        client.create_payload_index(collection_name=collection, field_name=field, field_schema=schema)


def upsert_chunks(client, collection: str, chunks: list[Chunk], embedder, batch_size: int = 64) -> int:
    """
    Embed and upsert chunks. Chunk ids are deterministic, so re-ingesting a
    period overwrites its point instead of duplicating it. Returns the count.
    """
    from qdrant_client import models

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        dense, sparse = embedder.embed_documents([c.text for c in batch])
        client.upsert(
            collection_name=collection,
            points=[
                models.PointStruct(
                    id=chunk.id,
                    vector={
                        DENSE_VECTOR: dense_vector,
                        SPARSE_VECTOR: models.SparseVector(indices=indices, values=values),
                    },
                    payload=chunk.payload,
                )
                for chunk, dense_vector, (indices, values) in zip(batch, dense, sparse)
            ],
        )
    return len(chunks)
