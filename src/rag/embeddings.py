"""
Embedding and reranking models. All three run locally through fastembed
(ONNX, CPU) — no embedding API key is needed; the first use downloads the
model files (~150 MB in total) to fastembed's cache.
"""

# Dense: semantic similarity. 384 dimensions, cosine.
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
DENSE_SIZE = 384

# Sparse: BM25 term weights, for exact matches on tickers, names, months and
# years, which dense embeddings blur ("March 2020" vs "March 2021").
SPARSE_MODEL = "Qdrant/bm25"

# Reranker: a cross-encoder that reads the query and one passage together.
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

# (indices, values) of a sparse vector.
SparseVector = tuple[list[int], list[float]]


class FastEmbedder:
    """Dense + sparse embeddings for documents and queries."""

    dense_size = DENSE_SIZE

    def __init__(self):
        from fastembed import SparseTextEmbedding, TextEmbedding

        self._dense = TextEmbedding(DENSE_MODEL)
        self._sparse = SparseTextEmbedding(SPARSE_MODEL)

    def embed_documents(self, texts: list[str]) -> tuple[list[list[float]], list[SparseVector]]:
        dense = [v.tolist() for v in self._dense.embed(texts)]
        sparse = [(s.indices.tolist(), s.values.tolist()) for s in self._sparse.embed(texts)]
        return dense, sparse

    def embed_query(self, text: str) -> tuple[list[float], SparseVector]:
        dense = next(iter(self._dense.query_embed(text))).tolist()
        sparse = next(iter(self._sparse.query_embed(text)))
        return dense, (sparse.indices.tolist(), sparse.values.tolist())


class CrossEncoderReranker:
    """Scores (query, passage) pairs; higher means more relevant."""

    def __init__(self):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(RERANK_MODEL)

    def score(self, query: str, passages: list[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, passages)]
