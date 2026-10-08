"""
Price-history RAG: 20 years of daily prices for the NIFTY 50, summarised into
text chunks and stored in Qdrant for hybrid retrieval.

    python -m src.rag.ingest                 # build / refresh the store
    python -m src.rag.query "question"       # retrieve + answer

See structure.md ("Price-History RAG") for the chunking and reranking design.
"""
