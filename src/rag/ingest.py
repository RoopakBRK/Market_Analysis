"""
Build or refresh the price-history store.

    python -m src.rag.ingest                          # NIFTY 50, 20 years
    python -m src.rag.ingest --tickers TMPV,ADANIPORTS
    python -m src.rag.ingest --recreate               # drop the collection first

Needs QDRANT_URL and QDRANT_API_KEY in .env. Safe to re-run: every chunk has a
deterministic id, so a refresh overwrites existing points.
"""

import argparse
import sys
import time

from config.settings import settings
from src.rag.chunking import build_chunks
from src.rag.loader import NIFTY50_SYMBOL, load_price_history
from src.rag.store import ensure_collection, get_client, upsert_chunks
from src.rag.universe import Company, resolve_universe


def ingest(client, embedder, companies: list[Company], years: int = 20, collection: str | None = None, recreate: bool = False) -> dict[str, int]:
    """
    Load, chunk, embed and upsert each company's history.
    Returns {symbol: chunks stored}; a company that fails is reported and skipped.
    """
    collection = collection or settings.QDRANT_COLLECTION
    ensure_collection(client, collection, embedder.dense_size, recreate=recreate)

    benchmark = load_price_history(NIFTY50_SYMBOL, years=years)
    if benchmark.empty:
        print("[RAG] NIFTY 50 history unavailable; chunks will omit the index comparison.", file=sys.stderr)
        benchmark = None

    stored = {}
    for position, company in enumerate(companies, 1):
        try:
            prices = load_price_history(company.symbol, years=years)
            if prices.empty:
                print(f"[{position}/{len(companies)}] {company.symbol}: no price history on Yahoo, skipped")
                continue

            chunks = build_chunks(company, prices, benchmark=benchmark)
            stored[company.symbol] = upsert_chunks(client, collection, chunks, embedder)
            print(
                f"[{position}/{len(companies)}] {company.symbol}: {len(prices)} sessions "
                f"({prices.index[0]:%Y-%m-%d} to {prices.index[-1]:%Y-%m-%d}) -> {len(chunks)} chunks"
            )
        except Exception as exc:
            print(f"[{position}/{len(companies)}] {company.symbol}: FAILED ({type(exc).__name__}: {exc})", file=sys.stderr)
    return stored


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest NIFTY 50 price history into Qdrant.")
    parser.add_argument("--tickers", help="Comma-separated NSE symbols (default: the NIFTY 50 plus the watchlist)")
    parser.add_argument("--years", type=int, default=20, help="Years of history to load (default: 20)")
    parser.add_argument("--recreate", action="store_true", help="Delete and rebuild the collection")
    args = parser.parse_args()

    client = get_client()
    if client is None:
        print("Set QDRANT_URL and QDRANT_API_KEY in .env, then run this again.", file=sys.stderr)
        return 1

    from src.rag.embeddings import FastEmbedder

    companies = resolve_universe(args.tickers.split(",") if args.tickers else None)
    print(f"Ingesting {len(companies)} companies, {args.years} years, into '{settings.QDRANT_COLLECTION}'")

    started = time.time()
    stored = ingest(client, FastEmbedder(), companies, years=args.years, recreate=args.recreate)

    total = client.count(settings.QDRANT_COLLECTION, exact=True).count
    print(
        f"\nDone in {time.time() - started:.0f}s: {sum(stored.values())} chunks from "
        f"{len(stored)}/{len(companies)} companies. Collection now holds {total} points."
    )
    return 0 if stored else 1


if __name__ == "__main__":
    sys.exit(main())
