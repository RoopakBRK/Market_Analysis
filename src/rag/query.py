"""
Ask the price-history store a question.

    python -m src.rag.query "How did Adani Ports do in 2020?"
    python -m src.rag.query "sharpest one-day falls" --ticker TMPV --granularity event
    python -m src.rag.query "best months" --ticker TMPV --no-answer   # retrieval only
"""

import argparse
import sys

from src.rag.retriever import HistoryRetriever, RetrievedChunk, get_retriever


def answer(question: str, chunks: list[RetrievedChunk]) -> str:
    """Generate an answer grounded in the retrieved chunks."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from src.llm.gateway import get_llm
    from src.prompts.history import SYSTEM_PROMPT

    excerpts = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks, 1))
    response = get_llm(agent_name="HistoryQA").invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}"),
    ])
    return str(response.content).strip()


def ask(retriever: HistoryRetriever, question: str, ticker=None, granularity=None, limit: int = 5):
    """Retrieve the passages for a question. Returns the reranked chunks."""
    return retriever.search(question, ticker=ticker, granularity=granularity, limit=limit)


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the NIFTY 50 price-history store.")
    parser.add_argument("question")
    parser.add_argument("--ticker", help="Restrict to one NSE symbol")
    parser.add_argument("--granularity", choices=["year", "month", "event"], help="Restrict to one chunk type")
    parser.add_argument("--top", type=int, default=5, help="Passages to keep after reranking (default: 5)")
    parser.add_argument("--no-answer", action="store_true", help="Show the retrieved passages only")
    args = parser.parse_args()

    retriever = get_retriever()
    if retriever is None:
        print("Set QDRANT_URL and QDRANT_API_KEY in .env, then run `python -m src.rag.ingest` first.", file=sys.stderr)
        return 1

    chunks = ask(retriever, args.question, ticker=args.ticker, granularity=args.granularity, limit=args.top)
    if not chunks:
        print("No matching passages. Has the store been ingested?")
        return 1

    for i, chunk in enumerate(chunks, 1):
        print(f"[{i}] score {chunk.score:.3f} | {chunk.payload.get('ticker')} {chunk.payload.get('granularity')} "
              f"{chunk.payload.get('period_start')}\n    {chunk.text}\n")

    if not args.no_answer:
        print("--- ANSWER ---")
        print(answer(args.question, chunks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
