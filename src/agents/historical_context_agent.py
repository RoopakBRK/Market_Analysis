from src.models.market import MarketData

# A session move at least this large (percent) is worth looking up precedents for.
_NOTABLE_MOVE_PCT = 2.0


class HistoricalContextAgent:
    """
    Retrieves passages about a company from the 20-year price-history store
    (src/rag) to put today's session in context.

    No LLM is used: the query is built from today's market data and the
    passages come back verbatim from the store. Returns [] when the store is
    not configured, so the pipeline runs unchanged without it.
    """

    def __init__(self, retriever=None):
        self._retriever = retriever
        self._resolved = retriever is not None

    @property
    def retriever(self):
        # Resolved on first use, not at import: building it loads the
        # embedding models, which is wasted work when Qdrant isn't configured.
        if not self._resolved:
            from src.rag.retriever import get_retriever
            self._retriever = get_retriever()
            self._resolved = True
        return self._retriever

    @staticmethod
    def build_query(company: str, market: MarketData | None) -> str:
        """Describe today's session in the same terms the stored chunks use."""
        move = market.day_change_percent if market is not None else None
        if move is not None and abs(move) >= _NOTABLE_MOVE_PCT:
            direction = "rose" if move > 0 else "fell"
            return f"{company} {direction} {abs(move):.1f}% in a single session and what followed"
        return f"{company} recent monthly performance against the NIFTY 50"

    def run(self, ticker: str, company: str, market: MarketData | None = None, limit: int = 3) -> list[str]:
        if self.retriever is None:
            return []
        hits = self.retriever.search(self.build_query(company, market), ticker=ticker, limit=limit)
        return [hit.text for hit in hits]
