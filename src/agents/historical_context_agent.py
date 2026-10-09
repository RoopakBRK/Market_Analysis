from datetime import date

from src.models.market import MarketData
from src.rag import history_stats as stats

# A session move at least this large (percent) is worth looking up precedents for.
_NOTABLE_MOVE_PCT = 2.0
_PRECEDENTS = 3

_MONTH_FIELDS = ["year", "month", "period_start", "in_progress", "return_pct", "sessions"]
_YEAR_FIELDS = ["year", "period_start", "period_end", "in_progress", "return_pct", "index_return_pct"]


class HistoricalContextAgent:
    """
    Puts today's session in context using the 20-year price-history store
    (src/rag). For one company it produces up to four kinds of line:

      - Comparable sessions: what followed past sessions of today's size,
        against how the stock does after any session
      - Precedent: the three closest past sessions, one per distinct episode
      - Seasonality: how the stock did in this calendar month in past years
      - Year context: this year so far against its own history and the index

    No LLM is used. Past sessions are selected by the size of the move, not
    by text similarity, and every figure is computed in code from the stored
    daily prices and returns, so the lines can be quoted as fact. Returns []
    when the store is not configured, so the pipeline runs unchanged without it.
    """

    def __init__(self, reader=None):
        self._reader = reader
        self._resolved = reader is not None

    @property
    def reader(self):
        # Resolved on first use, so importing the pipeline does not touch the
        # network when Qdrant isn't configured.
        if not self._resolved:
            from src.rag.reader import get_reader
            self._reader = get_reader()
            self._resolved = True
        return self._reader

    def run(
        self,
        ticker: str,
        company: str,
        market: MarketData | None = None,
        as_of: date | None = None,
    ) -> list[str]:
        reader = self.reader
        if reader is None:
            return []
        as_of = as_of or date.today()

        months = reader.scan(ticker, "month", fields=_MONTH_FIELDS)
        years = reader.scan(ticker, "year", fields=_YEAR_FIELDS)
        if not months and not years:
            return []

        closes = stats.closes_from_months(months)
        move = market.day_change_percent if market is not None else None

        lines = []
        if move is not None and abs(move) >= _NOTABLE_MOVE_PCT and not closes.empty:
            lines += self._session_lines(reader, ticker, closes, move)

        for line in (
            stats.seasonality(ticker, months, as_of.month),
            stats.year_context(ticker, years),
            stats.staleness_note(closes, as_of),
        ):
            if line:
                lines.append(line)
        return lines

    @staticmethod
    def _session_lines(reader, ticker: str, closes, move: float) -> list[str]:
        lines = []

        comparable = stats.comparable_sessions(closes, move)
        if comparable is not None:
            lines.append(stats.format_comparable(ticker, move, comparable))

        closest = stats.precedents(closes, move, count=_PRECEDENTS)
        # Sessions of 4% or more have their own stored chunk, which adds the
        # index move and volume; smaller ones are described from the prices alone.
        events = reader.events(ticker, [p.day.strftime("%Y-%m-%d") for p in closest])
        for precedent in closest:
            lines.append(stats.format_precedent(ticker, precedent, events.get(precedent.day.strftime("%Y-%m-%d"))))
        return lines
