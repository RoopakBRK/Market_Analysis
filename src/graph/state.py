from typing import TypedDict, Annotated
import operator

from src.models.company import CompanyNews
from src.models.financial_data import CompanyFinancials
from src.models.macro import MacroSummary
from src.models.market import MarketData
from src.models.reddit import RedditSignal
from src.models.report import DailyMarketReport
from src.models.sentiment import SentimentResult


class GraphState(TypedDict):

    watchlist: Annotated[list[str], operator.add]

    macro_summary: MacroSummary | None

    company_news: dict[str, CompanyNews]

    market_data: dict[str, MarketData]

    # New: structured financial data per company (from FinancialDataAgent).
    # Keys are ticker symbols.
    financial_data: dict[str, CompanyFinancials]

    # New: Reddit community sentiment signals per company.
    # Keys are ticker symbols.
    reddit_signals: dict[str, RedditSignal]

    sentiments: dict[str, SentimentResult]

    report: DailyMarketReport | None