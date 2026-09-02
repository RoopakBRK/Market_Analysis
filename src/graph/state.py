from typing import TypedDict

from src.models.company import CompanyNews
from src.models.macro import MacroSummary
from src.models.market import MarketData
from src.models.report import DailyMarketReport
from src.models.sentiment import SentimentResult


class GraphState(TypedDict):

    watchlist: list[str]

    macro_summary: MacroSummary | None

    company_news: dict[str, CompanyNews]

    market_data: dict[str, MarketData]

    sentiments: dict[str, SentimentResult]

    report: DailyMarketReport | None