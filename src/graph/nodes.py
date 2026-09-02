from src.agents.macro_agent import MacroAgent
from src.agents.company_news_agent import CompanyNewsAgent
from src.agents.market_data_agent import MarketDataAgent
from src.agents.sentiment_agent import SentimentAgent
from src.agents.report_agent import ReportAgent

report_agent = ReportAgent()

sentiment_agent = SentimentAgent()
market_agent = MarketDataAgent()

company_agent = CompanyNewsAgent()


macro_agent = MacroAgent()


def macro_node(state):
    macro_summary = macro_agent.run()
    return {"macro_summary": macro_summary}

import time

def company_news_node(state):
    watchlist = state.get("watchlist", [])
    news = {}
    for company in watchlist:
        news[company] = company_agent.run(company)
        time.sleep(2)  # Pace API requests to respect rate limits
    return {"company_news": news}

def market_data_node(state):
    watchlist = state.get("watchlist", [])
    market_data = {}
    for ticker in watchlist:
        market_data[ticker] = market_agent.run(ticker)
    return {"market_data": market_data}

def sentiment_node(state):
    sentiments = {}
    for ticker in state.get("company_news", {}):
        sentiments[ticker] = sentiment_agent.run(
            state["macro_summary"],
            state["company_news"][ticker],
            state["market_data"][ticker],
        )
        time.sleep(2)  # Pace API requests to respect rate limits
    return {"sentiments": sentiments}

def report_node(state):
    report = report_agent.run(
        state["macro_summary"],
        state["sentiments"],
    )
    return {"report": report}