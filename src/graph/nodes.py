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
    try:
        macro_summary = macro_agent.run()
    except Exception as e:
        print(f"[Graph Warning] MacroAgent failed: {e}")
        from src.models.macro import MacroSummary
        macro_summary = MacroSummary(
            overall_sentiment="Unknown",
            confidence=0,
            summary="Macro agent failed to run.",
            key_drivers=[],
            market_events=[],
            last_updated="Unknown",
            market_data={}
        )
    return {"macro_summary": macro_summary}

import time

def company_news_node(state):
    watchlist = state.get("watchlist", [])
    news = {}
    for company in watchlist:
        try:
            news[company] = company_agent.run(company)
        except Exception as e:
            print(f"[Graph Warning] CompanyNewsAgent failed for {company}: {e}")
            from src.models.company import CompanyNews
            news[company] = CompanyNews(ticker=company, company_name=company, articles=[], total_articles=0)
        time.sleep(2)  # Pace API requests
    return {"company_news": news}

def market_data_node(state):
    watchlist = state.get("watchlist", [])
    market_data = {}
    for ticker in watchlist:
        try:
            market_data[ticker] = market_agent.run(ticker)
        except Exception as e:
            print(f"[Graph Warning] MarketDataAgent failed for {ticker}: {e}")
            market_data[ticker] = {}
    return {"market_data": market_data}

def sentiment_node(state):
    sentiments = {}
    for ticker in state.get("company_news", {}):
        try:
            sentiments[ticker] = sentiment_agent.run(
                state["company_news"].get(ticker)
            )
        except Exception as e:
            print(f"[Graph Warning] SentimentAgent failed for {ticker}: {e}")
            from src.models.sentiment import SentimentResult
            sentiments[ticker] = SentimentResult(
                ticker=ticker,
                company_name=ticker,
                sentiment="Unknown",
                confidence=0,
                impact="Unknown",
                summary="Sentiment analysis failed.",
                positive_drivers=[],
                negative_drivers=[],
                articles_analyzed=0
            )
        time.sleep(2)  # Pace API requests
    return {"sentiments": sentiments}

def report_node(state):
    try:
        report = report_agent.run(
            state.get("macro_summary"),
            state.get("sentiments"),
        )
    except Exception as e:
        print(f"[Graph Warning] ReportAgent failed: {e}")
        from src.models.report import DailyMarketReport
        report = DailyMarketReport(
            report_date="Unknown",
            executive_summary="Report generation failed.",
            macro_overview="Unknown",
            top_positive_stocks=[],
            top_negative_stocks=[],
            important_events=[]
        )
    return {"report": report}