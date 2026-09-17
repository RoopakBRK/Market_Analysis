import time
from src.agents.macro_agent import MacroAgent
from src.agents.company_news_agent import CompanyNewsAgent
from src.agents.market_data_agent import MarketDataAgent
from src.agents.financial_data_agent import FinancialDataAgent
from src.agents.reddit_sentiment_agent import RedditSentimentAgent
from src.agents.sentiment_agent import SentimentAgent
from src.agents.report_agent import ReportAgent

report_agent = ReportAgent()
sentiment_agent = SentimentAgent()
market_agent = MarketDataAgent()
company_agent = CompanyNewsAgent()
macro_agent = MacroAgent()
financial_data_agent = FinancialDataAgent()
reddit_sentiment_agent = RedditSentimentAgent()


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
            from src.models.market import MarketData
            market_data[ticker] = MarketData(ticker=ticker, price=0.0, change=0.0)
    return {"market_data": market_data}


def financial_data_node(state):
    watchlist = state.get("watchlist", [])
    financial_data = {}
    for ticker in watchlist:
        try:
            financial_data[ticker] = financial_data_agent.run(ticker)
        except Exception as e:
            print(f"[Graph Warning] FinancialDataAgent failed for {ticker}: {e}")
            from src.models.financial_data import CompanyFinancials
            from src.tools.common.normalization import utc_now_iso
            financial_data[ticker] = CompanyFinancials(ticker=ticker, retrieved_at=utc_now_iso())
    return {"financial_data": financial_data}


def reddit_sentiment_node(state):
    watchlist = state.get("watchlist", [])
    reddit_signals = {}
    for ticker in watchlist:
        try:
            reddit_signals[ticker] = reddit_sentiment_agent.run(ticker)
        except Exception as e:
            print(f"[Graph Warning] RedditSentimentAgent failed for {ticker}: {e}")
            from src.models.reddit import RedditSignal
            from src.tools.common.normalization import utc_now_iso
            reddit_signals[ticker] = RedditSignal(ticker=ticker, company_name=ticker, retrieved_at=utc_now_iso())
    return {"reddit_signals": reddit_signals}


def sentiment_node(state):
    sentiments = {}
    watchlist = state.get("watchlist", [])
    for ticker in watchlist:
        try:
            sentiments[ticker] = sentiment_agent.run(
                company_news=state.get("company_news", {}).get(ticker),
                reddit_signal=state.get("reddit_signals", {}).get(ticker),
                financial_data=state.get("financial_data", {}).get(ticker)
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
            macro_summary=state.get("macro_summary"),
            company_news=state.get("company_news", {}),
            market_data=state.get("market_data", {}),
            financial_data=state.get("financial_data", {}),
            reddit_signals=state.get("reddit_signals", {}),
            sentiments=state.get("sentiments", {})
        )
    except Exception as e:
        print(f"[Graph Warning] ReportAgent failed: {e}")
        from src.models.report import DailyMarketReport
        from src.models.macro import MacroSummary
        from src.tools.common.normalization import utc_now_iso
        import datetime
        report = DailyMarketReport(
            date=datetime.date.today().isoformat(),
            overall_market_sentiment="Unknown",
            overall_confidence=0,
            macro_summary=MacroSummary(overall_sentiment="Unknown", confidence=0, summary="Report generation failed.", key_drivers=[], market_events=[], last_updated="Unknown", market_data={}),
            generated_at=utc_now_iso()
        )
    return {"report": report}