from src.agents.macro_agent import MacroAgent
from src.agents.company_news_agent import CompanyNewsAgent
from src.agents.market_data_agent import MarketDataAgent
from src.agents.sentiment_agent import SentimentAgent

sentiment_agent = SentimentAgent()
market_agent = MarketDataAgent()

company_agent = CompanyNewsAgent()


macro_agent = MacroAgent()


def macro_node(state):

    macro_summary = macro_agent.run()

    state["macro_summary"] = macro_summary

    return state

def company_news_node(state):

    watchlist = state.get("watchlist", [])

    news = {}

    for company in watchlist:
        news[company] = company_agent.run(company)

    state["company_news"] = news

    return state

def market_data_node(state):

    watchlist = state.get("watchlist", [])

    market_data = {}

    for ticker in watchlist:
        market_data[ticker] = market_agent.run(ticker)

    state["market_data"] = market_data

    return state

def sentiment_node(state):

    sentiments = {}

    for ticker in state["company_news"]:

        sentiments[ticker] = sentiment_agent.run(
            state["macro_summary"],
            state["company_news"][ticker],
            state["market_data"][ticker],
        )

    state["sentiments"] = sentiments

    return state