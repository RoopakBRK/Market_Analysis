from langchain_core.tools import tool
from src.tools.reddit import client as reddit_client

# Subreddits focused on Indian stocks and markets.
_INDIA_STOCK_SUBS = ["IndiaInvestments", "IndianStockMarket", "NSEIndia", "DalalStreet"]


@tool
def search_company_reddit(ticker: str, company: str) -> dict:
    """
    Search Reddit for discussions about a specific company/ticker.

    Searches Indian investment subreddits.
    Returns posts as community/alternative sentiment — NOT verified financial news.

    Returns a dict with 'ticker', 'company', and 'posts' (list of dicts).
    Returns empty posts list gracefully if Reddit is unavailable.
    """
    query = f"{ticker} OR {company}"
    posts = reddit_client.search_subreddit_posts(
        subreddits=_INDIA_STOCK_SUBS,
        query=query,
        limit=10,
        time_filter="week",
    )
    return {
        "ticker": ticker,
        "company": company,
        "posts": posts,
        "source_type": "reddit",
    }
