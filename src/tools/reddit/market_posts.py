from langchain_core.tools import tool
from src.tools.reddit import client as reddit_client

_INDIA_MARKET_SUBS = ["IndiaInvestments", "IndianStockMarket", "NSEIndia", "DalalStreet", "india"]


@tool
def search_market_reddit(query: str) -> dict:
    """
    Search Reddit for general market/economic discussions.

    Useful for detecting emerging retail-investor narratives around the
    broader Indian market (NIFTY, SENSEX, sector rotations, etc.).

    Returns a dict with 'query' and 'posts' (list of dicts).
    Returns empty posts list gracefully if Reddit is unavailable.
    """
    posts = reddit_client.search_subreddit_posts(
        subreddits=_INDIA_MARKET_SUBS,
        query=query,
        limit=10,
        time_filter="week",
    )
    return {
        "query": query,
        "posts": posts,
        "source_type": "reddit",
    }
