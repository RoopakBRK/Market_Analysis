from langchain_core.tools import tool
from src.tools.tavily import client as tavily_client


@tool
def search_sector_news_tavily(sector: str) -> dict:
    """
    Retrieve recent sector-level news using Tavily.

    Useful for providing sector context when evaluating individual companies.

    Returns a dict with 'sector' and 'articles' (list of normalised dicts).
    Returns an empty articles list gracefully if Tavily is unavailable.
    """
    query = f"{sector} sector India stock market news"
    results = tavily_client.search(query=query, max_results=5, days_back=3)
    return {
        "sector": sector,
        "articles": results,
        "source": "Tavily",
    }
