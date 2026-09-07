from langchain_core.tools import tool
from src.tools.tavily import client as tavily_client


@tool
def search_macro_news_tavily(query: str) -> dict:
    """
    Retrieve recent macroeconomic news using Tavily.

    Used to augment the MacroAgent's deterministic data collection with
    qualitative news context (RBI updates, government actions, global events
    affecting Indian markets, etc.).

    Returns a dict with 'query' and 'articles' (list of normalised dicts).
    Returns an empty articles list gracefully if Tavily is unavailable.
    """
    enriched_query = f"{query} India economy market 2025"
    results = tavily_client.search(query=enriched_query, max_results=5, days_back=3)
    return {
        "query": query,
        "articles": results,
        "source": "Tavily",
    }
