from langchain_core.tools import tool
from src.tools.firecrawl import client as firecrawl_client


@tool
def search_macro_news_firecrawl(query: str = "India economy markets RBI rupee") -> dict:
    """
    Retrieve recent macroeconomic news using Firecrawl.

    Runs alongside the Tavily macro search: the two draw on different search
    indexes, and the MacroAgent drops any article both return.

    Returns a dict with 'query' and 'articles' (list of normalised dicts).
    Returns an empty articles list gracefully if Firecrawl is unavailable.
    """
    # Sent as it is, and kept short: Firecrawl's news search returned nothing
    # for the nine-word query the Tavily tool builds.
    results = firecrawl_client.search(query=query, max_results=5, days_back=3)
    return {
        "query": query,
        "articles": results,
        "source": "Firecrawl",
    }
