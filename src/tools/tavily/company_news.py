from langchain_core.tools import tool
from src.tools.tavily import client as tavily_client


@tool
def search_company_news_tavily(company: str) -> dict:
    """
    Retrieve recent news articles for a specific company using Tavily.

    Returns a dict with 'company' and 'articles' (list of normalised dicts).
    Returns an empty articles list gracefully if Tavily is unavailable.
    """
    # Just the name: the search is already scoped to recent news, and words
    # like "stock" / "NSE" / "latest" pull in share-price pages instead.
    query = company
    results = tavily_client.search(query=query, max_results=5, days_back=3)
    return {
        "company": company,
        "articles": results,
        "source": "Tavily",
    }
