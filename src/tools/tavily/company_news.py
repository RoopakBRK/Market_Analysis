from langchain_core.tools import tool
from src.tools.tavily import client as tavily_client


@tool
def search_company_news_tavily(company: str) -> dict:
    """
    Retrieve recent news articles for a specific company using Tavily.

    Returns a dict with 'company' and 'articles' (list of normalised dicts).
    Returns an empty articles list gracefully if Tavily is unavailable.
    """
    query = f"{company} stock news India NSE latest"
    results = tavily_client.search(query=query, max_results=5, days_back=3)
    return {
        "company": company,
        "articles": results,
        "source": "Tavily",
    }
