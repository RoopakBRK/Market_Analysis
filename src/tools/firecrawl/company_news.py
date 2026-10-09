from langchain_core.tools import tool
from src.tools.firecrawl import client as firecrawl_client


@tool
def search_company_news_firecrawl(company: str) -> dict:
    """
    Retrieve recent news articles for a specific company using Firecrawl.

    Returns a dict with 'company' and 'articles' (list of normalised dicts).
    Returns an empty articles list gracefully if Firecrawl is unavailable.
    """
    # Just the name, as with Tavily: words like "stock" / "NSE" / "latest"
    # pull in share-price pages instead.
    results = firecrawl_client.search(query=company, max_results=5, days_back=3)
    return {
        "company": company,
        "articles": results,
        "source": "Firecrawl",
    }
