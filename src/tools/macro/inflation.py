from langchain_core.tools import tool
from src.tools.common.normalization import deduplicate_article_dicts
from src.tools.firecrawl import client as firecrawl_client
from src.tools.tavily import client as tavily_client


@tool
def get_inflation_data() -> dict:
    """
    Fetch recent news coverage of India's CPI / WPI inflation prints.

    No structured, reachable source for the official figures is wired up
    (MOSPI's API is not usable from here), so this returns dated news
    articles rather than numbers — it must never report a fixed value as if
    it were the latest reading.

    Returns a dict with 'articles' (list of normalised dicts) from Firecrawl
    and Tavily combined; the list is empty if neither is available.
    """
    query = "India CPI WPI inflation data"
    # CPI and WPI are published monthly, so look back further than daily news.
    results = deduplicate_article_dicts(
        firecrawl_client.search(query=query, max_results=3, days_back=35)
        + tavily_client.search(query=query, max_results=3, days_back=35)
    )
    return {
        "articles": results,
        "source": "Firecrawl, Tavily",
    }
