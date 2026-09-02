import requests

from langchain_core.tools import tool


@tool
def search_reuters_news(company: str) -> dict:
    """
    Search Reuters for company-specific news.
    """

    # TODO:
    # Replace with Reuters API or scraping implementation.

    return {
        "company": company,
        "articles": [],
        "source": "Reuters",
    }