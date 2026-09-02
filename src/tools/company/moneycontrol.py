import requests

from langchain_core.tools import tool


@tool
def search_moneycontrol_news(company: str) -> dict:
    """
    Search Moneycontrol for company-specific news.
    """

    # TODO:
    # Replace with scraping/API implementation.

    return {
        "company": company,
        "articles": [],
        "source": "Moneycontrol",
    }