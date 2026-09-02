import requests

from langchain_core.tools import tool


@tool
def search_mint_news(company: str) -> dict:
    """
    Search Mint for company-specific news.
    """

    # TODO:
    # Replace with scraping/API implementation.

    return {
        "company": company,
        "articles": [],
        "source": "Mint",
    }