import requests

from langchain_core.tools import tool


@tool
def get_investor_relations(company: str) -> dict:
    """
    Fetch latest Investor Relations updates.
    """

    # TODO:
    # Company website scraping/API.

    return {
        "company": company,
        "press_releases": [],
        "source": "Investor Relations",
    }