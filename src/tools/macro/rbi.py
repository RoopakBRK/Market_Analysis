import requests

from langchain_core.tools import tool


@tool
def get_rbi_updates() -> dict:
    """
    Fetch the latest RBI announcements.
    """

    # TODO:
    # Replace with RSS feed, RBI API, or web scraping.

    return {
        "headline": "No major RBI announcement today.",
        "policy_rate": 6.00,
        "impact": "Neutral",
        "source": "RBI",
    }