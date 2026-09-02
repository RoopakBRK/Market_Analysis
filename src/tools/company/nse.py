import requests

from langchain_core.tools import tool


@tool
def get_nse_announcements(company: str) -> dict:
    """
    Fetch latest NSE announcements for a company.
    """

    # TODO:
    # Replace with NSE implementation.

    return {
        "company": company,
        "announcements": [],
        "source": "NSE",
    }