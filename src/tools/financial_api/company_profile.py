from langchain_core.tools import tool
from src.tools.financial_api import client as fin_client


@tool
def get_company_profile(ticker: str) -> dict:
    """
    Retrieve basic company profile from the Financial Agent API.

    Returns a dict with profile fields, or an empty dict if unavailable.
    The Financial Agent API is currently a stub adapter — returns {} until
    the provider contract is documented and implemented.
    """
    data = fin_client.fetch_company_profile(ticker)
    if not data:
        return {"ticker": ticker, "available": False}

    return {
        "ticker": ticker,
        "available": True,
        "company_name": data.get("company_name"),
        "sector": data.get("sector"),
        "industry": data.get("industry"),
        "description": data.get("description", ""),
    }
