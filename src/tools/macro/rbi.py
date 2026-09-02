import requests
from langchain_core.tools import tool

@tool
def get_rbi_updates() -> dict:
    """
    Fetch the latest RBI announcements.
    """
    # Hardcoded as requested
    return {
        "headline": "RBI maintains repo rate at 5.25%",
        "policy_rate": 5.25,
        "impact": "Neutral",
        "source": "RBI",
    }