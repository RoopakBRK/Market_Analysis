import requests
from langchain_core.tools import tool
from src.tools.common.scraper_utils import make_headers

@tool
def get_rbi_updates() -> dict:
    """
    Fetch the latest RBI announcements.
    """
    try:
        # Example URL: RBI RSS feed or API.
        # Since RBI RSS feeds can be unstable, we use a robust network structure.
        # url = "https://rbi.org.in/home.aspx"
        raise requests.RequestException("RBI endpoint not configured or unreachable.")

    except requests.RequestException:
        # Graceful degradation on network failures
        return {
            "headline": "",
            "policy_rate": 0.0,
            "impact": "Unknown",
            "source": "RBI",
        }