import requests
from langchain_core.tools import tool

@tool
def get_inflation_data() -> dict:
    """
    Fetch the latest inflation data.
    """
    # Hardcoded as requested
    return {
        "cpi": 4.45,
        "wpi": 0.53,
        "trend": "Stable",
        "source": "MOSPI",
    }