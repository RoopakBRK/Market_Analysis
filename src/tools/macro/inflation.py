import requests

from langchain_core.tools import tool


@tool
def get_inflation_data() -> dict:
    """
    Fetch the latest inflation data.
    """

    # TODO:
    # Replace with a real API implementation.

    return {
        "cpi": 4.21,
        "wpi": 1.84,
        "trend": "Cooling",
        "source": "MOSPI",
    }