import requests

from langchain_core.tools import tool


@tool
def get_gold_price() -> dict:
    """
    Fetch the latest gold price.
    """

    # TODO:
    # Replace with a real API implementation.

    return {
        "price": 3525.40,
        "unit": "INR/Gram",
        "change_percent": 0.62,
        "trend": "Bullish",
        "source": "MCX",
    }