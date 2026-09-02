import requests

from langchain_core.tools import tool


@tool
def get_crude_price() -> dict:
    """
    Fetch the latest Brent crude oil price.
    """

    # TODO:
    # Replace with a real API implementation.

    return {
        "price": 74.83,
        "change_percent": -2.15,
        "unit": "USD/Barrel",
        "trend": "Bearish",
        "source": "Trading Economics",
    }