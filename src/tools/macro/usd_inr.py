import requests

from langchain_core.tools import tool


@tool
def get_usd_inr_rate() -> dict:
    """
    Fetch the latest USD/INR exchange rate.
    """

    # TODO:
    # Replace with a real API implementation.

    return {
        "exchange_rate": 83.17,
        "change_percent": -0.18,
        "trend": "Strengthening INR",
        "source": "RBI",
    }