import requests

from langchain_core.tools import tool


@tool
def get_us_market_summary() -> dict:
    """
    Fetch the latest US market performance.
    """

    # TODO:
    # Replace with a real API implementation.

    return {
        "date": "2026-09-02",
        "sp500": 0.84,
        "nasdaq": 1.23,
        "dow_jones": 0.51,
        "overall_sentiment": "Bullish",
        "source": "Yahoo Finance",
    }