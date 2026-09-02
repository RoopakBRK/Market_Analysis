import requests

from langchain_core.tools import tool


@tool
def get_stock_price(ticker: str) -> dict:
    """
    Fetch the latest stock price.
    """

    # TODO:
    # Integrate with Yahoo Finance / NSE / Polygon / etc.

    return {
        "ticker": ticker,
        "current_price": 1524.60,
        "previous_close": 1511.20,
        "change_percent": 0.88,
    }