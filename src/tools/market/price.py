from typing import Optional, Dict, Any
from langchain_core.tools import tool
from src.tools.common.yahoo_finance import get_latest_quote

@tool
def get_stock_price(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Fetch the latest stock price for a given ticker.
    """
    quote = get_latest_quote(ticker.upper())
    if not quote:
        return None
        
    return {
        "ticker": ticker,
        "current_price": round(quote["price"], 2),
        "previous_close": round(quote["previous_close"], 2),
        "change_percent": round(quote["change_percent"], 2),
        "source": "Yahoo Finance"
    }