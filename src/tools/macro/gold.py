from typing import Optional, Dict, Any
from langchain_core.tools import tool
from src.tools.common.yahoo_finance import get_latest_quote

@tool
def get_gold_price() -> Optional[Dict[str, Any]]:
    """
    Fetch the latest gold price.
    """
    quote = get_latest_quote("GC=F")
    if not quote:
        return None
        
    trend = "Bullish" if quote["change_percent"] > 0 else "Bearish"
    
    return {
        "price": round(quote["price"], 2),
        "change_percent": round(quote["change_percent"], 2),
        "unit": "USD/Oz",
        "trend": trend,
        "source": "Yahoo Finance",
    }