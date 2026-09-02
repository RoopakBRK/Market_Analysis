from typing import Optional, Dict, Any
from langchain_core.tools import tool
from src.tools.common.yahoo_finance import get_latest_quote

@tool
def get_usd_inr_rate() -> Optional[Dict[str, Any]]:
    """
    Fetch the latest USD/INR exchange rate.
    """
    quote = get_latest_quote("INR=X")
    if not quote:
        return None
        
    trend = "Weakening INR" if quote["change_percent"] > 0 else "Strengthening INR"
    
    return {
        "exchange_rate": round(quote["price"], 2),
        "change_percent": round(quote["change_percent"], 2),
        "trend": trend,
        "source": "Yahoo Finance",
    }