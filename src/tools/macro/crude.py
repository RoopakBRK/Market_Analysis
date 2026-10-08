from typing import Optional, Dict, Any
from langchain_core.tools import tool
from src.tools.common.yahoo_finance import get_latest_quote

@tool
def get_crude_price() -> Optional[Dict[str, Any]]:
    """
    Fetch the latest Brent crude oil price.
    """
    quote = get_latest_quote("BZ=F")
    if not quote:
        return None
        
    change = quote["change_percent"]
    trend = "Rising" if change > 0 else "Falling" if change < 0 else "Flat"
    # India imports most of its crude, so a higher oil price is a headwind
    # (import bill, inflation, rupee) and a lower one a tailwind.
    india_impact = "Negative" if change > 0 else "Positive" if change < 0 else "Neutral"
    
    return {
        "price": round(quote["price"], 2),
        "previous_close": round(quote["previous_close"], 2),
        "change": round(quote["change"], 2),
        "change_percent": round(quote["change_percent"], 2),
        "unit": "USD/Barrel",
        "trend": trend,
        "impact_on_india": india_impact,
        "source": "Yahoo Finance",
    }