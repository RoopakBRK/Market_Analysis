from typing import Optional, Dict, Any
from langchain_core.tools import tool
from src.tools.common.yahoo_finance import get_latest_quote
from datetime import date

@tool
def get_us_market_summary() -> Optional[Dict[str, Any]]:
    """
    Fetch the latest US market performance.
    This tool takes NO arguments.
    Do not provide a date, ticker, or any other arguments.
    The tool automatically determines the latest available
    US market data.
    """
    indices = {
        "^GSPC": "sp500",
        "^IXIC": "nasdaq",
        "^DJI": "dow_jones"
    }
    results = {}
    
    avg_change = 0.0
    valid_count = 0
    
    for symbol, name in indices.items():
        quote = get_latest_quote(symbol)
        if quote:
            results[name] = {
                "current": round(quote["price"], 2),
                "previous_close": round(quote["previous_close"], 2),
                "change": round(quote["change"], 2),
                "change_percent": round(quote["change_percent"], 2)
            }
            avg_change += quote["change_percent"]
            valid_count += 1
            
    if not results:
        return None
        
    avg_change = avg_change / valid_count
    overall_sentiment = "Bullish" if avg_change > 0 else "Bearish" if avg_change < 0 else "Neutral"
    
    # Merge the dicts into the root response
    response = {
        "date": date.today().isoformat(),
        "sp500": results.get("sp500", {}),
        "nasdaq": results.get("nasdaq", {}),
        "dow_jones": results.get("dow_jones", {}),
        "overall_sentiment": overall_sentiment,
        "source": "Yahoo Finance",
    }
    return response