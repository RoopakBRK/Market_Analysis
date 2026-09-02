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
    indices = ["^GSPC", "^IXIC", "^DJI"]
    results = {}
    
    for idx in indices:
        quote = get_latest_quote(idx)
        if quote:
            results[idx] = quote["change_percent"]
            
    if not results:
        return None
        
    sp500 = results.get("^GSPC", 0.0)
    nasdaq = results.get("^IXIC", 0.0)
    dow_jones = results.get("^DJI", 0.0)
    
    avg_change = (sp500 + nasdaq + dow_jones) / len(results)
    overall_sentiment = "Bullish" if avg_change > 0 else "Bearish" if avg_change < 0 else "Neutral"
    
    return {
        "date": date.today().isoformat(),
        "sp500": round(sp500, 2),
        "nasdaq": round(nasdaq, 2),
        "dow_jones": round(dow_jones, 2),
        "overall_sentiment": overall_sentiment,
        "source": "Yahoo Finance",
    }