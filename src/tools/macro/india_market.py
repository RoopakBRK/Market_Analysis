from typing import Optional, Dict, Any
from langchain_core.tools import tool
from src.tools.common.yahoo_finance import get_latest_quote
from datetime import date

@tool
def get_india_market_summary() -> Optional[Dict[str, Any]]:
    """
    Fetch the latest levels of India's benchmark indices (Nifty 50 and Sensex).
    This tool takes NO arguments.
    """
    indices = {
        "^NSEI": "nifty50",
        "^BSESN": "sensex",
    }
    results = {}

    for symbol, name in indices.items():
        quote = get_latest_quote(symbol)
        if quote:
            results[name] = {
                "current": round(quote["price"], 2),
                "previous_close": round(quote["previous_close"], 2),
                "change": round(quote["change"], 2),
                "change_percent": round(quote["change_percent"], 2)
            }

    if not results:
        return None

    avg_change = sum(r["change_percent"] for r in results.values()) / len(results)
    trend = "Rising" if avg_change > 0 else "Falling" if avg_change < 0 else "Flat"

    return {
        "date": date.today().isoformat(),
        "nifty50": results.get("nifty50", {}),
        "sensex": results.get("sensex", {}),
        "trend": trend,
        "source": "Yahoo Finance",
    }
