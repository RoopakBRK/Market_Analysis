import requests
from typing import Optional, Dict
from src.tools.common.scraper_utils import make_headers

_session = requests.Session()
_session.headers.update(make_headers())

def get_latest_quote(symbol: str) -> Optional[Dict[str, float]]:
    """
    Fetch the latest quote data for a symbol from Yahoo Finance.
    Returns a dict with 'price', 'previous_close', 'change', and 'change_percent'.
    Returns None gracefully on network failure, empty response, or malformed JSON.
    """
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=5d"
    
    try:
        response = _session.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        result = data.get("chart", {}).get("result", [])
        if not result:
            return None
            
        result_data = result[0]
        indicators = result_data.get("indicators", {}).get("quote", [])
        if not indicators:
            return None
            
        close_prices = indicators[0].get("close", [])
        
        # Filter out None values that Yahoo sometimes injects
        valid_closes = [p for p in close_prices if p is not None]
        if not valid_closes:
            return None
            
        current_price = valid_closes[-1]
        
        # Get previous close from meta if it exists, otherwise use the previous array value
        meta = result_data.get("meta", {})
        previous_close = meta.get("previousClose")
        if previous_close is None and len(valid_closes) > 1:
            previous_close = valid_closes[-2]
        elif previous_close is None:
            previous_close = current_price
            
        change = current_price - previous_close
        change_percent = (change / previous_close) * 100 if previous_close > 0 else 0.0
        
        return {
            "price": float(current_price),
            "previous_close": float(previous_close),
            "change": float(change),
            "change_percent": float(change_percent)
        }
    except Exception:
        return None
