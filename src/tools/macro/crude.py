import requests
from langchain_core.tools import tool
from src.tools.common.scraper_utils import make_headers

@tool
def get_crude_price() -> dict:
    """
    Fetch the latest Brent crude oil price.
    """
    try:
        # Dummy endpoint - real implementations would use Yahoo Finance (BZ=F) or EIA
        url = "https://query2.finance.yahoo.com/v8/finance/chart/BZ=F?interval=1d&range=2d"
        response = requests.get(url, headers=make_headers(), timeout=10)
        response.raise_for_status()
        
        data = response.json()
        result = data.get("chart", {}).get("result", [])
        
        if not result:
            raise ValueError("No crude price data found")
            
        meta = result[0].get("meta", {})
        current_price = meta.get("regularMarketPrice")
        previous_close = meta.get("previousClose")
        
        if current_price is None or previous_close is None:
            raise ValueError("Incomplete price data received from Yahoo Finance")
            
        change_percent = ((current_price - previous_close) / previous_close) * 100 if previous_close else 0
        trend = "Bearish" if change_percent < 0 else "Bullish"
        
        return {
            "price": round(current_price, 2),
            "change_percent": round(change_percent, 2),
            "unit": "USD/Barrel",
            "trend": trend,
            "source": "Yahoo Finance",
        }
    except requests.RequestException:
        # Graceful degradation on network failures
        return {
            "price": 0.0,
            "change_percent": 0.0,
            "unit": "USD/Barrel",
            "trend": "Unknown",
            "source": "Yahoo Finance",
        }