import requests
from langchain_core.tools import tool
from src.tools.common.scraper_utils import make_headers

@tool
def get_usd_inr_rate() -> dict:
    """
    Fetch the latest USD/INR exchange rate.
    """
    try:
        # Fetch USD/INR (INR=X) using Yahoo Finance
        url = "https://query2.finance.yahoo.com/v8/finance/chart/INR=X?interval=1d&range=2d"
        response = requests.get(url, headers=make_headers(), timeout=10)
        response.raise_for_status()
        
        data = response.json()
        result = data.get("chart", {}).get("result", [])
        
        if not result:
            raise ValueError("No exchange rate data found")
            
        meta = result[0].get("meta", {})
        current_rate = meta.get("regularMarketPrice")
        previous_close = meta.get("previousClose")
        
        if current_rate is None or previous_close is None:
            raise ValueError("Incomplete exchange rate data received from Yahoo Finance")
            
        change_percent = ((current_rate - previous_close) / previous_close) * 100 if previous_close else 0
        trend = "Weakening INR" if change_percent > 0 else "Strengthening INR"
        
        return {
            "exchange_rate": round(current_rate, 2),
            "change_percent": round(change_percent, 2),
            "trend": trend,
            "source": "Yahoo Finance",
        }
    except requests.RequestException:
        # Graceful degradation on network failures
        return {
            "exchange_rate": 0.0,
            "change_percent": 0.0,
            "trend": "Unknown",
            "source": "Yahoo Finance",
        }