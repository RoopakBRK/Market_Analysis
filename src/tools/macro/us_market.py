import requests
from langchain_core.tools import tool
from src.tools.common.scraper_utils import make_headers

@tool
def get_us_market_summary() -> dict:
    """
    Fetch the latest US market performance.
    """
    try:
        # Fetch S&P 500 (^GSPC), Nasdaq (^IXIC), Dow (^DJI) using Yahoo Finance
        indices = ["^GSPC", "^IXIC", "^DJI"]
        results = {}
        
        for idx in indices:
            url = f"https://query2.finance.yahoo.com/v8/finance/chart/{idx}?interval=1d&range=2d"
            response = requests.get(url, headers=make_headers(), timeout=10)
            response.raise_for_status()
            
            data = response.json()
            chart_result = data.get("chart", {}).get("result", [])
            
            if chart_result:
                meta = chart_result[0].get("meta", {})
                current_price = meta.get("regularMarketPrice")
                previous_close = meta.get("previousClose")
                
                if current_price is not None and previous_close is not None:
                    change_percent = ((current_price - previous_close) / previous_close) * 100 if previous_close else 0
                    results[idx] = change_percent
                    
        sp500 = results.get("^GSPC", 0.0)
        nasdaq = results.get("^IXIC", 0.0)
        dow_jones = results.get("^DJI", 0.0)
        
        avg_change = (sp500 + nasdaq + dow_jones) / 3
        overall_sentiment = "Bullish" if avg_change > 0 else "Bearish" if avg_change < 0 else "Neutral"
        
        return {
            "date": "",
            "sp500": round(sp500, 2),
            "nasdaq": round(nasdaq, 2),
            "dow_jones": round(dow_jones, 2),
            "overall_sentiment": overall_sentiment,
            "source": "Yahoo Finance",
        }
    except requests.RequestException:
        # Graceful degradation on network failures
        return {
            "date": "",
            "sp500": 0.0,
            "nasdaq": 0.0,
            "dow_jones": 0.0,
            "overall_sentiment": "Unknown",
            "source": "Yahoo Finance",
        }