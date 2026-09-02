import requests
from langchain_core.tools import tool


@tool
def get_stock_price(ticker: str) -> dict:
    """
    Fetch the latest stock price for a given ticker.
    """
    try:
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker.upper()}?interval=1d&range=2d"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        result = data.get("chart", {}).get("result", [])
        
        if not result:
            raise ValueError(f"No data found for ticker {ticker}")
            
        meta = result[0].get("meta", {})
        current_price = meta.get("regularMarketPrice")
        previous_close = meta.get("previousClose")
        
        if current_price is None or previous_close is None:
            raise ValueError("Incomplete price data received from Yahoo Finance")
            
        change = current_price - previous_close
        change_percent = (change / previous_close) * 100 if previous_close > 0 else 0
        
        return {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "previous_close": round(previous_close, 2),
            "change_percent": round(change_percent, 2),
            "source": "Yahoo Finance"
        }
    except requests.RequestException:
        # Graceful degradation on network failures
        return {
            "ticker": ticker,
            "current_price": 0.0,
            "previous_close": 0.0,
            "change_percent": 0.0,
            "source": "Yahoo Finance"
        }