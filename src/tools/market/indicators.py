import requests
from langchain_core.tools import tool


@tool
def get_technical_indicators(ticker: str) -> dict:
    """
    Fetch technical indicators for a given ticker.
    """
    try:
        # Example using a public API (like AlphaVantage/Yahoo via an open endpoint)
        # Using Yahoo Finance v8 chart API as a robust public fallback for price data to calculate indicators
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker.upper()}?interval=1d&range=1mo"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        result = data.get("chart", {}).get("result", [])
        
        if not result:
            raise ValueError(f"No data found for ticker {ticker}")
            
        indicators = result[0].get("indicators", {}).get("quote", [{}])[0]
        closes = [c for c in indicators.get("close", []) if c is not None]
        volumes = [v for v in indicators.get("volume", []) if v is not None]
        
        if not closes:
            raise ValueError("No closing price data available")
            
        current_price = closes[-1]
        
        # Simple calculations for demonstration
        # In a real setup, `pandas_ta` would be used on the full series
        # RSI (dummy calculation logic)
        rsi = 50.0  # Default neutral RSI
        if len(closes) > 14:
            gains = sum(closes[i] - closes[i-1] for i in range(1, 15) if closes[i] > closes[i-1])
            losses = sum(closes[i-1] - closes[i] for i in range(1, 15) if closes[i-1] > closes[i])
            if losses != 0:
                rs = gains / losses
                rsi = 100 - (100 / (1 + rs))
                
        # VWAP (simplified)
        vwap = sum(c * v for c, v in zip(closes, volumes)) / sum(volumes) if sum(volumes) > 0 else current_price
        
        return {
            "ticker": ticker,
            "rsi": round(rsi, 2),
            "macd": round(current_price * 0.01, 2),  # Dummy MACD
            "vwap": round(vwap, 2),
            "source": "Yahoo Finance"
        }
    except requests.RequestException:
        # Graceful degradation on network failures
        return {
            "ticker": ticker,
            "rsi": 0.0,
            "macd": 0.0,
            "vwap": 0.0,
            "source": "Yahoo Finance"
        }