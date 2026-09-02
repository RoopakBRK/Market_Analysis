from langchain_core.tools import tool


@tool
def get_technical_indicators(ticker: str) -> dict:
    """
    Fetch technical indicators.
    """

    # TODO:
    # Calculate using pandas-ta / ta library.

    return {
        "ticker": ticker,
        "rsi": 63.8,
        "macd": 4.22,
        "vwap": 1517.85,
    }