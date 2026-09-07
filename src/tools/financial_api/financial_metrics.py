from langchain_core.tools import tool
from src.tools.financial_api import client as fin_client


@tool
def get_financial_metrics(ticker: str) -> dict:
    """
    Retrieve financial metrics from the Financial Agent API.

    Returns a dict with metrics, or an empty dict if unavailable.
    All financial values come directly from the API — the LLM must not
    modify or reinterpret raw numeric values.

    The Financial Agent API is currently a stub adapter — returns {} until
    the provider contract is documented and implemented.
    """
    data = fin_client.fetch_financial_metrics(ticker)
    if not data:
        return {"ticker": ticker, "available": False}

    # Only extract known/expected keys — do not pass raw provider dict to LLM.
    # Missing values stay as None (not 0).
    return {
        "ticker": ticker,
        "available": True,
        "market_cap": data.get("market_cap"),
        "pe_ratio": data.get("pe_ratio"),
        "eps": data.get("eps"),
        "revenue": data.get("revenue"),
        "debt_to_equity": data.get("debt_to_equity"),
    }
