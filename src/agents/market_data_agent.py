import requests
from typing import TypedDict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from langgraph.graph import StateGraph, START, END
from pydantic import ValidationError

from src.models.market import MarketData
from src.tools.market.price import get_stock_price
from src.tools.market.indicators import get_technical_indicators
from src.tools.market.sector import get_sector_performance


class MarketDataState(TypedDict):
    ticker: str
    price_data: dict | None
    tech_data: dict | None
    sector_data: dict | None
    final_result: MarketData | None


def get_sector_for_ticker(ticker: str) -> str:
    """Fetch the actual sector name for a ticker from Yahoo Finance."""
    url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=assetProfile"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        result = data.get("quoteSummary", {}).get("result", [])
        if result and result[0]:
            sector = result[0].get("assetProfile", {}).get("sector")
            if sector:
                return sector
    except Exception:
        pass
    return ""


class MarketDataAgent:
    def __init__(self):
        # Graph Builder
        builder = StateGraph(MarketDataState)
        
        builder.add_node("collect_data_node", self.collect_data)
        builder.add_node("merge_data_node", self.merge_data)
        
        builder.add_edge(START, "collect_data_node")
        builder.add_edge("collect_data_node", "merge_data_node")
        builder.add_edge("merge_data_node", END)
        
        self.graph = builder.compile()

    def collect_data(self, state: MarketDataState):
        """Deterministically collect market data from tools without an LLM."""
        ticker = state["ticker"]
        
        # Dynamically determine the real sector string to avoid semantic misuse of the ticker
        resolved_sector = get_sector_for_ticker(ticker)
        
        # Define the exact execution arguments based on the tool signatures
        execution_plan = [
            ("price_data", get_stock_price, {"ticker": ticker}),
            ("tech_data", get_technical_indicators, {"ticker": ticker}),
            ("sector_data", get_sector_performance, {"sector": resolved_sector}),
        ]
        
        updates = {}
        
        def run_tool(state_key, tool_func, kwargs):
            try:
                # Call tool directly using .invoke with the correct schema kwargs
                result = tool_func.invoke(kwargs)
                return state_key, result
            except Exception as e:
                return state_key, {"error": str(e)}

        with ThreadPoolExecutor(max_workers=len(execution_plan)) as executor:
            futures = [executor.submit(run_tool, key, t, kw) for key, t, kw in execution_plan]
            for future in as_completed(futures):
                key, result = future.result()
                updates[key] = result
                
        return updates

    def merge_data(self, state: MarketDataState):
        """Merge raw tool dictionaries and validate with Pydantic."""
        ticker = state["ticker"]
        price_data = state.get("price_data") or {}
        tech_data = state.get("tech_data") or {}
        sector_data = state.get("sector_data") or {}
        
        merged: dict[str, Any] = {"ticker": ticker}
        
        # Merge Price Data
        if "error" not in price_data:
            merged["current_price"] = price_data.get("current_price")
            merged["previous_close"] = price_data.get("previous_close")
            merged["day_change_percent"] = price_data.get("change_percent")
            
        # Merge Technical Indicators
        if "error" not in tech_data:
            merged["rsi"] = tech_data.get("rsi")
            merged["macd"] = tech_data.get("macd")
            merged["vwap"] = tech_data.get("vwap")
            
        # Merge Sector Performance
        if "error" not in sector_data:
            merged["sector"] = sector_data.get("sector")
            merged["sector_performance"] = sector_data.get("performance")
            
        try:
            result = MarketData.model_validate(merged)
            return {"final_result": result}
        except ValidationError as e:
            raise ValueError(f"MarketData validation failed on deterministic merge.\\nError: {e}\\nMerged dict: {merged}") from e

    def run(self, ticker: str) -> MarketData:
        initial_state: MarketDataState = {
            "ticker": ticker,
            "price_data": None,
            "tech_data": None,
            "sector_data": None,
            "final_result": None
        }
        state = self.graph.invoke(initial_state)
        # LangGraph invoke returns the final state dict, or we can fallback if None
        return state.get("final_result") or MarketData(ticker=ticker)