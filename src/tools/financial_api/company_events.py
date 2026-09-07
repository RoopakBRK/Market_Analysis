from langchain_core.tools import tool
from src.tools.financial_api import client as fin_client


@tool
def get_company_events(ticker: str) -> dict:
    """
    Retrieve upcoming corporate events from the Financial Agent API.

    Returns a dict with 'events' (list of event strings), or empty if unavailable.
    Examples: earnings date, AGM, dividend, regulatory deadlines.

    The Financial Agent API is currently a stub adapter — returns {} until
    the provider contract is documented and implemented.
    """
    data = fin_client.fetch_company_events(ticker)
    if not data:
        return {"ticker": ticker, "available": False, "events": []}

    raw_events = data.get("events", [])
    # Normalise each event to a plain string for safe LLM consumption.
    events = []
    for ev in raw_events:
        if isinstance(ev, str):
            events.append(ev)
        elif isinstance(ev, dict):
            # Best-effort: combine type + date + description
            parts = [ev.get("type", ""), ev.get("date", ""), ev.get("description", "")]
            events.append(" — ".join(p for p in parts if p))

    return {
        "ticker": ticker,
        "available": True,
        "events": events,
    }
