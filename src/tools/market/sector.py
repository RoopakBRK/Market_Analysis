from langchain_core.tools import tool


@tool
def get_sector_performance(sector: str) -> dict:
    """
    Fetch sector performance.
    """

    # TODO:
    # Replace with live market implementation.

    return {
        "sector": sector,
        "performance": 1.84,
        "leader": "Reliance",
    }