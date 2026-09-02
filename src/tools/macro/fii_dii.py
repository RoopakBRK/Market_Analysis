import requests

from langchain_core.tools import tool


@tool
def get_fii_dii_flows() -> dict:
    """
    Fetch the latest FII and DII cash market activity.

    Returns:
        dict: Latest FII/DII flow information.
    """

    # TODO:
    # Replace this placeholder with NSE or other reliable data source.

    return {
        "date": "2026-09-02",
        "fii_net": 4523.41,
        "dii_net": -1892.34,
        "unit": "INR Crore",
        "source": "NSE",
    }