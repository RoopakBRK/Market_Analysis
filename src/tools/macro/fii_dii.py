import requests
from langchain_core.tools import tool
from src.tools.common.scraper_utils import make_headers

@tool
def get_fii_dii_flows() -> dict:
    """
    Fetch the latest FII and DII cash market activity.
    """
    try:
        # Note: True NSE API requires cookies. We fetch gracefully if cookies fail.
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=make_headers(), timeout=10)
        
        url = "https://www.nseindia.com/api/fiidiiTradeReact"
        response = session.get(url, headers=make_headers(), timeout=10)
        response.raise_for_status()
        
        data = response.json()
        # Data format is typically a list, we pick the first/latest entry
        latest = data[0] if isinstance(data, list) and len(data) > 0 else {}
        
        fii_net = float(latest.get("fii_net", 0.0) or 0.0)
        dii_net = float(latest.get("dii_net", 0.0) or 0.0)
        date = latest.get("date", "")
        
        return {
            "date": date,
            "fii_net": fii_net,
            "dii_net": dii_net,
            "unit": "INR Crore",
            "source": "NSE",
        }
    except requests.RequestException:
        # Graceful degradation on network failures
        return {
            "date": "",
            "fii_net": 0.0,
            "dii_net": 0.0,
            "unit": "INR Crore",
            "source": "NSE",
        }