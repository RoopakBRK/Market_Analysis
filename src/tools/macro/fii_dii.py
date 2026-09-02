import requests
from langchain_core.tools import tool
from src.tools.common.scraper_utils import make_headers

@tool
def get_fii_dii_flows() -> dict:
    """
    Fetch the latest FII and DII cash market activity.
    """
    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=make_headers(), timeout=10)
        
        url = "https://www.nseindia.com/api/fiidiiTradeReact"
        response = session.get(url, headers=make_headers(), timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        fii_net = 0.0
        dii_net = 0.0
        date = ""
        
        if isinstance(data, list):
            for item in data:
                if item.get("category") == "DII":
                    dii_net = float(item.get("netValue", 0.0) or 0.0)
                    date = item.get("date", "")
                elif item.get("category") == "FII/FPI":
                    fii_net = float(item.get("netValue", 0.0) or 0.0)
                    date = item.get("date", "")
        
        return {
            "date": date,
            "fii_net": fii_net,
            "dii_net": dii_net,
            "unit": "INR Crore",
            "source": "NSE",
        }
    except Exception:
        # Graceful degradation on network failures
        return {
            "date": "",
            "fii_net": 0.0,
            "dii_net": 0.0,
            "unit": "INR Crore",
            "source": "NSE",
        }