import requests
from src.tools.common.scraper_utils import make_headers
session = requests.Session()
session.get("https://www.nseindia.com", headers=make_headers(), timeout=10)
url = "https://www.nseindia.com/api/fiidiiTradeReact"
response = session.get(url, headers=make_headers(), timeout=10)
print(response.json())
