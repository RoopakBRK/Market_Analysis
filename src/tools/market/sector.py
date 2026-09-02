import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool


@tool
def get_sector_performance(sector: str) -> dict:
    """
    Fetch sector performance.
    """
    try:
        # A simple fallback scraper for sector performance
        url = "https://www.moneycontrol.com/stocks/marketstats/sector-scan/bse/today.html"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        tables = soup.find_all("table", class_="mctable1")
        
        performance = 0.0
        leader = "Unknown"
        
        # This is a very rudimentary fallback mapping to extract real data.
        # Ideally, we would map the exact sector name to Moneycontrol's sector list.
        if tables:
            rows = tables[0].find_all("tr")[1:5]
            for row in rows:
                cols = row.find_all("td")
                if len(cols) > 2:
                    sector_name = cols[0].text.strip()
                    if sector.lower() in sector_name.lower():
                        perf_text = cols[2].text.strip().replace("%", "")
                        performance = float(perf_text) if perf_text else 0.0
                        leader = cols[0].find("a").text.strip() if cols[0].find("a") else sector_name
                        break
        
        return {
            "sector": sector,
            "performance": performance,
            "leader": leader,
            "source": "Moneycontrol Sector Scan"
        }
    except requests.RequestException:
        # Graceful degradation on network failures
        return {
            "sector": sector,
            "performance": 0.0,
            "leader": "Unknown",
            "source": "Moneycontrol Sector Scan"
        }