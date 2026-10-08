from langchain_core.tools import tool
from src.tools.common.yahoo_finance import get_latest_quote

# Yahoo symbol -> display name of the NSE sector index used as the benchmark.
_INDEX_NAMES = {
    "^CNXAUTO": "Nifty Auto",
    "^NSEBANK": "Nifty Bank",
    "^CNXMETAL": "Nifty Metal",
    "^CNXINFRA": "Nifty Infrastructure",
    "^CNXIT": "Nifty IT",
    "NIFTY_FIN_SERVICE.NS": "Nifty Financial Services",
    "^CNXCONSUM": "Nifty India Consumption",
    "^CNXFMCG": "Nifty FMCG",
    "^CNXPHARMA": "Nifty Pharma",
    "^CNXENERGY": "Nifty Energy",
    "^CNXCMDT": "Nifty Commodities",
    "^CNXREALTY": "Nifty Realty",
    "^CNXMEDIA": "Nifty Media",
}

# Yahoo's sectors are broader than NSE's indices ("Consumer Cyclical" covers
# both carmakers and retailers), so the industry is checked first.
_INDUSTRY_KEYWORD_INDEX = [
    ("auto", "^CNXAUTO"),
    ("bank", "^NSEBANK"),
    ("steel", "^CNXMETAL"),
    ("metal", "^CNXMETAL"),
    ("alumin", "^CNXMETAL"),
    ("copper", "^CNXMETAL"),
    ("telecom", "^CNXINFRA"),
]

_SECTOR_INDEX = {
    "technology": "^CNXIT",
    "financial services": "NIFTY_FIN_SERVICE.NS",
    "consumer cyclical": "^CNXCONSUM",
    "consumer defensive": "^CNXFMCG",
    "healthcare": "^CNXPHARMA",
    "energy": "^CNXENERGY",
    "utilities": "^CNXENERGY",
    "basic materials": "^CNXCMDT",
    "industrials": "^CNXINFRA",
    "real estate": "^CNXREALTY",
    "communication services": "^CNXMEDIA",
}


def _benchmark_index(sector: str, industry: str) -> str | None:
    industry_l = industry.lower()
    for keyword, symbol in _INDUSTRY_KEYWORD_INDEX:
        if keyword in industry_l:
            return symbol
    return _SECTOR_INDEX.get(sector.lower().strip())


@tool
def get_sector_performance(sector: str, industry: str = "") -> dict:
    """
    Fetch today's move of the NSE sector index that best matches a company's
    sector/industry. 'performance' is None when no index matches or the quote
    is unavailable.
    """
    symbol = _benchmark_index(sector, industry)
    quote = get_latest_quote(symbol) if symbol else None

    return {
        "sector": sector,
        "industry": industry,
        "index": _INDEX_NAMES[symbol] if symbol else None,
        "performance": round(quote["change_percent"], 2) if quote else None,
        "source": "Yahoo Finance",
    }
