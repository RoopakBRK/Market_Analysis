import re
from typing import Optional, Dict, Any

import requests
from langchain_core.tools import tool
from src.tools.common.scraper_utils import make_headers

_RBI_HOME = "https://www.rbi.org.in/"

# Label on the RBI home page's "Current Rates" panel -> key in our output.
_RATE_LABELS = {
    "Policy Repo Rate": "policy_repo_rate",
    "Standing Deposit Facility Rate": "standing_deposit_facility_rate",
    "Marginal Standing Facility Rate": "marginal_standing_facility_rate",
    "Bank Rate": "bank_rate",
    "CRR": "crr",
    "SLR": "slr",
}


@tool
def get_rbi_updates() -> Optional[Dict[str, Any]]:
    """
    Fetch the RBI's current policy rates and reserve ratios (in percent) from
    the "Current Rates" panel on rbi.org.in.
    Returns None if the page is unreachable or the repo rate cannot be read.
    """
    try:
        response = requests.get(_RBI_HOME, headers=make_headers(), timeout=15)
        response.raise_for_status()
    except requests.RequestException:
        return None

    text = re.sub(r"<[^>]+>", " ", response.text).replace("&nbsp;", " ")
    text = re.sub(r"\s+", " ", text)

    rates: Dict[str, Any] = {}
    for label, key in _RATE_LABELS.items():
        match = re.search(rf"\b{re.escape(label)}\s*:\s*([\d.]+)\s*%", text)
        if match:
            rates[key] = float(match.group(1))

    # Without the repo rate the panel wasn't parsed; report nothing rather
    # than a partial or stale picture.
    if "policy_repo_rate" not in rates:
        return None

    rates["unit"] = "percent"
    rates["source"] = "RBI"
    return rates
