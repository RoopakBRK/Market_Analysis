import json
import sys

# Append the project root to sys.path to allow imports to work natively
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.macro.crude import get_crude_price
from src.tools.macro.fii_dii import get_fii_dii_flows
from src.tools.macro.gold import get_gold_price
from src.tools.macro.inflation import get_inflation_data
from src.tools.macro.rbi import get_rbi_updates
from src.tools.macro.us_market import get_us_market_summary
from src.tools.macro.usd_inr import get_usd_inr_rate

TOOLS = [
    get_crude_price,
    get_fii_dii_flows,
    get_gold_price,
    get_inflation_data,
    get_rbi_updates,
    get_us_market_summary,
    get_usd_inr_rate,
]

def format_data(data):
    if data is None:
        return "None"
    if not isinstance(data, dict):
        return str(data)
    try:
        return json.dumps(data, indent=2)
    except Exception:
        return str(data)

def is_empty(data):
    if not data:
        return True
    
    # Check if a dict contains only None, 0.0, or "Unknown" / empty strings
    if isinstance(data, dict):
        all_empty = True
        for k, v in data.items():
            # Source tags don't count as real data
            if k == "source" or k == "title":
                continue
            
            # Check for meaningful values
            if v and v not in (0, 0.0, "Unknown", "unknown", "N/A", "n/a", "Not Found"):
                all_empty = False
                break
        return all_empty
    return False

def run_tests():
    for tool in TOOLS:
        print("=" * 70)
        print(f"TOOL: {tool.name}")
        
        try:
            # Langchain @tool wrapper allows direct invocation via .invoke
            result = tool.invoke({})
            
            if is_empty(result):
                status = "EMPTY"
            else:
                status = "SUCCESS"
                
            print(f"STATUS: {status}")
            print("=" * 70)
            print(format_data(result))
            
        except Exception as e:
            import traceback
            print(f"STATUS: ERROR")
            print("=" * 70)
            print(f"{type(e).__name__}: {str(e)}")
            traceback.print_exc(file=sys.stdout)
            
        print("\\n")

if __name__ == "__main__":
    run_tests()
