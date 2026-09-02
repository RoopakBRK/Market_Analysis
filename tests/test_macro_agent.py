import json
from src.agents.macro_agent import MacroAgent

def test_macro():
    print("\n" + "=" * 60)
    print("MACRO AGENT TEST")
    print("=" * 60)

    agent = MacroAgent()
    
    try:
        result = agent.run()
        
        print("\n--- SENTIMENT SUMMARY ---")
        print(f"Overall Sentiment: {result.overall_sentiment}")
        print(f"Confidence:        {result.confidence}")
        print(f"Summary:           {result.summary}")
        print(f"Key Drivers:       {result.key_drivers}")
        print(f"Market Events:     {result.market_events}")
        print(f"Last Updated:      {result.last_updated}")
        
        print("\n--- DETERMINISTIC MARKET DATA ---")
        print(json.dumps(result.market_data, indent=2))
        
        print("\n" + "=" * 60)
        
    except ValueError as e:
        print("\n[DIAGNOSTIC ERROR CAUGHT]")
        print(str(e))
        print("\n" + "=" * 60)
        
if __name__ == "__main__":
    test_macro()
