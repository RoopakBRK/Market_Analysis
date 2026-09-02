"""
Worker entry point for running intelligence pipelines.
"""
from src.graph.workflow import graph


from src.graph.state import GraphState


def main():

    initial_state: GraphState = {
        "watchlist": ["RELIANCE", "TCS", "INFOSYS"],
        "macro_summary": None,
        "company_news": {},
        "market_data": {},
        "sentiments": {},
        "report": None,
    }

    result = graph.invoke(initial_state)

    print("\n--- MACRO ---")
    print(result.get("macro_summary"))
    
    print("\n--- SENTIMENTS ---")
    for ticker, sentiment in result.get("sentiments", {}).items():
        print(f"\n[{ticker}]")
        print(sentiment)

    from src.llm.gateway import print_usage_stats
    print_usage_stats()

if __name__ == "__main__":
    main()