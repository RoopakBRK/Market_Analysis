from src.graph.state import GraphState
from src.graph.workflow import graph

def main():
    print("==================================================")
    print("RUNNING THREE COMPANIES SENTIMENT TEST")
    print("==================================================")

    initial_state: GraphState = {
        "watchlist": ["RELIANCE", "TCS", "INFOSYS"],
        "macro_summary": None,
        "company_news": {},
        "market_data": {},
        "sentiments": {},
        "report": None,
    }

    result = graph.invoke(initial_state)

    sentiments = result.get("sentiments", {})

    for company in ["RELIANCE", "TCS", "INFOSYS"]:
        print(f"\n==================================================")
        print(f"{company}")
        print(f"==================================================")
        
        sentiment = sentiments.get(company)
        if sentiment:
            print(f"Sentiment: {sentiment.sentiment}")
            print(f"Confidence: {sentiment.confidence}")
            print(f"Impact: {sentiment.impact}")
            print(f"Summary: {sentiment.summary}")
            print(f"Positive Drivers: {sentiment.positive_drivers}")
            print(f"Negative Drivers: {sentiment.negative_drivers}")
            print(f"Articles Analyzed: {sentiment.articles_analyzed}")
        else:
            print("No sentiment generated.")

if __name__ == "__main__":
    main()
