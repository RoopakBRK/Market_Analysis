import json
from src.graph.state import GraphState
from src.graph.workflow import graph
from src.llm.gateway import print_usage_stats

def main():
    print("\n==================================================")
    print("RUNNING FULL PIPELINE TEST")
    print("==================================================")

    initial_state: GraphState = {
        "watchlist": ["RELIANCE", "TCS", "INFOSYS"],
        "macro_summary": None,
        "company_news": {},
        "market_data": {},
        "sentiments": {},
        "report": None,
    }

    print("\n[INFO] Starting pipeline execution. This will take some time due to rate limit pacing...")
    
    result = graph.invoke(initial_state)

    print("\n==================================================")
    print("1. MACRO SUMMARY")
    print("==================================================")
    macro = result.get("macro_summary")
    if macro:
        print(f"Overall Sentiment: {macro.overall_sentiment}")
        print(f"Confidence: {macro.confidence}")
        print(f"Summary: {macro.summary}")
    else:
        print("No Macro Summary generated.")

    print("\n==================================================")
    print("2. COMPANY NEWS")
    print("==================================================")
    news = result.get("company_news", {})
    for company, data in news.items():
        count = len(data.articles) if data and hasattr(data, 'articles') else 0
        print(f"{company}: {count} articles collected")

    print("\n==================================================")
    print("3. SENTIMENT ANALYSIS")
    print("==================================================")
    sentiments = result.get("sentiments", {})
    for company, sentiment in sentiments.items():
        if sentiment:
            print(f"[{company}] Sentiment: {sentiment.sentiment} | Articles Analyzed: {sentiment.articles_analyzed} | Confidence: {sentiment.confidence}")
        else:
            print(f"[{company}] No sentiment generated.")

    print("\n==================================================")
    print("4. FINAL REPORT")
    print("==================================================")
    report = result.get("report")
    if report:
        print(f"Date: {report.report_date}")
        print(f"Executive Summary: {report.executive_summary}")
        print(f"Top Positive Stocks: {len(report.top_positive_stocks)}")
        print(f"Top Negative Stocks: {len(report.top_negative_stocks)}")
    else:
        print("No Report generated.")

    print("\n==================================================")
    print("PIPELINE COMPLETE")
    print("==================================================")
    
    # Print the instrumentation stats
    print_usage_stats()

if __name__ == "__main__":
    main()
