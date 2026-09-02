"""
Worker entry point for running intelligence pipelines.
"""
from src.graph.workflow import graph


def main():

    initial_state = {
        "macro_summary": None,
        "company_news": {},
        "market_data": {},
        "sentiments": {},
        "report": None,
    }

    result = graph.invoke(initial_state)

    print(result["macro_summary"])


if __name__ == "__main__":
    main()