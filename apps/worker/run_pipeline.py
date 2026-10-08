"""
Worker entry point for running intelligence pipelines.
"""
import sys
from pathlib import Path

# Allow running this file directly (`python apps/worker/run_pipeline.py`), as
# documented in the README, without requiring `python -m apps.worker.run_pipeline`.
# Without this, `from src...` / `from config...` imports fail with
# `ModuleNotFoundError: No module named 'src'` because Python only puts the
# script's own directory on sys.path, not the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.graph.workflow import graph
from src.graph.state import GraphState
from src.utils.constants import WATCHLIST


def main():

    initial_state: GraphState = {
        "watchlist": list(WATCHLIST),
        "macro_summary": None,
        "company_news": {},
        "market_data": {},
        "financial_data": {},
        "reddit_signals": {},
        "historical_context": {},
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

    print("\n--- REPORT ---")
    print(result.get("report"))

    report = result.get("report")
    if report is not None:
        from src.services.report_service import build_report_pdf, default_report_path
        pdf_path = default_report_path(report)
        try:
            written = build_report_pdf(report, pdf_path)
            print(f"\n--- PDF REPORT ---\nWritten to: {written}")
        except Exception as e:
            print(f"\n[Warning] Failed to write PDF report: {e}")

        # Saved for tomorrow's "What Changed" comparison.
        from src.services.historical_service import build_snapshot, save_snapshot
        snapshot = build_snapshot(report.date, result.get("macro_summary"), result.get("sentiments", {}))
        print(f"\n--- HISTORY ---\nSnapshot saved to: {save_snapshot(snapshot)}")

    # Imported here, not at the top: importing the storage package opens the
    # database connection, which is only wanted once there is something to save.
    from src.storage.repositories import persist_pipeline_result
    print("\n--- STORAGE ---")
    if persist_pipeline_result(result):
        print("Run saved to the database.")
    else:
        print("Run not saved (database not configured, unreachable, or the save failed).")

    from src.llm.gateway import print_usage_stats
    print_usage_stats()

if __name__ == "__main__":
    main()
