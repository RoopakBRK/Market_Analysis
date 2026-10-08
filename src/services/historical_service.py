"""
Run history for day-over-day comparison.

Each pipeline run saves a small JSON snapshot of its headline figures to
data/history/<date>.json. A later run loads the most recent snapshot from an
earlier date to report what changed. A second run on the same day overwrites
that day's snapshot, and is still compared with the previous day's.

Prices are not stored: their day-over-day change comes straight from the
market data (previous close), so only figures the pipeline itself produces or
that carry no previous value (sentiment, FII/DII flows, the repo rate) need
remembering.
"""

import json
from pathlib import Path

from src.tools.common.normalization import utc_now_iso

HISTORY_DIR = Path(__file__).resolve().parents[2] / "data" / "history"


def build_snapshot(run_date: str, macro_summary, sentiments: dict) -> dict:
    """Extract the figures worth comparing tomorrow from one run's results."""
    market_data = macro_summary.market_data if macro_summary is not None else {}
    flows = market_data.get("fii_dii_flows") or {}
    rbi = market_data.get("rbi") or {}

    return {
        "date": run_date,
        "saved_at": utc_now_iso(),
        "macro": {
            "sentiment": macro_summary.overall_sentiment if macro_summary is not None else None,
            "confidence": macro_summary.confidence if macro_summary is not None else None,
            "fii_net": flows.get("fii_net"),
            "dii_net": flows.get("dii_net"),
            "flows_date": flows.get("date"),
            "repo_rate": rbi.get("policy_repo_rate"),
        },
        "companies": {
            ticker: {"sentiment": s.sentiment, "confidence": s.confidence}
            for ticker, s in sentiments.items()
        },
    }


def save_snapshot(snapshot: dict, history_dir: Path = HISTORY_DIR) -> Path:
    """Write a snapshot to <history_dir>/<date>.json and return the path."""
    history_dir.mkdir(parents=True, exist_ok=True)
    path = history_dir / f"{snapshot['date']}.json"
    path.write_text(json.dumps(snapshot, indent=2))
    return path


def load_previous_snapshot(before_date: str, history_dir: Path = HISTORY_DIR) -> dict | None:
    """
    Load the most recent snapshot dated strictly before `before_date`
    (YYYY-MM-DD), or None if there is none. Unreadable files are skipped.
    """
    if not history_dir.is_dir():
        return None

    # ISO dates sort chronologically as plain strings.
    earlier = sorted((p for p in history_dir.glob("*.json") if p.stem < before_date), reverse=True)
    for path in earlier:
        try:
            snapshot = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(snapshot, dict) and snapshot.get("date"):
            return snapshot
    return None
