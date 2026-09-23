"""
Repository layer — thin wrappers around storage models.

Each repository converts a Pydantic domain model into an ORM record
and persists it. All operations are no-ops when the DB is unavailable.

Design:
- Never raise exceptions into the pipeline. Storage failures are logged
  and silently skipped so the market intelligence pipeline always runs.
- All methods accept the domain Pydantic models and handle serialisation.
"""

import sys
from datetime import datetime

from src.storage.postgres import get_session
from src.storage.models import (
    PipelineRun,
    MacroSummaryRecord,
    CompanyNewsRecord,
    SentimentRecord,
    ReportRecord,
    RedditSignalRecord,
    CompanyFinancialsRecord,
)


# ── PipelineRun ───────────────────────────────────────────────────────────────

def create_pipeline_run(run_date: str) -> int | None:
    """
    Create a new pipeline run record. Returns the run_id or None if DB unavailable.
    """
    with get_session() as session:
        if session is None:
            return None
        try:
            run = PipelineRun(run_date=run_date, status="running")
            session.add(run)
            session.flush()  # get the id before commit
            run_id = run.id
            return run_id
        except Exception as exc:
            print(f"[Storage] Failed to create pipeline run: {exc}", file=sys.stderr)
            return None


def complete_pipeline_run(run_id: int, success: bool = True, error: str | None = None):
    """Mark a pipeline run as completed or failed."""
    with get_session() as session:
        if session is None:
            return
        try:
            run = session.get(PipelineRun, run_id)
            if run:
                run.status = "completed" if success else "failed"
                run.completed_at = datetime.utcnow()
                run.error_message = error
        except Exception as exc:
            print(f"[Storage] Failed to update pipeline run: {exc}", file=sys.stderr)


# ── MacroSummary ──────────────────────────────────────────────────────────────

def save_macro_summary(run_id: int, run_date: str, macro_summary) -> bool:
    """
    Persist a MacroSummary domain object.
    Returns True on success, False on failure.
    """
    with get_session() as session:
        if session is None:
            return False
        try:
            record = MacroSummaryRecord(
                run_id=run_id,
                run_date=run_date,
                overall_sentiment=macro_summary.overall_sentiment,
                confidence=macro_summary.confidence,
                summary=macro_summary.summary,
                key_drivers=macro_summary.key_drivers,
                market_events=macro_summary.market_events,
                market_data=macro_summary.market_data,
                last_updated=macro_summary.last_updated,
            )
            session.add(record)
            return True
        except Exception as exc:
            print(f"[Storage] Failed to save macro summary: {exc}", file=sys.stderr)
            return False


# ── CompanyNews ───────────────────────────────────────────────────────────────

def save_company_news(run_id: int, run_date: str, company_news_dict: dict) -> bool:
    """
    Persist CompanyNews objects for all companies.
    company_news_dict: {ticker: CompanyNews}
    """
    with get_session() as session:
        if session is None:
            return False
        try:
            for ticker, news in company_news_dict.items():
                articles_data = [a.model_dump() for a in news.articles]
                record = CompanyNewsRecord(
                    run_id=run_id,
                    run_date=run_date,
                    ticker=ticker,
                    company_name=news.company_name,
                    total_articles=news.total_articles,
                    articles=articles_data,
                )
                session.add(record)
            return True
        except Exception as exc:
            print(f"[Storage] Failed to save company news: {exc}", file=sys.stderr)
            return False


# ── Sentiments ────────────────────────────────────────────────────────────────

def save_sentiments(run_id: int, run_date: str, sentiments_dict: dict) -> bool:
    """
    Persist SentimentResult objects for all companies.
    sentiments_dict: {ticker: SentimentResult}
    """
    with get_session() as session:
        if session is None:
            return False
        try:
            for ticker, sentiment in sentiments_dict.items():
                record = SentimentRecord(
                    run_id=run_id,
                    run_date=run_date,
                    ticker=ticker,
                    company_name=sentiment.company_name,
                    sentiment=sentiment.sentiment,
                    confidence=sentiment.confidence,
                    impact=sentiment.impact,
                    summary=sentiment.summary,
                    positive_drivers=sentiment.positive_drivers,
                    negative_drivers=sentiment.negative_drivers,
                    articles_analyzed=sentiment.articles_analyzed,
                    verified_news_sentiment=sentiment.verified_news_sentiment,
                    financial_data_signal=sentiment.financial_data_signal,
                    reddit_sentiment=sentiment.reddit_sentiment,
                    source_breakdown=sentiment.source_breakdown,
                )
                session.add(record)
            return True
        except Exception as exc:
            print(f"[Storage] Failed to save sentiments: {exc}", file=sys.stderr)
            return False


# ── DailyMarketReport ─────────────────────────────────────────────────────────

def save_report(run_id: int, run_date: str, report) -> bool:
    """
    Persist the final DailyMarketReport.
    """
    with get_session() as session:
        if session is None:
            return False
        try:
            company_intel = [c.model_dump() for c in report.company_intelligence]
            record = ReportRecord(
                run_id=run_id,
                run_date=run_date,
                overall_market_sentiment=report.overall_market_sentiment,
                overall_confidence=report.overall_confidence,
                macro_overview=report.macro_overview,
                key_macro_drivers=report.key_macro_drivers,
                company_intelligence=company_intel,
                final_market_view=report.final_market_view,
                major_catalysts=report.major_catalysts,
                major_risks=report.major_risks,
                market_events=report.market_events,
                generated_at=report.generated_at,
            )
            session.add(record)
            return True
        except Exception as exc:
            print(f"[Storage] Failed to save report: {exc}", file=sys.stderr)
            return False


# ── RedditSignals ─────────────────────────────────────────────────────────────

def save_reddit_signals(run_date: str, reddit_signals_dict: dict) -> bool:
    """
    Persist RedditSignal objects for all companies.
    reddit_signals_dict: {ticker: RedditSignal}
    """
    with get_session() as session:
        if session is None:
            return False
        try:
            for ticker, signal in reddit_signals_dict.items():
                posts_data = [p.model_dump() for p in signal.posts]
                record = RedditSignalRecord(
                    run_date=run_date,
                    ticker=ticker,
                    overall_sentiment=signal.overall_sentiment,
                    post_count=signal.post_count,
                    avg_score=signal.avg_score,
                    posts=posts_data,
                    retrieved_at=signal.retrieved_at,
                )
                session.add(record)
            return True
        except Exception as exc:
            print(f"[Storage] Failed to save Reddit signals: {exc}", file=sys.stderr)
            return False


# ── CompanyFinancials ─────────────────────────────────────────────────────────

def save_financial_data(run_date: str, financial_data_dict: dict) -> bool:
    """
    Persist CompanyFinancials objects for all companies.
    financial_data_dict: {ticker: CompanyFinancials}
    """
    with get_session() as session:
        if session is None:
            return False
        try:
            for ticker, fin in financial_data_dict.items():
                record = CompanyFinancialsRecord(
                    run_date=run_date,
                    ticker=ticker,
                    company_name=fin.company_name,
                    sector=fin.sector,
                    market_cap=fin.market_cap,
                    pe_ratio=fin.pe_ratio,
                    eps=fin.eps,
                    revenue=fin.revenue,
                    debt_to_equity=fin.debt_to_equity,
                    upcoming_events=fin.upcoming_events,
                    data_source=fin.data_source,
                    retrieved_at=fin.retrieved_at,
                )
                session.add(record)
            return True
        except Exception as exc:
            print(f"[Storage] Failed to save financial data: {exc}", file=sys.stderr)
            return False


# ── Queries ───────────────────────────────────────────────────────────────────

def get_latest_report(run_date: str | None = None) -> ReportRecord | None:
    """
    Fetch the most recent report, or the report for a specific date.
    Returns None if DB is unavailable or no report found.
    """
    with get_session() as session:
        if session is None:
            return None
        try:
            query = session.query(ReportRecord)
            if run_date:
                return query.filter(ReportRecord.run_date == run_date).first()
            return query.order_by(ReportRecord.created_at.desc()).first()
        except Exception as exc:
            print(f"[Storage] Failed to fetch report: {exc}", file=sys.stderr)
            return None


def get_sentiment_history(ticker: str, limit: int = 30) -> list[SentimentRecord]:
    """
    Fetch recent sentiment records for a ticker. Useful for trend analysis.
    Returns [] if DB is unavailable.
    """
    with get_session() as session:
        if session is None:
            return []
        try:
            return (
                session.query(SentimentRecord)
                .filter(SentimentRecord.ticker == ticker)
                .order_by(SentimentRecord.created_at.desc())
                .limit(limit)
                .all()
            )
        except Exception as exc:
            print(f"[Storage] Failed to fetch sentiment history: {exc}", file=sys.stderr)
            return []
