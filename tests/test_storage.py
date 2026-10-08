"""Offline test for pipeline persistence, run against a throwaway SQLite file."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.models.company import CompanyNews
from src.models.macro import MacroSummary
from src.models.report import DailyMarketReport
from src.models.sentiment import SentimentResult

NOW = "2026-10-08T11:00:00+00:00"


@pytest.fixture
def storage(tmp_path, monkeypatch):
    from src.storage import postgres, repositories

    engine = create_engine(f"sqlite:///{tmp_path / 'pipeline.db'}")
    monkeypatch.setattr(postgres, "engine", engine)
    monkeypatch.setattr(postgres, "SessionLocal", sessionmaker(autocommit=False, autoflush=False, bind=engine))
    yield repositories
    engine.dispose()


def _state(view: str) -> dict:
    macro = MacroSummary(
        overall_sentiment="Bearish", confidence=70, summary="s", key_drivers=["d"], last_updated=NOW
    )
    sentiment = SentimentResult(
        ticker="TMPV", company_name="Tata Motors Passenger Vehicles", sentiment="Bearish",
        confidence=70, impact="Medium", summary="s", articles_analyzed=0,
    )
    report = DailyMarketReport(
        date="2026-10-08", overall_market_sentiment="Bearish", overall_confidence=70,
        macro_summary=macro, final_market_view=view, generated_at=NOW,
    )
    return {
        "macro_summary": macro,
        "company_news": {"TMPV": CompanyNews(ticker="TMPV", company_name="Tata Motors Passenger Vehicles")},
        "market_data": {},
        "financial_data": {},
        "reddit_signals": {},
        "sentiments": {"TMPV": sentiment},
        "report": report,
    }


def test_pipeline_run_is_persisted(storage):
    from src.storage.models import PipelineRun, ReportRecord, SentimentRecord
    from src.storage.postgres import get_session

    assert storage.persist_pipeline_result(_state("first view")) is True

    with get_session() as session:
        run = session.query(PipelineRun).one()
        assert run.status == "completed"
        assert session.query(SentimentRecord).one().ticker == "TMPV"
        assert session.query(ReportRecord).one().final_market_view == "first view"


def test_second_run_on_the_same_day_replaces_the_report(storage):
    from src.storage.models import PipelineRun, ReportRecord
    from src.storage.postgres import get_session

    assert storage.persist_pipeline_result(_state("first view")) is True
    assert storage.persist_pipeline_result(_state("second view")) is True

    with get_session() as session:
        assert session.query(PipelineRun).count() == 2
        assert session.query(ReportRecord).one().final_market_view == "second view"


def test_persistence_is_skipped_without_a_database(monkeypatch):
    from src.storage import postgres, repositories

    monkeypatch.setattr(postgres, "engine", None)
    monkeypatch.setattr(postgres, "SessionLocal", None)

    assert repositories.persist_pipeline_result(_state("view")) is False
