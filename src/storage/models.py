"""
SQLAlchemy ORM models for persisting market intelligence data.

These map the Pydantic domain models to PostgreSQL tables.
"""

import json
from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    Text,
    DateTime,
    JSON,
    ForeignKey,
    Index,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ── PipelineRun ───────────────────────────────────────────────────────────────

class PipelineRun(Base):
    """
    Tracks each full pipeline execution.
    """
    __tablename__ = "pipeline_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_date = Column(String(20), nullable=False, index=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    status = Column(String(20), default="running")  # running | completed | failed
    error_message = Column(Text, nullable=True)

    # Relationships
    macro_summary = relationship("MacroSummaryRecord", back_populates="run", uselist=False)
    company_news = relationship("CompanyNewsRecord", back_populates="run")
    sentiments = relationship("SentimentRecord", back_populates="run")
    reports = relationship("ReportRecord", back_populates="run", uselist=False)


# ── MacroSummary ──────────────────────────────────────────────────────────────

class MacroSummaryRecord(Base):
    __tablename__ = "macro_summaries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("pipeline_runs.id"), nullable=False, index=True)
    run_date = Column(String(20), nullable=False)

    overall_sentiment = Column(String(20))
    confidence = Column(Integer)
    summary = Column(Text)
    key_drivers = Column(JSON)        # list[str]
    market_events = Column(JSON)      # list[str]
    market_data = Column(JSON)        # raw dict of tool outputs
    last_updated = Column(String(50))

    created_at = Column(DateTime, default=datetime.utcnow)

    run = relationship("PipelineRun", back_populates="macro_summary")

    __table_args__ = (
        Index("ix_macro_run_date", "run_date"),
    )


# ── CompanyNews ───────────────────────────────────────────────────────────────

class CompanyNewsRecord(Base):
    __tablename__ = "company_news"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("pipeline_runs.id"), nullable=False, index=True)
    run_date = Column(String(20), nullable=False)

    ticker = Column(String(20), nullable=False, index=True)
    company_name = Column(String(200))
    total_articles = Column(Integer, default=0)
    articles = Column(JSON)   # list of NewsArticle dicts

    created_at = Column(DateTime, default=datetime.utcnow)

    run = relationship("PipelineRun", back_populates="company_news")

    __table_args__ = (
        Index("ix_company_news_ticker_date", "ticker", "run_date"),
    )


# ── SentimentResult ───────────────────────────────────────────────────────────

class SentimentRecord(Base):
    __tablename__ = "sentiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("pipeline_runs.id"), nullable=False, index=True)
    run_date = Column(String(20), nullable=False)

    ticker = Column(String(20), nullable=False, index=True)
    company_name = Column(String(200))

    sentiment = Column(String(20))
    confidence = Column(Integer)
    impact = Column(String(20))
    summary = Column(Text)
    positive_drivers = Column(JSON)   # list[str]
    negative_drivers = Column(JSON)   # list[str]
    articles_analyzed = Column(Integer, default=0)

    # Multi-signal fields
    verified_news_sentiment = Column(String(50), nullable=True)
    financial_data_signal = Column(String(50), nullable=True)
    reddit_sentiment = Column(String(50), nullable=True)
    source_breakdown = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    run = relationship("PipelineRun", back_populates="sentiments")

    __table_args__ = (
        Index("ix_sentiment_ticker_date", "ticker", "run_date"),
    )


# ── DailyMarketReport ─────────────────────────────────────────────────────────

class ReportRecord(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("pipeline_runs.id"), nullable=False, index=True)
    run_date = Column(String(20), nullable=False, unique=True, index=True)

    overall_market_sentiment = Column(String(20))
    overall_confidence = Column(Integer)

    macro_overview = Column(Text)
    key_macro_drivers = Column(JSON)     # list[str]
    company_intelligence = Column(JSON)  # list of CompanyIntelligence dicts
    final_market_view = Column(Text)
    major_catalysts = Column(JSON)       # list[str]
    major_risks = Column(JSON)           # list[str]
    market_events = Column(JSON)         # list[str]

    generated_at = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)

    run = relationship("PipelineRun", back_populates="reports")


# ── RedditSignal ──────────────────────────────────────────────────────────────

class RedditSignalRecord(Base):
    __tablename__ = "reddit_signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_date = Column(String(20), nullable=False, index=True)
    ticker = Column(String(20), nullable=False)

    overall_sentiment = Column(String(20))
    post_count = Column(Integer, default=0)
    avg_score = Column(Float, default=0.0)
    posts = Column(JSON)         # list of RedditPost dicts
    retrieved_at = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_reddit_ticker_date", "ticker", "run_date"),
    )


# ── CompanyFinancials ─────────────────────────────────────────────────────────

class CompanyFinancialsRecord(Base):
    __tablename__ = "company_financials"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_date = Column(String(20), nullable=False, index=True)
    ticker = Column(String(20), nullable=False)

    company_name = Column(String(200), nullable=True)
    sector = Column(String(100), nullable=True)
    market_cap = Column(Float, nullable=True)
    pe_ratio = Column(Float, nullable=True)
    eps = Column(Float, nullable=True)
    revenue = Column(Float, nullable=True)
    debt_to_equity = Column(Float, nullable=True)
    upcoming_events = Column(JSON)    # list[str]
    data_source = Column(String(100), default="unknown")
    retrieved_at = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_financials_ticker_date", "ticker", "run_date"),
    )
