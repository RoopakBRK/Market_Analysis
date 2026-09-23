"""
PostgreSQL database connection and session management.

Uses SQLAlchemy with connection pooling. Falls back gracefully when
DATABASE_URL is not configured — storage operations become no-ops.
"""

import sys
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from config.settings import settings
from src.storage.models import Base


# ── Engine ────────────────────────────────────────────────────────────────────

def _create_engine_safe():
    """
    Create a SQLAlchemy engine if DATABASE_URL is configured.
    Returns None if unconfigured — callers must handle gracefully.
    """
    db_url = settings.DATABASE_URL
    if not db_url:
        return None
    try:
        engine = create_engine(
            db_url,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_pre_ping=True,   # validate connections before use
            echo=False,
        )
        # Quick connectivity check
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as exc:
        print(
            f"[Storage] Database connection failed: {type(exc).__name__}: {exc}\n"
            f"[Storage] Running without persistence. Set DATABASE_URL to enable.",
            file=sys.stderr,
        )
        return None


engine = _create_engine_safe()

SessionLocal: sessionmaker | None = (
    sessionmaker(autocommit=False, autoflush=False, bind=engine)
    if engine else None
)


# ── Schema init ───────────────────────────────────────────────────────────────

def init_db() -> bool:
    """
    Create all tables if they don't exist yet.
    Returns True on success, False if DB is unavailable.
    """
    if engine is None:
        print("[Storage] Skipping DB init — DATABASE_URL not configured.", file=sys.stderr)
        return False
    try:
        Base.metadata.create_all(bind=engine)
        print("[Storage] Database tables initialised.", file=sys.stderr)
        return True
    except Exception as exc:
        print(f"[Storage] Failed to create tables: {exc}", file=sys.stderr)
        return False


# ── Session context manager ───────────────────────────────────────────────────

@contextmanager
def get_session() -> Generator[Session | None, None, None]:
    """
    Yield a database session.

    Usage:
        with get_session() as session:
            if session:
                session.add(record)
                session.commit()

    Yields None if DATABASE_URL is not configured — callers must check.
    Always commits on clean exit, rolls back on exception.
    """
    if SessionLocal is None:
        yield None
        return

    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
