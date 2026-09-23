"""
Storage package — PostgreSQL persistence layer for the Market Analysis pipeline.

Usage:
    from src.storage import repositories
    from src.storage.postgres import init_db

    init_db()  # create tables on startup
    repositories.save_report(run_id, run_date, report)
"""

from src.storage.postgres import init_db, get_session
from src.storage import repositories, models

__all__ = ["init_db", "get_session", "repositories", "models"]
