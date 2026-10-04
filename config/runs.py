"""
config/runs.py
Markers for finished pipeline jobs (raw.pipeline_runs): one row per job
and local date, written when the job reaches its end.

The news monitor (betting/news.py) reads the 'daily' marker: until
today's daily run has finished, last night's box scores, Elo and rolling
stats are not loaded yet, so a pick re-scored from news would rest on
stale data, and an issued pick is frozen (the daily run can never
re-decide it). News found before the marker is recorded, not bet on.
"""
from datetime import date, datetime, timezone
from typing import Optional

from sqlalchemy import text

from config.settings import engine, local_today


def mark_finished(job: str, run_date: Optional[date] = None) -> None:
    """Record that `job` finished for run_date (default: today, local).
    A second run the same day moves finished_at forward."""
    from config.migrate import ensure_schema
    ensure_schema()
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO raw.pipeline_runs (job, run_date, finished_at)
            VALUES (:job, :d, :now)
            ON CONFLICT (job, run_date) DO UPDATE SET finished_at = EXCLUDED.finished_at
        """), {"job": job, "d": run_date or local_today(),
               "now": datetime.now(timezone.utc)})


def finished(job: str, run_date: Optional[date] = None) -> bool:
    """Whether `job` has finished for run_date (default: today, local)."""
    from config.migrate import ensure_schema
    ensure_schema()
    with engine.connect() as conn:
        return conn.execute(text("""
            SELECT 1 FROM raw.pipeline_runs WHERE job = :job AND run_date = :d
        """), {"job": job, "d": run_date or local_today()}).first() is not None
