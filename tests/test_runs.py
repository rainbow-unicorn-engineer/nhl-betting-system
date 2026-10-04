"""
Tests for config/runs.py, the finished-job markers (raw.pipeline_runs)
the news monitor reads before it makes a pick. Needs a disposable
database (tests/conftest.py).
"""
from datetime import date

import pytest
from sqlalchemy import text

from config import runs
from config.settings import check_db_connection, engine

requires_db = pytest.mark.skipif(not check_db_connection(), reason="database not reachable")

DAY = date(2099, 1, 2)


@requires_db
def test_mark_then_read_and_a_second_mark_moves_the_time():
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM raw.pipeline_runs WHERE run_date = :d"), {"d": DAY})
    try:
        assert runs.finished("daily", DAY) is False
        runs.mark_finished("daily", DAY)
        assert runs.finished("daily", DAY) is True
        assert runs.finished("refresh", DAY) is False
        assert runs.finished("daily", date(2099, 1, 3)) is False
        with engine.connect() as conn:
            first = conn.execute(text("SELECT finished_at FROM raw.pipeline_runs "
                                      "WHERE job = 'daily' AND run_date = :d"),
                                 {"d": DAY}).scalar()
        runs.mark_finished("daily", DAY)
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT finished_at FROM raw.pipeline_runs "
                                     "WHERE job = 'daily' AND run_date = :d"),
                                {"d": DAY}).fetchall()
        assert len(rows) == 1 and rows[0][0] >= first
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM raw.pipeline_runs WHERE run_date = :d"),
                         {"d": DAY})
