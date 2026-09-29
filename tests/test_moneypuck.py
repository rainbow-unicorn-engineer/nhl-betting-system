"""
Tests for ingestion/moneypuck.py refresh_season: download only when the
cached CSV is missing or stale, then reload; skip before the season has a
completed game. No network, no database (both are faked).
"""
import os
import time
from types import SimpleNamespace

import pytest

from ingestion import moneypuck


class _FakeEngine:
    def __init__(self, finals):
        self.finals = finals

    def connect(self):
        engine = self

        class _Conn:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute(self, *a, **k):
                return SimpleNamespace(scalar=lambda: engine.finals)
        return _Conn()


@pytest.fixture()
def fake_io(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(moneypuck, "DATA_DIR", tmp_path)
    monkeypatch.setattr(moneypuck, "engine", _FakeEngine(finals=12))
    monkeypatch.setattr(moneypuck, "download_shots_csv",
                        lambda year, force=False: calls.append(("download", year, force)))
    monkeypatch.setattr(moneypuck, "load_shots_to_db",
                        lambda year, path=None: calls.append(("load", year)) or 345)
    return calls, tmp_path / "moneypuck_shots_2026.csv"


def test_missing_file_downloads_then_loads(fake_io):
    calls, _ = fake_io
    assert moneypuck.refresh_season(20262027) == 345
    assert calls == [("download", 2026, True), ("load", 2026)]


def test_fresh_file_is_reloaded_without_download(fake_io):
    calls, csv = fake_io
    csv.write_text("game_id\n")
    assert moneypuck.refresh_season(20262027, max_age_hours=20) == 345
    assert calls == [("load", 2026)]


def test_stale_file_is_downloaded_again(fake_io):
    calls, csv = fake_io
    csv.write_text("game_id\n")
    old = time.time() - 21 * 3600
    os.utime(csv, (old, old))
    moneypuck.refresh_season(20262027, max_age_hours=20)
    assert calls == [("download", 2026, True), ("load", 2026)]


def test_skips_before_the_first_completed_game(fake_io, monkeypatch):
    calls, _ = fake_io
    monkeypatch.setattr(moneypuck, "engine", _FakeEngine(finals=0))
    assert moneypuck.refresh_season(20262027) == 0
    assert calls == []


# ── load_shots_to_db takes a per-season lock before DELETE + INSERT ──

_CSV_COLUMNS = ["game_id", "period", "time", "teamCode", "shooterPlayerId",
                "goalieIdForShot", "arenaAdjustedXCord", "arenaAdjustedYCord",
                "shotType", "event", "goal", "xGoal", "homeSkatersOnIce",
                "awaySkatersOnIce", "isHomeTeam", "homeTeamGoals",
                "awayTeamGoals", "shotRebound", "shotRush", "shotDistance",
                "shotAngle"]


class _LockEngine:
    """connect(): the known-games read; begin(): records every statement."""

    def __init__(self):
        self.statements = []

    def connect(self):
        return _Ctx(lambda *a, **k: [(2026020001,)])

    def begin(self):
        engine = self

        def execute(stmt, params=None):
            engine.statements.append((str(stmt), params))
        return _Ctx(execute)


class _Ctx:
    def __init__(self, execute):
        self.execute = execute

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_reload_locks_the_season_before_deleting(monkeypatch, tmp_path):
    """Two overlapping reloads of one season must not both DELETE then
    both INSERT (duplicated shots): the transaction takes
    pg_advisory_xact_lock(key, season) first."""
    import pandas as pd
    csv = tmp_path / "shots.csv"
    row = [20001, 1, 30, "T.B", 8478010, 8476883, 60.0, 10.0, "WRIST", "SHOT",
           0, 0.05, 5, 5, 1, 0, 0, 0, 0, 30.0, 18.0]
    pd.DataFrame([row], columns=_CSV_COLUMNS).to_csv(csv, index=False)

    fake = _LockEngine()
    appended = []
    monkeypatch.setattr(moneypuck, "engine", fake)
    monkeypatch.setattr(pd.DataFrame, "to_sql",
                        lambda self, *a, **k: appended.append(len(self)))
    assert moneypuck.load_shots_to_db(2026, csv) == 1
    assert appended == [1]
    (lock_sql, lock_params), (delete_sql, _) = fake.statements[:2]
    assert "pg_advisory_xact_lock" in lock_sql
    assert lock_params == {"k": moneypuck.SHOTS_LOCK_KEY, "s": 20262027}
    assert delete_sql.startswith("DELETE FROM raw.shots")
