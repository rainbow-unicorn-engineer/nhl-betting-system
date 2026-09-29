"""
Tests for pipeline.py wiring that needs no database or network.
"""
import pytest

import pipeline


def test_close_is_a_moneyline_only_snapshot(monkeypatch):
    from ingestion import odds_api
    calls = []
    monkeypatch.setattr(pipeline, "_wait_for_network", lambda: True)
    monkeypatch.setattr(odds_api, "snapshot_odds",
                        lambda **kw: calls.append(kw) or 0)
    monkeypatch.setattr(pipeline, "recommend",
                        lambda: pytest.fail("close must not recommend"))
    monkeypatch.setattr(pipeline, "starters",
                        lambda: pytest.fail("close must not fetch starters"))
    pipeline.close()
    assert calls == [{"markets": "h2h"}]


def test_close_waits_for_network(monkeypatch):
    from ingestion import odds_api
    monkeypatch.setattr(pipeline, "_wait_for_network", lambda: False)
    monkeypatch.setattr(odds_api, "snapshot_odds",
                        lambda **kw: pytest.fail("no network, no snapshot"))
    pipeline.close()


def test_close_due_skips_when_not_due(monkeypatch):
    from ingestion import odds_api
    monkeypatch.setattr(odds_api, "close_is_due",
                        lambda: (False, "no game starts in the next 16 minutes"))
    monkeypatch.setattr(pipeline, "_wait_for_network",
                        lambda: pytest.fail("not due: no network wait"))
    monkeypatch.setattr(odds_api, "snapshot_odds",
                        lambda **kw: pytest.fail("not due: no snapshot"))
    pipeline.close(due=True)


def test_close_due_snapshots_moneyline_when_due(monkeypatch):
    from ingestion import odds_api
    calls = []
    monkeypatch.setattr(odds_api, "close_is_due", lambda: (True, "1 game soon"))
    monkeypatch.setattr(pipeline, "_wait_for_network", lambda: True)
    monkeypatch.setattr(odds_api, "snapshot_odds",
                        lambda **kw: calls.append(kw) or 0)
    pipeline.close(due=True)
    assert calls == [{"markets": "h2h"}]


def test_plain_close_does_not_check_timing(monkeypatch):
    from ingestion import odds_api
    calls = []
    monkeypatch.setattr(odds_api, "close_is_due",
                        lambda: pytest.fail("plain close never checks"))
    monkeypatch.setattr(pipeline, "_wait_for_network", lambda: True)
    monkeypatch.setattr(odds_api, "snapshot_odds",
                        lambda **kw: calls.append(kw) or 0)
    pipeline.close()
    assert calls == [{"markets": "h2h"}]


class _StatusConn:
    def __init__(self):
        self.sql = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, stmt, params=None):
        self.sql.append(str(stmt))

        class _Result:
            def scalar(self):
                return 0

            def fetchall(self):
                return []
        return _Result()


def test_status_counts_upcoming_games_by_state(monkeypatch, capsys):
    """The NHL API never sends SCHEDULED (upcoming games are FUT or PRE),
    so counting it always printed 0."""
    conn = _StatusConn()
    monkeypatch.setattr(pipeline, "db_ready", lambda: True)
    monkeypatch.setattr(pipeline, "engine",
                        type("E", (), {"connect": lambda self: conn})())
    pipeline.db_status()
    assert not any("'SCHEDULED'" in s for s in conn.sql)
    assert any("game_state NOT IN ('FINAL', 'OFF')" in s for s in conn.sql)
    assert "Games (upcoming)" in capsys.readouterr().out


def test_setup_seeds_venues_only_when_missing(monkeypatch):
    from config import migrate
    seeded = []
    monkeypatch.setattr(migrate, "seed_venues", lambda: seeded.append(1) or 1)
    monkeypatch.setattr(migrate, "venues_missing", lambda: True)
    assert pipeline.seed_venues_if_missing().startswith("SEEDED")
    monkeypatch.setattr(migrate, "venues_missing", lambda: False)
    assert pipeline.seed_venues_if_missing() == "OK"
    assert seeded == [1]
