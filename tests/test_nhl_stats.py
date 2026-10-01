"""
Tests for ingestion/nhl_stats.py — the NHL stats API fill of
raw.skater_games' power-play, penalty-kill and faceoff columns.

The fixture is a real api.nhle.com response for game 2025020001 (FLA-CHI,
2025-10-07), trimmed to five players. No network: requests.get and
fetch_report are mocked. The database tests use a synthetic game
(9999020101, dated 2031-11-15) and delete it afterwards.
"""
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import requests
from sqlalchemy import text

from config.settings import check_db_connection, engine
from ingestion import nhl_stats
from ingestion.nhl_stats import (COLUMNS, cayenne, merge_reports, windows,
                                 windows_for_dates)

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = json.loads((ROOT / "tests" / "fixtures" / "nhl_stats_game_2025020001.json")
                     .read_text(encoding="utf-8"))
REPORT_ROWS = {r: FIXTURE[r]["data"] for r in nhl_stats.REPORTS}

requires_db = pytest.mark.skipif(not check_db_connection(),
                                 reason="database not reachable")


def _by_player(rows):
    return {r["player_id"]: r for r in rows}


@pytest.fixture()
def no_pause(monkeypatch):
    monkeypatch.setattr(nhl_stats, "PAUSE_S", 0)
    monkeypatch.setattr(nhl_stats.time, "sleep", lambda s: None)


# ── Pure parsing ──────────────────────────────────────────────────

class TestMergeReports:
    def test_real_rows_fill_every_column(self):
        rows = _by_player(merge_reports(REPORT_ROWS))
        assert len(rows) == 5
        bedard = rows[8484144]
        assert bedard["game_id"] == 2025020001
        assert (bedard["pp_toi_seconds"], bedard["sh_toi_seconds"]) == (247, 5)
        assert (bedard["fow"], bedard["fol"]) == (3, 14)
        verhaeghe = rows[8477409]          # scored the power-play goal
        assert (verhaeghe["pp_goals"], verhaeghe["pp_assists"]) == (1, 0)
        assert rows[8482713]["pp_assists"] == 1   # Samoskevich assisted it
        # a real zero stays a zero, not "missing"
        assert rows[8475179]["pp_toi_seconds"] == 0
        assert rows[8475179]["sh_toi_seconds"] == 57

    def test_a_report_without_the_player_leaves_its_columns_none(self):
        rows = _by_player(merge_reports({
            "timeonice": REPORT_ROWS["timeonice"],
            "powerplay": [r for r in REPORT_ROWS["powerplay"]
                          if r["playerId"] != 8484144]}))
        assert rows[8484144]["pp_toi_seconds"] == 247
        assert rows[8484144]["pp_goals"] is None       # COALESCE keeps the stored value
        assert rows[8484144]["fow"] is None

    def test_rows_without_ids_are_dropped_and_values_coerced(self):
        rows = merge_reports({"timeonice": [
            {"gameId": None, "playerId": 1, "ppTimeOnIce": 5},
            {"gameId": 2025020001, "playerId": "8", "ppTimeOnIce": 12.0,
             "shTimeOnIce": "n/a"},
        ]})
        assert rows == [{"game_id": 2025020001, "player_id": 8,
                         "pp_toi_seconds": 12, "sh_toi_seconds": None,
                         "pp_goals": None, "pp_assists": None,
                         "fow": None, "fol": None}]

    def test_every_column_is_a_real_skater_games_column(self):
        assert COLUMNS == ["pp_toi_seconds", "sh_toi_seconds", "pp_goals",
                           "pp_assists", "fow", "fol"]


class TestWindows:
    def test_cayenne_filter(self):
        assert cayenne(dt.date(2025, 10, 7), dt.date(2025, 10, 13)) == (
            'gameDate>="2025-10-07" and gameDate<="2025-10-13" and gameTypeId>=2')
        assert cayenne(dt.date(2025, 10, 7), dt.date(2025, 10, 7), 20252026).endswith(
            " and seasonId=20252026")

    def test_windows_cover_the_range_without_gaps_or_overlap(self):
        lo, hi = dt.date(2025, 10, 7), dt.date(2026, 6, 14)
        ws = windows(lo, hi, days=30)
        assert ws[0][0] == lo and ws[-1][1] == hi
        for (a0, a1), (b0, _) in zip(ws, ws[1:]):
            assert b0 == a1 + dt.timedelta(days=1)
        assert all((b - a).days < 30 for a, b in ws)
        assert len(ws) == 9          # a full season: 9 windows x 3 reports
        assert windows(lo, lo) == [(lo, lo)]
        assert windows(hi, lo) == []

    def test_windows_for_dates_groups_nearby_dates(self):
        d = dt.date(2025, 10, 7)
        got = windows_for_dates([d + dt.timedelta(days=k) for k in (0, 3, 29, 30, 90)],
                                days=30)
        assert got == [(d, d + dt.timedelta(days=29)),
                       (d + dt.timedelta(days=30), d + dt.timedelta(days=30)),
                       (d + dt.timedelta(days=90), d + dt.timedelta(days=90))]
        assert windows_for_dates([]) == []


# ── Network (mocked) ──────────────────────────────────────────────

def _response(status, body):
    r = requests.Response()
    r.status_code, r.reason = status, "OK" if status == 200 else "Error"
    r._content = json.dumps(body).encode()
    return r


class TestFetchReport:
    def test_one_request_with_limit_minus_one(self, monkeypatch, no_pause):
        calls = []

        def fake_get(url, params=None, timeout=None):
            calls.append((url, dict(params)))
            return _response(200, FIXTURE["timeonice"])

        monkeypatch.setattr(nhl_stats.requests, "get", fake_get)
        rows = nhl_stats.fetch_report("timeonice", "gameId=2025020001")
        assert len(rows) == 5
        (url, params), = calls
        assert url == "https://api.nhle.com/stats/rest/en/skater/timeonice"
        assert params["isGame"] == "true" and params["isAggregate"] == "false"
        assert params["limit"] == -1 and params["start"] == 0
        assert params["cayenneExp"] == "gameId=2025020001"
        assert "gameId" in params["sort"]           # stable order for paging

    def test_pages_the_rest_when_a_response_is_cut_short(self, monkeypatch, no_pause):
        """An explicit limit above 100 comes back as 100 rows; if limit=-1
        ever does the same, the rest is paged with start=."""
        data = FIXTURE["timeonice"]["data"]
        calls = []

        def fake_get(url, params=None, timeout=None):
            calls.append((params["start"], params["limit"]))
            start = params["start"]
            size = 2 if params["limit"] == -1 else params["limit"]
            return _response(200, {"data": data[start:start + size], "total": 5})

        monkeypatch.setattr(nhl_stats.requests, "get", fake_get)
        monkeypatch.setattr(nhl_stats, "PAGE_SIZE", 2)
        rows = nhl_stats.fetch_report("timeonice", "x")
        assert [r["playerId"] for r in rows] == [r["playerId"] for r in data]
        assert calls == [(0, -1), (2, 2), (4, 2)]

    def test_failure_after_retries_is_none_and_logged(self, monkeypatch, no_pause, caplog):
        calls = []

        def fake_get(url, params=None, timeout=None):
            calls.append(1)
            return _response(503, {"message": "busy"})

        monkeypatch.setattr(nhl_stats.requests, "get", fake_get)
        assert nhl_stats.fetch_report("powerplay", "x") is None
        assert len(calls) == nhl_stats.RETRIES
        assert "NHL stats request failed after 3 attempts: HTTP 503" in caplog.text

    def test_retry_recovers(self, monkeypatch, no_pause):
        answers = [requests.ConnectionError("boom"), _response(200, FIXTURE["powerplay"])]

        def fake_get(url, params=None, timeout=None):
            a = answers.pop(0)
            if isinstance(a, Exception):
                raise a
            return a

        monkeypatch.setattr(nhl_stats.requests, "get", fake_get)
        assert len(nhl_stats.fetch_report("powerplay", "x")) == 5


class TestFillWindows:
    def test_a_failed_report_skips_the_whole_window(self, monkeypatch, no_pause, caplog):
        written = []
        monkeypatch.setattr(nhl_stats, "ensure_columns", lambda: None)
        monkeypatch.setattr(nhl_stats, "apply_updates",
                            lambda rows: written.append(rows) or (len(rows), len(rows)))

        def fetch(report, exp):
            if "2025-10-08" in exp and report == "powerplay":
                return None
            return REPORT_ROWS[report]

        monkeypatch.setattr(nhl_stats, "fetch_report", fetch)
        d = dt.date(2025, 10, 7)
        summary = nhl_stats.fill_windows([(d, d), (d + dt.timedelta(days=1),) * 2])
        assert summary["failed_windows"] == 1
        assert len(written) == 1                  # only the good window wrote
        assert summary["matched"] == summary["fetched"] == 5
        assert summary["requests"] == 5           # 3 + 2 (stopped at the failure)
        assert "2025-10-08 to 2025-10-08 skipped, the powerplay report failed" in caplog.text


# ── CLI ───────────────────────────────────────────────────────────

class TestCli:
    def test_dispatch(self, monkeypatch, capsys):
        calls = []
        empty = {"windows": 0, "requests": 0, "fetched": 0, "matched": 0,
                 "changed": 0, "unmatched": 0, "failed_windows": 0}
        monkeypatch.setattr(nhl_stats, "fill_season",
                            lambda s: calls.append(("season", s)) or empty)
        monkeypatch.setattr(nhl_stats, "fill_range",
                            lambda a, b, season=None: calls.append(("range", a, b)) or empty)
        monkeypatch.setattr(nhl_stats, "fill_missing",
                            lambda: calls.append(("missing",)) or dict(empty, failed_windows=1))
        assert nhl_stats.main(["--season", "20252026"]) == 0
        assert nhl_stats.main(["--from", "2025-10-07", "--to", "2025-10-13"]) == 0
        assert nhl_stats.main(["--missing"]) == 1      # a failed window: exit 1
        assert calls == [("season", 20252026),
                         ("range", dt.date(2025, 10, 7), dt.date(2025, 10, 13)),
                         ("missing",)]
        assert "request(s) over" in capsys.readouterr().out

    @pytest.mark.parametrize("argv", [
        [], ["--season", "20252027"], ["--season", "2025"],
        ["--to", "2025-10-13"], ["--season", "20252026", "--to", "2025-10-13"],
        ["--season", "20252026", "--from", "2025-10-07"],
        ["--from", "2025-10-13", "--to", "2025-10-07"],
    ])
    def test_bad_arguments_exit_2_and_fetch_nothing(self, argv, monkeypatch):
        for name in ("fill_season", "fill_missing", "fill_windows"):
            monkeypatch.setattr(nhl_stats, name,
                                lambda *a, **k: pytest.fail("fetched with bad arguments"))
        with pytest.raises(SystemExit) as exc:
            nhl_stats.main(argv)
        assert exc.value.code == 2

    def test_module_help_runs_nothing(self):
        out = subprocess.run(
            [sys.executable, "-m", "ingestion.nhl_stats", "--help"],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
            env={**os.environ, "PYTHONPATH": str(ROOT),
                 "POSTGRES_HOST": "127.0.0.1", "POSTGRES_PORT": "1"})
        assert out.returncode == 0, out.stderr
        assert "usage:" in out.stdout and "--season" in out.stdout
        assert "--missing" in out.stdout


# ── Database ──────────────────────────────────────────────────────

GAME_ID = 9_999_020_101          # synthetic, far outside real id ranges
GAME_DATE = dt.date(2031, 11, 15)
SEASON = 20312032


def _synthetic_rows():
    """The fixture's rows, moved onto the synthetic game."""
    return {rep: [dict(r, gameId=GAME_ID) for r in rows]
            for rep, rows in REPORT_ROWS.items()}


@requires_db
class TestDatabase:
    @pytest.fixture()
    def game(self):
        nhl_stats.ensure_columns()
        pids = [r["playerId"] for r in REPORT_ROWS["timeonice"]]
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM raw.skater_games WHERE game_id = :g"), {"g": GAME_ID})
            conn.execute(text("DELETE FROM raw.games WHERE game_id = :g"), {"g": GAME_ID})
            conn.execute(text("""
                INSERT INTO raw.games (game_id, season, game_type, date, home_team,
                                       away_team, home_score, away_score, game_state)
                VALUES (:g, :s, 2, :d, 'FLA', 'CHI', 3, 2, 'FINAL')
            """), {"g": GAME_ID, "s": SEASON, "d": GAME_DATE})
            conn.execute(text("""
                INSERT INTO raw.skater_games (player_id, game_id, team, toi_seconds)
                VALUES (:p, :g, 'FLA', 900)
            """), [{"p": p, "g": GAME_ID} for p in pids])
        yield pids
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM raw.skater_games WHERE game_id = :g"), {"g": GAME_ID})
            conn.execute(text("DELETE FROM raw.games WHERE game_id = :g"), {"g": GAME_ID})

    def _stored(self):
        with engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT player_id, {', '.join(COLUMNS)}, stats_filled_at
                FROM raw.skater_games WHERE game_id = :g
            """), {"g": GAME_ID}).mappings().all()
        return {r["player_id"]: dict(r) for r in rows}

    def test_ensure_columns_is_idempotent(self):
        nhl_stats._ensured = False
        nhl_stats.ensure_columns()
        nhl_stats._ensured = False
        nhl_stats.ensure_columns()
        with engine.connect() as conn:
            assert conn.execute(text("""
                SELECT COUNT(*) FROM information_schema.columns
                WHERE table_schema = 'raw' AND table_name = 'skater_games'
                  AND column_name = 'stats_filled_at'
            """)).scalar() == 1

    def test_fills_then_rewrites_nothing(self, game):
        before = self._stored()
        assert all(r["pp_toi_seconds"] == 0 and r["stats_filled_at"] is None
                   for r in before.values())       # the boxscore load's defaults
        extra = {"gameId": GAME_ID, "playerId": 1, "ppTimeOnIce": 9}   # no stored row
        rows = merge_reports({**_synthetic_rows(),
                              "timeonice": _synthetic_rows()["timeonice"] + [extra]})
        assert nhl_stats.apply_updates(rows) == (5, 5)
        after = self._stored()
        assert (after[8484144]["pp_toi_seconds"], after[8484144]["sh_toi_seconds"]) == (247, 5)
        assert (after[8484144]["fow"], after[8484144]["fol"]) == (3, 14)
        assert after[8477409]["pp_goals"] == 1 and after[8482713]["pp_assists"] == 1
        assert all(r["stats_filled_at"] is not None for r in after.values())

        assert nhl_stats.apply_updates(rows) == (5, 0)          # idempotent
        corrected = [dict(r, pp_goals=2) if r["player_id"] == 8477409 else r
                     for r in rows]
        assert nhl_stats.apply_updates(corrected) == (5, 1)     # a stats correction
        assert self._stored()[8477409]["pp_goals"] == 2

    def test_report_without_the_player_keeps_stored_values(self, game):
        nhl_stats.apply_updates(merge_reports(_synthetic_rows()))
        # A later run where the faceoff report lacks one player and his PP
        # time was corrected: the row is rewritten with the new PP time,
        # and his stored faceoffs are left alone (not NULLed)
        partial = _synthetic_rows()
        partial["faceoffwins"] = [r for r in partial["faceoffwins"]
                                  if r["playerId"] != 8484144]
        partial["timeonice"] = [dict(r, ppTimeOnIce=250) if r["playerId"] == 8484144 else r
                                for r in partial["timeonice"]]
        rows = merge_reports(partial)
        assert next(r for r in rows if r["player_id"] == 8484144)["fow"] is None
        assert nhl_stats.apply_updates(rows) == (5, 1)
        stored = self._stored()[8484144]
        assert (stored["fow"], stored["fol"]) == (3, 14)
        assert stored["pp_toi_seconds"] == 250

    def test_missing_mode_fetches_only_unfilled_dates(self, game, monkeypatch, no_pause):
        exps = []

        def fetch(report, exp):
            exps.append(exp)
            return _synthetic_rows()[report]

        monkeypatch.setattr(nhl_stats, "fetch_report", fetch)
        summary = nhl_stats.fill_missing(SEASON)
        assert summary["changed"] == 5 and summary["failed_windows"] == 0
        assert exps == [cayenne(GAME_DATE, GAME_DATE, SEASON)] * 3
        assert nhl_stats.fill_rates(GAME_DATE, GAME_DATE)["filled"] == 5

        exps.clear()
        assert nhl_stats.fill_missing(SEASON)["requests"] == 0   # nothing left
        assert exps == []
