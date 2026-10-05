"""
Tests for ingestion/nhl_game_info.py: parsing a recorded right-rail
response (tests/fixtures/nhl_right_rail_2025020500.json, its real gameInfo
block), the hand-off from ingestion/nhl_api.ingest_team_stats (no extra
request, never fatal), the fetch loop, and the command line. No network.
The database tests at the end run only against a disposable copy (see
tests/conftest.py) and use synthetic games in season 20302031, which they
delete.
"""
import datetime as dt
import json
from pathlib import Path

import pytest
from sqlalchemy import text

from config.settings import check_db_connection, engine
from ingestion import nhl_game_info as gi
from ingestion.polite import Reply

requires_db = pytest.mark.skipif(not check_db_connection(),
                                 reason="needs a disposable database (see tests/conftest.py)")

FIXTURE = Path(__file__).parent / "fixtures" / "nhl_right_rail_2025020500.json"
GAME = 2025020500


def body():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_parse_recorded_response():
    p = gi.parse_game_info(body(), GAME, "NYR", "MTL")
    assert p["status"] == "ok"
    info = p["info"]
    assert info["n_referees"] == 2 and info["n_linesmen"] == 2
    assert info["home_coach"] and info["away_coach"]
    assert info["n_scratches_home"] + info["n_scratches_away"] == len(p["scratches"]) == 5
    teams = {s["team"] for s in p["scratches"]}
    assert teams == {"NYR", "MTL"}
    assert all(s["player_name"] and s["player_id"] for s in p["scratches"])
    roles = sorted(o["role"] for o in p["officials"])
    assert roles == ["linesman", "linesman", "referee", "referee"]
    assert all(isinstance(o["sweater_number"], int) for o in p["officials"])


def test_parse_missing_block_and_odd_entries():
    assert gi.parse_game_info({}, GAME, "A", "B")["status"] == "empty"
    assert gi.parse_game_info({"gameInfo": {}}, GAME, "A", "B")["status"] == "empty"
    assert gi.parse_game_info(None, GAME, "A", "B")["status"] == "empty"
    odd = {"gameInfo": {
        "referees": [{"fullName": {"default": " Ref One "}, "sweaterNumber": "x"},
                     {"fullName": {"default": "Ref One"}},           # repeated
                     {"fullName": {"default": ""}}],                 # no name
        "homeTeam": {"scratches": [{"id": 1, "firstName": {"default": "A"},
                                    "lastName": {"default": "B"}},
                                   {"id": 1}, {"id": None}]},       # repeat, no id
        "awayTeam": {"headCoach": "Plain String"}}}
    p = gi.parse_game_info(odd, GAME, "HOM", "AWY")
    assert p["status"] == "ok"
    assert p["officials"] == [{"game_id": GAME, "role": "referee",
                               "official_name": "Ref One", "sweater_number": None}]
    assert p["scratches"] == [{"game_id": GAME, "team": "HOM", "player_id": 1,
                               "player_name": "A B"}]
    assert p["info"]["away_coach"] == "Plain String" and p["info"]["home_coach"] is None


def test_team_stats_hands_its_response_over_without_a_second_request(monkeypatch):
    from ingestion import nhl_api
    gets, handed = [], []

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {**body(), "teamGameStats": []}    # stop before the team-stats write
    monkeypatch.setattr(nhl_api.requests, "get", lambda *a, **k: gets.append(a) or Resp())
    monkeypatch.setattr(gi, "store_payload",
                        lambda gid, h, a, b, db=None: handed.append((gid, h, a, "gameInfo" in b)))
    assert nhl_api.ingest_team_stats(GAME, "NYR", "MTL") is False
    assert len(gets) == 1
    assert handed == [(GAME, "NYR", "MTL", True)]


def test_game_info_failure_never_fails_team_stats(monkeypatch, caplog):
    from ingestion import nhl_api

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"gameInfo": {}, "teamGameStats": []}

    def boom(*a, **k):
        raise RuntimeError("database gone")
    monkeypatch.setattr(nhl_api.requests, "get", lambda *a, **k: Resp())
    monkeypatch.setattr(gi, "store_payload", boom)
    assert nhl_api.ingest_team_stats(GAME, "NYR", "MTL") is False   # no team stats
    assert "Game info (scratches, officials) not stored" in caplog.text


class StubClient:
    def __init__(self, replies):
        self.replies = list(replies)

    def get_json(self, url, params=None):
        return self.replies.pop(0)


def test_fetch_games_counts_and_logs_errors(monkeypatch):
    stored = []
    monkeypatch.setattr(gi, "ensure_tables", lambda db=None: None)
    monkeypatch.setattr(gi, "store", lambda gid, parsed, problem=None, db=None:
                        stored.append((gid, parsed["status"], problem)))
    client = StubClient([Reply("ok", body()), Reply("ok", {}),
                         Reply("error", problem="HTTP 503")])
    counts = gi.fetch_games([(1, "NYR", "MTL"), (2, "A", "B"), (3, "C", "D")], client=client)
    assert (counts["ok"], counts["empty"], counts["error"]) == (1, 1, 1)
    assert counts["scratches"] == 5 and counts["officials"] == 4
    assert stored == [(1, "ok", None), (2, "empty", None), (3, "error", "HTTP 503")]


def test_help(capsys):
    with pytest.raises(SystemExit) as exc:
        gi.main(["--help"])
    assert exc.value.code == 0
    assert "scratched" in capsys.readouterr().out


# ── Database (disposable copy only) ───────────────────────────────

SEASON = 20302031
G1 = 2030020011


@pytest.fixture
def game():
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO raw.games (game_id, season, game_type, date, home_team,
                                   away_team, game_state)
            VALUES (:g, :s, 2, :d, 'NYR', 'MTL', 'OFF') ON CONFLICT (game_id) DO NOTHING
        """), {"g": G1, "s": SEASON, "d": dt.date(2030, 11, 1)})
    yield
    with engine.begin() as conn:
        for t in ("raw.game_scratches", "raw.game_officials", "raw.game_info"):
            conn.execute(text(f"DELETE FROM {t} WHERE game_id = :g"), {"g": G1})
        conn.execute(text("DELETE FROM raw.games WHERE game_id = :g"), {"g": G1})


@requires_db
def test_store_replaces_and_a_later_failure_keeps_the_good_rows(game):
    gi.ensure_tables(engine)
    assert gi.store_payload(G1, "NYR", "MTL", body(), db=engine) == "ok"
    assert gi.store_payload(G1, "NYR", "MTL", body(), db=engine) == "ok"   # no duplicates
    gi.store(G1, {"status": "error"}, problem="HTTP 503", db=engine)
    with engine.connect() as conn:
        n_s = conn.execute(text("SELECT COUNT(*) FROM raw.game_scratches WHERE game_id = :g"),
                           {"g": G1}).scalar()
        n_o = conn.execute(text("SELECT COUNT(*) FROM raw.game_officials WHERE game_id = :g"),
                           {"g": G1}).scalar()
        st = conn.execute(text("SELECT status, attempts FROM raw.game_info WHERE game_id = :g"),
                          {"g": G1}).one()
    assert (n_s, n_o) == (5, 4)
    assert tuple(st) == ("ok", 3)
    assert gi.games_to_fetch(season=SEASON) == []
