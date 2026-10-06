"""
Tests for ingestion/nhl_shifts.py: parsing a recorded shiftcharts
response (tests/fixtures/nhl_shiftcharts_2025020500.json, four real
shifts and one goal marker), the ok / partial / empty / error outcomes,
the early stop after repeated failures, and the command line. No network
(the client is a stub). The database tests at the end run only against a
disposable copy (see tests/conftest.py) and use synthetic games in
season 20302031, which they delete.
"""
import datetime as dt
import json
from pathlib import Path

import pytest
from sqlalchemy import text

from config.settings import check_db_connection, engine
from ingestion import nhl_shifts as ns
from ingestion.polite import Reply

requires_db = pytest.mark.skipif(not check_db_connection(),
                                 reason="needs a disposable database (see tests/conftest.py)")

FIXTURE = Path(__file__).parent / "fixtures" / "nhl_shiftcharts_2025020500.json"
GAME = 2025020500


def body():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class StubClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.urls = []

    def get_json(self, url, params=None):
        self.calls.append(params)
        self.urls.append(url)
        return self.replies.pop(0)

    get_text = get_json


TOI_FIXTURE = Path(__file__).parent / "fixtures" / "nhl_toi_report_2024021235_TH_excerpt.htm"
BOX_FIXTURE = Path(__file__).parent / "fixtures" / "nhl_boxscore_2024021235_excerpt.json"
HTML_GAME = 2024021235


def shift(pid, start="00:00", end="00:40", sid=None, game=GAME, period=1, dur="00:40"):
    return {"id": sid, "gameId": game, "typeCode": 517, "playerId": pid, "period": period,
            "startTime": start, "endTime": end, "duration": dur, "shiftNumber": 1,
            "teamAbbrev": "NYR"}


def full_game(n_players=36, per_player=14):
    data, sid = [], 1
    for p in range(n_players):
        for k in range(per_player):
            data.append(shift(8470000 + p, f"{k:02d}:00", f"{k:02d}:40", sid=sid))
            sid += 1
    return {"data": data, "total": len(data)}


def test_mmss():
    assert ns.mmss("07:58") == 478
    assert ns.mmss("00:00") == 0
    assert ns.mmss(None) is None and ns.mmss("7.58") is None and ns.mmss("") is None


def test_parse_recorded_response():
    rows, goals = ns.parse_shifts(body(), GAME)
    assert goals == 1                       # the goal marker is counted, not stored
    assert len(rows) == 4
    r = rows[0]
    assert set(r) == {"game_id", "player_id", "period", "start_time", "end_time",
                      "duration", "team", "nhl_shift_id", "shift_number"}
    assert r["game_id"] == GAME and r["period"] >= 1 and r["start_time"] >= 0
    assert r["duration"] == r["end_time"] - r["start_time"]
    assert all(x["nhl_shift_id"] for x in rows)


def test_parse_drops_other_games_bad_rows_and_repeats():
    data = [shift(1, sid=10), shift(1, sid=10),              # repeated NHL row id
            shift(2, sid=11, game=GAME + 1),                 # another game
            {**shift(3, sid=12), "playerId": None},          # no player
            {**shift(4, sid=13), "startTime": "bad"},        # no start
            {**shift(5, sid=14), "typeCode": 999},           # unknown row type
            shift(6, "01:00", "01:30", sid=15, dur=None)]    # duration from end - start
    rows, goals = ns.parse_shifts({"data": data}, GAME)
    assert [r["player_id"] for r in rows] == [1, 6]
    assert rows[1]["duration"] == 30 and goals == 0


def test_classify():
    assert ns.classify([]) == "empty"
    rows, _ = ns.parse_shifts(body(), GAME)
    assert ns.classify(rows) == "partial"
    rows, _ = ns.parse_shifts(full_game(), GAME)
    assert ns.classify(rows) == "ok"
    rows, _ = ns.parse_shifts(full_game(n_players=20, per_player=30), GAME)
    assert ns.classify(rows) == "partial"   # enough shifts, too few players


def test_fetch_game_outcomes():
    ok = StubClient([Reply("ok", full_game())])
    status, rows, _, problem = ns.fetch_game(ok, GAME)
    assert status == "ok" and problem is None and len(rows) == 36 * 14
    assert ok.calls == [{"cayenneExp": f"gameId={GAME}"}]

    status, rows, _, problem = ns.fetch_game(StubClient([Reply("ok", body())]), GAME)
    assert status == "partial" and "4 shifts" in problem

    truncated = full_game()
    truncated["total"] = len(truncated["data"]) + 50
    status, _, _, problem = ns.fetch_game(StubClient([Reply("ok", truncated)]), GAME)
    assert status == "partial" and "of" in problem

    assert ns.fetch_game(StubClient([Reply("ok", {"data": [], "total": 0})]), GAME)[0] == "empty"
    status, rows, _, problem = ns.fetch_game(StubClient([Reply("error", problem="HTTP 503")]), GAME)
    assert (status, rows, problem) == ("error", [], "HTTP 503")
    assert ns.fetch_game(StubClient([Reply("ok", ["not", "a", "dict"])]), GAME)[0] == "error"


def test_fetch_games_stops_after_repeated_errors(monkeypatch):
    stored = []
    monkeypatch.setattr(ns, "ensure_tables", lambda db=None: None)
    monkeypatch.setattr(ns, "store_game", lambda gid, rows, goals, status, problem=None,
                        db=None, source="api": stored.append((gid, status)))
    client = StubClient([Reply("error", problem="down")] * 10)
    counts = ns.fetch_games(list(range(10)), client=client, max_consecutive_errors=3)
    assert counts["stopped_early"] == 1 and counts["error"] == 3
    assert [s for _, s in stored] == ["error"] * 3     # each failure is logged


def test_parse_recorded_toi_report():
    # Two real players from the home report of 2024021235 (BUF), each with
    # a shift table and a per-period summary table that must be skipped
    shifts = ns.parse_toi_report(TOI_FIXTURE.read_text(encoding="utf-8"))
    by_num = {}
    for r in shifts:
        by_num.setdefault(r["sweater"], []).append(r)
    assert set(by_num) == {4, 9}
    assert len(by_num[4]) == 27 and by_num[4][-1]["shift_number"] == 27
    assert by_num[4][0] == {"sweater": 4, "shift_number": 1, "period": 1,
                            "start_time": 36, "end_time": 76, "duration": 40}
    box = json.loads(BOX_FIXTURE.read_text(encoding="utf-8"))
    toi = {p["sweaterNumber"]: ns.mmss(p["toi"])
           for g in ("forwards", "defense") for p in box["playerByGameStats"]["homeTeam"][g]}
    for num, rows in by_num.items():      # the shifts add up to the box-score ice time
        assert sum(r["duration"] for r in rows) == toi[num]
    assert ns.parse_toi_report("") == [] and ns.parse_toi_report(None) == []


def test_html_rows_map_sweaters_to_players():
    page = TOI_FIXTURE.read_text(encoding="utf-8")
    box = json.loads(BOX_FIXTURE.read_text(encoding="utf-8"))
    rows, unmatched = ns.html_rows(HTML_GAME, {"H": page}, box)
    assert unmatched == 0
    assert {r["player_id"] for r in rows} == {8481524, 8484145}
    assert {r["team"] for r in rows} == {"BUF"} and all(r["nhl_shift_id"] is None for r in rows)
    # The same report read as the visitors': sweater 4 and 9 are not CAR's
    rows, unmatched = ns.html_rows(HTML_GAME, {"V": page}, box)
    assert rows == [] and unmatched > 0
    assert ns.sweater_map({}) == {}


def test_empty_api_falls_back_to_the_html_reports(monkeypatch):
    stored = []
    monkeypatch.setattr(ns, "ensure_tables", lambda db=None: None)
    monkeypatch.setattr(ns, "store_game", lambda gid, rows, goals, status, problem=None,
                        db=None, source="api": stored.append((gid, status, len(rows),
                                                              source, problem)))
    monkeypatch.setattr(ns, "MIN_SHIFTS", 10)
    monkeypatch.setattr(ns, "MIN_PLAYERS", 2)
    page = TOI_FIXTURE.read_text(encoding="utf-8")
    box = json.loads(BOX_FIXTURE.read_text(encoding="utf-8"))
    client = StubClient([Reply("ok", {"data": [], "total": 0}), Reply("ok", box),
                         Reply("ok", page), Reply("ok", "<html></html>")])
    counts = ns.fetch_games([HTML_GAME], client=client)
    assert counts["from_html"] == 1 and counts["ok"] == 1
    assert stored[0][1:4] == ("ok", 47, "html") and "HTML" in stored[0][4]
    assert client.urls[1].endswith("/2024021235/boxscore")
    assert client.urls[2] == "https://www.nhl.com/scores/htmlreports/20242025/TH021235.HTM"
    assert client.urls[3] == "https://www.nhl.com/scores/htmlreports/20242025/TV021235.HTM"

    # A failed report keeps the game 'empty' (retried later) and says why
    stored.clear()
    client = StubClient([Reply("ok", {"data": [], "total": 0}), Reply("ok", box),
                         Reply("error", problem="HTTP 503")])
    counts = ns.fetch_games([HTML_GAME], client=client)
    assert counts["empty"] == 1 and stored[0][3] == "api"
    assert "TH report: HTTP 503" in stored[0][4]

    # Fallback off: one request only
    stored.clear()
    client = StubClient([Reply("ok", {"data": [], "total": 0})])
    ns.fetch_games([HTML_GAME], client=client, html_fallback=False)
    assert stored[0][1] == "empty" and len(client.urls) == 1


def test_season_argument_and_help(capsys):
    assert ns._season_arg("20252026") == 20252026
    with pytest.raises(Exception):
        ns._season_arg("2025")
    with pytest.raises(SystemExit) as exc:
        ns.main(["--help"])
    assert exc.value.code == 0
    assert "raw.shifts" in capsys.readouterr().out


# ── Database (disposable copy only) ───────────────────────────────

SEASON = 20302031
G1, G2, G3 = 2030020001, 2030020002, 2030020003


@pytest.fixture
def games():
    now = dt.datetime.now(dt.timezone.utc)
    with engine.begin() as conn:
        for gid, state, start in ((G1, "OFF", now - dt.timedelta(days=30)),
                                  (G2, "OFF", now - dt.timedelta(hours=2)),
                                  (G3, "FUT", now + dt.timedelta(days=1))):
            conn.execute(text("""
                INSERT INTO raw.games (game_id, season, game_type, date, start_time_utc,
                                       home_team, away_team, game_state)
                VALUES (:g, :s, 2, :d, :t, 'BOS', 'OTT', :st)
                ON CONFLICT (game_id) DO NOTHING
            """), {"g": gid, "s": SEASON, "d": start.date(), "t": start, "st": state})
    yield
    with engine.begin() as conn:
        for t in ("raw.shifts", "raw.shift_fetches"):
            conn.execute(text(f"DELETE FROM {t} WHERE game_id IN (:a, :b, :c)"),
                         {"a": G1, "b": G2, "c": G3})
        conn.execute(text("DELETE FROM raw.games WHERE season = :s"), {"s": SEASON})


@requires_db
def test_store_replaces_and_failures_keep_rows(games):
    ns.ensure_tables(engine)
    rows, goals = ns.parse_shifts(full_game(), G1)
    for r in rows:
        r["game_id"] = G1
    ns.store_game(G1, rows, goals, "ok")
    ns.store_game(G1, rows, goals, "ok")                 # re-fetch: no duplicates
    ns.store_game(G1, [], 0, "error", "HTTP 503")        # a failure deletes nothing
    with engine.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM raw.shifts WHERE game_id = :g"),
                         {"g": G1}).scalar()
        f = conn.execute(text("SELECT status, n_shifts, attempts, source FROM "
                              "raw.shift_fetches WHERE game_id = :g"), {"g": G1}).one()
    assert n == len(rows)
    assert tuple(f) == ("error", len(rows), 3, "api")   # a failure keeps the source


@requires_db
def test_games_to_fetch_resumes_and_skips_unsettled(games):
    ns.ensure_tables(engine)
    due = ns.games_to_fetch(season=SEASON)
    assert due == [G1]                       # G2 started 2 hours ago, G3 is not played
    ns.store_game(G1, [], 0, "empty")
    assert ns.games_to_fetch(season=SEASON) == []           # old empty: left alone
    assert ns.games_to_fetch(season=SEASON, retry_empty=True) == [G1]
    ns.store_game(G1, [], 0, "error", "x")
    assert ns.games_to_fetch(season=SEASON) == [G1]         # errors always retried
