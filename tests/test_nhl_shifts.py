"""
Tests for ingestion/nhl_shifts.py: parsing a recorded shiftcharts
response (tests/fixtures/nhl_shiftcharts_2025020500.json, four real
shifts and one goal marker), the ok / partial / empty / error outcomes,
cleaning bad source rows (another team's shifts, the same shift twice),
the QA check against box-score ice time and the 'suspect' outcome, the
early stop after repeated failures, and the command line. No network
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


def shift(pid, start="00:00", end="00:40", sid=None, game=GAME, period=1, dur="00:40",
          team="NYR"):
    return {"id": sid, "gameId": game, "typeCode": 517, "playerId": pid, "period": period,
            "startTime": start, "endTime": end, "duration": dur, "shiftNumber": 1,
            "teamAbbrev": team}


def full_game(n_players=36, per_player=14, teams=("NYR",), game=GAME):
    """n_players x per_player shifts of 40 s each (560 s a player with the
    defaults); players alternate between the given teams."""
    data, sid = [], 1
    for p in range(n_players):
        for k in range(per_player):
            data.append(shift(8470000 + p, f"{k:02d}:00", f"{k:02d}:40", sid=sid, game=game,
                              team=teams[p % len(teams)]))
            sid += 1
    return {"data": data, "total": len(data)}


def box_of(n_players=36, toi=14 * 40):
    """Box-score ice time that matches full_game()."""
    return {8470000 + p: toi for p in range(n_players)}


def no_db(monkeypatch, stored, teams=("NYR", "BOS"), box=None):
    """fetch_games without a database: each store is recorded in `stored`
    as (game, status, rows, source, problem), and the game's teams and box
    score come from the arguments."""
    monkeypatch.setattr(ns, "ensure_tables", lambda db=None: None)
    monkeypatch.setattr(ns, "store_game", lambda gid, rows, goals, status, problem=None,
                        db=None, source="api": stored.append((gid, status, len(rows),
                                                              source, problem)))
    monkeypatch.setattr(ns, "game_context", lambda gid, db=None: (
        list(teams), box_of() if box is None else box))


def mmss_text(seconds):
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def html_game():
    """(the recorded HTML report's rows for 2024021235, their summed ice
    time per player, the report page, the boxscore excerpt)."""
    page = TOI_FIXTURE.read_text(encoding="utf-8")
    box = json.loads(BOX_FIXTURE.read_text(encoding="utf-8"))
    rows, _ = ns.html_rows(HTML_GAME, {"H": page}, box)
    toi = {}
    for r in rows:
        toi[r["player_id"]] = toi.get(r["player_id"], 0) + r["duration"]
    return rows, toi, page, box


def test_mmss():
    assert ns.mmss("07:58") == 478
    assert ns.mmss("00:00") == 0
    assert ns.mmss(None) is None and ns.mmss("7.58") is None and ns.mmss("") is None


def test_parse_recorded_response():
    rows, goals, dropped = ns.parse_shifts(body(), GAME)
    assert dropped == {"wrong_team": 0, "duplicates": 0}
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
    rows, goals, _ = ns.parse_shifts({"data": data}, GAME)
    assert [r["player_id"] for r in rows] == [1, 6]
    assert rows[1]["duration"] == 30 and goals == 0


def test_clean_rows_drops_other_teams_and_repeated_shifts():
    # As the API sent them for 2021020513 (NYI-WSH): STL and MIN shifts
    # mixed in, and the same shift under two NHL row ids
    data = [shift(1, "00:00", "00:40", sid=21, team="NYI"),
            shift(1, "00:00", "00:45", sid=20, team="NYI", dur="00:45"),   # repeat, lower id
            shift(1, "01:00", "01:40", sid=22, team="NYI"),
            shift(2, "00:00", "00:40", sid=23, team="WSH"),
            shift(3, "00:00", "00:40", sid=24, team="STL"),
            shift(4, "00:00", "00:40", sid=25, team="MIN"),
            shift(5, "00:00", "00:40", sid=26, team="WSH", period=2),
            shift(5, "00:00", "00:40", sid=27, team="WSH", period=3)]       # other period: kept
    rows, _, dropped = ns.parse_shifts({"data": data}, GAME, teams=["WSH", "NYI"])
    assert dropped == {"wrong_team": 2, "duplicates": 1}
    assert {r["team"] for r in rows} == {"NYI", "WSH"}
    assert sorted(r["nhl_shift_id"] for r in rows) == [20, 22, 23, 26, 27]  # lowest id kept
    keys = [(r["player_id"], r["period"], r["start_time"]) for r in rows]
    assert len(keys) == len(set(keys))
    # No teams known: only the repeats go
    rows, _, dropped = ns.parse_shifts({"data": data}, GAME)
    assert dropped == {"wrong_team": 0, "duplicates": 1} and len(rows) == 7
    # Rows with no NHL id (the HTML reports): the first one seen is kept
    plain = [{"player_id": 1, "period": 1, "start_time": 0, "duration": d, "team": "NYI",
              "nhl_shift_id": None} for d in (40, 45)]
    kept, dropped = ns.clean_rows(plain, ["NYI"])
    assert [r["duration"] for r in kept] == [40] and dropped["duplicates"] == 1
    assert ns.dropped_note(dropped) == "dropped 1 repeated shifts"
    assert ns.dropped_note({"wrong_team": 3, "duplicates": 0}) == "dropped 3 rows of another team"
    assert ns.dropped_note({"wrong_team": 0, "duplicates": 0}) is None


def test_qa_check_against_box_score_ice_time():
    rows, _, _ = ns.parse_shifts(full_game(), GAME)
    assert ns.qa_problem(rows, box_of()) is None
    assert "1 skater(s) more than 60 s" in ns.qa_problem(rows, {**box_of(), 8470003: 560 + 61})
    assert ns.qa_problem(rows, {**box_of(), 8470003: 560 + 60}) is None   # at the tolerance
    missing = {**box_of(), 9999999: 300}                                 # played, no shift
    assert "player 9999999, shifts 0 s against 300 s" in ns.qa_problem(rows, missing)
    assert ns.qa_problem(rows, {}) == "no box-score ice time to check against"
    # A repeated shift that got through would double a skater's time
    doubled = rows + [dict(r) for r in rows if r["player_id"] == 8470000]
    assert "player 8470000, shifts 1120 s" in ns.qa_problem(doubled, box_of())


def test_classify():
    assert ns.classify([]) == "empty"
    rows, _, _ = ns.parse_shifts(body(), GAME)
    assert ns.classify(rows) == "partial"
    rows, _, _ = ns.parse_shifts(full_game(), GAME)
    assert ns.classify(rows) == "ok"
    rows, _, _ = ns.parse_shifts(full_game(n_players=20, per_player=30), GAME)
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

    # Only another game's teams (2025020565, NJD-BUF, held VGK and SJS): empty
    status, rows, _, problem = ns.fetch_game(
        StubClient([Reply("ok", full_game(teams=("VGK", "SJS")))]), GAME, ["BUF", "NJD"])
    assert (status, rows) == ("empty", []) and "504 rows of another team" in problem


def test_fetch_games_marks_a_game_that_fails_qa_suspect(monkeypatch):
    stored = []
    no_db(monkeypatch, stored, box={**box_of(), 8470005: 900})
    client = StubClient([Reply("ok", full_game(teams=("NYR", "BOS")))])
    counts = ns.fetch_games([GAME], client=client, html_fallback=False)
    assert counts["suspect"] == 1 and counts["ok"] == 0
    _, status, n, source, problem = stored[0]
    assert (status, n, source) == ("suspect", 504, "api")
    assert problem.startswith("QA: 1 skater(s)") and "player 8470005" in problem

    # A clean game passes; the repeat and the wrong-team row are noted
    stored.clear()
    no_db(monkeypatch, stored)
    game = full_game(teams=("NYR", "BOS"))
    game["data"] += [dict(game["data"][0], id=99999),            # a repeat
                     shift(7, sid=99998, team="STL")]            # another team
    counts = ns.fetch_games([GAME], client=StubClient([Reply("ok", game)]), html_fallback=False)
    assert counts["ok"] == 1
    assert stored[0][1:4] == ("ok", 504, "api")
    assert stored[0][4] == "dropped 1 rows of another team, 1 repeated shifts"


def test_suspect_api_shifts_are_replaced_only_by_html_that_passes(monkeypatch):
    stored = []
    monkeypatch.setattr(ns, "MIN_SHIFTS", 10)
    monkeypatch.setattr(ns, "MIN_PLAYERS", 2)
    rows, toi, page, box = html_game()
    # The API's copy of the game has every shift twice as long (QA fails)
    api = {"data": [shift(r["player_id"], mmss_text(r["start_time"]), mmss_text(r["end_time"]),
                          sid=i + 1, game=HTML_GAME, period=r["period"], team="BUF",
                          dur=mmss_text(2 * r["duration"]))
                    for i, r in enumerate(rows)]}

    def client():
        return StubClient([Reply("ok", api), Reply("ok", box), Reply("ok", page),
                           Reply("ok", "<html></html>")])
    no_db(monkeypatch, stored, teams=("BUF", "CAR"), box=toi)
    counts = ns.fetch_games([HTML_GAME], client=client())
    assert counts["ok"] == 1 and counts["from_html"] == 1
    assert stored[0][1:4] == ("ok", 47, "html")
    assert "failed the QA check" in stored[0][4]

    # HTML that fails too: the API's shifts stay, 'suspect', and it says so
    stored.clear()
    no_db(monkeypatch, stored, teams=("BUF", "CAR"), box={**toi, 8481524: 1})
    counts = ns.fetch_games([HTML_GAME], client=client())
    assert counts["suspect"] == 1 and counts["from_html"] == 0
    assert stored[0][1:4] == ("suspect", 47, "api")
    assert "the HTML reports did not pass either" in stored[0][4]

    # No box score yet: suspect, and the HTML reports are not tried
    stored.clear()
    no_db(monkeypatch, stored, teams=("BUF", "CAR"), box={})
    one = StubClient([Reply("ok", api)])
    ns.fetch_games([HTML_GAME], client=one)
    assert stored[0][1] == "suspect" and len(one.urls) == 1
    assert "no box-score ice time" in stored[0][4]


def test_fetch_games_stops_after_repeated_errors(monkeypatch):
    stored = []
    no_db(monkeypatch, stored)
    client = StubClient([Reply("error", problem="down")] * 10)
    counts = ns.fetch_games(list(range(10)), client=client, max_consecutive_errors=3)
    assert counts["stopped_early"] == 1 and counts["error"] == 3
    assert [s[1] for s in stored] == ["error"] * 3     # each failure is logged


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
    monkeypatch.setattr(ns, "MIN_SHIFTS", 10)
    monkeypatch.setattr(ns, "MIN_PLAYERS", 2)
    _, toi, page, box = html_game()
    no_db(monkeypatch, stored, teams=("BUF", "CAR"), box=toi)
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


def test_game_argument_refetches_just_those_games(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(ns, "fetch_games", lambda ids: seen.append(ids) or ns._empty_counts(len(ids)))
    monkeypatch.setattr(ns, "fetch_missing", lambda *a, **k: pytest.fail("not a full run"))
    assert ns.main(["--game", "2021020513", "--game", "2025020565"]) == 0
    assert seen == [[2021020513, 2025020565]]


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
    rows, goals, _ = ns.parse_shifts(full_game(teams=("BOS", "OTT"), game=G1), G1)
    assert len(rows) == 504
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


def _box(conn, gid, toi):
    """raw.skater_games rows for a synthetic game: {player: ice time}."""
    for pid, t in toi.items():
        conn.execute(text("""
            INSERT INTO raw.skater_games (game_id, player_id, team, toi_seconds)
            VALUES (:g, :p, 'BOS', :t)
        """), {"g": gid, "p": pid, "t": t})


@pytest.fixture
def box_cleanup():
    yield
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM raw.skater_games WHERE game_id IN (:a, :b, :c)"),
                     {"a": G1, "b": G2, "c": G3})


@requires_db
def test_store_guard_drops_other_teams_and_reclassifies(games):
    ns.ensure_tables(engine)
    # Rows that were never cleaned (another team's only): nothing is stored
    # and the game is logged 'empty', not 'ok'
    rows, goals, _ = ns.parse_shifts(full_game(teams=("VGK", "SJS"), game=G1), G1)
    ns.store_game(G1, rows, goals, "ok")
    with engine.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM raw.shifts WHERE game_id = :g"),
                         {"g": G1}).scalar()
        f = conn.execute(text("SELECT status, n_shifts FROM raw.shift_fetches "
                              "WHERE game_id = :g"), {"g": G1}).one()
    assert n == 0 and tuple(f) == ("empty", 0)


@requires_db
def test_recheck_repairs_stored_rows(games, box_cleanup):
    """The repair for rows loaded before the cleaning existed: repeated
    shifts and another team's rows are deleted, the status follows the QA
    check, and a second run changes nothing."""
    ns.ensure_tables(engine)
    clean, _, _ = ns.parse_shifts(full_game(teams=("BOS", "OTT"), game=G1), G1)
    bad = clean + [dict(r, nhl_shift_id=r["nhl_shift_id"] + 100000)
                   for r in clean if r["player_id"] == 8470000]
    bad += [dict(clean[0], player_id=1, team="STL", nhl_shift_id=900001)]
    cols = ["game_id", "player_id", "period", "start_time", "end_time", "duration", "team",
            "nhl_shift_id", "shift_number"]
    with engine.begin() as conn:            # as the old loader stored them
        conn.execute(ns.INSERT_SHIFTS, {c: [r[c] for r in bad] for c in cols})
        conn.execute(ns.UPSERT_FETCH, {"game_id": G1, "status": "ok", "n_shifts": len(bad),
                                       "n_players": 37, "n_goal_events": 0,
                                       "problem": None, "source": "api"})
        _box(conn, G1, box_of())
    counts = ns.recheck_stored(game_ids=[G1], db=engine)
    assert counts["games"] == 1 and counts["ok"] == 1
    assert counts["duplicates_deleted"] == 14 and counts["wrong_team_deleted"] == 1
    with engine.connect() as conn:
        kept = conn.execute(text("SELECT COUNT(*), MAX(nhl_shift_id) FROM raw.shifts "
                                 "WHERE game_id = :g"), {"g": G1}).one()
        f = conn.execute(text("SELECT status, n_shifts, n_players, problem FROM "
                              "raw.shift_fetches WHERE game_id = :g"), {"g": G1}).one()
    assert tuple(kept) == (504, max(r["nhl_shift_id"] for r in clean))   # lowest ids kept
    assert tuple(f) == ("ok", 504, 36, "dropped 1 rows of another team, 14 repeated shifts")
    rep = {r["season"]: r for r in ns.coverage(engine)}[SEASON]
    assert rep["repeated_shift_rows"] == 0 and rep["wrong_team_rows"] == 0
    assert rep["skater_games_over_tolerance"] == 0

    # A box score that disagrees: 'suspect', with the reason; then the same
    # check again gives the same text (no repeated notes)
    with engine.begin() as conn:
        conn.execute(text("UPDATE raw.skater_games SET toi_seconds = 900 "
                          "WHERE game_id = :g AND player_id = 8470001"), {"g": G1})
    for _ in range(2):
        counts = ns.recheck_stored(game_ids=[G1], db=engine)
        with engine.connect() as conn:
            status, problem = conn.execute(text(
                "SELECT status, problem FROM raw.shift_fetches WHERE game_id = :g"),
                {"g": G1}).one()
        assert counts["suspect"] == 1 and status == "suspect"
        assert problem == ("dropped 1 rows of another team, 14 repeated shifts; QA: 1 skater(s) "
                           "more than 60 s from box-score ice time (worst: player 8470001, "
                           "shifts 560 s against 900 s)")
    rep = {r["season"]: r for r in ns.coverage(engine)}[SEASON]
    assert rep["suspect"] == 1 and rep["ok"] == 0


@requires_db
def test_recheck_empties_a_game_holding_only_another_games_shifts(games, box_cleanup):
    ns.ensure_tables(engine)
    other, _, _ = ns.parse_shifts(full_game(teams=("VGK", "SJS"), game=G1), G1)
    cols = ["game_id", "player_id", "period", "start_time", "end_time", "duration", "team",
            "nhl_shift_id", "shift_number"]
    with engine.begin() as conn:
        conn.execute(ns.INSERT_SHIFTS, {c: [r[c] for r in other] for c in cols})
        conn.execute(ns.UPSERT_FETCH, {"game_id": G1, "status": "ok", "n_shifts": len(other),
                                       "n_players": 36, "n_goal_events": 0,
                                       "problem": None, "source": "api"})
    counts = ns.recheck_stored(game_ids=[G1], db=engine)
    assert counts["empty"] == 1 and counts["wrong_team_deleted"] == 504
    assert ns.games_to_fetch(season=SEASON, retry_empty=True) == [G1]   # re-fetched later


@requires_db
def test_games_to_fetch_retries_recent_suspect_games(games):
    ns.ensure_tables(engine)
    assert ns.games_to_fetch(season=SEASON) == [G1]
    ns.store_game(G1, [], 0, "suspect")
    assert ns.games_to_fetch(season=SEASON) == []            # 30 days old: left alone
    assert ns.games_to_fetch(season=SEASON, retry_empty=True) == [G1]
    recent = dt.date.today() - dt.timedelta(days=2)
    with engine.begin() as conn:
        conn.execute(text("UPDATE raw.games SET date = :d WHERE game_id = :g"),
                     {"g": G1, "d": recent})
    assert ns.games_to_fetch(season=SEASON) == [G1]          # recent suspect: retried
