"""
Tests for ingestion/props_odds.py — live player-prop lines from The Odds API.

No network: requests.get is mocked. The event-odds fixture follows the
documented response shape (the v4 guide's player-props example: outcome
name Over/Under, the player in description, the line in point), with NHL
players and made-up prices; the guide's own example is checked verbatim
too. The snapshot tests replace the database helpers with fakes; the
database tests (clone or a _test copy only) use a synthetic game,
9999020201, and delete it afterwards.
"""
import datetime as dt
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest
import requests
from sqlalchemy import text

from config.settings import check_db_connection, engine
from ingestion import odds_api, props_odds
from ingestion.props_odds import (build_player_index, match_player, parse_event_odds,
                                  props_due, select_events)

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = json.loads((ROOT / "tests" / "fixtures" / "odds_api_event_odds_nhl.json")
                     .read_text(encoding="utf-8"))
UTC = dt.timezone.utc
FAKE_KEY = "fakekey0123456789abcdef0123456789"
NOW = dt.datetime(2026, 1, 15, 17, 0, tzinfo=UTC)          # noon ET

requires_db = pytest.mark.skipif(not check_db_connection(),
                                 reason="database not reachable")

# The v4 guide's "GET event odds" example, verbatim
DOCS_EXAMPLE = {
    "id": "a512a48a58c4329048174217b2cc7ce0",
    "sport_key": "americanfootball_nfl",
    "sport_title": "NFL",
    "commence_time": "2023-01-01T18:00:00Z",
    "home_team": "Atlanta Falcons",
    "away_team": "Arizona Cardinals",
    "bookmakers": [{
        "key": "draftkings",
        "title": "DraftKings",
        "markets": [{
            "key": "player_pass_tds",
            "last_update": "2023-01-01T05:31:29Z",
            "outcomes": [
                {"name": "Over", "description": "David Blough", "price": -205, "point": 0.5},
                {"name": "Under", "description": "David Blough", "price": 150, "point": 0.5},
            ],
        }],
    }],
}


def _rows_by(rows):
    return {(r["book"], r["market"], r["player_name"], r["line"]): r for r in rows}


# ── Parsing ───────────────────────────────────────────────────────

class TestParse:
    def test_documented_example(self):
        (row,) = parse_event_odds(DOCS_EXAMPLE)
        assert row == {"book": "draftkings", "market": "player_pass_tds",
                       "player_name": "David Blough", "line": 0.5,
                       "over_price": -205, "under_price": 150,
                       "book_updated_at": dt.datetime(2023, 1, 1, 5, 31, 29)}

    def test_nhl_fixture(self):
        rows = parse_event_odds(FIXTURE)
        assert len(rows) == 10                    # kalshi posted nothing
        by = _rows_by(rows)
        dk = by[("draftkings", "player_shots_on_goal", "Tage Thompson", 3.5)]
        assert (dk["over_price"], dk["under_price"]) == (105, -135)
        assert dk["book_updated_at"] == dt.datetime(2026, 1, 15, 21, 31, 7)   # naive UTC
        # a side the book doesn't offer is NULL, not dropped
        hutson = by[("fanduel", "player_shots_on_goal", "Lane Hutson", 1.5)]
        assert (hutson["over_price"], hutson["under_price"]) == (-125, None)
        # alternate lines: one row per line
        alt = [r for r in rows if r["market"] == "player_shots_on_goal_alternate"]
        assert sorted(r["line"] for r in alt) == [2.5, 3.5, 4.5]
        # yes/no: Yes is the over, No the under, no line
        cole = by[("draftkings", "player_goal_scorer_anytime", "Cole Caufield", None)]
        assert (cole["over_price"], cole["under_price"]) == (160, -210)
        tage = by[("draftkings", "player_goal_scorer_anytime", "Tage Thompson", None)]
        assert (tage["over_price"], tage["under_price"]) == (150, None)

    def test_unusable_outcomes_are_skipped(self):
        payload = {"bookmakers": [{"key": "b", "markets": [{"key": "m", "outcomes": [
            {"name": "Over", "price": -110, "point": 2.5},                       # no player
            {"name": "Buffalo Sabres", "description": "X Y", "price": -110},     # no side
            {"name": "Over", "description": "X Y", "price": None, "point": 2.5},  # no price
            {"name": "under", "description": " X Y ", "price": "120", "point": "2.5"},
        ]}]}]}
        assert parse_event_odds(payload) == [{
            "book": "b", "market": "m", "player_name": "X Y", "line": 2.5,
            "over_price": None, "under_price": 120, "book_updated_at": None}]
        assert parse_event_odds({}) == [] and parse_event_odds(None) == []


# ── Settings ──────────────────────────────────────────────────────

class TestSettings:
    def test_default_books_are_the_moneyline_default_and_one_region(self):
        sel = props_odds.book_selection({})
        assert sel == {"bookmakers": ",".join(props_odds.DEFAULT_BOOKMAKERS)}
        assert len(props_odds.DEFAULT_BOOKMAKERS) == 10
        assert props_odds.region_units(sel) == 1           # 1 credit per market
        assert {"kalshi", "polymarket"} <= set(props_odds.DEFAULT_BOOKMAKERS)
        shared = getattr(odds_api, "DEFAULT_BOOKMAKERS", None)
        if shared is not None:           # keep the two defaults the same list
            assert tuple(shared) == props_odds.DEFAULT_BOOKMAKERS

    def test_more_than_ten_books_warns(self, caplog):
        books = ",".join(f"book{i}" for i in range(11))
        sel = props_odds.book_selection({"PROPS_BOOKMAKERS": books})
        assert props_odds.region_units(sel) == 2
        assert "each market costs 2 credits a game instead of 1" in caplog.text

    def test_empty_books_fall_back_to_regions(self, caplog):
        assert props_odds.book_selection({"PROPS_BOOKMAKERS": ""}) == {"regions": "us"}
        assert props_odds.book_selection({"PROPS_BOOKMAKERS": " ",
                                          "PROPS_REGIONS": "US, us2"}) == {"regions": "us,us2"}
        assert props_odds.book_selection({"PROPS_BOOKMAKERS": "Draft Kings!"}) == {
            "regions": "us"}
        assert "names no usable bookmaker" in caplog.text

    def test_markets(self, caplog):
        assert props_odds.markets_setting(None, {}) == "player_shots_on_goal"
        env = {"PROPS_MARKETS": " player_points, PLAYER_ASSISTS,player_points "}
        assert props_odds.markets_setting(None, env) == "player_points,player_assists"
        assert props_odds.markets_setting("player_total_saves", env) == "player_total_saves"
        assert props_odds.markets_setting("bad market!", {}) == "player_shots_on_goal"
        assert "No usable market" in caplog.text

    def test_minutes(self, caplog):
        f = props_odds.minutes_setting
        assert f("X", 16, False, {}) == dt.timedelta(minutes=16)
        assert f("X", 16, False, {"X": " 30 "}) == dt.timedelta(minutes=30)
        assert f("X", 16, True, {"X": "0"}) == dt.timedelta(0)
        for bad in ("0", "-5", "nan", "4O"):
            assert f("X", 16, False, {"X": bad}) == dt.timedelta(minutes=16)
        assert "X='4O' is not a number of minutes" in caplog.text


# ── Which games, when ─────────────────────────────────────────────

def ev(eid, minutes_from_now, home="Buffalo Sabres", away="Montréal Canadiens", now=NOW):
    start = now + dt.timedelta(minutes=minutes_from_now)
    return {"id": eid, "sport_key": "icehockey_nhl", "home_team": home,
            "away_team": away, "commence_time": start.strftime("%Y-%m-%dT%H:%M:%SZ")}


class TestTiming:
    LEAD = GAP = dt.timedelta(minutes=16)

    def test_select_events_never_takes_a_started_game(self):
        events = [ev("a", 60), ev("b", -30), ev("c", 0), ev("d", 24 * 60 + 1),
                  dict(ev("e", 60), id=None), dict(ev("f", 60), commence_time="junk")]
        assert [e["id"] for e in select_events(events, NOW, dt.timedelta(hours=24))] == ["a"]

    def test_due_needs_a_start_within_the_lead_and_no_recent_snapshot(self):
        events = [ev("a", 10), ev("b", 12), ev("c", 40)]
        due, why = props_due(events, {"b": NOW - dt.timedelta(minutes=5)}, NOW,
                             self.LEAD, self.GAP)
        assert [e["id"] for e in due] == ["a"] and "first starting in 10" in why
        due, _ = props_due(events, {"b": NOW - dt.timedelta(minutes=20)}, NOW,
                           self.LEAD, self.GAP)
        assert [e["id"] for e in due] == ["a", "b"]
        assert props_due([ev("c", 40)], {}, NOW, self.LEAD, self.GAP) == (
            [], "no game starts in the next 16 minutes")
        due, why = props_due([ev("a", 10)], {"a": NOW}, NOW, self.LEAD, self.GAP)
        assert due == [] and "prop snapshot from the last 16 minutes" in why

    @pytest.mark.parametrize("late_s", [0, 30, 59])
    def test_each_game_gets_one_pregame_snapshot_on_the_15_minute_cycle(self, late_s):
        start = dt.datetime(2026, 1, 16, 0, 0, tzinfo=UTC)
        games = [ev("early", 0, now=start), ev("late", 30, now=start)]
        for offset in range(15):          # whatever minute the cycle starts on
            t = dt.datetime(2026, 1, 15, 22, offset, tzinfo=UTC)
            caps, taken = {}, []
            while t < start + dt.timedelta(hours=1):
                now = t + dt.timedelta(seconds=late_s)
                due, _ = props_due(games, caps, now, self.LEAD, self.GAP)
                for g in due:
                    caps[g["id"]] = now
                    taken.append((g["id"], now))
                t += dt.timedelta(minutes=15)
            for g in games:
                mine = [n for gid, n in taken if gid == g["id"]]
                assert len(mine) == 1, (offset, g["id"], mine)
                before = props_odds._parse_time(g["commence_time"]) - mine[0]
                assert dt.timedelta(0) < before <= dt.timedelta(minutes=16)


# ── Player names ──────────────────────────────────────────────────

INDEX = build_player_index(
    [(1, "T. Thompson"), (2, "C. Caufield"), (3, "E. Pettersson"), (4, "E. Pettersson"),
     (5, "S. Aho"), (6, "S. Aho"), (7, "T. Stützle"), (8, "R. O'Reilly"),
     (9, "E. Sharangovich"), (10, "J.J. Moser"), (11, "Z. Thompson")],
    [(1, "BUF"), (2, "MTL"), (3, "VAN"), (4, "VAN"), (5, "CAR"), (6, "NYI"),
     (7, "OTT"), (8, "NSH"), (9, "CGY"), (10, "TBL"), (11, "SEA")])


class TestMatchPlayer:
    @pytest.mark.parametrize("name,teams,expected", [
        ("Tage Thompson", ("BUF", "MTL"), (1, "ok")),        # full name -> "T. Thompson"
        ("T. Thompson", ("BUF", "MTL"), (1, "ok")),          # exact
        ("Tim Stutzle", ("OTT", "BOS"), (7, "ok")),          # accents stripped
        ("Ryan O’Reilly", ("NSH", "DAL"), (8, "ok")),   # curly apostrophe
        ("JJ Moser", ("TBL", "FLA"), (10, "ok")),
        ("Sebastian Aho", ("CAR", "BOS"), (5, "ok")),        # split by team
        ("Sebastian Aho", ("NYI", "BOS"), (6, "ok")),
        ("Sebastian Aho", ("CAR", "NYI"), (None, "ambiguous")),
        ("Elias Pettersson", ("VAN", "SEA"), (None, "ambiguous")),   # same team
        ("Yegor Sharangovich", ("CGY", "EDM"), (9, "ok")),   # surname, on-team only
        ("Yegor Sharangovich", ("BOS", "EDM"), (None, "not in raw.players")),
        ("Zach Thompson", ("BUF", "SEA"), (11, "ok")),        # initial splits the Thompsons
        ("Nobody Here", ("BUF", "MTL"), (None, "not in raw.players")),
    ])
    def test_cases(self, name, teams, expected):
        assert match_player(name, teams, INDEX) == expected


# ── HTTP and the key ──────────────────────────────────────────────

def _response(status, body, headers=None, reason=None):
    r = requests.Response()
    r.status_code = status
    r.reason = reason or ("OK" if status == 200 else "Error")
    r._content = (body if isinstance(body, str) else json.dumps(body)).encode()
    r.headers.update(headers or {})
    return r


def _logged(caplog):
    return caplog.text + "\n".join(r.getMessage() for r in caplog.records)


@pytest.fixture()
def fake_key(monkeypatch):
    monkeypatch.setattr(odds_api, "ODDS_API_KEY", FAKE_KEY)
    monkeypatch.setattr(props_odds, "PAUSE_S", 0)
    monkeypatch.setattr(props_odds.time, "sleep", lambda s: None)


class TestGet:
    def test_401_logs_the_api_message_never_the_key(self, monkeypatch, caplog, fake_key):
        sent = []

        def fake_get(url, params=None, timeout=None):
            sent.append(params)
            return _response(401, {"message": f"API key {FAKE_KEY} is not valid"},
                             reason="Unauthorized")

        monkeypatch.setattr(props_odds.requests, "get", fake_get)
        with caplog.at_level(logging.DEBUG):
            assert props_odds._get("/sports/icehockey_nhl/events", {}) == (None, {}, 401)
        assert sent[0]["apiKey"] == FAKE_KEY
        assert FAKE_KEY not in _logged(caplog)
        assert "HTTP 401 Unauthorized: API key *** is not valid" in caplog.text

    def test_connection_error_never_logs_the_url(self, monkeypatch, caplog, fake_key):
        def boom(*a, **k):
            raise requests.ConnectionError(
                f"Max retries exceeded with url: /v4/sports/icehockey_nhl/events?apiKey={FAKE_KEY}")

        monkeypatch.setattr(props_odds.requests, "get", boom)
        with caplog.at_level(logging.DEBUG):
            assert props_odds._get("/x", {})[0] is None
        assert FAKE_KEY not in _logged(caplog)
        assert "could not reach api.the-odds-api.com (ConnectionError)" in caplog.text

    def test_success_logs_credits(self, monkeypatch, caplog, fake_key):
        headers = {"x-requests-remaining": "480", "x-requests-used": "20",
                   "x-requests-last": "1"}
        monkeypatch.setattr(props_odds.requests, "get",
                            lambda *a, **k: _response(200, [], headers))
        with caplog.at_level(logging.INFO):
            body, _, status = props_odds._get("/x", {})
        assert (body, status) == ([], 200)
        assert "credits remaining 480, used 20, this call cost 1" in caplog.text

    def test_429_is_retried_once(self, monkeypatch, fake_key):
        answers = [_response(429, {"message": "slow down"}), _response(200, [])]
        monkeypatch.setattr(props_odds.requests, "get", lambda *a, **k: answers.pop(0))
        assert props_odds._get("/x", {})[2] == 200

    def test_placeholder_key_makes_no_request(self, monkeypatch, caplog):
        monkeypatch.setattr(odds_api, "ODDS_API_KEY", "your_key_here")
        monkeypatch.setattr(props_odds.requests, "get",
                            lambda *a, **k: pytest.fail("request made"))
        assert props_odds._get("/x", {}) == (None, {}, None)
        assert "ODDS_API_KEY is not set" in caplog.text


# ── A whole run, database faked ───────────────────────────────────

def _game(game_id, home, away, minutes_from_now):
    start = NOW + dt.timedelta(minutes=minutes_from_now)
    return {"game_id": game_id, "home_team": home, "away_team": away,
            "date": start.date(), "start_time_utc": start}


class _Run:
    """Fakes for every database helper plus a scripted API."""

    def __init__(self, monkeypatch, events, games, captures=None, odds=None):
        self.events, self.written, self.calls = events, [], []
        self.odds = odds or {}
        for k in ("PROPS_BOOKMAKERS", "PROPS_MARKETS", "PROPS_REGIONS",
                  "PROPS_CLOSE_LEAD_MINUTES", "PROPS_CLOSE_MIN_GAP_MINUTES"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setattr(props_odds, "ensure_table", lambda: None)
        monkeypatch.setattr(props_odds, "upcoming_starts",
                            lambda now, horizon: [g["start_time_utc"] for g in games
                                                  if now < g["start_time_utc"] <= now + horizon])
        monkeypatch.setattr(props_odds, "candidate_games", lambda now, horizon: games)
        monkeypatch.setattr(props_odds, "last_captures",
                            lambda ids: {i: t for i, t in (captures or {}).items() if i in ids})
        monkeypatch.setattr(props_odds, "load_player_index", lambda: INDEX)
        monkeypatch.setattr(props_odds, "write_rows",
                            lambda rows: self.written.extend(rows) or len(rows))
        monkeypatch.setattr(props_odds.requests, "get", self.get)

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params)))
        if url.endswith("/sports/icehockey_nhl/events"):
            return _response(200, self.events)
        eid = url.split("/events/")[1].split("/")[0]
        answer = self.odds.get(eid, dict(FIXTURE, id=eid))
        if isinstance(answer, requests.Response):
            return answer
        return _response(200, answer, {"x-requests-remaining": "499",
                                       "x-requests-used": "1", "x-requests-last": "1"})

    def odds_calls(self):
        return [(u.split("/events/")[1].split("/")[0], p) for u, p in self.calls
                if "/odds" in u]


class TestSnapshot:
    def test_morning_run(self, monkeypatch, caplog, fake_key):
        events = [ev("bufmtl", 120), ev("torbos", -30, "Toronto Maple Leafs", "Boston Bruins"),
                  ev("nojson", 180, "Seattle Kraken", "Vegas Golden Knights")]
        games = [_game(101, "BUF", "MTL", 120), _game(102, "TOR", "BOS", -30)]
        run = _Run(monkeypatch, events, games)
        with caplog.at_level(logging.INFO):
            stored = props_odds.snapshot_props(now=NOW)
        assert stored == 10
        # the free events call: a time window, no markets
        url, params = run.calls[0]
        assert url == "https://api.the-odds-api.com/v4/sports/icehockey_nhl/events"
        assert params["commenceTimeFrom"] == "2026-01-15T17:00:00Z"
        assert params["commenceTimeTo"] == "2026-01-16T17:00:00Z"
        assert "markets" not in params
        # one paid call: not the game under way, not the unmatched one
        (eid, params), = run.odds_calls()
        assert eid == "bufmtl"
        assert params["markets"] == "player_shots_on_goal"
        assert params["oddsFormat"] == "american"
        assert params["bookmakers"] == ",".join(props_odds.DEFAULT_BOOKMAKERS)
        assert "regions" not in params
        rows = _rows_by(run.written)
        tage = rows[("draftkings", "player_shots_on_goal", "Tage Thompson", 3.5)]
        assert (tage["game_id"], tage["event_id"], tage["player_id"]) == (101, "bufmtl", 1)
        assert tage["captured_at"].tzinfo is None                  # naive UTC
        hutson = rows[("fanduel", "player_shots_on_goal", "Lane Hutson", 1.5)]
        assert hutson["player_id"] is None                          # not in the index
        assert "1 event(s) matched no raw.games row and were not requested" in caplog.text
        assert "Lane Hutson (not in raw.players)" in caplog.text
        assert "this run cost 1 credit(s), 499 remaining" in caplog.text
        assert FAKE_KEY not in _logged(caplog)

    def test_placeholder_key_requests_nothing(self, monkeypatch, caplog, fake_key):
        run = _Run(monkeypatch, [ev("a", 60)], [_game(101, "BUF", "MTL", 60)])
        monkeypatch.setattr(odds_api, "ODDS_API_KEY", "your_key_here")
        assert props_odds.snapshot_props(now=NOW) == 0
        assert run.calls == []
        assert "ODDS_API_KEY is not set" in caplog.text

    def test_markets_option_reaches_the_request(self, monkeypatch, fake_key):
        run = _Run(monkeypatch, [ev("a", 60)], [_game(101, "BUF", "MTL", 60)])
        props_odds.snapshot_props(markets="player_points,player_total_saves", now=NOW)
        assert run.odds_calls()[0][1]["markets"] == "player_points,player_total_saves"

    def test_no_game_in_the_schedule_means_no_request(self, monkeypatch, caplog, fake_key):
        run = _Run(monkeypatch, [ev("a", 60)], [_game(101, "BUF", "MTL", 25 * 60)])
        with caplog.at_level(logging.INFO):
            assert props_odds.snapshot_props(now=NOW) == 0
        assert run.calls == []
        assert "no request made" in caplog.text

    def test_due_requests_only_games_about_to_start(self, monkeypatch, caplog, fake_key):
        events = [ev("soon", 10), ev("done", 12, "Ottawa Senators", "Boston Bruins"),
                  ev("later", 40, "Seattle Kraken", "Vegas Golden Knights")]
        games = [_game(101, "BUF", "MTL", 10), _game(102, "OTT", "BOS", 12),
                 _game(103, "SEA", "VGK", 40)]
        run = _Run(monkeypatch, events, games,
                   captures={"done": NOW - dt.timedelta(minutes=5)})
        with caplog.at_level(logging.INFO):
            assert props_odds.snapshot_props(due=True, now=NOW) == 10
        assert [eid for eid, _ in run.odds_calls()] == ["soon"]
        assert run.calls[0][1]["commenceTimeTo"] == "2026-01-15T17:16:00Z"
        assert "props --due: 1 game(s) due, the first starting in 10 minute(s)" in caplog.text

    def test_due_with_nothing_due_pays_nothing(self, monkeypatch, caplog, fake_key):
        run = _Run(monkeypatch, [ev("a", 10)], [_game(101, "BUF", "MTL", 10)],
                   captures={"a": NOW - dt.timedelta(minutes=3)})
        with caplog.at_level(logging.INFO):
            assert props_odds.snapshot_props(due=True, now=NOW) == 0
        assert run.odds_calls() == []
        assert "props --due: no snapshot" in caplog.text

    def test_stops_after_a_401(self, monkeypatch, caplog, fake_key):
        events = [ev("a", 60), ev("b", 90, "Ottawa Senators", "Boston Bruins")]
        games = [_game(101, "BUF", "MTL", 60), _game(102, "OTT", "BOS", 90)]
        run = _Run(monkeypatch, events, games, odds={"a": _response(
            401, {"message": "Usage quota has been reached"}, reason="Unauthorized")})
        assert props_odds.snapshot_props(now=NOW) == 0
        assert [eid for eid, _ in run.odds_calls()] == ["a"]
        assert "stopping after HTTP 401; the remaining 1 game(s) were not requested" in caplog.text

    def test_empty_response_stores_nothing_and_goes_on(self, monkeypatch, fake_key):
        events = [ev("a", 60), ev("b", 90, "Ottawa Senators", "Boston Bruins")]
        games = [_game(101, "BUF", "MTL", 60), _game(102, "OTT", "BOS", 90)]
        run = _Run(monkeypatch, events, games,
                   odds={"a": dict(FIXTURE, id="a", bookmakers=[])})
        assert props_odds.snapshot_props(now=NOW) == 10
        assert {r["game_id"] for r in run.written} == {102}


# ── CLI ───────────────────────────────────────────────────────────

def test_main_passes_options(monkeypatch):
    calls = []
    monkeypatch.setattr(props_odds, "snapshot_props", lambda **kw: calls.append(kw) or 0)
    props_odds.main(["--markets", "player_points", "--due"])
    props_odds.main([])
    assert calls == [{"markets": "player_points", "due": True},
                     {"markets": None, "due": False}]


def test_module_help_runs_nothing():
    out = subprocess.run(
        [sys.executable, "-m", "ingestion.props_odds", "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
        env={**os.environ, "PYTHONPATH": str(ROOT), "ODDS_API_KEY": "your_key_here",
             "POSTGRES_HOST": "127.0.0.1", "POSTGRES_PORT": "1"})
    assert out.returncode == 0, out.stderr
    assert "usage:" in out.stdout and "--markets" in out.stdout and "--due" in out.stdout


# ── Database ──────────────────────────────────────────────────────

GAME_ID = 9_999_020_201


@requires_db
class TestDatabase:
    @pytest.fixture()
    def game(self):
        props_odds.ensure_table()
        start = dt.datetime.now(UTC).replace(microsecond=0) + dt.timedelta(hours=3)
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM raw.prop_snapshots WHERE game_id = :g"), {"g": GAME_ID})
            conn.execute(text("DELETE FROM raw.games WHERE game_id = :g"), {"g": GAME_ID})
            conn.execute(text("""
                INSERT INTO raw.games (game_id, season, game_type, date, start_time_utc,
                                       home_team, away_team, game_state)
                VALUES (:g, 20992100, 2, :d, :t, 'BUF', 'MTL', 'FUT')
            """), {"g": GAME_ID, "d": start.date(), "t": start})
        yield start
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM raw.prop_snapshots WHERE game_id = :g"), {"g": GAME_ID})
            conn.execute(text("DELETE FROM raw.games WHERE game_id = :g"), {"g": GAME_ID})

    def test_ensure_table_is_idempotent(self):
        props_odds.ensure_table()
        props_odds.ensure_table()
        with engine.connect() as conn:
            cols = [r[0] for r in conn.execute(text("""
                SELECT column_name FROM information_schema.columns
                WHERE table_schema = 'raw' AND table_name = 'prop_snapshots'
                ORDER BY ordinal_position"""))]
        assert cols == ["snapshot_id", "captured_at", "game_id", "event_id", "book",
                        "market", "player_name", "player_id", "line", "over_price",
                        "under_price", "book_updated_at"]

    def test_write_read_and_schedule_helpers(self, game):
        now = dt.datetime.now(UTC)
        assert game in props_odds.upcoming_starts(now, dt.timedelta(hours=24))
        assert GAME_ID in [g["game_id"] for g in
                           props_odds.candidate_games(now, dt.timedelta(hours=24))]
        captured = now.replace(tzinfo=None, microsecond=0)
        rows = [dict(r, captured_at=captured, game_id=GAME_ID, event_id="evt-test",
                     player_id=None) for r in parse_event_odds(FIXTURE)]
        assert props_odds.write_rows(rows) == 10
        assert props_odds.last_captures(["evt-test", "other"]) == {
            "evt-test": captured.replace(tzinfo=UTC)}
        with engine.connect() as conn:
            got = conn.execute(text("""
                SELECT line, over_price, under_price FROM raw.prop_snapshots
                WHERE game_id = :g AND book = 'draftkings'
                  AND market = 'player_shots_on_goal' AND player_name = 'Tage Thompson'
            """), {"g": GAME_ID}).one()
        assert (float(got.line), got.over_price, got.under_price) == (3.5, 105, -135)

    def test_book_style_names_resolve_against_raw_players(self):
        """raw.players holds 'C. McDavid'; books send 'Connor McDavid'.
        Build a book-style name from a real row with a unique key."""
        index = props_odds.load_player_index()
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT p.player_id, p.full_name FROM raw.players p
                JOIN raw.skater_games sg USING (player_id)
                WHERE p.full_name LIKE '_. %' GROUP BY 1, 2 LIMIT 200
            """)).fetchall()
        pid, name = next((p, n) for p, n in rows
                         if len(index["by_key"][props_odds._key(n)]) == 1)
        initial, surname = name.split(". ", 1)
        team = index["last_team"][pid]
        assert match_player(f"{initial}ohnfake {surname}", (team, "ZZZ"), index) == (pid, "ok")
