"""
Tests for betting/news.py, the news monitor: the scheduling window, the
diffs for starters, ESPN injuries and Daily Faceoff lines (the lines from
the saved pages in tests/fixtures), the market-move check and the
re-score plan, all pure. An end-to-end run with every network source
stubbed needs a disposable database (tests/conftest.py).
"""
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from betting import news
from ingestion import dailyfaceoff_lines as dfl

CT = ZoneInfo("America/Chicago")
DAY = date(2026, 10, 10)
FIXTURES = Path(__file__).resolve().parent / "fixtures"
BOS = (FIXTURES / "dailyfaceoff_lines_bos.html").read_text(encoding="utf-8")


def at(h, m=0):
    return datetime(2026, 10, 10, h, m, tzinfo=CT)


STARTS = [at(18, 0), at(19, 0), at(21, 30)]


# ── When to run ────────────────────────────────────────────────────

class TestNewsDue:
    kw = {"start_hour": 8, "min_gap_minutes": 14}

    def test_no_games_no_run(self):
        assert news.news_due(at(12), [], **self.kw) == (False, "no game today")
        assert news.news_due(at(12), [None], **self.kw)[0] is False

    def test_window_opens_at_eight_local(self):
        due, why = news.news_due(at(7, 59), STARTS, **self.kw)
        assert not due and "before 08:00" in why
        assert news.news_due(at(8, 0), STARTS, **self.kw)[0]

    def test_window_closes_at_the_last_puck_drop(self):
        assert news.news_due(at(21, 30), STARTS, **self.kw)[0]
        due, why = news.news_due(at(21, 45), STARTS, **self.kw)
        assert not due and "last puck drop" in why

    def test_starts_in_utc_compare_correctly(self):
        utc = [s.astimezone(timezone.utc) for s in STARTS]
        assert news.news_due(at(21, 0), utc, **self.kw)[0]
        assert not news.news_due(at(22, 0), utc, **self.kw)[0]

    def test_minimum_gap_between_runs(self):
        due, why = news.news_due(at(12, 5), STARTS, last_run=at(12, 0), **self.kw)
        assert not due and "5 minute(s) ago" in why
        assert news.news_due(at(12, 15), STARTS, last_run=at(12, 0), **self.kw)[0]

    def test_quarter_hour_runs_over_a_game_day(self):
        """Task Scheduler fires every 15 minutes all day; count the runs
        that work: 8:00 through 21:30 inclusive."""
        ran, last = [], None
        t = datetime(2026, 10, 10, 0, 0, tzinfo=CT)
        while t.date() == DAY:
            if news.news_due(t, STARTS, last_run=last, **self.kw)[0]:
                ran.append(t)
                last = t
            t += timedelta(minutes=15)
        assert ran[0] == at(8, 0) and ran[-1] == at(21, 30)
        assert len(ran) == (21 * 60 + 30 - 8 * 60) // 15 + 1

    def test_settings_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("NEWS_START_HOUR", "10")
        monkeypatch.setenv("NEWS_MIN_GAP_MINUTES", "x")         # bad: default 14
        assert not news.news_due(at(9, 45), STARTS)[0]
        assert news.news_due(at(10, 0), STARTS)[0]
        assert not news.news_due(at(10, 10), STARTS, last_run=at(10, 0))[0]


# ── Starters ───────────────────────────────────────────────────────

def starter(goalie, conf=None, d="2026-10-10", gid=None):
    return {"game_date": d, "goalie": goalie, "goalie_id": gid, "confirmation": conf}


class TestDiffStarters:
    def test_first_projection_is_not_news_but_a_confirmation_is(self):
        ev = news.diff_starters({}, {"BOS": starter("J. Swayman", "Likely"),
                                     "TOR": starter("A. Stolarz", "Confirmed", gid=7)})
        assert [(e["team"], e["kind"]) for e in ev] == [("TOR", "STARTER_CONFIRMED")]
        assert ev[0]["player_id"] == 7 and ev[0]["previous"] is None
        assert ev[0]["current"] == "A. Stolarz (Confirmed)"

    def test_same_goalie_becomes_confirmed(self):
        ev = news.diff_starters({"BOS": starter("Jeremy Swayman", "Likely")},
                                {"BOS": starter("Jeremy Swayman", "Confirmed")})
        assert [e["kind"] for e in ev] == ["STARTER_CONFIRMED"]
        assert ev[0]["previous"] == "Jeremy Swayman (Likely)"

    def test_a_different_goalie_is_a_change(self):
        ev = news.diff_starters({"BOS": starter("Jeremy Swayman", "Likely")},
                                {"BOS": starter("Joonas Korpisalo", "Confirmed")})
        assert [e["kind"] for e in ev] == ["STARTER_CHANGED"]
        assert "Joonas Korpisalo is now expected to start instead of Jeremy Swayman" \
            in ev[0]["detail"] and "(confirmed)" in ev[0]["detail"]

    def test_losing_confirmation_is_a_change(self):
        ev = news.diff_starters({"BOS": starter("J. Swayman", "Confirmed")},
                                {"BOS": starter("J. Swayman", "Unconfirmed")})
        assert [e["kind"] for e in ev] == ["STARTER_CHANGED"]

    def test_nothing_new_is_no_news_and_accents_do_not_count(self):
        prev = {"MTL": starter("Samuel Montembeault", "Confirmed")}
        assert news.diff_starters(prev, {"MTL": starter("Samuel  Montembeault ".strip(),
                                                        "Confirmed")}) == []
        prev = {"MTL": starter("Jakub Dobeš", "Likely")}
        assert news.diff_starters(prev, {"MTL": starter("Jakub Dobes", "Likely")}) == []

    def test_a_new_day_starts_fresh(self):
        prev = {"BOS": starter("J. Swayman", "Confirmed", d="2026-10-09")}
        assert news.diff_starters(prev, {"BOS": starter("J. Korpisalo", "Likely")}) == []
        ev = news.diff_starters(prev, {"BOS": starter("J. Swayman", "Confirmed")})
        assert [e["kind"] for e in ev] == ["STARTER_CONFIRMED"]


# ── ESPN injuries ──────────────────────────────────────────────────

def hurt(name, status, injury="Upper Body", pid=None):
    return {"name": name, "status": status, "injury": injury, "player_id": pid}


class TestDiffInjuries:
    def test_first_run_is_only_remembered(self):
        cur = {"BOS": {"1": hurt("Charlie McAvoy", "Out")}}
        assert news.diff_injuries({}, cur, has_baseline=False) == []

    def test_added_removed_and_status_changes(self):
        prev = {"BOS": {"1": hurt("Charlie McAvoy", "Out", pid=11),
                        "2": hurt("Brad Marchand", "Day-To-Day")},
                "MTL": {"3": hurt("Kaiden Guhle", "Injured Reserve")}}
        cur = {"BOS": {"1": hurt("Charlie McAvoy", "Day-To-Day", pid=11),
                       "2": hurt("Brad Marchand", "Out")},
               "TOR": {"4": hurt("Max Domi", "Out", "Knee")}}
        ev = {(e["team"], e["player_name"]): e for e in news.diff_injuries(prev, cur)}
        assert ev[("BOS", "Charlie McAvoy")]["kind"] == "PLAYER_IN"        # better
        assert ev[("BOS", "Charlie McAvoy")]["player_id"] == 11
        assert ev[("BOS", "Brad Marchand")]["kind"] == "PLAYER_OUT"        # worse
        assert ev[("MTL", "Kaiden Guhle")]["kind"] == "PLAYER_IN"          # off the list
        assert ev[("MTL", "Kaiden Guhle")]["current"] == "not listed"
        added = ev[("TOR", "Max Domi")]
        assert added["kind"] == "PLAYER_OUT" and added["current"] == "Out (Knee)"
        assert len(ev) == 4

    def test_unchanged_list_is_no_news(self):
        lst = {"BOS": {"1": hurt("Charlie McAvoy", "Out")}}
        assert news.diff_injuries(lst, {"BOS": dict(lst["BOS"])}) == []


# ── Daily Faceoff lines ────────────────────────────────────────────

def bos_rows():
    return [dict(r) for r in dfl.parse_lineup(BOS)["rows"]]


def _move(rows, name, unit=None, slot=None, new_name=None):
    for r in rows:
        if r["player_name"] == name and (unit is None or r["unit"] == unit) \
                and (slot is None or r["slot"] == slot):
            if new_name:
                r["player_name"] = new_name
            return rows
    raise AssertionError(name)


class TestDiffLineups:
    def test_state_from_rows(self):
        state = news.lineup_state(bos_rows())
        assert len(state["units"]["F1"]) == 3 and "IR" not in state["units"]
        assert state["ir"] == ["Charlie McAvoy"]
        assert set(news._dressed(state)) and len(news._dressed(state)) == 18

    def test_no_baseline_or_no_change_is_no_news(self):
        state = news.lineup_state(bos_rows())
        assert news.diff_lineups("BOS", None, state) == []
        assert news.diff_lineups("BOS", state, news.lineup_state(bos_rows())) == []

    def test_a_player_swapped_in_from_outside(self):
        before = news.lineup_state(bos_rows())
        rows = bos_rows()
        f2 = [r for r in rows if r["unit"] == "F2"]
        out_name = f2[0]["player_name"]
        f2[0]["player_name"] = "New Callup"
        f2[0]["game_time_decision"] = True
        ev = news.diff_lineups("BOS", before, news.lineup_state(rows))
        kinds = sorted((e["kind"], e.get("player_name")) for e in ev)
        assert kinds == [("LINE_CHANGE", None), ("PLAYER_IN", "New Callup"),
                         ("PLAYER_OUT", out_name)]
        player_in = next(e for e in ev if e["kind"] == "PLAYER_IN")
        assert player_in["current"] == "F2" and "game-time decision" in player_in["detail"]
        line = next(e for e in ev if e["kind"] == "LINE_CHANGE")
        assert line["detail"] == "forward line 2 changed"
        assert out_name in line["previous"] and "New Callup" in line["current"]

    def test_a_player_moved_to_injured_reserve(self):
        before = news.lineup_state(bos_rows())
        rows = [r for r in bos_rows()]
        d1 = next(r for r in rows if r["unit"] == "D1")
        gone = d1["player_name"]
        rows = [r for r in rows if r is not d1]
        rows.append({**d1, "unit": "IR", "slot": "ir2", "injury_status": "out"})
        ev = news.diff_lineups("BOS", before, news.lineup_state(rows))
        out = next(e for e in ev if e["kind"] == "PLAYER_OUT")
        assert out["player_name"] == gone and out["current"] == "IR"
        assert "injured reserve" in out["detail"]
        assert any(e["kind"] == "LINE_CHANGE" and e["detail"] == "defence pair 1 changed"
                   for e in ev)

    def test_two_lines_swapping_players_is_two_line_changes_and_no_one_out(self):
        before = news.lineup_state(bos_rows())
        rows = bos_rows()
        f1 = next(r for r in rows if r["unit"] == "F1" and r["slot"] == "rw")
        f3 = next(r for r in rows if r["unit"] == "F3" and r["slot"] == "rw")
        f1["player_name"], f3["player_name"] = f3["player_name"], f1["player_name"]
        ev = news.diff_lineups("BOS", before, news.lineup_state(rows))
        assert sorted(e["detail"] for e in ev) == ["forward line 1 changed",
                                                   "forward line 3 changed"]

    def test_power_play_change_and_order_within_a_unit_ignored(self):
        before = news.lineup_state(bos_rows())
        rows = bos_rows()
        pp1 = [r for r in rows if r["unit"] == "PP1"]
        pp2 = [r for r in rows if r["unit"] == "PP2"]
        pp1[0]["player_name"], pp2[0]["player_name"] = pp2[0]["player_name"], pp1[0]["player_name"]
        ev = news.diff_lineups("BOS", before, news.lineup_state(rows))
        assert sorted((e["kind"], e["detail"]) for e in ev) == [
            ("PP_UNIT_CHANGE", "power-play unit 1 changed"),
            ("PP_UNIT_CHANGE", "power-play unit 2 changed")]
        # the same five in another order is no change
        rows = bos_rows()
        pp1 = [r for r in rows if r["unit"] == "PP1"]
        pp1[0]["player_name"], pp1[1]["player_name"] = pp1[1]["player_name"], pp1[0]["player_name"]
        assert news.diff_lineups("BOS", before, news.lineup_state(rows)) == []

    def test_goalie_and_penalty_kill_changes_are_not_news_here(self):
        before = news.lineup_state(bos_rows())
        rows = bos_rows()
        g = [r for r in rows if r["unit"] == "G"]
        g[0]["player_name"], g[1]["player_name"] = g[1]["player_name"], "Third Goalie"
        pk = next(r for r in rows if r["unit"] == "PK1")
        pk["player_name"] = "Someone Else"
        assert news.diff_lineups("BOS", before, news.lineup_state(rows)) == []

    def test_state_survives_a_json_round_trip(self):
        import json
        state = news.lineup_state(bos_rows())
        assert news.diff_lineups("BOS", json.loads(json.dumps(state)), state) == []


# ── Market check and re-score plan ─────────────────────────────────

class TestPriceMove:
    def test_not_moved(self):
        moved, note = news.price_move({"draftkings": (-150, 130)},
                                      {"draftkings": (-155, 135)}, threshold_pts=1.0)
        assert moved is False and note.startswith("not moved")

    def test_moved(self):
        moved, note = news.price_move({"draftkings": (-150, 130), "fanduel": (-145, 125)},
                                      {"draftkings": (-110, -110), "fanduel": (-148, 128)},
                                      threshold_pts=1.0)
        assert moved is True and "draftkings" in note and "2 book(s)" in note
        assert "-150/+130 → -110/-110" in note

    def test_threshold_is_inclusive(self):
        b = news.fair_home(-150, 130)
        # find a price pair exactly a point away is fiddly: use the threshold instead
        delta = abs(news.fair_home(-160, 140) - b) * 100
        assert news.price_move({"x": (-150, 130)}, {"x": (-160, 140)},
                               threshold_pts=delta)[0] is True
        assert news.price_move({"x": (-150, 130)}, {"x": (-160, 140)},
                               threshold_pts=delta + 0.01)[0] is False

    def test_nothing_to_compare(self):
        moved, note = news.price_move({"draftkings": (-150, 130)}, {"fanduel": (-150, 130)},
                                      threshold_pts=1.0)
        assert moved is None and "can't be ruled out" in note
        assert news.price_move({}, {}, threshold_pts=1.0)[0] is None

    def test_threshold_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("NEWS_MOVE_PTS", "50")
        assert news.price_move({"x": (-150, 130)}, {"x": (-110, -110)})[0] is False


class TestRescorePlan:
    games = {1: {"date": DAY}, 2: {"date": DAY}, 3: {"date": DAY},
             4: {"date": DAY + timedelta(days=1)}}

    def test_frozen_games_are_not_rescored_and_only_unmoved_ones_may_pick(self):
        moves = {1: (False, ""), 2: (True, ""), 3: (None, ""), 4: (False, "")}
        plan = news.rescore_plan([1, 2, 3, 4, 9], self.games, frozen={3}, moves=moves)
        assert plan == {DAY: ([1, 2], {1}), DAY + timedelta(days=1): ([4], {4})}

    def test_every_game_frozen_means_nothing_to_rescore(self):
        plan = news.rescore_plan([1], self.games, frozen={1}, moves={1: (False, "")})
        assert plan == {DAY: ([], set())}


def test_injury_list_may_be_saved_only_before_the_first_puck_drop():
    assert news.may_save_injuries(at(17, 59), STARTS) is True
    assert news.may_save_injuries(at(18, 0), STARTS) is False
    assert news.may_save_injuries(at(20, 0), [None, at(21, 30), at(18, 0)]) is False
    assert news.may_save_injuries(at(20, 0), [None]) is True     # no start known


# ── The run, without a database ────────────────────────────────────

def test_due_run_outside_the_window_touches_nothing(monkeypatch, caplog):
    import logging
    caplog.set_level(logging.INFO)
    import config.migrate
    monkeypatch.setattr(config.migrate, "ensure_schema", lambda: None)
    monkeypatch.setattr(news, "ensure_table", lambda: None)
    monkeypatch.setattr(news, "todays_games", lambda: [
        {"game_id": 1, "date": DAY, "start_time_utc": at(19), "home_team": "BOS",
         "away_team": "MTL", "started": False}])
    monkeypatch.setattr(news, "last_run_start", lambda: None)
    monkeypatch.setattr(news, "local_now", lambda: at(6, 0))
    for step in ("check_starters", "check_lineups", "check_injuries", "rescore"):
        monkeypatch.setattr(news, step, lambda *a, **k: pytest.fail("ran outside the window"))
    assert news.run_news(due=True) == 0
    assert "news --due: nothing to do, before 08:00" in caplog.text


# ── End to end on a disposable database ────────────────────────────

from config.settings import check_db_connection, engine  # noqa: E402
from sqlalchemy import text  # noqa: E402

requires_db = pytest.mark.skipif(not check_db_connection(), reason="database not reachable")


@requires_db
class TestRunOnDatabase:
    """Two news runs with every network source stubbed: the first only
    remembers, the second sees a starter change, a line change and an
    injury and re-scores the game (generate_recommendations stubbed).
    It empties the five news tables (raw.news_events, news_state,
    news_runs, lineups, lineup_fetches) of the disposable database, and
    sees only its own synthetic game (2099020001, BOS vs MTL, today)."""

    GAME = 2099020001

    @pytest.fixture(autouse=True)
    def world(self, monkeypatch):
        from config.migrate import ensure_schema
        ensure_schema()
        news._table_ready = dfl._table_ready = False
        news.ensure_table()
        dfl.ensure_table()
        start = datetime.now(timezone.utc) + timedelta(hours=4)
        today = news.local_today()
        with engine.begin() as conn:
            for t in ("raw.news_events", "raw.news_state", "raw.news_runs",
                      "raw.lineups", "raw.lineup_fetches"):
                conn.execute(text(f"DELETE FROM {t}"))
            conn.execute(text("DELETE FROM raw.starting_goalies WHERE game_date = :d "
                              "AND team IN ('BOS', 'MTL')"), {"d": today})
            conn.execute(text("DELETE FROM raw.nhl_feed_snapshots WHERE game_id = :g"),
                         {"g": self.GAME})
            conn.execute(text("DELETE FROM raw.odds_snapshots WHERE game_id = :g"),
                         {"g": self.GAME})
            conn.execute(text("DELETE FROM raw.games WHERE game_id = :g"), {"g": self.GAME})
            for team in ("BOS", "MTL"):
                conn.execute(text("""
                    INSERT INTO raw.teams (team_abbrev, team_name)
                    VALUES (:t, :t) ON CONFLICT DO NOTHING"""), {"t": team})
            conn.execute(text("""
                INSERT INTO raw.games (game_id, season, game_type, date, home_team,
                                       away_team, game_state, start_time_utc)
                VALUES (:g, 20262027, 2, :d, 'BOS', 'MTL', 'FUT', :s)
            """), {"g": self.GAME, "d": today, "s": start})
        self.today, self.start = today, start
        real_games = news.todays_games
        monkeypatch.setattr(news, "todays_games", lambda on_date=None: [
            g for g in real_games(on_date) if g["game_id"] == self.GAME])
        self.goalies = {"BOS": ("Jeremy Swayman", "Likely"),
                        "MTL": ("Sam Montembeault", "Likely")}
        self.lines = {"BOS": BOS, "MTL": BOS.replace('"Boston Bruins"', '"Montreal Canadiens"')}
        self.injuries = [("BOS", 1, "Charlie McAvoy", "Out")]
        self.feed_now = (-150, 130)
        self.recs = []

        import ingestion.dailyfaceoff as df
        from ingestion import espn_injuries, nhl_odds
        from betting import recommend

        def starters(d=None):
            with engine.begin() as conn:
                for team, (name, conf) in self.goalies.items():
                    conn.execute(text("""
                        INSERT INTO raw.starting_goalies (game_date, team, goalie_name,
                                                          confirmation)
                        VALUES (:d, :t, :n, :c) ON CONFLICT (game_date, team) DO UPDATE
                        SET goalie_name = EXCLUDED.goalie_name,
                            confirmation = EXCLUDED.confirmation"""),
                        {"d": d, "t": team, "n": name, "c": conf})
            return len(self.goalies)
        monkeypatch.setattr(df, "ingest_starting_goalies", starters)
        monkeypatch.setattr(dfl, "fetch_page", lambda team: self.lines[team])
        for name in ("LINEUPS_MIN_GAP_MINUTES", "LINEUPS_FAR_GAP_MINUTES",
                     "LINEUPS_PAUSE_SECONDS"):
            monkeypatch.setenv(name, "0")

        def payload():
            return {"injuries": [{"displayName": "x", "injuries": [
                {"status": status, "date": "2026-10-10T12:00Z",
                 "details": {"type": "Upper Body"},
                 "athlete": {"id": str(aid), "displayName": name,
                             "position": {"abbreviation": "D"},
                             "team": {"abbreviation": team}}}
                for team, aid, name, status in self.injuries]}]}
        monkeypatch.setattr(espn_injuries, "fetch_injuries", payload)

        def feed_snapshot(skip_when_idle=False):
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO raw.nhl_feed_snapshots (captured_at, game_id, source, book,
                                                        market, home_price, away_price)
                    VALUES (:c, :g, 'partner-US', 'draftkings', 'ml', :h, :a)"""),
                    {"c": datetime.now(timezone.utc).replace(tzinfo=None), "g": self.GAME,
                     "h": self.feed_now[0], "a": self.feed_now[1]})
        monkeypatch.setattr(nhl_odds, "snapshot", feed_snapshot)

        def generate(d, only_games=None, **kw):
            self.recs.append((d, set(only_games)))
            import pandas as pd
            return pd.DataFrame({"game_id": sorted(only_games)})
        monkeypatch.setattr(recommend, "generate_recommendations", generate)
        monkeypatch.setattr(recommend, "load_issued_picks", lambda d: None)
        monkeypatch.setattr(recommend, "frozen_games", lambda issued: set())
        from config import runs
        self.daily_done = True
        monkeypatch.setattr(runs, "finished",
                            lambda job, run_date=None: job == "daily" and self.daily_done)
        yield
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM raw.news_events WHERE game_id = :g"), {"g": self.GAME})
            conn.execute(text("DELETE FROM raw.nhl_feed_snapshots WHERE game_id = :g"),
                         {"g": self.GAME})
            conn.execute(text("DELETE FROM raw.odds_snapshots WHERE game_id = :g"),
                         {"g": self.GAME})
            conn.execute(text("DELETE FROM raw.games WHERE game_id = :g"), {"g": self.GAME})

    def _paid_snapshot(self, minutes_ago=60):
        when = (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).replace(tzinfo=None)
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO raw.odds_snapshots (game_id, captured_at, book_name, market_type,
                                                home_price, away_price)
                VALUES (:g, :c, 'draftkings', 'ml', -150, 130)"""), {"g": self.GAME, "c": when})
            conn.execute(text("""
                INSERT INTO raw.nhl_feed_snapshots (captured_at, game_id, source, book,
                                                    market, home_price, away_price)
                VALUES (:c, :g, 'partner-US', 'draftkings', 'ml', -150, 130)"""),
                {"c": when + timedelta(minutes=1), "g": self.GAME})

    def _events(self):
        with engine.connect() as conn:
            return [dict(r) for r in conn.execute(text(
                "SELECT * FROM raw.news_events ORDER BY event_id")).mappings()]

    def test_first_run_remembers_then_news_is_found_and_rescored(self):
        self._paid_snapshot()
        assert news.run_news() == 0                     # baseline only
        with engine.connect() as conn:
            assert conn.execute(text("SELECT COUNT(*) FROM raw.news_state")).scalar() == 5
            assert conn.execute(text("SELECT COUNT(*) FROM raw.lineups")).scalar() == 78
            run = conn.execute(text("SELECT events, notes, finished_at FROM raw.news_runs")).one()
        assert run.events == 0 and run.finished_at is not None

        # news arrives: BOS confirm a different goalie, a forward is
        # replaced, and MTL's player is added to ESPN's list
        self.goalies["BOS"] = ("Joonas Korpisalo", "Confirmed")
        self.lines["BOS"] = BOS.replace("Morgan Geekie", "New Callup")
        self.injuries.append(("MTL", 2, "Kaiden Guhle", "Injured Reserve"))
        self.feed_now = (-152, 132)                     # a price that barely moved
        n = news.run_news()
        ev = self._events()
        assert n == len(ev)
        kinds = sorted((e["team"], e["kind"]) for e in ev)
        assert ("BOS", "STARTER_CHANGED") in kinds
        assert ("BOS", "PLAYER_IN") in kinds and ("BOS", "PLAYER_OUT") in kinds
        assert ("BOS", "PP_UNIT_CHANGE") in kinds and ("BOS", "LINE_CHANGE") in kinds
        assert ("MTL", "PLAYER_OUT") in kinds
        assert all(e["game_id"] == self.GAME for e in ev)
        starter = next(e for e in ev if e["kind"] == "STARTER_CHANGED")
        assert starter["rescored"] is True and starter["new_pick"] is True
        assert starter["market_moved"] is False and "not moved" in starter["market_note"]
        assert self.recs == [(self.today, {self.GAME})]
        other = next(e for e in ev if e["kind"] == "LINE_CHANGE")
        assert other["rescored"] is None and other["market_moved"] is None

        # a third run with nothing new finds nothing
        assert news.run_news() == 0

    def test_a_moved_price_blocks_a_new_pick(self):
        self._paid_snapshot()
        news.run_news()
        self.goalies["MTL"] = ("Jakub Dobes", "Confirmed")
        self.feed_now = (-110, -110)                    # the market already reacted
        news.run_news()
        ev = [e for e in self._events() if e["kind"] == "STARTER_CHANGED"]
        assert len(ev) == 1 and ev[0]["team"] == "MTL"
        assert ev[0]["market_moved"] is True and ev[0]["rescored"] is True
        assert ev[0]["new_pick"] is False
        assert "the stored price may be gone" in ev[0]["market_note"]
        assert self.recs == [(self.today, set())]

    def test_no_pick_from_news_before_todays_daily_run(self):
        """At 8:00 last night's games are not loaded yet: the news is
        recorded with its market check, but nothing is re-scored."""
        self._paid_snapshot()
        self.daily_done = False
        news.run_news()
        self.goalies["BOS"] = ("Joonas Korpisalo", "Confirmed")
        news.run_news()
        ev = [e for e in self._events() if e["kind"] == "STARTER_CHANGED"]
        assert len(ev) == 1
        assert ev[0]["rescored"] is False and ev[0]["new_pick"] is False
        assert ev[0]["market_moved"] is False
        assert "Waiting for today's daily run" in ev[0]["market_note"]
        assert self.recs == []

    def _feed_row(self, minutes_ago, home=-150, away=130):
        when = (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).replace(tzinfo=None)
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO raw.nhl_feed_snapshots (captured_at, game_id, source, book,
                                                    market, home_price, away_price)
                VALUES (:c, :g, 'partner-US', 'draftkings', 'ml', :h, :a)"""),
                {"c": when, "g": self.GAME, "h": home, "a": away})

    def test_an_old_feed_price_is_not_this_runs_price(self, monkeypatch):
        """This run's free snapshot fails; a feed price from an earlier run
        (before the news) must not count as "the market has not moved"."""
        from ingestion import nhl_odds
        self._paid_snapshot(minutes_ago=180)
        self._feed_row(minutes_ago=60)                  # an earlier news run's price
        news.run_news()

        def down(skip_when_idle=False):
            raise RuntimeError("feed down")
        monkeypatch.setattr(nhl_odds, "snapshot", down)
        self.goalies["BOS"] = ("Joonas Korpisalo", "Confirmed")
        news.run_news()
        ev = [e for e in self._events() if e["kind"] == "STARTER_CHANGED"]
        assert len(ev) == 1 and ev[0]["market_moved"] is None
        assert "no free NHL-feed price from this run" in ev[0]["market_note"]
        assert ev[0]["new_pick"] is False
        assert self.recs == [(self.today, set())]

    def test_the_before_price_must_be_taken_with_the_paid_snapshot(self):
        """A feed price from long before the paid snapshot is not its pair."""
        when = (datetime.now(timezone.utc) - timedelta(minutes=60)).replace(tzinfo=None)
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO raw.odds_snapshots (game_id, captured_at, book_name, market_type,
                                                home_price, away_price)
                VALUES (:g, :c, 'draftkings', 'ml', -150, 130)"""), {"g": self.GAME, "c": when})
        self._feed_row(minutes_ago=60 * 48)             # two days earlier
        since = datetime.now(timezone.utc).replace(tzinfo=None)
        self._feed_row(minutes_ago=-1)                  # this run's price
        moved, note = news.market_check(self.GAME, since=since)
        assert moved is None and "taken with the last paid odds snapshot" in note
        self._feed_row(minutes_ago=55, home=-110, away=-110)   # the real pair
        moved, note = news.market_check(self.GAME, since=since)
        assert moved is True

    def _injury_rows_today(self):
        with engine.connect() as conn:
            return conn.execute(text("SELECT COUNT(*) FROM raw.injuries "
                                     "WHERE snapshot_date = :d"), {"d": self.today}).scalar()

    @pytest.mark.parametrize("started", [False, True])
    def test_the_injury_list_is_saved_only_before_the_first_puck_drop(self, started):
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM raw.injuries WHERE snapshot_date = :d"),
                         {"d": self.today})
            if started:      # the day's game began an hour ago
                conn.execute(text("UPDATE raw.games SET start_time_utc = :s "
                                  "WHERE game_id = :g"),
                             {"s": datetime.now(timezone.utc) - timedelta(hours=1),
                              "g": self.GAME})
        try:
            news.run_news()
            assert self._injury_rows_today() == (0 if started else 1)
            with engine.connect() as conn:       # compared either way
                assert conn.execute(text("SELECT COUNT(*) FROM raw.news_state "
                                         "WHERE source = 'espn'")).scalar() >= 1
        finally:
            with engine.begin() as conn:
                conn.execute(text("DELETE FROM raw.injuries WHERE snapshot_date = :d"),
                             {"d": self.today})

    def test_no_paid_snapshot_means_no_new_pick(self):
        news.run_news()
        self.goalies["BOS"] = ("Jeremy Swayman", "Confirmed")
        news.run_news()
        ev = [e for e in self._events() if e["kind"] == "STARTER_CONFIRMED"]
        assert len(ev) == 1 and ev[0]["market_moved"] is None
        assert "no paid odds snapshot" in ev[0]["market_note"]
        assert self.recs == [(self.today, set())]

    def test_a_failing_source_does_not_stop_the_others(self, monkeypatch, caplog):
        from ingestion import espn_injuries

        def down():
            raise RuntimeError("ESPN down")
        monkeypatch.setattr(espn_injuries, "fetch_injuries", down)
        news.run_news()
        assert "News injuries check failed (non-fatal)" in caplog.text
        with engine.connect() as conn:
            notes = conn.execute(text("SELECT notes FROM raw.news_runs")).scalar()
        assert "injuries: failed" in notes and "starters: 0" in notes
