"""
Tests for ingestion/dailyfaceoff_lines.py: the parser on two saved Daily
Faceoff pages (tests/fixtures/dailyfaceoff_lines_bos.html and _mtl.html,
fetched 2026-10-04 and trimmed to the fields the parser reads), the team
slugs, the change fingerprint, the politeness rules, and the fetch loop
with the network and database stubbed. The database round trip runs only
against a disposable database (tests/conftest.py).
"""
import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from ingestion import dailyfaceoff_lines as dfl
from ingestion.espn_injuries import PlayerIndex
from ingestion.odds_api import _TEAM_NAME_TO_ABBREV

FIXTURES = Path(__file__).resolve().parent / "fixtures"
BOS = (FIXTURES / "dailyfaceoff_lines_bos.html").read_text(encoding="utf-8")
MTL = (FIXTURES / "dailyfaceoff_lines_mtl.html").read_text(encoding="utf-8")
NOW = datetime(2026, 10, 10, 14, 0, tzinfo=timezone.utc)


def _units(rows):
    out = {}
    for r in rows:
        out.setdefault(r["unit"], []).append(r)
    return out


class TestParser:
    def test_full_lineup(self):
        page = dfl.parse_lineup(BOS)
        assert page["team_name"] == "Boston Bruins"
        assert page["updated_at"] == datetime(2026, 10, 3, 23, 41, 19, 791000,
                                              tzinfo=timezone.utc)
        units = _units(page["rows"])
        assert {u: len(r) for u, r in units.items()} == {
            "F1": 3, "F2": 3, "F3": 3, "F4": 3, "D1": 2, "D2": 2, "D3": 2,
            "G": 2, "PP1": 5, "PP2": 5, "PK1": 4, "PK2": 4, "IR": 1}
        assert dfl.dressed_count(page["rows"]) == 18

    def test_slots_positions_and_goalies(self):
        units = _units(dfl.parse_lineup(BOS)["rows"])
        f1 = {r["slot"]: r for r in units["F1"]}
        assert f1["lw"]["player_name"] == "JJ Peterka" and f1["lw"]["position"] == "L"
        assert {r["position"] for r in units["D1"]} == {"D"}
        goalies = {r["slot"]: r["player_name"] for r in units["G"]}
        assert goalies == {"g1": "Michael DiPietro", "g2": "Jeremy Swayman"}

    def test_special_teams_take_the_even_strength_position(self):
        pp1 = {r["player_name"]: r["position"] for r in _units(dfl.parse_lineup(BOS)["rows"])["PP1"]}
        assert pp1 == {"Morgan Geekie": "R", "Pavel Zacha": "C", "JJ Peterka": "L",
                       "David Pastrnak": "R", "Hampus Lindholm": "D"}

    def test_injured_reserve_and_game_time_decisions(self):
        rows = dfl.parse_lineup(MTL)["rows"]
        ir = {r["player_name"]: r["injury_status"] for r in _units(rows)["IR"]}
        assert ir == {"Kaiden Guhle": "out", "Oliver Kapanen": "dtd",
                      "Alexandre Carrier": "dtd"}
        # a player not in the lines has no known position
        assert next(r for r in rows if r["player_name"] == "Kaiden Guhle")["position"] is None
        gtd = {(r["unit"], r["player_name"]) for r in rows if r["game_time_decision"]}
        assert gtd == {("D2", "Alexandre Carrier"), ("PK1", "Alexandre Carrier")}

    def test_unit_names(self):
        assert dfl.unit_name("f1") == "F1" and dfl.unit_name("d3") == "D3"
        assert dfl.unit_name("g") == "G" and dfl.unit_name("pp2") == "PP2"
        assert dfl.unit_name("ir") == "IR" and dfl.unit_name("scratches") == "SCRATC"
        assert dfl.unit_name(None) is None and dfl.unit_name("") is None

    def test_layout_change_is_an_error_not_an_empty_lineup(self):
        with pytest.raises(ValueError, match="layout changed"):
            dfl.parse_lineup("<html>nothing here</html>")
        empty = ('<script id="__NEXT_DATA__" type="application/json">'
                 + json.dumps({"props": {"pageProps": {}}}) + "</script>")
        with pytest.raises(ValueError, match="no line combinations"):
            dfl.parse_lineup(empty)

    def test_duplicate_slots_and_nameless_rows_are_dropped(self):
        players = [
            {"name": "A One", "categoryIdentifier": "ev", "groupIdentifier": "f1",
             "positionIdentifier": "c"},
            {"name": "B Two", "categoryIdentifier": "ev", "groupIdentifier": "f1",
             "positionIdentifier": "c"},
            {"name": "", "categoryIdentifier": "ev", "groupIdentifier": "f1",
             "positionIdentifier": "lw"},
        ]
        html = ('<script id="__NEXT_DATA__" type="application/json">'
                + json.dumps({"props": {"pageProps": {"combinations": {
                    "teamName": "X", "players": players}}}}) + "</script>")
        rows = dfl.parse_lineup(html)["rows"]
        assert [r["player_name"] for r in rows] == ["A One"]


class TestSlugs:
    def test_every_team_has_the_slug_daily_faceoff_lists(self):
        data = json.loads(re.search(r'<script id="__NEXT_DATA__" type="application/json">'
                                    r'(.*?)</script>', BOS, re.S).group(1))
        teams = data["props"]["pageProps"]["sortedTeams"]
        assert len(teams) == 32
        listed = {_TEAM_NAME_TO_ABBREV[t["name"]]: t["slug"] for t in teams}
        assert listed == dfl.DF_SLUGS


class TestFingerprint:
    def test_order_does_not_matter_but_a_change_does(self):
        rows = dfl.parse_lineup(BOS)["rows"]
        assert dfl.lines_hash(rows) == dfl.lines_hash(list(reversed(rows)))
        moved = [dict(r) for r in rows]
        moved[0]["player_name"] = "Someone Else"
        assert dfl.lines_hash(moved) != dfl.lines_hash(rows)
        flagged = [dict(r) for r in rows]
        flagged[0]["game_time_decision"] = True
        assert dfl.lines_hash(flagged) != dfl.lines_hash(rows)


class TestPoliteness:
    def test_gap_is_long_far_from_puck_drop_and_short_near_it(self):
        far = NOW + timedelta(hours=6)
        near = NOW + timedelta(hours=2)
        kw = {"near_gap": 14, "far_gap": 55, "near_hours": 3}
        assert dfl.fetch_gap(far, NOW, **kw) == timedelta(minutes=55)
        assert dfl.fetch_gap(near, NOW, **kw) == timedelta(minutes=14)
        assert dfl.fetch_gap(None, NOW, **kw) == timedelta(minutes=14)

    def test_gap_settings_come_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("LINEUPS_FAR_GAP_MINUTES", "90")
        monkeypatch.setenv("LINEUPS_MIN_GAP_MINUTES", "abc")      # bad: default
        assert dfl.fetch_gap(NOW + timedelta(hours=8), NOW) == timedelta(minutes=90)
        assert dfl.fetch_gap(NOW + timedelta(hours=1), NOW) == timedelta(minutes=29)

    def test_teams_due(self):
        starts = {"BOS": NOW + timedelta(hours=6), "MTL": NOW + timedelta(hours=1),
                  "TOR": NOW + timedelta(hours=1), "XXX": None}
        last = {"BOS": NOW - timedelta(minutes=30),     # far, 30 < 55: wait
                "MTL": NOW - timedelta(minutes=20)}     # near, 20 >= 14: due
        kw = {"near_gap": 14, "far_gap": 55, "near_hours": 3}
        assert dfl.teams_to_fetch(starts, last, NOW, **kw) == ["MTL", "TOR"]
        assert dfl.teams_to_fetch(starts, last, NOW, force=True, **kw) == ["BOS", "MTL", "TOR"]

    def test_a_refusal_holds_teams_back_until_its_time(self):
        starts = {"MTL": NOW + timedelta(hours=1), "TOR": NOW + timedelta(hours=1)}
        kw = {"near_gap": 14, "far_gap": 55, "near_hours": 3}
        hold = {"MTL": NOW + timedelta(minutes=5), "TOR": NOW - timedelta(minutes=1)}
        assert dfl.teams_to_fetch(starts, {}, NOW, not_before=hold, **kw) == ["TOR"]
        assert dfl.teams_to_fetch(starts, {}, NOW, force=True, not_before=hold,
                                  **kw) == ["MTL", "TOR"]

    def test_retry_after_and_the_back_off(self):
        assert dfl.retry_after_seconds("120") == 120
        assert dfl.retry_after_seconds(None) is None
        assert dfl.retry_after_seconds("soon") is None
        assert dfl.retry_after_seconds("Sat, 10 Oct 2026 16:00:00 GMT", now=NOW) == 7200
        assert dfl.retry_after_seconds("Sat, 10 Oct 2026 13:00:00 GMT", now=NOW) == 0
        assert dfl.refused_until(NOW, None, 60) == NOW + timedelta(minutes=60)
        assert dfl.refused_until(NOW, 7200, 60) == NOW + timedelta(hours=2)
        assert dfl.refused_until(NOW, 30, 60) == NOW + timedelta(minutes=60)

    @pytest.mark.parametrize("status", [429, 403])
    def test_fetch_page_raises_on_a_refusal(self, monkeypatch, status):
        class Resp:
            status_code, text = status, ""
            headers = {"Retry-After": "120"}

            def raise_for_status(self):
                raise AssertionError("a refusal is not a plain HTTP error")
        monkeypatch.setattr(dfl.requests, "get", lambda *a, **k: Resp())
        with pytest.raises(dfl.SiteRefused) as e:
            dfl.fetch_page("BOS")
        assert (e.value.status, e.value.retry_after) == (status, 120)


class TestIngestLoop:
    """ingest_lineups with the page fetch and the database stubbed."""

    @pytest.fixture()
    def stubbed(self, monkeypatch):
        written, pauses = {}, []
        monkeypatch.setattr(dfl, "ensure_table", lambda: None)
        monkeypatch.setattr(dfl, "_last_fetches", lambda: {
            "BOS": {"fetched_at": datetime.now(timezone.utc) - timedelta(hours=2),
                    "lines_hash": None}})

        def write(team, game_date, parsed, now, previous_hash):
            written[team] = parsed
            return "broken" if dfl.dressed_count(parsed["rows"]) < dfl.MIN_DRESSED else "new"
        monkeypatch.setattr(dfl, "write_snapshot", write)
        monkeypatch.setattr(dfl.time, "sleep", lambda s: pauses.append(s))
        return written, pauses

    def test_one_request_per_due_team_with_a_pause_between(self, stubbed):
        written, pauses = stubbed
        asked = []

        def fetch(team):
            asked.append(team)
            return {"BOS": BOS, "MTL": MTL}[team]
        index = PlayerIndex([(8477956, "D. Pastrnak", "R"), (8480039, "N. Suzuki", "C")],
                            {8477956: "BOS", 8480039: "MTL"})
        soon = datetime.now(timezone.utc) + timedelta(hours=1)
        got = dfl.ingest_lineups({"BOS": soon, "MTL": soon}, game_date=date(2026, 10, 10),
                                 fetch=fetch, pause_s=2, index=index)
        assert got == {"BOS": "new", "MTL": "new"}
        assert asked == ["BOS", "MTL"] and pauses == [2]
        pp1 = {r["player_name"]: r["player_id"] for r in written["BOS"]["rows"]
               if r["unit"] == "PP1"}
        assert pp1["David Pastrnak"] == 8477956 and pp1["Pavel Zacha"] is None
        assert any(r["player_id"] == 8480039 for r in written["MTL"]["rows"])

    def test_a_failing_team_does_not_stop_the_others(self, stubbed, monkeypatch, caplog):
        monkeypatch.setattr(dfl, "_record_fetch", lambda *a, **k: None)

        def fetch(team):
            if team == "BOS":
                raise RuntimeError("503")
            return MTL
        soon = datetime.now(timezone.utc) + timedelta(hours=1)
        got = dfl.ingest_lineups({"BOS": soon, "MTL": soon}, fetch=fetch, pause_s=0,
                                 index=PlayerIndex([]))
        assert got == {"BOS": "failed", "MTL": "new"}
        assert "Lineups BOS: fetch or parse failed (non-fatal)" in caplog.text

    def test_a_refusal_stops_the_run_and_holds_every_team(self, stubbed, monkeypatch, caplog):
        from contextlib import contextmanager
        recorded, held, asked = [], [], []

        class FakeEngine:
            @contextmanager
            def begin(self):
                yield None
        monkeypatch.setattr(dfl, "engine", FakeEngine())
        monkeypatch.setattr(dfl, "_record_fetch", lambda conn, team, now, u, h, status,
                            not_before=None: recorded.append((team, status, not_before)))
        monkeypatch.setattr(dfl, "_hold", lambda conn, team, now, until: held.append(
            (team, until)))
        monkeypatch.setenv("LINEUPS_REFUSED_BACKOFF_MINUTES", "60")

        def fetch(team):
            asked.append(team)
            raise dfl.SiteRefused(429, 7200)
        soon = datetime.now(timezone.utc) + timedelta(hours=1)
        teams = {"MTL": soon, "TOR": soon, "VAN": soon, "BOS": soon}
        got = dfl.ingest_lineups(teams, fetch=fetch, pause_s=0, index=PlayerIndex([]))
        assert asked == ["BOS"]                      # MTL, TOR and VAN not asked
        assert got == {"BOS": "failed", "MTL": "failed", "TOR": "failed", "VAN": "failed"}
        (team, status, until), = recorded
        assert (team, status) == ("BOS", "refused")
        assert timedelta(minutes=119) < until - datetime.now(timezone.utc) <= timedelta(hours=2)
        assert sorted(t for t, _ in held) == ["MTL", "TOR", "VAN"]
        assert all(u == until for _, u in held)
        assert "Stopping: 3 other team(s) not asked" in caplog.text

    def test_a_page_with_too_few_players_is_not_stored(self, stubbed):
        thin = BOS.replace('"groupIdentifier": "f2"', '"groupIdentifier": "x2"') \
                  .replace('"groupIdentifier": "f3"', '"groupIdentifier": "x3"') \
                  .replace('"groupIdentifier": "f4"', '"groupIdentifier": "x4"')
        soon = datetime.now(timezone.utc) + timedelta(hours=1)
        got = dfl.ingest_lineups({"BOS": soon}, fetch=lambda t: thin, pause_s=0,
                                 index=PlayerIndex([]))
        assert got == {"BOS": "broken"}

    def test_no_teams_no_requests(self, stubbed):
        assert dfl.ingest_lineups({}, fetch=lambda t: pytest.fail("no request")) == {}


# ── Database round trip (disposable database only) ─────────────────

from config.settings import check_db_connection, engine  # noqa: E402
from sqlalchemy import text  # noqa: E402

requires_db = pytest.mark.skipif(not check_db_connection(), reason="database not reachable")


@requires_db
def test_snapshot_is_stored_only_when_the_lines_change():
    dfl._table_ready = False
    dfl.ensure_table()
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM raw.lineups WHERE team = 'ZZZ'"))
        conn.execute(text("DELETE FROM raw.lineup_fetches WHERE team = 'ZZZ'"))
    page = dfl.parse_lineup(BOS)
    t1 = datetime.now(timezone.utc).replace(microsecond=0)
    assert dfl.write_snapshot("ZZZ", date(2026, 10, 10), page, t1, None) == "new"
    digest = dfl.lines_hash(page["rows"])
    t2 = t1 + timedelta(minutes=15)
    assert dfl.write_snapshot("ZZZ", date(2026, 10, 10), page, t2, digest) == "same"
    with engine.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*), COUNT(DISTINCT snapshot_ts) "
                              "FROM raw.lineups WHERE team = 'ZZZ'")).one()
        fetch = conn.execute(text("SELECT fetched_at, status, lines_hash "
                                  "FROM raw.lineup_fetches WHERE team = 'ZZZ'")).one()
    assert tuple(n) == (39, 1)
    assert fetch.status == "same" and fetch.fetched_at == t2 and fetch.lines_hash == digest
    latest = dfl.latest_lineups(["ZZZ"])["ZZZ"]
    assert len(latest) == 39

    # a refusal holds the team back; the next good fetch clears the hold
    until = t2 + timedelta(hours=1)
    with engine.begin() as conn:
        dfl._record_fetch(conn, "ZZZ", t2, None, None, "refused", until)
    got = dfl._last_fetches()["ZZZ"]
    assert got["next_fetch_after"] == until and got["lines_hash"] == digest
    assert dfl.write_snapshot("ZZZ", date(2026, 10, 10), page,
                              t2 + timedelta(hours=2), digest) == "same"
    assert dfl._last_fetches()["ZZZ"]["next_fetch_after"] is None
    # a hold on a team with an earlier fetch keeps that fetch
    with engine.begin() as conn:
        dfl._hold(conn, "ZZZ", t2 + timedelta(hours=3), until)
    got = dfl._last_fetches()["ZZZ"]
    assert got["fetched_at"] == t2 + timedelta(hours=2) and got["next_fetch_after"] == until
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM raw.lineups WHERE team = 'ZZZ'"))
        conn.execute(text("DELETE FROM raw.lineup_fetches WHERE team = 'ZZZ'"))
