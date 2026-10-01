"""
Tests for ingestion/espn_injuries.py.

No network: the fixture (tests/fixtures/espn_injuries.json) is 12 real
entries from ESPN's injury feed of 2026-09-28, trimmed. The database tests
(clone or a _test copy only) write snapshots dated 2031-01-14 and
2031-01-15, which no real run can produce, and delete them afterwards.
"""
import datetime as dt
import json
from pathlib import Path

import pytest
from sqlalchemy import text

from config.settings import check_db_connection, engine, local_today
from ingestion import espn_injuries
from ingestion.espn_injuries import (PlayerIndex, athlete_id, espn_team_abbrev,
                                     parse_injuries, position_group)

FIXTURE = json.loads((Path(__file__).resolve().parent / "fixtures" / "espn_injuries.json")
                     .read_text(encoding="utf-8"))

requires_db = pytest.mark.skipif(not check_db_connection(),
                                 reason="database not reachable")


def by_name(rows):
    return {r["player_name"]: r for r in rows}


class TestParse:
    def test_every_entry_becomes_a_row(self):
        rows = parse_injuries(FIXTURE)
        assert len(rows) == 12
        assert len({r["espn_athlete_id"] for r in rows}) == 12

    def test_fields(self):
        greer = by_name(parse_injuries(FIXTURE))["A.J. Greer"]
        assert greer == {
            "espn_athlete_id": 3648015, "player_name": "A.J. Greer",
            "team_abbrev": "ANA", "position": "L", "status": "Day-To-Day",
            "injury_type": "Upper Body", "injury_detail": None,
            "return_date": dt.date(2026, 10, 2),
            "short_comment": "day-to-day", "long_comment": "day-to-day",
            "reported_at": dt.datetime(2026, 9, 27, 13, 59, tzinfo=dt.timezone.utc),
        }

    def test_detail_status_and_goalies(self):
        rows = by_name(parse_injuries(FIXTURE))
        assert rows["Troy Terry"]["injury_detail"] == "Surgery"
        assert rows["Troy Terry"]["position"] == "R"
        assert rows["Charlie McAvoy"]["status"] == "Suspension"
        assert rows["Thatcher Demko"]["position"] == "G"
        assert rows["Thatcher Demko"]["status"] == "Injured Reserve"

    def test_espn_abbreviations_become_nhl_ones(self):
        teams = {r["player_name"]: r["team_abbrev"] for r in parse_injuries(FIXTURE)}
        assert teams["Kevin Fiala"] == "LAK"          # ESPN: LA
        assert teams["Nick Bjugstad"] == "NJD"        # ESPN: NJ
        assert teams["Kyle Keyser"] == "SJS"          # ESPN: SJ
        assert teams["Yanni Gourde"] == "TBL"         # ESPN: TB
        assert teams["Maveric Lamoureux"] == "UTA"    # Utah Mammoth

    def test_accents_survive(self):
        assert "Melvin Fernström" in by_name(parse_injuries(FIXTURE))

    def test_athlete_id_is_not_the_injury_id(self):
        """The entry's own id (593968) belongs to the injury note."""
        greer = by_name(parse_injuries(FIXTURE))["A.J. Greer"]
        assert greer["espn_athlete_id"] == 3648015

    def test_entry_without_an_id_is_skipped(self):
        payload = {"injuries": [{"displayName": "Boston Bruins", "injuries": [
            {"status": "Out", "athlete": {"displayName": "No Id"}}]}]}
        assert parse_injuries(payload) == []

    def test_listed_twice_keeps_the_latest(self):
        a = {"displayName": "Joe Test", "links": [{"href": "https://x/nhl/player/_/id/77"}],
             "team": {"displayName": "Boston Bruins"}}
        payload = {"injuries": [{"displayName": "Boston Bruins", "injuries": [
            {"status": "Out", "date": "2026-09-20T10:00Z", "athlete": a},
            {"status": "Day-To-Day", "date": "2026-09-25T10:00Z", "athlete": a},
            {"status": "Injured Reserve", "date": "2026-09-22T10:00Z", "athlete": a},
        ]}]}
        rows = parse_injuries(payload)
        assert [(r["espn_athlete_id"], r["status"]) for r in rows] == [(77, "Day-To-Day")]

    def test_team_falls_back_to_the_group(self):
        payload = {"injuries": [{"displayName": "Tampa Bay Lightning", "injuries": [
            {"status": "Out", "athlete": {"displayName": "Joe Test", "id": "5"}}]}]}
        assert parse_injuries(payload)[0]["team_abbrev"] == "TBL"

    def test_changed_feed_is_loud(self):
        with pytest.raises(ValueError, match="injuries"):
            parse_injuries({"teams": []})


class TestAthleteId:
    @pytest.mark.parametrize("athlete, want", [
        ({"id": "123"}, 123),
        ({"links": [{"href": "https://www.espn.com/nhl/player/_/id/3648015"}]}, 3648015),
        ({"headshot": {"href": "https://a.espncdn.com/i/headshots/nhl/players/full/3942905.png"}},
         3942905),
        ({"notes": {"items": [{"injury": {"$ref": "http://sports.core.api.espn.pvt/v2/sports/"
                                                  "hockey/leagues/nhl/seasons/2027/athletes/"
                                                  "3648015/injuries/593968?lang=en"}}]}}, 3648015),
        ({"uid": "s:70~l:90~a:4233889"}, 4233889),
        ({"displayName": "No Id"}, None),
        (None, None),
    ])
    def test_sources(self, athlete, want):
        assert athlete_id(athlete) == want


class TestTeamsAndPositions:
    def test_team_abbrev(self):
        assert espn_team_abbrev("Los Angeles Kings", "LA") == "LAK"
        assert espn_team_abbrev(None, "LA") == "LAK"
        assert espn_team_abbrev(None, "TB") == "TBL"
        assert espn_team_abbrev(None, "NJ") == "NJD"
        assert espn_team_abbrev("", "sj") == "SJS"
        assert espn_team_abbrev(None, "BOS") == "BOS"
        assert espn_team_abbrev("St. Louis Blues", None) == "STL"
        assert espn_team_abbrev("Quebec Nordiques", "QUE") is None
        assert espn_team_abbrev(None, None) is None

    @pytest.mark.parametrize("pos, group", [("C", "F"), ("LW", "F"), ("R", "F"),
                                            ("D", "D"), ("G", "G"), (None, None),
                                            ("", None), ("X", None)])
    def test_position_group(self, pos, group):
        assert position_group(pos) == group


class TestPlayerIndex:
    PLAYERS = [
        (1, "J. Swayman", "G"),
        (2, "S. Aho", "C"),          # Sebastian Aho, CAR forward
        (3, "S. Aho", "D"),          # Sebastian Aho, NYI defenceman
        (4, "T. Stützle", "C"),
        (5, "M. Fernström", "R"),
        (6, "E. Lindholm", "C"),     # two E. Lindholms on one team
        (7, "E. Lindholm", "C"),
        (8, "A. Greer", "L"),
        (9, "J. Hughes", "C"),
        (10, "J. Hughes", "G"),      # a goalie sharing a skater's key
        (11, "M. Murray", "G"),      # two goalies, same key, different teams
        (12, "M. Murray", "G"),
    ]
    LAST_TEAM = {1: "BOS", 2: "CAR", 3: "NYI", 4: "OTT", 5: "PIT", 6: "BOS", 7: "BOS",
                 8: "ANA", 9: "NJD", 10: "NJD", 11: "PIT", 12: "TOR"}

    @pytest.fixture()
    def index(self):
        return PlayerIndex(self.PLAYERS, self.LAST_TEAM)

    def test_full_name_matches_abbreviated(self, index):
        assert index.match("Jeremy Swayman", "BOS", "G") == (1, "initial")

    def test_exact_abbreviated_name(self, index):
        assert index.match("J. Swayman", "BOS", "G") == (1, "exact")

    def test_initials_with_periods(self, index):
        assert index.match("A.J. Greer", "ANA", "LW") == (8, "initial")

    def test_accents_are_ignored(self, index):
        assert index.match("Tim Stutzle", "OTT", "C") == (4, "initial")
        assert index.match("Melvin Fernström", "PIT", "RW") == (5, "initial")

    def test_team_breaks_a_tie(self, index):
        assert index.match("Sebastian Aho", "NYI", "D") == (3, "initial")
        assert index.match("Sebastian Aho", "CAR", "C") == (2, "initial")

    def test_team_alone_breaks_a_tie(self, index):
        assert index.match("Matt Murray", "TOR", "G") == (12, "initial")
        assert index.match("Matt Murray", "PIT") == (11, "initial")   # no position given
        assert index.match("Matt Murray", None, "G") == (None, "ambiguous")

    def test_position_breaks_a_tie_when_the_team_does_not(self, index):
        assert index.match("Sebastian Aho", None, "D") == (3, "initial")

    def test_true_tie_stays_unresolved(self, index):
        assert index.match("Elias Lindholm", "BOS", "C") == (None, "ambiguous")

    def test_goalie_never_matches_a_skater(self, index):
        assert index.match("Jack Hughes", "NJD", "C") == (9, "initial")
        assert index.match("Jake Hughes", "NJD", "G") == (10, "initial")
        assert index.match("Jeremy Swayman", "BOS", "C") == (None, "unknown")

    def test_surname_on_the_same_team(self, index):
        """A listed first name whose initial differs from raw.players' (a
        nickname or another spelling) still matches on the surname, but
        only among players last seen on the same team."""
        assert index.match("Joe Greer", "ANA", "LW") == (8, "surname")
        assert index.match("Joe Greer", "BOS", "LW") == (None, "unknown")

    def test_unknown(self, index):
        assert index.match("Nobody Here", "BOS", "C") == (None, "unknown")
        assert index.match("", "BOS") == (None, "unknown")
        assert index.match(None) == (None, "unknown")

    def test_resolve_player_ids_sets_every_row(self, index):
        rows = [{"player_name": "Jeremy Swayman", "team_abbrev": "BOS", "position": "G"},
                {"player_name": "Nobody Here", "team_abbrev": "BOS", "position": "C"}]
        espn_injuries.resolve_player_ids(rows, index)
        assert [r["player_id"] for r in rows] == [1, None]


class TestCommandLine:
    def test_help_runs_nothing(self, monkeypatch, capsys):
        monkeypatch.setattr(espn_injuries, "ingest_injuries",
                            lambda *a, **k: pytest.fail("--help ran the job"))
        with pytest.raises(SystemExit) as exc:
            espn_injuries.main(["--help"])
        assert exc.value.code == 0
        assert "--date" in capsys.readouterr().out

    def test_default_is_today(self, monkeypatch):
        calls = []
        monkeypatch.setattr(espn_injuries, "ingest_injuries", lambda d=None: calls.append(d) or 0)
        espn_injuries.main([])
        espn_injuries.main(["--date", str(local_today())])
        assert calls == [None, local_today()]

    @pytest.mark.parametrize("days", [-30, 5])
    def test_a_date_far_from_today_is_refused_before_any_request(self, monkeypatch, days):
        monkeypatch.setattr(espn_injuries, "fetch_injuries",
                            lambda: pytest.fail("fetched with a bad --date"))
        monkeypatch.setattr(espn_injuries, "ingest_injuries",
                            lambda *a, **k: pytest.fail("ran with a bad --date"))
        with pytest.raises(SystemExit) as exc:
            espn_injuries.main(["--date", str(local_today() + dt.timedelta(days=days))])
        assert exc.value.code == 2


# ── Database (clone or _test copy only) ────────────────────────────

DAY = dt.date(2031, 1, 15)
OTHER_DAY = dt.date(2031, 1, 14)


@requires_db
class TestStore:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        espn_injuries.ensure_table()
        yield
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM raw.injuries WHERE snapshot_date IN (:a, :b)"),
                         {"a": DAY, "b": OTHER_DAY})

    def _snapshot(self, day):
        with engine.connect() as conn:
            return {r.espn_athlete_id: r for r in conn.execute(text(
                "SELECT * FROM raw.injuries WHERE snapshot_date = :d"), {"d": day})}

    def test_rerun_replaces_only_that_day(self):
        rows = parse_injuries(FIXTURE)
        for r in rows:
            r["player_id"] = None
        assert espn_injuries.write_injuries(rows, OTHER_DAY) == 12
        assert espn_injuries.write_injuries(rows, DAY) == 12
        # later that day Greer is off the list and Demko's status changed
        later = [dict(r) for r in rows if r["player_name"] != "A.J. Greer"]
        for r in later:
            if r["player_name"] == "Thatcher Demko":
                r["status"] = "Out"
        assert espn_injuries.write_injuries(later, DAY) == 11
        today = self._snapshot(DAY)
        assert len(today) == 11 and 3648015 not in today
        assert today[3096217].status == "Out"
        assert len(self._snapshot(OTHER_DAY)) == 12        # untouched

    def test_empty_list_writes_nothing(self, monkeypatch):
        rows = parse_injuries(FIXTURE)
        for r in rows:
            r["player_id"] = None
        espn_injuries.write_injuries(rows, DAY)
        monkeypatch.setattr(espn_injuries, "fetch_injuries", lambda: {"injuries": []})
        assert espn_injuries.ingest_injuries(DAY) == 0
        assert len(self._snapshot(DAY)) == 12

    def test_ingest_resolves_real_players(self, monkeypatch):
        """Build an ESPN-style full name from a real raw.players goalie
        (same initial and surname) and require it to resolve."""
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT p.player_id, p.full_name, gg.team
                FROM raw.players p JOIN raw.goalie_games gg USING (player_id)
                JOIN raw.games g USING (game_id)
                WHERE p.position = 'G' AND p.full_name LIKE '_. %'
                ORDER BY g.date DESC LIMIT 1
            """)).fetchone()
        initial, surname = row.full_name.split(". ", 1)
        payload = {"injuries": [{"displayName": "x", "injuries": [{
            "status": "Out", "date": "2026-09-27T13:59Z",
            "athlete": {"displayName": f"{initial}ohnfake {surname}", "id": "999000111",
                        "position": {"abbreviation": "G"},
                        "team": {"abbreviation": row.team}}}]}]}
        monkeypatch.setattr(espn_injuries, "fetch_injuries", lambda: payload)
        assert espn_injuries.ingest_injuries(DAY) == 1
        assert self._snapshot(DAY)[999000111].player_id == row.player_id
