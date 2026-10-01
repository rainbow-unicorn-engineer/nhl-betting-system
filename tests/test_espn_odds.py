"""
Tests for ingestion/espn_odds.py.

No network. The pickcenter fixture (tests/fixtures/espn_pickcenter.json)
holds two real blocks from ESPN's summary API, trimmed: DraftKings for
TB@BUF 2026-04-06 (game 2025021229), and the older Unibet layout for
PIT@NYI 2024-04-17 (game 2023021303), whose "current" prices were captured
in play. The database tests (clone or a _test copy only) use a synthetic
game, 9999020301, and delete it afterwards.
"""
import datetime as dt
import json
from pathlib import Path

import pytest
from sqlalchemy import text

from config.settings import check_db_connection, engine
from ingestion import espn_odds
from ingestion.espn_odds import (
    PRICE_FIELDS, _espn_name_to_abbrev, _match_events_to_games, close_changed,
    match_events, parse_american, parse_line, parse_pickcenter, refresh_decision,
)

FIXTURE = json.loads((Path(__file__).resolve().parent / "fixtures" / "espn_pickcenter.json")
                     .read_text(encoding="utf-8"))
DK = FIXTURE["draftkings_2025021229"]["block"]
UNIBET = FIXTURE["unibet_2023021303"]["block"]

requires_db = pytest.mark.skipif(not check_db_connection(),
                                 reason="database not reachable")


def make_event(event_id, home_name, away_name, date="2026-04-06T23:00Z"):
    return {
        "id": event_id,
        "date": date,
        "competitions": [{
            "competitors": [
                {"homeAway": "home", "team": {"displayName": home_name}},
                {"homeAway": "away", "team": {"displayName": away_name}},
            ],
        }],
    }


class TestNameMapping:
    def test_shared_map_and_espn_extras(self):
        assert _espn_name_to_abbrev("Boston Bruins") == "BOS"
        assert _espn_name_to_abbrev("Montreal Canadiens") == "MTL"
        assert _espn_name_to_abbrev("Utah Mammoth") == "UTA"      # 2025-26 rename
        assert _espn_name_to_abbrev("Utah Hockey Club") == "UTA"  # 2024-25 name
        assert _espn_name_to_abbrev("Quebec Nordiques") is None


class TestMatchEvents:
    def test_matches_on_home_team(self):
        events = [make_event("e1", "Boston Bruins", "Ottawa Senators"),
                  make_event("e2", "Utah Mammoth", "Dallas Stars")]
        games = [{"game_id": 101, "home_team": "BOS"},
                 {"game_id": 102, "home_team": "UTA"},
                 {"game_id": 103, "home_team": "SEA"}]  # not on ESPN's slate
        mapping = _match_events_to_games(events, games)
        assert mapping == {101: "e1", 102: "e2"}

    def test_unknown_espn_name_is_skipped(self):
        events = [make_event("e1", "Mystery Team", "Boston Bruins")]
        assert _match_events_to_games(events, [{"game_id": 1, "home_team": "BOS"}]) == {}

    def test_match_events_carries_the_start_time(self):
        events = [make_event("401803580", "Buffalo Sabres", "Tampa Bay Lightning")]
        m = match_events(events, [{"game_id": 2025021229, "home_team": "BUF"}])
        assert m[2025021229]["event_id"] == "401803580"
        assert m[2025021229]["start"] == dt.datetime(2026, 4, 6, 23, 0, tzinfo=dt.timezone.utc)

    def test_missing_or_bad_start_is_none(self):
        ev = make_event("e1", "Boston Bruins", "Ottawa Senators", date=None)
        assert match_events([ev], [{"game_id": 1, "home_team": "BOS"}])[1]["start"] is None
        ev = make_event("e1", "Boston Bruins", "Ottawa Senators", date="soon")
        assert match_events([ev], [{"game_id": 1, "home_team": "BOS"}])[1]["start"] is None


class TestParseHelpers:
    @pytest.mark.parametrize("raw, want", [
        (-115.0, -115), (102, 102), ("+102", 102), ("-105", -105),
        ("Even", 100), ("EV", 100), (" -3500 ", -3500), ("−110", -110),
        (None, None), ("", None), ("OFF", None), (50, None), (-99.0, None),
        (True, None), (float("nan"), None),
    ])
    def test_parse_american(self, raw, want):
        assert parse_american(raw) == want

    @pytest.mark.parametrize("raw, want", [
        (6.5, 6.5), ("6.5", 6.5), (".5", 0.5), ("o6.5", 6.5), ("u5.5", 5.5),
        ("+1.5", 1.5), ("-1.5", -1.5), ("PK", 0.0), (None, None), ("", None),
        ("3+", None), (True, None),
    ])
    def test_parse_line(self, raw, want):
        assert parse_line(raw) == want


class TestParsePickcenter:
    def test_draftkings_block_keeps_opening_and_over_under_prices(self):
        """Game 2025021229 (TB@BUF): ML open -105/-115 close +102/-122,
        O6.5 open +110 close -115, U6.5 open -130 close -105."""
        row = parse_pickcenter(DK)
        assert row == {
            "provider": "DraftKings", "home_ml": 102, "away_ml": -122,
            "spread": 1.5, "over_under": 6.5, "details": "TB -122",
            "home_ml_open": -105, "away_ml_open": -115,
            "total_open": 6.5,
            "over_price": -115, "under_price": -105,
            "over_price_open": 110, "under_price_open": -130,
            "spread_home_price": -250, "spread_away_price": 205,
            "spread_open": 1.5,
            "spread_home_price_open": -265, "spread_away_price_open": 215,
        }

    def test_every_price_field_is_returned(self):
        assert set(PRICE_FIELDS) <= set(parse_pickcenter(DK))

    def test_opening_total_can_differ_from_the_close(self):
        block = json.loads(json.dumps(DK))
        block["total"]["over"]["open"] = {"line": "o5.5", "odds": "-135"}
        block["total"]["under"]["open"] = {"line": "u5.5", "odds": "+110"}
        row = parse_pickcenter(block)
        assert row["over_under"] == 6.5 and row["total_open"] == 5.5
        assert (row["over_price_open"], row["under_price_open"]) == (-135, 110)
        assert (row["over_price"], row["under_price"]) == (-115, -105)

    def test_unibet_layout_opening_prices(self):
        """2023-24 Unibet blocks keep the open in a top-level `open` and in
        each team's `open`. Its current prices here are in-play, which the
        module docstring warns about; they are still what ESPN reports."""
        row = parse_pickcenter(UNIBET)
        assert row["provider"] == "Unibet"
        assert (row["home_ml"], row["away_ml"]) == (-500, 4500)
        assert (row["home_ml_open"], row["away_ml_open"]) == (143, 145)
        assert row["total_open"] == 5.5
        assert (row["over_price_open"], row["under_price_open"]) == (-127, 105)
        assert (row["over_price"], row["under_price"]) == (300, -526)
        assert row["spread_open"] == 1.5
        assert (row["spread_home_price_open"], row["spread_away_price_open"]) == (-278, 215)
        assert (row["spread_home_price"], row["spread_away_price"]) == (128, -167)

    def test_close_falls_back_to_the_total_block_on_the_same_line(self):
        block = json.loads(json.dumps(DK))
        del block["overOdds"], block["underOdds"]
        row = parse_pickcenter(block)
        assert (row["over_price"], row["under_price"]) == (-115, -105)

    def test_no_fallback_when_the_total_block_is_on_another_line(self):
        block = json.loads(json.dumps(DK))
        del block["overOdds"], block["underOdds"]
        block["total"]["over"]["close"]["line"] = "o5.5"
        block["total"]["under"]["close"]["line"] = "u5.5"
        row = parse_pickcenter(block)
        assert row["over_price"] is None and row["under_price"] is None

    def test_puck_line_prices_fall_back_to_the_spread_block(self):
        block = json.loads(json.dumps(DK))
        del block["homeTeamOdds"]["spreadOdds"], block["awayTeamOdds"]["spreadOdds"]
        row = parse_pickcenter(block)
        assert (row["spread_home_price"], row["spread_away_price"]) == (-250, 205)

    def test_missing_fields_become_none(self):
        row = parse_pickcenter({"homeTeamOdds": None, "awayTeamOdds": None})
        assert all(v is None for v in row.values())

    def test_impossible_opening_pair_is_dropped(self):
        """Seen in ESPN data (game 2025020390): the total opened at over
        +114 AND under +114, a pair no two-way market can quote."""
        block = json.loads(json.dumps(DK))
        block["total"]["over"]["open"] = {"line": "o6.5", "odds": "+114"}
        block["total"]["under"]["open"] = {"line": "u6.5", "odds": "+114"}
        row = parse_pickcenter(block)
        assert row["over_price_open"] is None and row["under_price_open"] is None
        assert row["total_open"] == 6.5
        assert (row["over_price"], row["under_price"]) == (-115, -105)   # close untouched

    def test_a_side_marked_off_keeps_the_other(self):
        """Seen in ESPN data (game 2025020362): underOdds null, "OFF" in the
        total block."""
        block = json.loads(json.dumps(DK))
        block["underOdds"] = None
        block["total"]["under"]["close"]["odds"] = "OFF"
        row = parse_pickcenter(block)
        assert (row["over_price"], row["under_price"]) == (-115, None)


class TestSanePair:
    @pytest.mark.parametrize("a, b, ok", [
        (-115, -105, True), (-110, -110, True), (100, 100, True),
        (300, -526, True),                 # lopsided but real (in-play Unibet)
        (114, 114, False), (150, 120, False),   # sums below 1
        (-300, -250, False),               # sums far above 1
    ])
    def test_pairs(self, a, b, ok):
        assert espn_odds.sane_pair(a, b) == ((a, b) if ok else (None, None))

    def test_one_side_is_kept(self):
        assert espn_odds.sane_pair(-125, None) == (-125, None)
        assert espn_odds.sane_pair(None, None) == (None, None)

    def test_moneylines_are_not_checked(self):
        """Unibet-era moneylines are 3-way (sum near 0.83) and must survive."""
        row = parse_pickcenter(UNIBET)
        assert (row["home_ml_open"], row["away_ml_open"]) == (143, 145)


class TestRefreshDecision:
    def test_same_book_fills(self):
        assert refresh_decision({"provider": "DraftKings"}, parse_pickcenter(DK)) == "fill"

    def test_no_block_is_remembered(self):
        assert refresh_decision({"provider": "DraftKings"}, None) == "no_block"

    def test_other_book_is_never_mixed_in(self):
        stored = {"provider": "espn-kaggle-onesided"}
        assert refresh_decision(stored, parse_pickcenter(DK)) == "provider_mismatch"

    def test_close_changed(self):
        record = parse_pickcenter(DK)
        assert not close_changed({"home_ml": 102, "away_ml": -122}, record)
        assert close_changed({"home_ml": 110, "away_ml": -122}, record)
        assert not close_changed({"home_ml": None, "away_ml": None}, record)


class TestCommandLine:
    def test_help_runs_nothing(self, monkeypatch, capsys):
        monkeypatch.setattr(espn_odds, "backfill_historical_odds",
                            lambda *a, **k: pytest.fail("--help ran the backfill"))
        monkeypatch.setattr(espn_odds, "refresh_historical_odds",
                            lambda *a, **k: pytest.fail("--help ran the refresh"))
        with pytest.raises(SystemExit) as exc:
            espn_odds.main(["--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "season" in out and "--refresh" in out

    def test_optional_season(self, monkeypatch):
        calls = []
        monkeypatch.setattr(espn_odds, "backfill_historical_odds",
                            lambda season, limit=None: calls.append((season, limit)) or 0)
        espn_odds.main([])
        espn_odds.main(["20252026"])
        espn_odds.main(["--season", "20252026", "--limit", "5"])
        assert calls == [(None, None), (20252026, None), (20252026, 5)]

    def test_refresh_routes_to_refresh(self, monkeypatch):
        calls = []
        monkeypatch.setattr(espn_odds, "backfill_historical_odds",
                            lambda *a, **k: pytest.fail("--refresh ran the backfill"))
        monkeypatch.setattr(espn_odds, "refresh_historical_odds",
                            lambda season, limit=None: calls.append((season, limit)) or 0)
        espn_odds.main(["--refresh", "--season", "20252026"])
        espn_odds.main(["--refresh", "--limit", "25"])
        assert calls == [(20252026, None), (None, 25)]

    @pytest.mark.parametrize("argv", [["20242025", "--season", "20252026"],
                                      ["--limit", "0"]])
    def test_bad_arguments_exit(self, monkeypatch, argv):
        monkeypatch.setattr(espn_odds, "backfill_historical_odds",
                            lambda *a, **k: pytest.fail("ran with bad arguments"))
        with pytest.raises(SystemExit) as exc:
            espn_odds.main(argv)
        assert exc.value.code == 2


# ── Database (clone or _test copy only) ────────────────────────────

GAME_ID = 9999020301


@requires_db
class TestRefreshOnDatabase:
    @pytest.fixture()
    def stored_row(self):
        """A synthetic finished game with a closing-only DraftKings row, as
        the first version of the loader wrote it."""
        espn_odds.ensure_columns()
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO raw.games (game_id, season, game_type, date, home_team,
                                       away_team, home_score, away_score, game_state)
                VALUES (:g, 20302031, 2, '2031-01-15', 'BUF', 'TBL', 4, 2, 'OFF')
            """), {"g": GAME_ID})
            conn.execute(text("""
                INSERT INTO raw.historical_odds
                    (game_id, provider, home_ml, away_ml, spread, over_under, details)
                VALUES (:g, 'DraftKings', 102, -122, 1.5, NULL, 'TB -122')
            """), {"g": GAME_ID})
        yield GAME_ID
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM raw.historical_odds WHERE game_id = :g"), {"g": GAME_ID})
            conn.execute(text("DELETE FROM raw.games WHERE game_id = :g"), {"g": GAME_ID})

    def _read(self, game_id):
        with engine.connect() as conn:
            return conn.execute(text("SELECT * FROM raw.historical_odds WHERE game_id = :g"),
                                {"g": game_id}).mappings().one()

    def test_refresh_fills_prices_and_is_resumable(self, stored_row, monkeypatch):
        calls = {"scoreboard": 0, "summary": 0}

        def scoreboard(yyyymmdd):
            calls["scoreboard"] += 1
            assert yyyymmdd == "20310115"
            return [make_event("401999999", "Buffalo Sabres", "Tampa Bay Lightning")]

        def summary(event_id):
            calls["summary"] += 1
            assert event_id == "401999999"
            return DK

        monkeypatch.setattr(espn_odds, "fetch_scoreboard", scoreboard)
        monkeypatch.setattr(espn_odds, "fetch_game_odds", summary)
        monkeypatch.setattr(espn_odds, "REQUEST_PAUSE_S", 0)

        assert espn_odds.refresh_historical_odds(game_ids=[stored_row]) == 1
        row = self._read(stored_row)
        assert (row["home_ml_open"], row["away_ml_open"]) == (-105, -115)
        assert (row["over_price"], row["under_price"]) == (-115, -105)
        assert (row["over_price_open"], row["under_price_open"]) == (110, -130)
        assert float(row["total_open"]) == 6.5
        assert (row["spread_home_price"], row["spread_away_price"]) == (-250, 205)
        assert row["espn_event_id"] == "401999999"
        assert row["prices_fetched_at"] is not None
        # the stored close is kept; an empty closing column is filled
        assert (row["home_ml"], row["away_ml"]) == (102, -122)
        assert float(row["over_under"]) == 6.5

        # a second run finds nothing left to do and makes no request
        assert espn_odds.refresh_historical_odds(game_ids=[stored_row]) == 0
        assert calls == {"scoreboard": 1, "summary": 1}

    def test_other_book_is_marked_not_filled(self, stored_row, monkeypatch):
        with engine.begin() as conn:
            conn.execute(text("UPDATE raw.historical_odds SET provider = 'espn-kaggle-onesided' "
                              "WHERE game_id = :g"), {"g": stored_row})
        monkeypatch.setattr(espn_odds, "fetch_scoreboard", lambda d: [
            make_event("401999999", "Buffalo Sabres", "Tampa Bay Lightning")])
        monkeypatch.setattr(espn_odds, "fetch_game_odds", lambda e: DK)
        monkeypatch.setattr(espn_odds, "REQUEST_PAUSE_S", 0)
        assert espn_odds.refresh_historical_odds(game_ids=[stored_row]) == 0
        row = self._read(stored_row)
        assert row["prices_fetched_at"] is not None
        assert all(row[c] is None for c in PRICE_FIELDS)

    def test_failed_download_is_retried_next_run(self, stored_row, monkeypatch):
        def boom(event_id):
            raise ConnectionError("network down")
        monkeypatch.setattr(espn_odds, "fetch_scoreboard", lambda d: [
            make_event("401999999", "Buffalo Sabres", "Tampa Bay Lightning")])
        monkeypatch.setattr(espn_odds, "fetch_game_odds", boom)
        monkeypatch.setattr(espn_odds, "REQUEST_PAUSE_S", 0)
        assert espn_odds.refresh_historical_odds(game_ids=[stored_row]) == 0
        assert self._read(stored_row)["prices_fetched_at"] is None

    def test_backfill_stores_prices_for_a_new_game(self, stored_row, monkeypatch):
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM raw.historical_odds WHERE game_id = :g"),
                         {"g": stored_row})
        monkeypatch.setattr(espn_odds, "fetch_scoreboard", lambda d: [
            make_event("401999999", "Buffalo Sabres", "Tampa Bay Lightning")])
        monkeypatch.setattr(espn_odds, "fetch_game_odds", lambda e: DK)
        monkeypatch.setattr(espn_odds, "REQUEST_PAUSE_S", 0)
        assert espn_odds.backfill_historical_odds(season=20302031) == 1
        row = self._read(stored_row)
        assert (row["home_ml"], row["away_ml"]) == (102, -122)
        assert (row["over_price"], row["under_price"]) == (-115, -105)
        assert row["prices_fetched_at"] is not None
