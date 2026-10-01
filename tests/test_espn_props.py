"""
Tests for ingestion/espn_props.py.

No network. The fixture (tests/fixtures/espn_props.json) is real ESPN data,
trimmed: DraftKings props for MTL@BUF 2026-01-15 (Jason Zucker, Cole
Caufield, Colten Ellis; all last updated before puck drop) and ESPN BET
props for NYR@CGY 2025-10-26 (Mikael Backlund, Nazem Kadri, Dustin Wolf;
updated in play), each with its boxscore names. The database tests (clone
or a _test copy only) use a synthetic game, 9999020302, and delete it
afterwards.
"""
import copy
import datetime as dt
import json
from pathlib import Path

import pytest
from sqlalchemy import text

from config.settings import check_db_connection, engine
from ingestion import espn_props
from ingestion.espn_injuries import PlayerIndex
from ingestion.espn_props import (milestone_line, name_rows, parse_prop_bets,
                                  parse_roster)

FIXTURE = json.loads((Path(__file__).resolve().parent / "fixtures" / "espn_props.json")
                     .read_text(encoding="utf-8"))
DK = FIXTURE["draftkings_401803091"]
EB = FIXTURE["espn_bet_401802497"]
UTC = dt.timezone.utc


def start(group):
    return dt.datetime.fromisoformat(group["event_start"].replace("Z", "+00:00"))


def keyed(rows):
    return {(r["espn_athlete_id"], r["market"], r["line"]): r for r in rows}


def prices(row):
    return (row["over_price"], row["under_price"], row["over_price_open"], row["under_price_open"])


requires_db = pytest.mark.skipif(not check_db_connection(),
                                 reason="database not reachable")


class TestDraftKings:
    """Unlabelled pairs, first entry = over; all lines updated pre-game."""

    @pytest.fixture()
    def rows(self):
        rows, counts = parse_prop_bets(DK["items"], "DraftKings", start(DK))
        assert counts["skipped_market"] == 3        # team total, period total, first goal
        assert not counts["in_play_prices_dropped"]
        return keyed(rows)

    def test_over_under_pairs(self, rows):
        # Zucker, points 0.5: over -120 (open -110), under -110 (open -120)
        assert prices(rows[(2593315, "player_points", 0.5)]) == (-120, -110, -110, -120)
        assert prices(rows[(2593315, "player_shots_on_goal", 1.5)]) == (-168, 130, -154, 120)
        assert prices(rows[(2593315, "player_assists", 0.5)]) == (180, -240, 210, -285)
        assert prices(rows[(4736758, "player_total_saves", 23.5)]) == (-115, -115, -115, -115)

    def test_one_sided_markets_are_overs(self, rows):
        assert prices(rows[(2593315, "player_goal_scorer_anytime", 0.5)]) == (275, None, 270, None)
        assert prices(rows[(2593315, "player_points_alternate", 0.5)]) == (-115, None, -105, None)
        assert prices(rows[(2593315, "player_goals_alternate", 1.5)]) == (2200, None, 2000, None)

    def test_milestone_open_from_another_line_is_dropped(self, rows):
        """ESPN keeps one entry per player, market and side: Zucker's shots
        milestone is '3+' now but its 'open' is the '2+' price (-160)."""
        assert prices(rows[(2593315, "player_shots_on_goal_alternate", 2.5)]) == (170, None, None, None)

    def test_two_milestone_lines_for_one_player(self, rows):
        assert rows[(4565236, "player_shots_on_goal_alternate", 2.5)]["over_price"] == -155
        assert rows[(4565236, "player_shots_on_goal_alternate", 3.5)]["over_price"] == 155

    def test_last_updated_and_book(self, rows):
        row = rows[(2593315, "player_points", 0.5)]
        assert row["last_updated"] == dt.datetime(2026, 1, 15, 19, 56, tzinfo=UTC)
        assert row["book"] == "DraftKings"

    def test_every_row_is_unique(self):
        rows, _ = parse_prop_bets(DK["items"], "DraftKings", start(DK))
        assert len(rows) == len(keyed(rows)) == 13

    def test_after_puck_drop_only_opening_prices_are_kept(self):
        early_start = dt.datetime(2026, 1, 15, 19, 0, tzinfo=UTC)   # before the 19:56 update
        rows, counts = parse_prop_bets(DK["items"], "DraftKings", early_start)
        row = keyed(rows)[(2593315, "player_points", 0.5)]
        assert prices(row) == (None, None, -110, -120)
        assert counts["in_play_prices_dropped"] > 0

    def test_unknown_start_is_treated_as_in_play(self):
        rows, _ = parse_prop_bets(DK["items"], "DraftKings", None)
        assert all(r["over_price"] is None and r["under_price"] is None for r in rows)

    def test_a_lone_unlabelled_entry_is_skipped(self):
        items = [it for it in DK["items"]
                 if it["type"]["name"] == "Total Points" and "2593315" in it["athlete"]["$ref"]][:1]
        rows, counts = parse_prop_bets(items, "DraftKings", start(DK))
        assert rows == [] and counts["skipped_unpaired"] == 1

    def test_unlabelled_pairs_from_other_books_are_skipped(self):
        """The first-is-over order is verified for DraftKings only."""
        rows, counts = parse_prop_bets(DK["items"], "Some Other Book", start(DK))
        markets = {r["market"] for r in rows}
        assert "player_points" not in markets and "player_shots_on_goal" not in markets
        assert "player_goal_scorer_anytime" in markets        # one-sided, no pairing needed
        assert counts["skipped_unlabelled"] == 12

    def test_impossible_pair_is_dropped(self):
        items = copy.deepcopy([it for it in DK["items"] if it["type"]["name"] == "Total Saves"])
        for it in items:
            it["odds"]["american"]["value"] = "+114"
        rows, _ = parse_prop_bets(items, "DraftKings", start(DK))
        assert prices(rows[0]) == (None, None, -115, -115)


class TestEspnBet:
    """Labelled entries in random order; every line updated in play."""

    @pytest.fixture()
    def result(self):
        return parse_prop_bets(EB["items"], "ESPN BET", start(EB))

    def test_labels_decide_the_side_not_the_order(self, result):
        rows = keyed(result[0])
        # listed under first: -650 is the under, +360 the over
        assert prices(rows[(3797, "player_goals", 0.5)]) == (None, None, 360, -650)
        assert prices(rows[(3797, "player_shots_on_goal", 1.5)]) == (None, None, -170, 135)
        assert prices(rows[(3797, "player_assists", 0.5)]) == (None, None, 290, -475)

    def test_relined_in_play_entries_collapse_to_the_opening_line(self, result):
        """Backlund's points: 0.5 at open, re-lined to 1.5 in play
        (+900/-2000). Only the 0.5 opening prices survive."""
        rows = keyed(result[0])
        assert prices(rows[(3797, "player_points", 0.5)]) == (None, None, 150, -190)
        assert not any(k[1] == "player_points" and k[2] == 1.5 for k in rows)
        assert result[1]["conflicting_duplicates"] == 0

    def test_skips_and_drops_are_counted(self, result):
        rows, counts = result
        assert counts["skipped_market"] == 8     # "Hockey Player Prop", first goal, 2+ goals
        assert counts["in_play_prices_dropped"] == 12
        assert len(rows) == 5

    def test_labels_are_read_when_present(self):
        items = copy.deepcopy([it for it in EB["items"] if it["type"]["name"] == "Total Goals"])
        items.reverse()
        rows, _ = parse_prop_bets(items, "ESPN BET", start(EB))
        assert prices(rows[0]) == (None, None, 360, -650)

    def test_labels_missing_on_espn_bet_means_skipped(self):
        items = copy.deepcopy([it for it in EB["items"] if it["type"]["name"] == "Total Goals"])
        for it in items:
            it["current"].pop("over", None), it["current"].pop("under", None)
            it["open"].pop("over", None), it["open"].pop("under", None)
        rows, counts = parse_prop_bets(items, "ESPN BET", start(EB))
        assert rows == [] and counts["skipped_unlabelled"] == 2


class TestHelpers:
    @pytest.mark.parametrize("raw, want", [("3+", 2.5), ("1+", 0.5), (2, 1.5), ("2.0", 1.5),
                                           ("0+", None), ("", None), (None, None), ("x", None)])
    def test_milestone_line(self, raw, want):
        assert milestone_line(raw) == want

    def test_parse_roster(self):
        roster = parse_roster(DK["summary"])
        assert roster[2593315] == {"name": "Jason Zucker", "short_name": "J. Zucker",
                                   "team": "BUF", "position": "L"}
        assert roster[4736758]["position"] == "G"
        assert roster[4565236]["team"] == "MTL"
        assert parse_roster({}) == {}

    def test_name_rows_prefers_the_game_roster(self):
        rows, _ = parse_prop_bets(DK["items"], "DraftKings", start(DK))
        game = PlayerIndex([(8475722, "J. Zucker", "L"), (8481540, "C. Caufield", "R"),
                            (8480045, "C. Ellis", "G")],
                           {8475722: "BUF", 8481540: "MTL", 8480045: "BUF"})
        league = PlayerIndex([(1, "J. Zucker", "L")], {1: "BUF"})   # would give the wrong id
        counts = name_rows(rows, parse_roster(DK["summary"]), game, league)
        by = {r["espn_athlete_id"]: r for r in rows}
        assert by[2593315]["player_id"] == 8475722
        assert by[2593315]["player_name"] == "Jason Zucker"
        assert by[4736758]["player_id"] == 8480045
        assert counts == {"resolved": len(rows)}

    def test_name_rows_falls_back_to_the_league(self):
        rows, _ = parse_prop_bets(DK["items"], "DraftKings", start(DK))
        league = PlayerIndex([(1, "J. Zucker", "L")], {1: "BUF"})
        counts = name_rows(rows, parse_roster(DK["summary"]), PlayerIndex([]), league)
        assert {r["player_id"] for r in rows if r["espn_athlete_id"] == 2593315} == {1}
        assert counts["unresolved"] == sum(1 for r in rows if r["espn_athlete_id"] != 2593315)


class TestCommandLine:
    def test_help_runs_nothing(self, monkeypatch, capsys):
        monkeypatch.setattr(espn_props, "backfill_props",
                            lambda *a, **k: pytest.fail("--help ran the backfill"))
        with pytest.raises(SystemExit) as exc:
            espn_props.main(["--help"])
        assert exc.value.code == 0
        assert "--season" in capsys.readouterr().out

    @pytest.mark.parametrize("argv", [[], ["--season", "20252026", "--limit", "0"]])
    def test_bad_arguments_exit(self, monkeypatch, argv):
        monkeypatch.setattr(espn_props, "backfill_props",
                            lambda *a, **k: pytest.fail("ran with bad arguments"))
        with pytest.raises(SystemExit) as exc:
            espn_props.main(argv)
        assert exc.value.code == 2

    def test_arguments_reach_the_backfill(self, monkeypatch):
        calls = []
        monkeypatch.setattr(espn_props, "backfill_props",
                            lambda season, limit=None, refresh=False:
                            calls.append((season, limit, refresh)) or 0)
        espn_props.main(["--season", "20252026"])
        espn_props.main(["--season", "20252026", "--limit", "20", "--refresh"])
        assert calls == [(20252026, None, False), (20252026, 20, True)]


# ── Database (clone or _test copy only) ────────────────────────────

GAME_ID = 9999020302


@requires_db
class TestBackfillOnDatabase:
    @pytest.fixture()
    def game(self, monkeypatch):
        espn_props.ensure_tables()
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO raw.games (game_id, season, game_type, date, home_team,
                                       away_team, home_score, away_score, game_state)
                VALUES (:g, 20302031, 2, '2031-01-15', 'BUF', 'MTL', 3, 2, 'OFF')
            """), {"g": GAME_ID})
        calls = {"props": 0}

        def props(event_id, provider_id):
            calls["props"] += 1
            return DK["items"] if provider_id == "100" else []

        monkeypatch.setattr(espn_props, "REQUEST_PAUSE_S", 0)
        monkeypatch.setattr(espn_props, "fetch_scoreboard", lambda d: [{
            "id": "401999998", "date": "2026-01-16T00:00Z",
            "competitions": [{"competitors": [
                {"homeAway": "home", "team": {"displayName": "Buffalo Sabres"}},
                {"homeAway": "away", "team": {"displayName": "Montreal Canadiens"}}]}]}])
        monkeypatch.setattr(espn_props, "fetch_books",
                            lambda e: [("100", "DraftKings"), ("58", "ESPN BET")])
        monkeypatch.setattr(espn_props, "fetch_prop_bets", props)
        monkeypatch.setattr(espn_props, "fetch_summary", lambda e: DK["summary"])
        monkeypatch.setattr(espn_props, "fetch_athlete",
                            lambda a: pytest.fail("every athlete is in the boxscore"))
        yield calls
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM raw.prop_odds_hist WHERE game_id = :g"), {"g": GAME_ID})
            conn.execute(text("DELETE FROM raw.prop_odds_fetches WHERE game_id = :g"), {"g": GAME_ID})
            conn.execute(text("DELETE FROM raw.games WHERE game_id = :g"), {"g": GAME_ID})

    def _rows(self):
        with engine.connect() as conn:
            return conn.execute(text("SELECT * FROM raw.prop_odds_hist WHERE game_id = :g"),
                                {"g": GAME_ID}).mappings().all()

    def _log(self):
        with engine.connect() as conn:
            return conn.execute(text("SELECT * FROM raw.prop_odds_fetches WHERE game_id = :g"),
                                {"g": GAME_ID}).mappings().one_or_none()

    def test_backfill_writes_rows_and_is_resumable(self, game):
        assert espn_props.backfill_props(20302031, game_ids=[GAME_ID]) == 13
        rows = {(r["espn_athlete_id"], r["market"], float(r["line"])): r for r in self._rows()}
        zp = rows[(2593315, "player_points", 0.5)]
        assert (zp["over_price"], zp["under_price"]) == (-120, -110)
        assert zp["player_name"] == "Jason Zucker"
        assert zp["book"] == "DraftKings" and zp["espn_event_id"] == "401999998"
        assert zp["event_start"] == dt.datetime(2026, 1, 16, 0, 0, tzinfo=UTC)
        log = self._log()
        assert (log["n_rows"], log["books"], log["espn_event_id"]) == (13, "DraftKings", "401999998")
        # a second run skips the game: no request at all
        assert espn_props.backfill_props(20302031, game_ids=[GAME_ID]) == 0
        assert game["props"] == 2

    def test_refresh_refetches_and_keeps_rows_when_espn_has_none(self, game, monkeypatch):
        espn_props.backfill_props(20302031, game_ids=[GAME_ID])
        assert espn_props.backfill_props(20302031, game_ids=[GAME_ID], refresh=True) == 13
        assert len(self._rows()) == 13
        monkeypatch.setattr(espn_props, "fetch_prop_bets", lambda e, p: [])
        assert espn_props.backfill_props(20302031, game_ids=[GAME_ID], refresh=True) == 0
        assert len(self._rows()) == 13 and self._log()["n_rows"] == 13

    def test_failed_download_is_not_recorded(self, game, monkeypatch):
        def boom(event_id):
            raise ConnectionError("network down")
        monkeypatch.setattr(espn_props, "fetch_books", boom)
        assert espn_props.backfill_props(20302031, game_ids=[GAME_ID]) == 0
        assert self._log() is None and self._rows() == []

    def test_game_without_props_is_recorded_as_zero(self, game, monkeypatch):
        monkeypatch.setattr(espn_props, "fetch_prop_bets", lambda e, p: [])
        assert espn_props.backfill_props(20302031, game_ids=[GAME_ID]) == 0
        assert self._log()["n_rows"] == 0
