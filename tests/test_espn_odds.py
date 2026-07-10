"""
Tests for ingestion/espn_odds.py — pure parsing/matching logic only
(no network); the backfill itself is resumable and audited separately.
"""
import pytest

from ingestion.espn_odds import (
    _espn_name_to_abbrev, _match_events_to_games, parse_pickcenter,
)


def make_event(event_id, home_name, away_name):
    return {
        "id": event_id,
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


class TestParsePickcenter:
    def test_extracts_all_columns(self):
        block = {
            "provider": {"name": "DraftKings"},
            "details": "MTL -125", "overUnder": 6.5, "spread": -1.5,
            "homeTeamOdds": {"moneyLine": 105, "favorite": False},
            "awayTeamOdds": {"moneyLine": -125, "favorite": True},
        }
        row = parse_pickcenter(block)
        assert row == {"provider": "DraftKings", "home_ml": 105,
                       "away_ml": -125, "spread": -1.5, "over_under": 6.5,
                       "details": "MTL -125"}

    def test_missing_fields_become_none(self):
        row = parse_pickcenter({"homeTeamOdds": None, "awayTeamOdds": None})
        assert all(v is None for v in row.values())
