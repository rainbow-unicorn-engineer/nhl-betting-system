"""
Tests for ingestion/dailyfaceoff.py — parser on a synthetic __NEXT_DATA__
fixture (no network), name resolution against real raw.players, and the
recommend-job integration (confirmed starter beats the heuristic).
"""
import datetime as dt
import json

import pytest
from sqlalchemy import text

from config.settings import check_db_connection, engine
from ingestion.dailyfaceoff import (_normalize, parse_starting_goalies,
                                    resolve_goalie_ids)

requires_db = pytest.mark.skipif(not check_db_connection(),
                                 reason="database not reachable")


def _fixture(games: list) -> str:
    payload = {"props": {"pageProps": {"data": games}}}
    return ('<html><script id="__NEXT_DATA__" type="application/json">'
            + json.dumps(payload) + "</script></html>")


GAME = {
    "date": "2026-01-15",
    "homeTeamName": "Buffalo Sabres", "awayTeamName": "Montreal Canadiens",
    "homeGoalieName": "Colten Ellis", "awayGoalieName": "Jacob Fowler",
    "homeNewsStrengthName": "Confirmed", "awayNewsStrengthName": None,
}


class TestParser:
    def test_parses_both_sides(self):
        rows = parse_starting_goalies(_fixture([GAME]))
        assert len(rows) == 2
        home = next(r for r in rows if r["team"] == "BUF")
        away = next(r for r in rows if r["team"] == "MTL")
        assert home["goalie_name"] == "Colten Ellis"
        assert home["confirmation"] == "Confirmed"
        assert away["confirmation"] is None
        assert home["game_date"] == "2026-01-15"

    def test_unknown_team_skipped_not_fatal(self):
        bad = dict(GAME, homeTeamName="Quebec Nordiques")
        rows = parse_starting_goalies(_fixture([bad]))
        assert [r["team"] for r in rows] == ["MTL"]

    def test_missing_goalie_skipped(self):
        bad = dict(GAME, awayGoalieName=None)
        rows = parse_starting_goalies(_fixture([bad]))
        assert [r["team"] for r in rows] == ["BUF"]

    def test_layout_change_is_loud(self):
        with pytest.raises(ValueError, match="__NEXT_DATA__"):
            parse_starting_goalies("<html>redesigned</html>")

    def test_normalize_strips_accents_and_case(self):
        assert _normalize("Štěpán  Lukeš ") == _normalize("stepan  lukes")


@requires_db
class TestResolution:
    def test_resolves_df_style_full_name(self):
        """raw.players stores 'J. Swayman'; Daily Faceoff sends
        'Jeremy Swayman'. Build a DF-style name from a real DB row
        (same initial + surname) and require it to resolve."""
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT p.player_id, p.full_name, gg.team
                FROM raw.players p
                JOIN raw.goalie_games gg USING (player_id)
                WHERE p.position = 'G' AND p.full_name LIKE '_. %' LIMIT 1
            """)).fetchone()
        initial, surname = row.full_name.split(". ", 1)
        df_style = f"{initial}ohnfake {surname}"
        rows = resolve_goalie_ids([{"game_date": "2026-01-15",
                                    "team": row.team,
                                    "goalie_name": df_style,
                                    "confirmation": "Confirmed"}])
        assert rows[0]["goalie_id"] == row.player_id

    def test_resolves_exact_db_name_too(self):
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT p.player_id, p.full_name, gg.team
                FROM raw.players p
                JOIN raw.goalie_games gg USING (player_id)
                WHERE p.position = 'G' LIMIT 1
            """)).fetchone()
        rows = resolve_goalie_ids([{"game_date": "2026-01-15",
                                    "team": row.team,
                                    "goalie_name": row.full_name,
                                    "confirmation": "Confirmed"}])
        assert rows[0]["goalie_id"] == row.player_id

    def test_unknown_name_stored_unresolved(self):
        rows = resolve_goalie_ids([{"game_date": "2026-01-15",
                                    "team": "BUF",
                                    "goalie_name": "Nonexistent Goalie",
                                    "confirmation": "Confirmed"}])
        assert rows[0]["goalie_id"] is None


@requires_db
class TestRecommendIntegration:
    SIM_DATE = dt.date(2026, 1, 15)

    @pytest.fixture()
    def confirmed_starter(self):
        """Plant a DF-confirmed starter who is NOT the heuristic pick for
        some team on the sim slate."""
        from betting.recommend import load_slate, project_starters
        from ingestion.dailyfaceoff import ensure_table
        ensure_table()
        slate = load_slate(self.SIM_DATE, simulate=True)
        baseline = project_starters(slate, 20252026, self.SIM_DATE)
        team = baseline.iloc[0]["team"]
        heuristic_pick = int(baseline.iloc[0]["goalie_id"])
        with engine.connect() as conn:
            other = conn.execute(text("""
                SELECT DISTINCT gg.player_id
                FROM raw.goalie_games gg JOIN raw.games g USING (game_id)
                WHERE gg.team = :t AND g.season = 20252026
                  AND gg.player_id <> :pick LIMIT 1
            """), {"t": team, "pick": heuristic_pick}).scalar()
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO raw.starting_goalies
                    (game_date, team, goalie_name, goalie_id, confirmation)
                VALUES (:d, :t, 'Test Starter', :gid, 'Confirmed')
                ON CONFLICT (game_date, team) DO UPDATE
                    SET goalie_id = :gid, confirmation = 'Confirmed'
            """), {"d": self.SIM_DATE, "t": team, "gid": other})
        yield {"slate": slate, "team": team, "df_pick": other,
               "heuristic_pick": heuristic_pick}
        with engine.begin() as conn:
            conn.execute(text("""
                DELETE FROM raw.starting_goalies
                WHERE game_date = :d AND team = :t
            """), {"d": self.SIM_DATE, "t": team})

    def test_confirmed_starter_overrides_heuristic(self, confirmed_starter):
        from betting.recommend import project_starters
        s = confirmed_starter
        starters = project_starters(s["slate"], 20252026, self.SIM_DATE)
        row = starters[starters["team"] == s["team"]].iloc[0]
        assert int(row["goalie_id"]) == s["df_pick"]
        assert int(row["starter_fallback"]) == 0    # confirmed
        # teams with NO Daily Faceoff row keep the heuristic, fallback=1
        with engine.connect() as conn:
            df_teams = {r[0] for r in conn.execute(text("""
                SELECT team FROM raw.starting_goalies
                WHERE game_date = :d AND goalie_id IS NOT NULL
            """), {"d": self.SIM_DATE})}
        untouched = starters[~starters["team"].isin(df_teams)]
        assert (untouched["starter_fallback"] == 1).all()
