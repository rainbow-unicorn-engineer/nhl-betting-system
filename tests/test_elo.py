"""
Tests for features/elo.py (Phase 2, Task 5).

Unit tests drive compute_elo() on synthetic schedules: winners gain, stored
ratings are PRE-game (the plan-mandated leakage property), home advantage
asymmetry, season regression, and franchise continuation. Integration tests
verify full-history properties (zero-sum conservation, sane spread) and that
features.matchup rows carry ratings.
"""
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from features.elo import (
    FRANCHISE_CONTINUATIONS, HOME_ADVANTAGE, INITIAL_RATING, K,
    build_elo, compute_elo, expected_home_score,
)


def make_games(rows, season=20202021):
    """Each row: (game_id, date, home, away, home_score, away_score[, season])."""
    df = pd.DataFrame(
        [r if len(r) == 7 else r + (season,) for r in rows],
        columns=["game_id", "date", "home_team", "away_team",
                 "home_score", "away_score", "season"])
    df["date"] = pd.to_datetime(df["date"])
    df["game_state"] = "FINAL"
    return df


def elo_of(df, game_id):
    row = df[df["game_id"] == game_id]
    assert len(row) == 1
    return row.iloc[0]


class TestExpectedScore:
    def test_equal_ratings_home_favored(self):
        e = expected_home_score(1500, 1500)
        # 50 points of home ice ~ 57.1% expected score
        assert e == pytest.approx(1 / (1 + 10 ** (-50 / 400)))
        assert e > 0.5

    def test_symmetry(self):
        assert expected_home_score(1600, 1500) > expected_home_score(1500, 1600)


class TestComputeElo:
    def test_ratings_are_pregame_not_postgame(self):
        """The plan-mandated test: game 1's stored ratings must be the
        initial 1500s even though the result is known; the update shows up
        in game 2."""
        games = make_games([
            (1, "2021-01-01", "AAA", "BBB", 5, 2),   # AAA wins at home
            (2, "2021-01-03", "BBB", "AAA", 3, 1),
        ])
        elos, _ = compute_elo(games)
        g1 = elo_of(elos, 1)
        assert g1["home_elo"] == INITIAL_RATING
        assert g1["away_elo"] == INITIAL_RATING
        g2 = elo_of(elos, 2)
        assert g2["away_elo"] > INITIAL_RATING   # AAA gained
        assert g2["home_elo"] < INITIAL_RATING   # BBB lost

    def test_result_does_not_affect_own_pregame_rating(self):
        """Flipping a game's result must not change its own stored ratings."""
        base = make_games([
            (1, "2021-01-01", "AAA", "BBB", 5, 2),
            (2, "2021-01-03", "BBB", "AAA", 3, 1),
        ])
        flipped = base.copy()
        flipped.loc[flipped["game_id"] == 2, ["home_score", "away_score"]] = [1, 3]
        e1, _ = compute_elo(base)
        e2, _ = compute_elo(flipped)
        pd.testing.assert_series_equal(elo_of(e1, 2), elo_of(e2, 2))

    def test_winner_gains_loser_drops_zero_sum(self):
        games = make_games([(1, "2021-01-01", "AAA", "BBB", 4, 1)])
        _, final = compute_elo(games)
        assert final["AAA"] > INITIAL_RATING > final["BBB"]
        assert final["AAA"] + final["BBB"] == pytest.approx(2 * INITIAL_RATING)

    def test_home_advantage_asymmetry(self):
        """At equal ratings a home win earns less than an away win."""
        home_win = make_games([(1, "2021-01-01", "AAA", "BBB", 3, 2)])
        _, f_home = compute_elo(home_win)
        away_win = make_games([(1, "2021-01-01", "BBB", "AAA", 2, 3)])
        _, f_away = compute_elo(away_win)
        gain_at_home = f_home["AAA"] - INITIAL_RATING
        gain_on_road = f_away["AAA"] - INITIAL_RATING
        assert 0 < gain_at_home < gain_on_road < K

    def test_ot_win_counts_full(self):
        reg = make_games([(1, "2021-01-01", "AAA", "BBB", 4, 1)])
        ot = make_games([(1, "2021-01-01", "AAA", "BBB", 2, 1)])
        _, f_reg = compute_elo(reg)
        _, f_ot = compute_elo(ot)
        assert f_reg["AAA"] == pytest.approx(f_ot["AAA"])  # no margin scaling

    def test_season_regression_toward_mean(self):
        games = make_games([
            (1, "2021-01-01", "AAA", "BBB", 5, 0, 20202021),
            (2, "2021-10-15", "AAA", "BBB", 3, 2, 20212022),
        ])
        elos, _ = compute_elo(games)
        post_g1_home = INITIAL_RATING + K * (1 - expected_home_score(1500, 1500))
        expected_regressed = post_g1_home + (INITIAL_RATING - post_g1_home) / 3
        assert elo_of(elos, 2)["home_elo"] == pytest.approx(expected_regressed, abs=0.05)

    def test_scheduled_games_get_ratings_but_no_update(self):
        games = make_games([
            (1, "2021-01-01", "AAA", "BBB", 5, 2),
            (2, "2021-01-03", "AAA", "BBB", None, None),
            (3, "2021-01-05", "AAA", "BBB", 1, 0),
        ])
        games.loc[games["game_id"] == 2, "game_state"] = "SCHEDULED"
        elos, _ = compute_elo(games)
        # game 2 carries game 1's update; game 3 identical (no update from 2)
        assert elo_of(elos, 2)["home_elo"] > INITIAL_RATING
        pd.testing.assert_series_equal(
            elo_of(elos, 2)[["home_elo", "away_elo"]],
            elo_of(elos, 3)[["home_elo", "away_elo"]],
            check_names=False)

    def test_franchise_continuation(self):
        assert FRANCHISE_CONTINUATIONS.get("UTA") == "ARI"
        games = make_games([
            (1, "2024-01-01", "ARI", "BBB", 0, 5, 20232024),   # ARI loses
            (2, "2024-10-15", "UTA", "BBB", 2, 1, 20242025),   # UTA debuts
        ])
        elos, final = compute_elo(games)
        uta_pre = elo_of(elos, 2)["home_elo"]
        # inherited ARI's depressed rating, regressed 1/3 toward 1500
        assert uta_pre < INITIAL_RATING
        ari_post = INITIAL_RATING - K * expected_home_score(1500, 1500)
        assert uta_pre == pytest.approx(ari_post + (INITIAL_RATING - ari_post) / 3, abs=0.05)
        assert "ARI" not in final


# ─────────────────────────────────────────────
# Integration (requires populated DB)
# ─────────────────────────────────────────────
@pytest.fixture(scope="module")
def db():
    from config.settings import engine
    try:
        with engine.connect() as conn:
            n = conn.execute(text(
                "SELECT COUNT(*) FROM raw.games WHERE game_state IN ('FINAL','OFF')"
            )).scalar()
    except Exception as e:
        pytest.skip(f"database unavailable: {e}")
    if n < 800:
        pytest.skip(f"raw.games too sparse ({n})")
    return engine


@pytest.fixture(scope="module")
def built(db):
    build_elo()
    return db


class TestIntegration:
    def test_all_games_rated(self, built):
        with built.connect() as conn:
            missing = conn.execute(text("""
                SELECT COUNT(*) FROM raw.games g
                LEFT JOIN features.matchup m USING (game_id)
                WHERE m.home_elo IS NULL OR m.away_elo IS NULL
            """)).scalar()
        assert missing == 0

    def test_openers_of_first_season_are_1500(self, built):
        with built.connect() as conn:
            bad = conn.execute(text("""
                WITH first_season AS (SELECT MIN(season) s FROM raw.games),
                openers AS (
                    SELECT DISTINCT ON (t.team) t.game_id, t.team, t.is_home
                    FROM (
                        SELECT game_id, date, home_team AS team, TRUE AS is_home, season
                        FROM raw.games
                        UNION ALL
                        SELECT game_id, date, away_team, FALSE, season FROM raw.games
                    ) t
                    WHERE t.season = (SELECT s FROM first_season)
                    ORDER BY t.team, t.date, t.game_id
                )
                SELECT COUNT(*) FROM openers o
                JOIN features.matchup m USING (game_id)
                WHERE (o.is_home AND m.home_elo <> 1500.0)
                   OR (NOT o.is_home AND m.away_elo <> 1500.0)
            """)).scalar()
        assert bad == 0

    def test_ratings_plausible_spread(self, built):
        """Elo is zero-sum around 1500. Bounds anchored to the observed
        historical extremes: the 2023-24 Sharks (worst cap-era team) bottom
        out ~1274 and the record-setting 2022-23 Bruins (65-12-5) peak
        ~1720. Values outside [1250, 1750] would signal a computation bug."""
        with built.connect() as conn:
            lo, hi, avg = conn.execute(text("""
                SELECT MIN(home_elo), MAX(home_elo), AVG(home_elo)
                FROM features.matchup
            """)).one()
        assert 1250 < float(lo) and float(hi) < 1750
        assert float(avg) == pytest.approx(1500, abs=15)
