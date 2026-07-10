"""
Tests for features/team_features.py (Phase 2, Task 2).

Unit tests exercise compute_rolling() on synthetic frames — including the
critical no-leakage property. The integration test hand-computes a rolling
GF/60 from raw tables with independent SQL and compares it to the built row.
"""
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from features.team_features import build_team_rolling, compute_rolling
from features.util import WINDOWS


# ─────────────────────────────────────────────
# Synthetic-frame helpers (unit tests, no DB)
# ─────────────────────────────────────────────
def make_base(rows):
    """
    Build a minimal base frame. Each row: (team, season, date, game_id, gf, ga).
    All other stat columns get simple deterministic values so ratios are
    computable but the tests only assert on the goal-based features.
    """
    df = pd.DataFrame(rows, columns=["team", "season", "date", "game_id", "gf", "ga"])
    df["date"] = pd.to_datetime(df["date"])
    df["is_home"] = True
    df["sog_for"] = 30
    df["sog_against"] = 28
    df["fow"] = 25
    df["fo_total"] = 50
    df["pp_goals"] = 1
    df["pp_opps"] = 3
    df["ppga"] = 1
    df["opp_pp_opps"] = 4
    df["pim"] = 8
    df["cf"] = 55
    df["ca"] = 50
    df["ff"] = 42
    df["fa"] = 40
    df["xgf"] = 2.5
    df["xga"] = 2.2
    df["pp_xgf"] = 0.6
    df["pk_xga"] = 0.5
    df["toi_sec"] = 3600
    df["pp_toi"] = 300.0
    df["sh_toi"] = 280.0
    return df


def one_team_six_games():
    return make_base([
        ("TST", 20202021, "2021-01-01", 1, 2, 1),
        ("TST", 20202021, "2021-01-03", 2, 4, 2),
        ("TST", 20202021, "2021-01-05", 3, 1, 3),
        ("TST", 20202021, "2021-01-07", 4, 5, 0),
        ("TST", 20202021, "2021-01-09", 5, 3, 3),
        ("TST", 20202021, "2021-01-11", 6, 0, 2),
    ])


def get_row(feats, game_id, window):
    rows = feats[(feats["game_id"] == game_id) & (feats["window_size"] == window)]
    assert len(rows) == 1
    return rows.iloc[0]


class TestComputeRolling:
    def test_hand_computed_gf_per60(self):
        feats = compute_rolling(one_team_six_games())
        # Game 6, window 5: prior games 1-5, ΣGF = 2+4+1+5+3 = 15 over 5*3600s
        row = get_row(feats, 6, 5)
        assert row["games_played"] == 5
        assert row["gf_per60"] == pytest.approx(15 * 3600 / (5 * 3600))  # 3.0

    def test_window_smaller_than_history(self):
        feats = compute_rolling(one_team_six_games())
        # Game 6, window 82: all 5 prior games available, games_played = 5
        row = get_row(feats, 6, 82)
        assert row["games_played"] == 5
        # Game 4, window 2: only games 2-3 in window -> ΣGF = 4+1
        row = get_row(compute_rolling(one_team_six_games(), windows=[2]), 4, 2)
        assert row["games_played"] == 2
        assert row["gf_per60"] == pytest.approx(5 * 3600 / (2 * 3600))

    def test_first_game_has_no_features(self):
        feats = compute_rolling(one_team_six_games())
        for w in WINDOWS:
            row = get_row(feats, 1, w)
            assert row["games_played"] == 0
            assert np.isnan(row["gf_per60"])
            assert np.isnan(row["pdo"])

    def test_no_leakage_from_target_game(self):
        """The critical test: a game's own stats must not affect its features."""
        base = one_team_six_games()
        feats_before = compute_rolling(base)

        poisoned = base.copy()
        poisoned.loc[poisoned["game_id"] == 4, ["gf", "ga", "xgf", "sog_for"]] = 999
        feats_after = compute_rolling(poisoned)

        for w in WINDOWS:
            before = get_row(feats_before, 4, w)
            after = get_row(feats_after, 4, w)
            for col in ("gf_per60", "ga_per60", "sh_pct", "xgf_pct", "pdo"):
                b, a = before[col], after[col]
                assert (np.isnan(b) and np.isnan(a)) or b == pytest.approx(a), \
                    f"game 4's own stats leaked into its window-{w} {col}"

    def test_no_leakage_from_future_games(self):
        base = one_team_six_games()
        feats_before = compute_rolling(base)

        poisoned = base.copy()
        poisoned.loc[poisoned["game_id"].isin([5, 6]), ["gf", "ga"]] = 999
        feats_after = compute_rolling(poisoned)

        for w in WINDOWS:
            b = get_row(feats_before, 4, w)
            a = get_row(feats_after, 4, w)
            assert b["gf_per60"] == pytest.approx(a["gf_per60"]), \
                f"future games leaked into window-{w} features"

    def test_windows_do_not_cross_seasons(self):
        base = make_base([
            ("TST", 20202021, "2021-01-01", 1, 2, 1),
            ("TST", 20202021, "2021-01-03", 2, 3, 2),
            ("TST", 20212022, "2021-10-15", 3, 4, 0),  # next season opener
        ])
        feats = compute_rolling(base, windows=[5])
        row = get_row(feats, 3, 5)
        assert row["games_played"] == 0
        assert np.isnan(row["gf_per60"])

    def test_two_teams_are_independent(self):
        base = pd.concat([
            one_team_six_games(),
            make_base([
                ("OTH", 20202021, "2021-01-02", 101, 1, 0),
                ("OTH", 20202021, "2021-01-04", 102, 2, 5),
            ]),
        ], ignore_index=True)
        feats = compute_rolling(base, windows=[5])
        row = get_row(feats, 102, 5)
        assert row["games_played"] == 1
        assert row["gf_per60"] == pytest.approx(1.0)  # only OTH game 101


# ─────────────────────────────────────────────
# Integration (requires populated DB)
# ─────────────────────────────────────────────
SEASON = 20202021


@pytest.fixture(scope="module")
def db():
    from config.settings import engine
    try:
        with engine.connect() as conn:
            n = conn.execute(text(
                "SELECT COUNT(*) FROM raw.games WHERE season = :s AND game_state IN ('FINAL','OFF')"
            ), {"s": SEASON}).scalar()
    except Exception as e:
        pytest.skip(f"database unavailable: {e}")
    if n < 800:
        pytest.skip(f"season {SEASON} not fully backfilled ({n} games)")
    return engine


@pytest.fixture(scope="module")
def built(db):
    build_team_rolling(SEASON)
    return db


class TestIntegration:
    def test_hand_computed_rolling_gf60_matches(self, built):
        """
        Independent recomputation: COL's 30th game of 2020-21, window 10.
        Expected GF/60 = 3600 * (GF over prior games 20-29) / (goalie TOI sum
        over those games), computed here with SQL that never touches the
        features schema.
        """
        with built.connect() as conn:
            games = conn.execute(text("""
                SELECT g.game_id,
                       CASE WHEN g.home_team = 'COL' THEN g.home_score ELSE g.away_score END AS gf,
                       (SELECT SUM(gg.toi_seconds) FROM raw.goalie_games gg
                        WHERE gg.game_id = g.game_id AND gg.team = 'COL') AS toi
                FROM raw.games g
                WHERE g.season = :s AND g.game_state IN ('FINAL','OFF')
                  AND 'COL' IN (g.home_team, g.away_team)
                ORDER BY g.date, g.game_id
            """), {"s": SEASON}).fetchall()

            assert len(games) >= 30
            target = games[29]  # 30th game
            window = games[19:29]  # prior 10 games
            expected = 3600.0 * sum(g.gf for g in window) / float(sum(g.toi for g in window))

            actual = conn.execute(text("""
                SELECT gf_per60, games_played FROM features.team_rolling
                WHERE game_id = :gid AND team = 'COL' AND window_size = 10
            """), {"gid": target.game_id}).one()

        assert actual.games_played == 10
        assert float(actual.gf_per60) == pytest.approx(expected, abs=0.002)

    def test_season_openers_have_zero_games_played(self, built):
        with built.connect() as conn:
            bad = conn.execute(text("""
                WITH openers AS (
                    SELECT DISTINCT ON (t.team) t.team, t.game_id
                    FROM (
                        SELECT home_team AS team, game_id, date FROM raw.games WHERE season = :s
                        UNION ALL
                        SELECT away_team, game_id, date FROM raw.games WHERE season = :s
                    ) t ORDER BY t.team, t.date, t.game_id
                )
                SELECT COUNT(*) FROM features.team_rolling tr
                JOIN openers o ON o.game_id = tr.game_id AND o.team = tr.team
                WHERE tr.games_played <> 0 OR tr.gf_per60 IS NOT NULL
            """), {"s": SEASON}).scalar()
        assert bad == 0

    def test_row_counts(self, built):
        """Every completed game × 2 teams × len(WINDOWS) rows."""
        with built.connect() as conn:
            n_games = conn.execute(text(
                "SELECT COUNT(*) FROM raw.games WHERE season = :s AND game_state IN ('FINAL','OFF')"
            ), {"s": SEASON}).scalar()
            n_rows = conn.execute(text("""
                SELECT COUNT(*) FROM features.team_rolling tr
                JOIN raw.games g ON g.game_id = tr.game_id WHERE g.season = :s
            """), {"s": SEASON}).scalar()
        assert n_rows == n_games * 2 * len(WINDOWS)
