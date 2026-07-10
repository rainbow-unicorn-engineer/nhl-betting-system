"""
Tests for features/goalie_features.py (Phase 2, Task 3).

Unit tests exercise compute_goalie_rolling() on synthetic frames — no-leakage,
window behavior, and the Buhlmann shrinkage contract (1 start shrinks to the
league mean; a workhorse retains materially more of their raw number). The
integration test hand-computes a rolling SV% with independent SQL.
"""
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from features.goalie_features import (
    build_goalie_rolling, compute_goalie_rolling, estimate_k,
)
from features.util import WINDOWS


# ─────────────────────────────────────────────
# Synthetic-frame helpers (unit tests, no DB)
# ─────────────────────────────────────────────
def make_base(rows):
    """Each row: (goalie_id, season, date, game_id, saves, shots_against)."""
    df = pd.DataFrame(rows, columns=["goalie_id", "season", "date", "game_id",
                                     "saves", "shots_against"])
    df["date"] = pd.to_datetime(df["date"])
    df["is_starter"] = True
    df["goals_against"] = df["shots_against"] - df["saves"]
    df["toi_seconds"] = 3600
    df["fen_att"] = df["shots_against"] + 10   # misses on top of SOG
    df["fen_goals"] = df["goals_against"]
    df["xga_shots"] = 2.5
    df["hd_att"] = 8
    df["hd_goals"] = df["goals_against"].clip(upper=8)
    df["gsax"] = df["xga_shots"] - df["fen_goals"]
    return df


def five_starts():
    # SV%: .900, .933, .867, .967, .900 — 30 shots each
    return make_base([
        (900, 20202021, "2021-01-01", 1, 27, 30),
        (900, 20202021, "2021-01-03", 2, 28, 30),
        (900, 20202021, "2021-01-05", 3, 26, 30),
        (900, 20202021, "2021-01-07", 4, 29, 30),
        (900, 20202021, "2021-01-09", 5, 27, 30),
    ])


def get_row(feats, game_id, window, goalie_id=900):
    rows = feats[(feats["game_id"] == game_id)
                 & (feats["window_size"] == window)
                 & (feats["goalie_id"] == goalie_id)]
    assert len(rows) == 1
    return rows.iloc[0]


class TestComputeGoalieRolling:
    def test_hand_computed_rolling_sv(self):
        feats = compute_goalie_rolling(five_starts())
        # Game 5, window 5: prior starts 1-4, SV% = (27+28+26+29)/120
        row = get_row(feats, 5, 5)
        assert row["starts_in_window"] == 4
        assert row["sv_pct"] == pytest.approx(110 / 120)

    def test_first_appearance_is_pure_prior(self):
        league = 0.910
        feats = compute_goalie_rolling(five_starts(), league_sv=league)
        for w in WINDOWS:
            row = get_row(feats, 1, w)
            assert row["starts_in_window"] == 0
            assert row["credibility_z"] == 0.0
            assert np.isnan(row["sv_pct"])
            assert row["shrunk_sv_pct"] == pytest.approx(league)  # never NaN

    def test_no_leakage_from_target_game(self):
        base = five_starts()
        before = compute_goalie_rolling(base)

        poisoned = base.copy()
        poisoned.loc[poisoned["game_id"] == 3,
                     ["saves", "shots_against", "gsax"]] = [0, 30, -5.0]
        after = compute_goalie_rolling(poisoned)

        for w in WINDOWS:
            b, a = get_row(before, 3, w), get_row(after, 3, w)
            for col in ("sv_pct", "gsax_per60", "shrunk_sv_pct", "credibility_z"):
                bv, av = b[col], a[col]
                assert (np.isnan(bv) and np.isnan(av)) or bv == pytest.approx(av), \
                    f"game 3's own stats leaked into its window-{w} {col}"

    def test_shrinkage_contract(self):
        """1 start -> essentially league average; 40 starts -> materially
        closer to the raw number (Z = 40/(40+k) vs 1/(1+k))."""
        k, league = 57.6, 0.905
        # A .930 goalie: same per-start line repeated 41 times
        rows = [(77, 20202021, f"2021-01-{d:02d}" if d <= 28 else f"2021-02-{d-28:02d}",
                 d, 27.9, 30) for d in range(1, 42)]
        base = make_base(rows)
        base["saves"] = 27.9  # keep exact .930 per start
        feats = compute_goalie_rolling(base, windows=[82], k=k, league_sv=league)

        after_1 = get_row(feats, 2, 82, goalie_id=77)   # 1 prior start
        after_40 = get_row(feats, 41, 82, goalie_id=77)  # 40 prior starts
        raw = 0.930

        z1, z40 = 1 / (1 + k), 40 / (40 + k)
        assert after_1["credibility_z"] == pytest.approx(z1)
        assert after_40["credibility_z"] == pytest.approx(z40)
        # 1 start: within a hair of league average
        assert abs(after_1["shrunk_sv_pct"] - league) < 0.001
        # 40 starts: recovered >40% of the raw-vs-league gap, and is far
        # closer to raw than the 1-start estimate is
        assert after_40["shrunk_sv_pct"] == pytest.approx(league + z40 * (raw - league))
        assert abs(after_40["shrunk_sv_pct"] - raw) < abs(after_1["shrunk_sv_pct"] - raw)

    def test_windows_do_not_cross_seasons(self):
        base = make_base([
            (900, 20202021, "2021-01-01", 1, 27, 30),
            (900, 20202021, "2021-01-03", 2, 28, 30),
            (900, 20212022, "2021-10-15", 3, 26, 30),
        ])
        feats = compute_goalie_rolling(base, windows=[10])
        row = get_row(feats, 3, 10)
        assert row["starts_in_window"] == 0
        assert np.isnan(row["sv_pct"])


# ─────────────────────────────────────────────
# Integration (requires populated DB)
# ─────────────────────────────────────────────
SEASON = 20202021


@pytest.fixture(scope="module")
def db():
    from config.settings import engine
    try:
        with engine.connect() as conn:
            n = conn.execute(text("""
                SELECT COUNT(*) FROM raw.goalie_games gg
                JOIN raw.games g USING (game_id)
                WHERE g.season = :s AND gg.toi_seconds > 0
            """), {"s": SEASON}).scalar()
    except Exception as e:
        pytest.skip(f"database unavailable: {e}")
    if n < 1500:
        pytest.skip(f"season {SEASON} goalie logs not fully backfilled ({n})")
    return engine


@pytest.fixture(scope="module")
def built(db):
    build_goalie_rolling(SEASON)
    return db


class TestIntegration:
    def test_hand_computed_rolling_sv_matches(self, built):
        """Busiest 2020-21 goalie, 20th appearance, window 10: recompute
        SV% over appearances 10-19 with SQL independent of features.*"""
        with built.connect() as conn:
            goalie_id = conn.execute(text("""
                SELECT gg.player_id FROM raw.goalie_games gg
                JOIN raw.games g USING (game_id)
                WHERE g.season = :s AND gg.toi_seconds > 0
                GROUP BY gg.player_id ORDER BY COUNT(*) DESC LIMIT 1
            """), {"s": SEASON}).scalar()

            apps = conn.execute(text("""
                SELECT gg.game_id, gg.saves, gg.shots_against
                FROM raw.goalie_games gg JOIN raw.games g USING (game_id)
                WHERE g.season = :s AND gg.player_id = :gid AND gg.toi_seconds > 0
                ORDER BY g.date, gg.game_id
            """), {"s": SEASON, "gid": goalie_id}).fetchall()

            assert len(apps) >= 20
            target = apps[19]
            window = apps[9:19]
            expected = sum(a.saves for a in window) / sum(a.shots_against for a in window)

            actual = conn.execute(text("""
                SELECT sv_pct, starts_in_window, credibility_z
                FROM features.goalie_rolling
                WHERE game_id = :gid AND goalie_id = :goalie AND window_size = 10
            """), {"gid": target.game_id, "goalie": goalie_id}).one()

        assert actual.starts_in_window == 10
        assert float(actual.sv_pct) == pytest.approx(expected, abs=0.0005)

    def test_shrunk_values_never_null(self, built):
        with built.connect() as conn:
            bad = conn.execute(text("""
                SELECT COUNT(*) FROM features.goalie_rolling gr
                JOIN raw.games g USING (game_id)
                WHERE g.season = :s
                  AND (gr.shrunk_sv_pct IS NULL OR gr.credibility_z IS NULL)
            """), {"s": SEASON}).scalar()
        assert bad == 0

    def test_z_bounds_and_monotonicity(self, built):
        """Z in [0,1) and increases with starts_in_window (fixed k)."""
        with built.connect() as conn:
            rows = conn.execute(text("""
                SELECT DISTINCT starts_in_window, credibility_z
                FROM features.goalie_rolling gr JOIN raw.games g USING (game_id)
                WHERE g.season = :s ORDER BY starts_in_window
            """), {"s": SEASON}).fetchall()
        assert rows[0].starts_in_window == 0 and float(rows[0].credibility_z) == 0.0
        zs = [float(r.credibility_z) for r in rows]
        assert all(0.0 <= z < 1.0 for z in zs)
        assert zs == sorted(zs)

    def test_estimate_k_is_stable(self, db):
        k = estimate_k(SEASON)
        assert 30 < k < 90  # documented estimate: 57.6
