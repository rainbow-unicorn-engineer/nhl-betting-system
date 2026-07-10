"""
Tests for features/build_vectors.py (Phase 2, Task 6).

Unit tests drive assemble() on synthetic frames — diff/impute semantics and
the starter-fallback path (never exercised by real data, where every game
has a flagged starter). Integration tests build real vectors and verify the
plan-mandated invariants: names/vector alignment, fully-finite vectors for
completed games, label correctness, and that every vector element traces
back to the (already leakage-tested) upstream feature tables.
"""
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from features.build_vectors import (
    FEATURE_NAMES, GOALIE_STATS, TEAM_STATS, assemble, build_game_vectors,
    feature_names,
)
from features.util import WINDOWS


# ─────────────────────────────────────────────
# Synthetic frames (unit tests, no DB)
# ─────────────────────────────────────────────
def synth_games():
    return pd.DataFrame([{
        "game_id": 1, "season": 20202021, "date": "2021-01-15",
        "home_team": "HHH", "away_team": "AAA",
        "home_score": 4, "away_score": 2,
        "home_rest_days": 1, "away_rest_days": 0,
        "home_b2b": False, "away_b2b": True,
        "home_travel_km": 0.0, "away_travel_km": 1200.0,
        "home_tz_shift": 0, "away_tz_shift": -1,
        "home_game_num": 10, "away_game_num": 12,
        "season_stage": "EARLY", "home_elo": 1520.0, "away_elo": 1490.0,
    }])


def synth_team_wide(gf60={"HHH": 3.0, "AAA": 2.5}):
    """Both teams, every stat_w{w} filled; gf_per60 configurable."""
    rows = []
    for team in ("HHH", "AAA"):
        row = {"game_id": 1, "team": team}
        for w in WINDOWS:
            for s in TEAM_STATS:
                row[f"{s}_w{w}"] = gf60[team] if s == "gf_per60" else 0.5
            row[f"games_played_w{w}"] = min(w, 9)
        rows.append(row)
    return pd.DataFrame(rows)


def synth_starters(include_away=True):
    rows = [{"game_id": 1, "team": "HHH", "goalie_id": 71, "starter_fallback": 0}]
    if include_away:
        rows.append({"game_id": 1, "team": "AAA", "goalie_id": 72, "starter_fallback": 0})
    return pd.DataFrame(rows)


def synth_goalie_wide(sv={71: 0.915, 72: 0.905}):
    rows = []
    for gid, v in sv.items():
        row = {"game_id": 1, "goalie_id": gid}
        for w in WINDOWS:
            row[f"shrunk_sv_pct_w{w}"] = v
            row[f"shrunk_gsax_w{w}"] = 0.1
            row[f"credibility_z_w{w}"] = 0.2
        rows.append(row)
    return pd.DataFrame(rows)


def vec_of(df):
    row = df.iloc[0]
    return {name: row[name] for name in FEATURE_NAMES}


class TestFeatureNames:
    def test_stable_and_unique(self):
        names = feature_names()
        assert names == FEATURE_NAMES
        assert len(names) == len(set(names))
        # 14 team stats x 5 windows + 2 gp x 5 + 3 goalie x 5 + 2 flags
        # + 10 context + 2 market
        assert len(names) == 14 * 5 + 2 * 5 + 3 * 5 + 2 + 10 + 2


class TestAssemble:
    def test_diffs_and_context(self):
        df = assemble(synth_games(), synth_team_wide(),
                      synth_starters(), synth_goalie_wide())
        v = vec_of(df)
        assert v["gf_per60_diff_w10"] == pytest.approx(0.5)   # 3.0 - 2.5
        assert v["pdo_diff_w10"] == 0.0                        # equal sides
        assert v["goalie_shrunk_sv_pct_diff_w20"] == pytest.approx(0.010)
        assert v["elo_diff"] == pytest.approx(30.0)
        assert v["rest_diff"] == 1.0 and v["b2b_away"] == 1.0
        assert v["travel_diff"] == pytest.approx(-1200.0)
        assert v["game_num_diff"] == -2.0
        assert v["stage_early"] == 1.0 and v["stage_late"] == 0.0
        assert df.iloc[0]["home_win"] == True

    def test_missing_side_imputes_zero_not_null(self):
        team_wide = synth_team_wide()
        team_wide = team_wide[team_wide["team"] == "HHH"]  # away side missing
        df = assemble(synth_games(), team_wide,
                      synth_starters(), synth_goalie_wide())
        v = vec_of(df)
        assert v["gf_per60_diff_w10"] == 0.0
        assert v["gp_away_w10"] == 0.0
        assert np.isfinite(list(vec_of(df).values())).all()

    def test_starter_fallback_flagged(self):
        df = assemble(synth_games(), synth_team_wide(),
                      synth_starters(include_away=False), synth_goalie_wide())
        v = vec_of(df)
        assert v["starter_fallback_away"] == 1.0
        assert v["starter_fallback_home"] == 0.0
        assert v["goalie_shrunk_sv_pct_diff_w10"] == 0.0  # imputed, not NaN
        assert np.isfinite(list(v.values())).all()

    def test_market_feature_joined(self):
        market = pd.DataFrame([{"game_id": 1, "market_home_prob": 0.58}])
        df = assemble(synth_games(), synth_team_wide(),
                      synth_starters(), synth_goalie_wide(), market=market)
        v = vec_of(df)
        assert v["market_home_prob"] == pytest.approx(0.58)
        assert v["market_available"] == 1.0

    def test_market_missing_is_neutral_and_flagged(self):
        for market in (None, pd.DataFrame(columns=["game_id", "market_home_prob"]),
                       pd.DataFrame([{"game_id": 999, "market_home_prob": 0.7}])):
            df = assemble(synth_games(), synth_team_wide(),
                          synth_starters(), synth_goalie_wide(), market=market)
            v = vec_of(df)
            assert v["market_home_prob"] == 0.5
            assert v["market_available"] == 0.0


# ─────────────────────────────────────────────
# Integration (requires fully built feature tables)
# ─────────────────────────────────────────────
@pytest.fixture(scope="module")
def db():
    from config.settings import engine
    try:
        with engine.connect() as conn:
            checks = {
                "team_rolling": "SELECT COUNT(*) FROM features.team_rolling",
                "goalie_rolling": "SELECT COUNT(*) FROM features.goalie_rolling",
                "matchup_elo": "SELECT COUNT(*) FROM features.matchup WHERE home_elo IS NOT NULL",
            }
            for name, sql in checks.items():
                if conn.execute(text(sql)).scalar() < 1000:
                    pytest.skip(f"{name} not built")
    except Exception as e:
        pytest.skip(f"database unavailable: {e}")
    return engine


@pytest.fixture(scope="module")
def built(db):
    build_game_vectors()
    return db


class TestIntegration:
    def test_row_count_and_alignment(self, built):
        with built.connect() as conn:
            n_final = conn.execute(text(
                "SELECT COUNT(*) FROM raw.games WHERE game_state IN ('FINAL','OFF')"
            )).scalar()
            n_rows, n_misaligned = conn.execute(text("""
                SELECT COUNT(*),
                       COUNT(*) FILTER (WHERE
                           array_length(feature_vector, 1) <> array_length(feature_names, 1)
                           OR array_length(feature_vector, 1) <> :n)
                FROM features.game_vector
            """), {"n": len(FEATURE_NAMES)}).one()
        assert n_rows == n_final
        assert n_misaligned == 0

    def test_vectors_fully_finite(self, built):
        with built.connect() as conn:
            rows = conn.execute(text(
                "SELECT feature_vector FROM features.game_vector")).fetchall()
        matrix = np.array([r.feature_vector for r in rows], dtype=float)
        assert np.isfinite(matrix).all()

    def test_labels_match_scores(self, built):
        with built.connect() as conn:
            bad = conn.execute(text("""
                SELECT COUNT(*) FROM features.game_vector v
                JOIN raw.games g USING (game_id)
                WHERE v.home_win <> (g.home_score > g.away_score)
                   OR v.home_goals <> g.home_score
                   OR v.away_goals <> g.away_score
            """)).scalar()
        assert bad == 0

    def test_elements_trace_to_upstream_tables(self, built):
        """Assembly must be a pure join of the (leakage-tested) upstream
        tables: for sampled games, spot-check vector elements against
        features.matchup, team_rolling and goalie_rolling directly."""
        idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
        with built.connect() as conn:
            sample = conn.execute(text("""
                SELECT v.game_id, v.feature_vector, g.home_team, g.away_team,
                       m.home_elo, m.away_elo, m.home_starter_id, m.away_starter_id
                FROM features.game_vector v
                JOIN raw.games g USING (game_id)
                JOIN features.matchup m USING (game_id)
                ORDER BY md5(v.game_id::text) LIMIT 25
            """)).fetchall()
            assert len(sample) == 25

            for row in sample:
                vec = row.feature_vector
                assert vec[idx["elo_diff"]] == pytest.approx(
                    float(row.home_elo) - float(row.away_elo), abs=0.05)

                tr = {r.team: r for r in conn.execute(text("""
                    SELECT team, gf_per60, games_played FROM features.team_rolling
                    WHERE game_id = :gid AND window_size = 10
                """), {"gid": row.game_id})}
                h, a = tr[row.home_team], tr[row.away_team]
                expected = (float(h.gf_per60) - float(a.gf_per60)
                            if h.gf_per60 is not None and a.gf_per60 is not None
                            else 0.0)
                assert vec[idx["gf_per60_diff_w10"]] == pytest.approx(expected, abs=1e-6)
                assert vec[idx["gp_home_w10"]] == float(h.games_played)

                gr = {r.goalie_id: r for r in conn.execute(text("""
                    SELECT goalie_id, shrunk_sv_pct FROM features.goalie_rolling
                    WHERE game_id = :gid AND window_size = 40
                      AND goalie_id IN (:hs, :as_)
                """), {"gid": row.game_id, "hs": row.home_starter_id,
                       "as_": row.away_starter_id})}
                expected_g = (float(gr[row.home_starter_id].shrunk_sv_pct)
                              - float(gr[row.away_starter_id].shrunk_sv_pct))
                assert vec[idx["goalie_shrunk_sv_pct_diff_w40"]] == \
                    pytest.approx(expected_g, abs=1e-6)

    def test_starter_ids_written_back(self, built):
        with built.connect() as conn:
            missing = conn.execute(text("""
                SELECT COUNT(*) FROM features.game_vector v
                JOIN features.matchup m USING (game_id)
                WHERE m.home_starter_id IS NULL OR m.away_starter_id IS NULL
            """)).scalar()
        assert missing == 0
