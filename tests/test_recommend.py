"""
Tests for betting/recommend.py — the daily recommendation job.

The load-bearing guarantee tested here: slate vectors built as-of a date
via the appended-stats-less-row trick must equal what the historical build
later stored for those same games, within DB NUMERIC rounding (the stored
values round-trip through NUMERIC(5,2)..(5,4) columns; the in-memory path
keeps full float precision).
"""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from betting.engine import EDGE_MIN_ML, MAX_STAKE_PCT, evaluate_market
from config.settings import check_db_connection

requires_db = pytest.mark.skipif(not check_db_connection(),
                                 reason="database not reachable")

SIM_DATE = dt.date(2026, 1, 15)      # mid-season 2025-26 slate
SIM_SEASON = 20252026


class TestEvaluateMarket:
    def test_side_priced_none_is_not_bettable(self):
        # Huge home edge but no home price -> falls through to away (no edge)
        assert evaluate_market(0.70, 0.50, None, -110) is None

    def test_consensus_fair_with_shopped_price(self):
        # fair 0.50 consensus; best home price +105 from some other book.
        # model .58 -> edge .08 vs consensus; kelly at +105
        d = evaluate_market(0.58, 0.50, 105, -115)
        assert d.side == "HOME" and d.price == 105
        assert d.edge == pytest.approx(0.08)
        assert d.market_prob == pytest.approx(0.50)
        b = 1.05
        assert d.kelly == pytest.approx((b * 0.58 - 0.42) / b)

    def test_away_edge_uses_complement_of_fair(self):
        # fair home .60 -> fair away .40; model home .55 -> away edge .05
        d = evaluate_market(0.55, 0.60, -150, 140)
        assert d.side == "AWAY"
        assert d.edge == pytest.approx(0.05)

    def test_moneyline_wrapper_unchanged(self):
        # evaluate_moneyline must behave exactly as before the refactor
        from betting.engine import evaluate_moneyline, no_vig_probs
        d = evaluate_moneyline(0.58, -110, -110)
        fair, _ = no_vig_probs(-110, -110)
        assert d.market_prob == pytest.approx(fair)
        assert d.stake_pct == MAX_STAKE_PCT


@requires_db
class TestSlateVectors:
    @pytest.fixture(scope="class")
    def slate(self):
        from betting.recommend import load_slate
        s = load_slate(SIM_DATE, simulate=True)
        assert not s.empty, "expected a 2025-26 slate on the sim date"
        return s

    @pytest.fixture(scope="class")
    def vectors(self, slate):
        from betting.recommend import build_slate_vectors
        asof = dt.datetime.combine(SIM_DATE, dt.datetime.min.time())
        return build_slate_vectors(slate, SIM_DATE, asof=asof)

    def test_vectors_finite_and_complete(self, slate, vectors):
        from features.build_vectors import FEATURE_NAMES
        assert len(vectors) == len(slate)
        m = vectors[FEATURE_NAMES].to_numpy(dtype=float)
        assert np.isfinite(m).all()

    def test_matches_historical_build_within_db_rounding(self, slate, vectors):
        """Non-goalie features must equal the stored game_vector rows for
        the same games; the only allowed difference is NUMERIC rounding
        (worst stored precision is NUMERIC(5,2) -> diff of diffs <= 0.01)."""
        from sqlalchemy import text
        from config.settings import engine
        from features.build_vectors import FEATURE_NAMES

        with engine.connect() as conn:
            stored = pd.read_sql(text("""
                SELECT game_id, feature_vector FROM features.game_vector
                WHERE game_id = ANY(:ids)
            """), conn, params={"ids": slate["game_id"].tolist()})
        stored_map = dict(zip(stored["game_id"], stored["feature_vector"]))

        skip = [i for i, n in enumerate(FEATURE_NAMES)
                if n.startswith("goalie_") or n.startswith("starter_fallback")
                or n.startswith("market_")]  # starters projected, odds source differs
        keep = [i for i in range(len(FEATURE_NAMES)) if i not in skip]

        for r in vectors.itertuples():
            mine = np.array([getattr(r, n) for n in FEATURE_NAMES], dtype=float)
            ref = np.array(stored_map[r.game_id], dtype=float)
            np.testing.assert_allclose(mine[keep], ref[keep], atol=0.011,
                                       err_msg=f"game {r.game_id}")

    def test_starters_projected_with_fallback_flag(self, slate):
        from betting.recommend import project_starters
        st = project_starters(slate, SIM_SEASON, SIM_DATE)
        assert len(st) == 2 * len(slate)
        assert (st["starter_fallback"] == 1).all()
        # mid-season: every team has a start history to project from
        assert st["goalie_id"].notna().all()


@requires_db
class TestMarketLoading:
    def test_historical_fallback_prices(self):
        """With no snapshots (offseason DB), load_market must fall back to
        the two-sided historical reference line."""
        from betting.recommend import load_market, load_slate
        slate = load_slate(SIM_DATE, simulate=True)
        m = load_market(slate["game_id"].tolist(),
                        asof=dt.datetime.combine(SIM_DATE, dt.datetime.min.time()))
        assert not m.empty
        assert set(m["game_id"]).issubset(set(slate["game_id"]))
        assert (m["fair_home_prob"] > 0).all() and (m["fair_home_prob"] < 1).all()
        assert m["home_price"].notna().all() and m["away_price"].notna().all()


@requires_db
class TestEndToEnd:
    def test_simulated_slate_dry_run(self):
        from betting.recommend import generate_recommendations
        recs = generate_recommendations(SIM_DATE, dry_run=True, simulate=True)
        if not recs.empty:
            assert (recs["edge_pct"] >= EDGE_MIN_ML - 1e-9).all()
            assert (recs["recommended_stake"] > 0).all()
            # per-bet cap: quarter-Kelly capped at 2% of the default bankroll
            from betting.recommend import BANKROLL
            assert (recs["recommended_stake"]
                    <= BANKROLL * MAX_STAKE_PCT + 1e-6).all()

    def test_write_and_cleanup(self):
        """Non-dry run writes predictions + recommendations; rerunning
        replaces PENDING rows instead of duplicating them."""
        from sqlalchemy import text
        from betting.recommend import generate_recommendations
        from config.settings import engine

        recs = generate_recommendations(SIM_DATE, simulate=True)
        game_ids = None
        try:
            with engine.connect() as conn:
                slate_ids = [r[0] for r in conn.execute(text(
                    "SELECT game_id FROM raw.games WHERE date = :d"
                    " AND game_type IN (2,3)"), {"d": SIM_DATE})]
                game_ids = slate_ids
                n_pred = conn.execute(text("""
                    SELECT COUNT(*) FROM models.predictions
                    WHERE game_id = ANY(:ids) AND market_type = 'ml'
                """), {"ids": slate_ids}).scalar()
                n_rec = conn.execute(text("""
                    SELECT COUNT(*) FROM betting.recommendations
                    WHERE game_id = ANY(:ids) AND status = 'PENDING'
                """), {"ids": slate_ids}).scalar()
            assert n_pred == len(slate_ids)     # every scored game audited
            assert n_rec == len(recs)

            # Idempotence: a second run must not duplicate PENDING rows
            generate_recommendations(SIM_DATE, simulate=True)
            with engine.connect() as conn:
                n_rec2 = conn.execute(text("""
                    SELECT COUNT(*) FROM betting.recommendations
                    WHERE game_id = ANY(:ids) AND status = 'PENDING'
                """), {"ids": slate_ids}).scalar()
            assert n_rec2 == n_rec
        finally:
            if game_ids:
                with engine.begin() as conn:
                    conn.execute(text("""
                        DELETE FROM betting.recommendations
                        WHERE game_id = ANY(:ids)"""), {"ids": game_ids})
                    conn.execute(text("""
                        DELETE FROM models.predictions
                        WHERE game_id = ANY(:ids) AND market_type = 'ml'
                    """), {"ids": game_ids})
