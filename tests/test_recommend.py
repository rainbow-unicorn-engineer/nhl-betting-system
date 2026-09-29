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
from betting.recommend import (cap_daily_exposure, committed_stake,
                               frozen_games, summarize_market)
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


def _snaps(rows):
    return pd.DataFrame(rows, columns=["game_id", "book_name", "captured_at",
                                       "home_price", "away_price"])


NO_HIST = pd.DataFrame(columns=["game_id", "book_name", "home_price",
                                "away_price"])
T1 = pd.Timestamp("2026-01-15 15:00")
T2 = pd.Timestamp("2026-01-15 16:00")


class TestBettableBooks:
    """summarize_market is load_market's pure half: fair odds from every
    book, best price only from BETTABLE_BOOKS, priced_at carried."""

    SNAPS = _snaps([
        (1, "draftkings", T1, -120, 105),
        (1, "fanduel", T2, -115, 100),
        (1, "pinnacle", T2, -110, 110),     # best on both sides, not bettable
        (2, "pinnacle", T2, -130, 115),     # only an off-list book prices it
    ])

    def test_unset_means_every_book(self):
        m = summarize_market(self.SNAPS, NO_HIST).set_index("game_id")
        assert m.loc[1, "home_price"] == -110 and m.loc[1, "home_book"] == "pinnacle"
        assert m.loc[1, "away_price"] == 110 and m.loc[1, "away_book"] == "pinnacle"
        assert m.loc[1, "home_priced_at"] == T2.to_pydatetime()

    def test_best_price_only_from_allowed_books(self):
        m = summarize_market(self.SNAPS, NO_HIST,
                             frozenset({"draftkings", "fanduel"})).set_index("game_id")
        assert m.loc[1, "home_price"] == -115 and m.loc[1, "home_book"] == "fanduel"
        assert m.loc[1, "away_price"] == 105 and m.loc[1, "away_book"] == "draftkings"
        # priced_at is the captured_at of the row each best price came from
        assert m.loc[1, "home_priced_at"] == T2.to_pydatetime()
        assert m.loc[1, "away_priced_at"] == T1.to_pydatetime()

    def test_fair_prob_still_uses_every_book(self):
        from features.util import american_implied_prob as imp
        allowed = summarize_market(self.SNAPS, NO_HIST, frozenset({"draftkings"}))
        everyone = summarize_market(self.SNAPS, NO_HIST)
        fair = allowed.set_index("game_id").loc[1, "fair_home_prob"]
        assert fair == everyone.set_index("game_id").loc[1, "fair_home_prob"]
        novig = [imp(h) / (imp(h) + imp(a)) for h, a in ((-120, 105), (-115, 100), (-110, 110))]
        assert fair == pytest.approx(float(np.median(novig)))
        assert allowed.set_index("game_id").loc[1, "n_books"] == 3

    def test_game_no_allowed_book_prices_cannot_be_bet(self):
        m = summarize_market(self.SNAPS, NO_HIST,
                             frozenset({"draftkings"})).set_index("game_id")
        assert pd.isna(m.loc[2, "home_price"]) and pd.isna(m.loc[2, "away_price"])
        assert pd.notna(m.loc[2, "fair_home_prob"])
        # the engine treats an unpriced side as not bettable
        assert evaluate_market(0.90, m.loc[2, "fair_home_prob"], None, None) is None

    def test_historical_reference_line_is_not_filtered(self):
        hist = pd.DataFrame([(3, "ESPN BET", -140, 120)], columns=NO_HIST.columns)
        m = summarize_market(_snaps([]), hist,
                             frozenset({"draftkings"})).set_index("game_id")
        assert m.loc[3, "home_price"] == -140 and m.loc[3, "away_price"] == 120
        assert m.loc[3, "home_priced_at"] is None       # no snapshot -> NULL priced_at

    def test_bettable_books_env_parsing(self):
        import os
        import subprocess
        import sys
        from pathlib import Path
        root = Path(__file__).parent.parent
        out = subprocess.run(
            [sys.executable, "-c",
             "from betting.recommend import BETTABLE_BOOKS as b; print(sorted(b))"],
            cwd=root, capture_output=True, text=True, check=True,
            env={**os.environ, "PYTHONPATH": str(root),
                 "BETTABLE_BOOKS": " DraftKings, fanduel ,,"})
        assert out.stdout.strip().splitlines()[-1] == "['draftkings', 'fanduel']"


def _cand(game_id, edge, stake):
    return {"game_id": game_id, "edge_pct": edge, "recommended_stake": stake}


class TestDailyCap:
    """cap_daily_exposure: the one allocation rule, run when deciding and
    again under the write lock. Bankroll 1000 at 10% = a 100 budget."""

    def test_picked_game_is_excluded(self):
        kept = cap_daily_exposure([_cand(1, 0.05, 10), _cand(2, 0.04, 10)],
                                  committed=0, bankroll=1000, picked={1})
        assert [r["game_id"] for r in kept] == [2]

    def test_new_stakes_fit_what_is_left_of_the_budget(self):
        cands = [_cand(i, 0.03 + i / 1000, 20) for i in range(10)]
        for committed in (0, 15, 40, 55, 99.99, 100):
            kept = cap_daily_exposure(cands, committed=committed, bankroll=1000)
            assert sum(r["recommended_stake"] for r in kept) <= 100 - committed + 1e-9

    def test_committed_40_leaves_room_for_two_stakes_of_20(self):
        kept = cap_daily_exposure([_cand(1, 0.05, 20), _cand(2, 0.04, 20)],
                                  committed=40, bankroll=1000)
        assert [r["game_id"] for r in kept] == [1, 2]

    def test_committed_85_blocks_them(self):
        assert cap_daily_exposure([_cand(1, 0.05, 20), _cand(2, 0.04, 20)],
                                  committed=85, bankroll=1000) == []

    def test_picked_game_with_the_best_edge_is_skipped(self):
        cands = [_cand(1, 0.03, 20), _cand(2, 0.09, 20), _cand(3, 0.05, 20)]
        kept = cap_daily_exposure(cands, committed=60, bankroll=1000, picked={2})
        # game 2 has the best edge but is already picked; 3 then 1 by edge,
        # and only 40 of budget is left
        assert [r["game_id"] for r in kept] == [3, 1]

    def test_strongest_edges_first_and_a_smaller_stake_can_still_fit(self):
        cands = [_cand(1, 0.03, 5), _cand(2, 0.08, 20), _cand(3, 0.06, 20)]
        kept = cap_daily_exposure(cands, committed=70, bankroll=1000)
        assert [r["game_id"] for r in kept] == [2, 1]


class TestCommittedStake:
    def test_skipped_and_voided_picks_commit_nothing(self):
        issued = pd.DataFrame({
            "game_id": [1, 2, 3, 4],
            "status": ["PENDING", "SKIPPED", "SETTLED", "SETTLED"],
            "recommended_stake": [10.0, 20.0, 30.0, 40.0],
            "voided": [False, False, False, True],
        })
        assert committed_stake(issued) == pytest.approx(40.0)

    def test_nothing_issued(self):
        empty = pd.DataFrame(columns=["game_id", "status",
                                      "recommended_stake", "voided"])
        assert committed_stake(empty) == 0.0


class TestFrozenGames:
    """Which games already have their pick: a voided pick (postponed
    game) doesn't count, so the game can be picked again when played."""

    def test_voided_pick_does_not_freeze_its_game(self):
        issued = pd.DataFrame({
            "game_id": [1, 2, 3, 4, 4],
            "status": ["PENDING", "SKIPPED", "SETTLED", "SETTLED", "PENDING"],
            "recommended_stake": [10.0, 20.0, 30.0, 40.0, 5.0],
            "voided": [False, False, True, True, False],
        })
        # 3: only a voided pick -> open. 4: voided, then picked again -> frozen
        assert frozen_games(issued) == {1, 2, 4}

    def test_unknown_void_flag_counts_as_a_pick(self):
        issued = pd.DataFrame({"game_id": [7], "status": ["PENDING"],
                               "recommended_stake": [10.0], "voided": [None]})
        assert frozen_games(issued) == {7}

    def test_nothing_issued(self):
        empty = pd.DataFrame(columns=["game_id", "status",
                                      "recommended_stake", "voided"])
        assert frozen_games(empty) == set()


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
        from sqlalchemy import text
        from config.settings import engine as db_engine
        from betting.recommend import project_starters
        st = project_starters(slate, SIM_SEASON, SIM_DATE)
        assert len(st) == 2 * len(slate)
        # mid-season: every team has a start history to project from
        assert st["goalie_id"].notna().all()
        # fallback=0 is allowed ONLY where a Daily Faceoff row confirms
        with db_engine.connect() as conn:
            confirmed = {r[0] for r in conn.execute(text("""
                SELECT team FROM raw.starting_goalies
                WHERE game_date = :d AND goalie_id IS NOT NULL
                  AND confirmation = 'Confirmed'
            """), {"d": SIM_DATE})}
        heuristic = st[~st["team"].isin(confirmed)]
        assert (heuristic["starter_fallback"] == 1).all()


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
class TestLiveSlateFilter:
    def test_postponed_suspended_cancelled_games_leave_the_live_slate(self):
        """A SIM_DATE game made to look upcoming is on the live slate while
        its schedule state is OK (or unknown), and off it when PPD, SUSP or
        CNCL. The row is restored afterwards."""
        from sqlalchemy import text
        from betting.recommend import load_slate
        from config.migrate import ensure_schema
        from config.settings import engine
        ensure_schema()
        with engine.connect() as conn:
            g = conn.execute(text("""
                SELECT g.game_id, g.game_state, g.start_time_utc, g.schedule_state
                FROM raw.games g JOIN features.matchup m USING (game_id)
                WHERE g.date = :d ORDER BY g.game_id LIMIT 1
            """), {"d": SIM_DATE}).one()

        def set_state(state, schedule_state, start):
            with engine.begin() as conn:
                conn.execute(text("""
                    UPDATE raw.games SET game_state = :s, schedule_state = :ss,
                           start_time_utc = :t WHERE game_id = :g
                """), {"s": state, "ss": schedule_state, "t": start,
                       "g": g.game_id})

        later = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=6)
        try:
            for schedule_state, on_slate in ((None, True), ("OK", True),
                                             ("PPD", False), ("SUSP", False),
                                             ("CNCL", False)):
                set_state("FUT", schedule_state, later)
                ids = set(load_slate(SIM_DATE)["game_id"])
                assert (g.game_id in ids) == on_slate, schedule_state
        finally:
            set_state(g.game_state, g.schedule_state, g.start_time_utc)


def _existing_ids(conn, sql, ids):
    from sqlalchemy import text
    return {r[0] for r in conn.execute(text(sql), {"ids": ids})}


_REC_IDS = "SELECT rec_id FROM betting.recommendations WHERE game_id = ANY(:ids)"
_PRED_IDS = ("SELECT prediction_id FROM models.predictions "
             "WHERE game_id = ANY(:ids) AND market_type IN ('ml', 'total')")


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
        keeps the issued picks (frozen) instead of duplicating them.
        Teardown deletes only the rows this test created (predictions it
        merely updated keep the values from this run)."""
        from sqlalchemy import text
        from betting.recommend import generate_recommendations
        from config.settings import engine

        with engine.connect() as conn:
            slate_ids = [r[0] for r in conn.execute(text(
                "SELECT game_id FROM raw.games WHERE date = :d"
                " AND game_type IN (2,3)"), {"d": SIM_DATE})]
            recs_before = _existing_ids(conn, _REC_IDS, slate_ids)
            preds_before = _existing_ids(conn, _PRED_IDS, slate_ids)
        try:
            recs = generate_recommendations(SIM_DATE, simulate=True)
            with engine.connect() as conn:
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
            with engine.begin() as conn:
                new_recs = _existing_ids(conn, _REC_IDS, slate_ids) - recs_before
                new_preds = _existing_ids(conn, _PRED_IDS, slate_ids) - preds_before
                conn.execute(text("DELETE FROM betting.recommendations "
                                  "WHERE rec_id = ANY(:r)"),
                             {"r": sorted(new_recs)})
                conn.execute(text("DELETE FROM models.predictions "
                                  "WHERE prediction_id = ANY(:p)"),
                             {"p": sorted(new_preds)})


def _rec(game_id, side="HOME", price=110, stake=10.0, priced_at=None):
    return {"game_id": int(game_id), "prediction_id": None, "side": side,
            "model_prob": 0.55, "best_book": "testbook", "best_price": price,
            "implied_prob_novig": 0.50, "edge_pct": 0.05,
            "kelly_fraction": 0.10, "recommended_stake": stake,
            "priced_at": priced_at}


@requires_db
class TestFrozenPicks:
    @pytest.fixture()
    def two_games(self):
        from sqlalchemy import text
        from config.migrate import ensure_schema
        from config.settings import engine
        ensure_schema()
        with engine.connect() as conn:
            ids = [r[0] for r in conn.execute(text("""
                SELECT game_id FROM raw.games
                WHERE date = :d AND game_type IN (2, 3)
                ORDER BY game_id LIMIT 2"""), {"d": SIM_DATE})]
            before = _existing_ids(conn, _REC_IDS, ids)
        assert len(ids) == 2
        yield ids
        with engine.begin() as conn:     # only the recommendations the test added
            new = _existing_ids(conn, _REC_IDS, ids) - before
            conn.execute(text("DELETE FROM betting.recommendations "
                              "WHERE rec_id = ANY(:r)"), {"r": sorted(new)})

    def _recs(self, ids):
        from sqlalchemy import text
        from config.settings import engine
        with engine.connect() as conn:
            return pd.read_sql(text("""
                SELECT game_id, side, best_price, priced_at, status
                FROM betting.recommendations WHERE game_id = ANY(:ids)
                ORDER BY game_id"""), conn, params={"ids": ids})

    def test_issued_pick_is_never_repriced_or_deleted(self, two_games):
        from betting.recommend import write_recommendations
        g1, g2 = two_games
        t0 = dt.datetime(2026, 1, 15, 15, 0)
        assert write_recommendations([_rec(g1, "HOME", 110, priced_at=t0)],
                                     two_games) == 1
        # a later run at newer prices: g1 must stay as issued, g2 is new
        t1 = dt.datetime(2026, 1, 15, 22, 0)
        n = write_recommendations([_rec(g1, "AWAY", -150, priced_at=t1),
                                   _rec(g2, "HOME", 120, priced_at=t1)],
                                  two_games)
        assert n == 1
        recs = self._recs(two_games).set_index("game_id")
        assert len(recs) == 2
        assert recs.loc[g1, "side"] == "HOME" and recs.loc[g1, "best_price"] == 110
        assert pd.Timestamp(recs.loc[g1, "priced_at"]) == pd.Timestamp(t0)
        assert recs.loc[g2, "best_price"] == 120
        assert pd.Timestamp(recs.loc[g2, "priced_at"]) == pd.Timestamp(t1)

    @pytest.mark.parametrize("status", ["SKIPPED", "SETTLED", "PLACED",
                                        "APPROVED"])
    def test_decided_or_settled_pick_blocks_new_one(self, two_games, status):
        from sqlalchemy import text
        from betting.recommend import write_recommendations
        from config.settings import engine
        g1, _ = two_games
        assert write_recommendations([_rec(g1)], two_games) == 1
        with engine.begin() as conn:
            conn.execute(text("UPDATE betting.recommendations SET status = "
                              ":s WHERE game_id = :g AND best_book = 'testbook'"),
                         {"s": status, "g": g1})
        assert write_recommendations([_rec(g1, "AWAY", -130)], two_games) == 0
        assert len(self._recs([g1])) == 1

    def test_cap_is_rechecked_under_the_write_lock(self, two_games):
        """A run that decided before another run's pick landed: the budget
        is re-read inside write_recommendations, so its new pick is
        trimmed instead of pushing the day past the cap."""
        from betting.engine import MAX_DAILY_PCT
        from betting.recommend import (BANKROLL, committed_stake,
                                       load_issued_picks, write_recommendations)
        g1, g2 = two_games
        budget = BANKROLL * MAX_DAILY_PCT
        already = committed_stake(load_issued_picks(SIM_DATE))
        first = round(budget - already - 5.0, 2)
        assert first > 0
        # the other run's pick, leaving 5 of the day's budget
        assert write_recommendations([_rec(g1, stake=first)], two_games) == 1
        # this run decided a stake of 10 for g2 while the budget looked free
        assert write_recommendations([_rec(g2, stake=10.0)], two_games,
                                     slate_date=SIM_DATE) == 0
        # a stake that fits what is left still goes in
        assert write_recommendations([_rec(g2, stake=5.0)], two_games,
                                     slate_date=SIM_DATE) == 1
