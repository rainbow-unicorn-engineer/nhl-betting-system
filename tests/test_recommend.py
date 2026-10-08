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
from betting.recommend import (allocate_exposure, cap_daily_exposure,
                               committed_stake, frozen_games, game_exposure,
                               summarize_market)
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


def test_pregame_lines_drop_inplay_fallback_lines():
    """The raw.historical_odds fallback must not price a game off a line
    captured during it (features.market_prices.inplay_mask)."""
    from betting.recommend import pregame_lines
    hist = pd.DataFrame({
        "game_id": [1, 2], "book_name": "Unibet", "home_price": [-10000, -150],
        "away_price": [9000, 130], "provider": "Unibet", "season": 20232024,
        "date": ["2024-03-01", "2024-03-01"], "home_ml": [-10000, -150],
        "away_ml": [9000, 130], "over_under": [5.5, 6.0]})
    out = pregame_lines(hist)
    assert out["game_id"].tolist() == [2]
    assert list(out.columns) == ["game_id", "book_name", "home_price", "away_price"]
    assert list(pregame_lines(hist.iloc[:0]).columns) == list(out.columns)


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

    def test_exchange_quote_is_ranked_after_its_fee(self):
        # Kalshi +102 (contract 0.495) beats fanduel +100 on the quote, but
        # after the 0.07 * p * (1 - p) fee it pays 1.932 < 2.0: fanduel wins
        snaps = _snaps([(4, "kalshi", T1, -110, 102),
                        (4, "fanduel", T1, -120, 100)])
        m = summarize_market(snaps, NO_HIST).set_index("game_id")
        assert m.loc[4, "away_book"] == "fanduel" and m.loc[4, "away_price"] == 100
        # a big enough price gap still goes to the exchange
        snaps = _snaps([(5, "kalshi", T1, -110, 110),
                        (5, "fanduel", T1, -120, 100)])
        m = summarize_market(snaps, NO_HIST).set_index("game_id")
        assert m.loc[5, "away_book"] == "kalshi"

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


def _mcand(game_id, market, edge, stake):
    return {"game_id": game_id, "market_type": market, "edge_pct": edge,
            "recommended_stake": stake}


class TestPerGameCaps:
    """PROJECT_CONTEXT §7: max 3 correlated bets per game, plus at most
    MAX_GAME_STAKE_PCT (4%: 40 of a 1000 bankroll) across one game's
    bets. Every market counts; strongest edges are kept. The daily
    budget (100) is ample in these cases unless a test says otherwise."""

    def test_at_most_three_bets_per_game_strongest_edges_kept(self):
        cands = [_mcand(1, "ml", 0.03, 5), _mcand(1, "total", 0.06, 5),
                 _mcand(1, "prop_a", 0.05, 5), _mcand(1, "prop_b", 0.08, 5),
                 _mcand(2, "ml", 0.04, 5)]
        kept = cap_daily_exposure(cands, committed=0, bankroll=1000)
        assert [(r["game_id"], r["market_type"]) for r in kept] == [
            (1, "prop_b"), (1, "total"), (1, "prop_a"), (2, "ml")]

    def test_bets_already_issued_count_toward_the_three(self):
        # game 1 already carries 2 live bets (any market): 1 more fits
        cands = [_mcand(1, "total", 0.06, 5), _mcand(1, "prop", 0.05, 5)]
        kept = cap_daily_exposure(cands, committed=10, bankroll=1000,
                                  game_bets={1: 2}, game_stakes={1: 10.0})
        assert [r["market_type"] for r in kept] == ["total"]
        # already at 3: nothing more on game 1, other games unaffected
        kept = cap_daily_exposure(cands + [_mcand(2, "ml", 0.03, 5)],
                                  committed=15, bankroll=1000,
                                  game_bets={1: 3}, game_stakes={1: 15.0})
        assert [r["game_id"] for r in kept] == [2]

    def test_game_stake_limit_and_a_smaller_stake_can_still_fit(self):
        # 20 already on game 1; limit 40: the 15 fits (35), the 10 would
        # pass it (45) and is skipped, the 5 after it still fits (40)
        cands = [_mcand(1, "total", 0.09, 15), _mcand(1, "prop", 0.07, 10),
                 _mcand(1, "prop2", 0.05, 5)]
        kept = cap_daily_exposure(cands, committed=20, bankroll=1000,
                                  game_bets={1: 1}, game_stakes={1: 20.0})
        assert [r["market_type"] for r in kept] == ["total", "prop2"]

    def test_limits_are_parameters(self):
        cands = [_mcand(1, "ml", 0.06, 5), _mcand(1, "total", 0.05, 5)]
        assert len(cap_daily_exposure(cands, 0, 1000,
                                      max_bets_per_game=1)) == 1
        assert len(cap_daily_exposure(cands, 0, 1000,
                                      max_game_stake_pct=0.004)) == 0

    def test_daily_cap_still_applies_across_games(self):
        cands = [_mcand(g, "ml", 0.05 + g / 1000, 20) for g in range(8)]
        kept = cap_daily_exposure(cands, committed=0, bankroll=1000)
        assert sum(r["recommended_stake"] for r in kept) <= 100 + 1e-9
        assert len(kept) == 5

    def test_skip_reasons(self):
        cands = [_mcand(1, "ml", 0.06, 15), _mcand(1, "total", 0.05, 5),
                 _mcand(2, "ml", 0.04, 20), _mcand(3, "ml", 0.03, 5)]
        out = allocate_exposure(cands, committed=70, bankroll=1000,
                                picked={3}, game_bets={1: 2},
                                game_stakes={1: 20.0})
        reasons = {(r["game_id"], r["market_type"]): why for r, why in out}
        assert (3, "ml") not in reasons            # frozen: never re-decided
        assert reasons[(1, "ml")] is None          # 85 of 100; game 1 at 35
        assert "per-game limit" in reasons[(1, "total")]
        assert "daily cap" in reasons[(2, "ml")]

    def test_moneyline_only_slate_is_unchanged(self):
        # one ml candidate per game, nothing issued: the per-game limits
        # never bind, so the result equals the daily cap alone
        cands = [_cand(i, 0.03 + i / 1000, 20) for i in range(10)]
        assert cap_daily_exposure(cands, committed=0, bankroll=1000) == \
            cap_daily_exposure(cands, committed=0, bankroll=1000,
                               max_bets_per_game=99, max_game_stake_pct=1.0)


class TestGameExposure:
    def test_counts_every_market_except_skipped_and_voided(self):
        issued = pd.DataFrame({
            "game_id": [1, 1, 1, 1, 2],
            "market_type": ["ml", "total", "prop", "pl", "ml"],
            "status": ["PENDING", "SETTLED", "SKIPPED", "PENDING", "SETTLED"],
            "recommended_stake": [10.0, 5.0, 7.0, 3.0, 20.0],
            "voided": [False, False, False, True, False],
        })
        bets, stakes = game_exposure(issued)
        assert bets == {1: 2, 2: 1}
        assert stakes == {1: pytest.approx(15.0), 2: pytest.approx(20.0)}

    def test_nothing_issued(self):
        empty = pd.DataFrame(columns=["game_id", "market_type", "status",
                                      "recommended_stake", "voided"])
        assert game_exposure(empty) == ({}, {})


class TestExposureSettings:
    """MAX_BETS_PER_GAME / MAX_GAME_STAKE_PCT from the environment: blank
    = default; a malformed or out-of-range value logs an error and falls
    back instead of breaking the import (and the daily chain)."""

    def _read(self, bets: str, pct: str):
        import os
        import subprocess
        import sys
        from pathlib import Path
        root = Path(__file__).parent.parent
        out = subprocess.run(
            [sys.executable, "-c",
             "import betting.recommend as r; "
             "print(r.MAX_BETS_PER_GAME, r.MAX_GAME_STAKE_PCT)"],
            cwd=root, capture_output=True, text=True, check=True,
            env={**os.environ, "PYTHONPATH": str(root),
                 "MAX_BETS_PER_GAME": bets, "MAX_GAME_STAKE_PCT": pct})
        n, p = out.stdout.strip().splitlines()[-1].split()
        return int(n), float(p), out.stderr

    def test_blank_means_the_engine_defaults(self):
        n, p, err = self._read("", "")
        assert (n, p) == (3, 0.04) and "ERROR" not in err

    def test_valid_values(self):
        n, p, _ = self._read(" 2 ", "0.05")
        assert (n, p) == (2, 0.05)

    @pytest.mark.parametrize("bets,pct", [("abc", "4"), ("0", "nan"),
                                          ("2.5", "0"), ("-1", "-0.1")])
    def test_bad_values_fall_back_with_an_error(self, bets, pct):
        n, p, err = self._read(bets, pct)
        assert (n, p) == (3, 0.04)
        assert f"MAX_BETS_PER_GAME={bets!r}" in err
        assert f"MAX_GAME_STAKE_PCT={pct!r}" in err


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

    def test_every_market_counts_toward_the_day(self):
        issued = pd.DataFrame({
            "game_id": [1, 1, 2],
            "market_type": ["ml", "total", "prop"],
            "status": ["PENDING", "PENDING", "SKIPPED"],
            "recommended_stake": [10.0, 5.0, 7.0],
            "voided": [False, False, False],
        })
        assert committed_stake(issued) == pytest.approx(15.0)


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

    def test_other_markets_do_not_freeze_the_moneyline(self):
        # load_issued_picks now returns every market; only an ml pick
        # makes a game's moneyline frozen
        issued = pd.DataFrame({
            "game_id": [1, 2, 3],
            "market_type": ["total", "ml", "prop"],
            "status": ["PENDING", "PENDING", "SKIPPED"],
            "recommended_stake": [5.0, 10.0, 3.0],
            "voided": [False, False, False],
        })
        assert frozen_games(issued) == {2}

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

    def test_cap_is_rechecked_under_the_write_lock(self, two_games,
                                                  monkeypatch):
        """A run that decided before another run's pick landed: the budget
        is re-read inside write_recommendations, so its new pick is
        trimmed instead of pushing the day past the cap. (The per-game
        stake limit is lifted here: this test puts most of a day's budget
        on one game to isolate the daily cap.)"""
        import betting.recommend as R
        from betting.engine import MAX_DAILY_PCT
        from betting.recommend import (BANKROLL, committed_stake,
                                       load_issued_picks, write_recommendations)
        monkeypatch.setattr(R, "MAX_GAME_STAKE_PCT", 1.0)
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

    def test_per_game_limits_count_other_markets_under_the_write_lock(
            self, two_games):
        """A totals pick already on the game counts toward the per-game
        limits when the moneyline pick is written: 30 on the total + 15
        would pass the 40 (4% of 1000) game limit; 10 fits. The totals
        row does not freeze the moneyline."""
        from sqlalchemy import text
        from betting.recommend import write_recommendations
        from config.settings import engine
        g1, _ = two_games
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO betting.recommendations
                    (game_id, market_type, side, model_prob, best_book,
                     best_price, recommended_stake, status)
                VALUES (:g, 'total', 'OVER', 0.55, 'testbook', -110, 30.0,
                        'PENDING')"""), {"g": g1})
        assert write_recommendations([_rec(g1, stake=15.0)], two_games,
                                     slate_date=SIM_DATE, bankroll=1000) == 0
        assert write_recommendations([_rec(g1, stake=10.0)], two_games,
                                     slate_date=SIM_DATE, bankroll=1000) == 1


class TestTotalsGateSwitch:
    """The daily job reads models.totals.GATE_PASSED once per run, with
    everything else faked (no database, no model training)."""

    @pytest.fixture()
    def job(self, monkeypatch):
        import betting.recommend as R
        calls = {"recs": [], "totals_scored": 0, "totals_written": 0}
        slate = pd.DataFrame({"game_id": [1, 2], "season": [20262027] * 2,
                              "date": [SIM_DATE] * 2,
                              "home_team": ["BOS", "NYR"],
                              "away_team": ["TOR", "PHI"]})
        monkeypatch.setattr(R, "ensure_schema", lambda: None)
        monkeypatch.setattr(R, "load_slate", lambda d, simulate=False: slate)
        monkeypatch.setattr(R, "build_slate_vectors",
                            lambda s, d, asof=None: s[["game_id"]])
        monkeypatch.setattr(R, "score_slate", lambda v, cutoff_date=None:
                            pd.DataFrame({"game_id": [1, 2],
                                          "prob_home": [0.60, 0.50],
                                          "market_available": [1.0, 1.0]}))
        monkeypatch.setattr(R, "load_market", lambda ids, asof=None:
                            pd.DataFrame({
                                "game_id": [1, 2], "fair_home_prob": [0.5, 0.5],
                                "n_books": [3, 3], "home_price": [-105, -110],
                                "home_book": ["b", "b"], "home_priced_at": [None] * 2,
                                "away_price": [-110, -110], "away_book": ["b", "b"],
                                "away_priced_at": [None] * 2}))
        monkeypatch.setattr(R, "load_issued_picks", lambda d, conn=None:
                            pd.DataFrame(columns=["game_id", "market_type",
                                                  "status", "recommended_stake",
                                                  "voided"]))
        monkeypatch.setattr(R, "write_predictions",
                            lambda scored: {1: 11, 2: 12})

        def fake_write_recs(recs, ids, slate_date=None, bankroll=None):
            calls["recs"].extend(recs)
            return len(recs)

        def fake_score_totals(slate, d, cutoff_date=None):
            calls["totals_scored"] += 1
            return pd.DataFrame({"game_id": [1, 2],
                                 "expected_total": [6.1, 5.9]})

        def fake_write_totals(scored, lines):
            calls["totals_written"] += 1
            return len(scored)

        monkeypatch.setattr(R, "write_recommendations", fake_write_recs)
        monkeypatch.setattr(R, "score_totals", fake_score_totals)
        monkeypatch.setattr(R, "load_total_lines", lambda ids, asof=None:
                            pd.DataFrame(columns=["game_id", "line"]))
        monkeypatch.setattr(R, "write_total_predictions", fake_write_totals)

        def run(gate_passed):
            import models.totals as T
            monkeypatch.setattr(T, "GATE_PASSED", gate_passed)
            recs = R.generate_recommendations(SIM_DATE, bankroll=1000,
                                              edge_min=0.025)
            return recs, calls
        return run

    @staticmethod
    def _gate_records(caplog):
        return [r for r in caplog.records if "GATE_PASSED" in r.getMessage()]

    def test_gate_closed_keeps_totals_predictions_only(self, job, caplog):
        import logging
        with caplog.at_level(logging.INFO, logger="nhl.betting.recommend"):
            recs, calls = job(False)
        gate = self._gate_records(caplog)
        assert len(gate) == 1 and gate[0].levelno == logging.INFO
        assert "predictions-only" in gate[0].getMessage()
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
        # totals scored and stored as predictions; the only pick is ml
        assert calls["totals_scored"] == 1 and calls["totals_written"] == 1
        assert [r["game_id"] for r in calls["recs"]] == [1]
        assert list(recs["side"]) == ["HOME"]

    def test_gate_open_logs_an_error_and_still_bets_no_totals(self, job,
                                                             caplog):
        import logging
        with caplog.at_level(logging.INFO, logger="nhl.betting.recommend"):
            recs, calls = job(True)
        gate = self._gate_records(caplog)
        assert len(gate) == 1 and gate[0].levelno == logging.ERROR
        assert "no totals betting path" in gate[0].getMessage()
        # nothing else changes: predictions stored, moneyline pick only
        assert calls["totals_scored"] == 1 and calls["totals_written"] == 1
        assert [r["game_id"] for r in calls["recs"]] == [1]
        assert "market_type" not in recs or set(recs["market_type"]) <= {"ml"}


class TestOnlyGames:
    """generate_recommendations(only_games=...) (the news monitor's
    re-score): every game is scored and its prediction written, but only
    the listed games may get a new pick. Everything else is faked."""

    @pytest.fixture()
    def run(self, monkeypatch):
        import betting.recommend as R
        seen = {"recs": [], "predicted": []}
        slate = pd.DataFrame({"game_id": [1, 2], "season": [20262027] * 2,
                              "date": [SIM_DATE] * 2,
                              "home_team": ["BOS", "NYR"],
                              "away_team": ["TOR", "PHI"]})
        monkeypatch.setattr(R, "ensure_schema", lambda: None)
        monkeypatch.setattr(R, "load_slate", lambda d, simulate=False: slate)
        monkeypatch.setattr(R, "build_slate_vectors", lambda s, d, asof=None: s[["game_id"]])
        # both games clear the minimum edge on the home side
        monkeypatch.setattr(R, "score_slate", lambda v, cutoff_date=None: pd.DataFrame(
            {"game_id": [1, 2], "prob_home": [0.60, 0.60], "market_available": [1.0, 1.0]}))
        monkeypatch.setattr(R, "load_market", lambda ids, asof=None: pd.DataFrame({
            "game_id": [1, 2], "fair_home_prob": [0.5, 0.5], "n_books": [3, 3],
            "home_price": [-105, -105], "home_book": ["b", "b"],
            "home_priced_at": [None] * 2, "away_price": [-110, -110],
            "away_book": ["b", "b"], "away_priced_at": [None] * 2}))
        monkeypatch.setattr(R, "load_issued_picks", lambda d, conn=None: pd.DataFrame(
            columns=["game_id", "market_type", "status", "recommended_stake", "voided"]))

        def predictions(scored):
            seen["predicted"].append(sorted(scored["game_id"]))
            return {1: 11, 2: 12}
        monkeypatch.setattr(R, "write_predictions", predictions)

        def write_recs(recs, ids, slate_date=None, bankroll=None):
            seen["recs"].append(sorted(r["game_id"] for r in recs))
            return len(recs)
        monkeypatch.setattr(R, "write_recommendations", write_recs)
        monkeypatch.setattr(R, "score_totals", lambda *a, **k: pd.DataFrame(
            {"game_id": [1, 2], "expected_total": [6.0, 6.0]}))
        monkeypatch.setattr(R, "load_total_lines", lambda ids, asof=None:
                            pd.DataFrame(columns=["game_id", "line"]))
        monkeypatch.setattr(R, "write_total_predictions", lambda s, l: len(s))

        def go(only_games):
            recs = R.generate_recommendations(SIM_DATE, bankroll=1000, edge_min=0.025,
                                              only_games=only_games)
            return recs, seen
        return go

    def test_default_lets_every_game_pick(self, run):
        recs, seen = run(None)
        assert sorted(recs["game_id"]) == [1, 2] and seen["recs"] == [[1, 2]]

    def test_only_the_listed_games_may_pick(self, run):
        recs, seen = run({2})
        assert list(recs["game_id"]) == [2]
        assert seen["recs"] == [[2]] and seen["predicted"] == [[1, 2]]

    def test_an_empty_list_scores_but_picks_nothing(self, run):
        recs, seen = run(set())
        assert recs.empty and seen["recs"] == [[]] and seen["predicted"] == [[1, 2]]
