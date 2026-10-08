"""
Tests for models/moneyline_v3.py's pure parts (no database): the adoption
rule, the market overlay, bet making with and without exchange fees, the
Kelly simulation's caps, the bootstrap and the timing-study statistics.
"""
import math

import numpy as np
import pandas as pd
import pytest

from betting.engine import decimal_odds, effective_decimal
from models import moneyline_v3 as V


def _p(mean, se):
    return {"mean": mean, "se": se, "n": 1000, "z": mean / se}


GOOD_MKT = _p(0.0005, 0.001)       # not worse than the market at 95%


def test_paired_mean_and_se():
    d = np.array([1.0, 2.0, 3.0, 4.0])
    r = V.paired(d)
    assert r["mean"] == 2.5
    assert r["se"] == pytest.approx(d.std(ddof=1) / 2)
    assert r["n"] == 4


def test_eligible_needs_both_conditions():
    assert V.eligible(_p(-0.003, 0.001), GOOD_MKT)
    assert not V.eligible(_p(-0.0015, 0.001), GOOD_MKT)      # under 2 SE
    assert not V.eligible(_p(-0.003, 0.001), _p(0.002, 0.001))  # market wins


def test_adopt_walks_the_chain():
    vs_v0 = {"V1": _p(-0.01, 0.001), "V2": _p(-0.0105, 0.001),
             "V3": _p(-0.013, 0.001)}
    vs_mkt = {k: GOOD_MKT for k in vs_v0}
    pw = {("V2", "V1"): _p(-0.0005, 0.0004), ("V3", "V1"): _p(-0.003, 0.001),
          ("V3", "V2"): _p(-0.0025, 0.001), ("V1", "V2"): _p(0.0005, 0.0004)}
    r = V.adopt(vs_v0, vs_mkt, pw)
    assert r["chosen"] == "V3"            # V2 not by 2 SE, V3 beats V1
    pw[("V3", "V1")] = _p(-0.001, 0.001)
    assert V.adopt(vs_v0, vs_mkt, pw)["chosen"] == "V1"


def test_adopt_keeps_v0_when_nothing_is_eligible():
    vs_v0 = {k: _p(-0.001, 0.001) for k in V.ADOPTABLE}
    vs_mkt = {k: GOOD_MKT for k in V.ADOPTABLE}
    assert V.adopt(vs_v0, vs_mkt, {})["chosen"] == "V0"


def test_adopt_v2_can_win_when_v1_is_not_eligible():
    vs_v0 = {"V1": _p(-0.001, 0.001), "V2": _p(-0.004, 0.001),
             "V3": _p(-0.001, 0.001)}
    vs_mkt = {k: GOOD_MKT for k in vs_v0}
    assert V.adopt(vs_v0, vs_mkt, {})["chosen"] == "V2"


def test_beats_market():
    assert V.beats_market(_p(-0.003, 0.001))
    assert not V.beats_market(_p(-0.001, 0.001))


def test_overlay_market_only_touches_listed_games():
    names = ["a", "market_home_prob", "market_available"]
    X = np.array([[1.0, 0.5, 0.0], [2.0, 0.6, 1.0], [3.0, 0.5, 0.0]])
    out = V.overlay_market(X, names, [10, 11, 12],
                           pd.Series({10: 0.55, 12: np.nan}))
    assert out[0].tolist() == [1.0, 0.55, 1.0]
    assert out[1].tolist() == X[1].tolist()
    assert out[2].tolist() == X[2].tolist()
    assert X[0, 1] == 0.5                       # the input is not modified


def test_role_diffs_is_home_starter_minus_away_starter():
    from features.goalie_role import ROLE_FEATURES
    home_def = pd.DataFrame([[1.0] * 7], columns=ROLE_FEATURES)   # away starter
    away_def = pd.DataFrame([[3.0] * 7], columns=ROLE_FEATURES)   # home starter
    d = V.role_diffs(home_def, away_def)
    assert (d.to_numpy() == 2.0).all()
    assert list(d.columns)[0] == "role_start_share_10_starter_diff"


def test_american_from_prob_round_trip():
    assert V.american_from_prob(0.5) == 100
    assert V.american_from_prob(0.6) == -150
    assert V.american_from_prob(0.4) == 150


def _games(prob, fair, home_win=True, hp=-110, ap=-110, **kw):
    return pd.DataFrame([{"game_id": 1, "date": pd.Timestamp("2025-01-01"),
                          "prob_home": prob, "nv_consensus": fair,
                          "home_win": home_win, "hp": hp, "ap": ap, **kw}])


def test_make_bets_takes_the_side_with_edge_and_settles():
    b = V.make_bets(_games(0.56, 0.50), "prob_home", "nv_consensus", "hp", "ap")
    assert len(b) == 1 and b.iloc[0]["side"] == "HOME"
    assert b.iloc[0]["flat_pnl"] == pytest.approx(decimal_odds(-110) - 1)
    # closing EV of -110 against a 50% close: 0.5 * 1.909 - 1
    assert b.iloc[0]["close_ev"] == pytest.approx(0.5 * decimal_odds(-110) - 1)
    b = V.make_bets(_games(0.40, 0.50, home_win=True), "prob_home",
                    "nv_consensus", "hp", "ap")
    assert b.iloc[0]["side"] == "AWAY" and b.iloc[0]["flat_pnl"] == -1.0


def test_make_bets_skips_small_edges_and_missing_prices():
    assert V.make_bets(_games(0.52, 0.50), "prob_home", "nv_consensus",
                       "hp", "ap").empty
    assert V.make_bets(_games(0.60, 0.50, hp=np.nan), "prob_home",
                       "nv_consensus", "hp", "ap").empty


def test_make_bets_charges_the_exchange_fee():
    g = V.exchange_frame(_games(0.535, 0.50))
    plain = V.make_bets(g, "prob_home", "nv_consensus", "ex_home", "ex_away")
    fee = V.make_bets(g, "prob_home", "nv_consensus", "ex_home", "ex_away",
                      "kalshi", "kalshi")
    assert fee.iloc[0]["book"] == "kalshi"
    assert fee.iloc[0]["decimal"] == pytest.approx(
        effective_decimal(int(g.iloc[0]["ex_home"]), "kalshi"))
    assert fee.iloc[0]["flat_pnl"] < plain.iloc[0]["flat_pnl"]
    assert fee.iloc[0]["stake_pct"] < plain.iloc[0]["stake_pct"]


def test_make_bets_reads_the_book_from_a_column():
    g = _games(0.56, 0.50, hb="polymarket", ab="fanduel")
    b = V.make_bets(g, "prob_home", "nv_consensus", "hp", "ap", "hb", "ab")
    assert b.iloc[0]["book"] == "polymarket"


def _bets(edges, stake_pct, won, date="2025-01-01", game_ids=None):
    n = len(edges)
    return pd.DataFrame({
        "game_id": game_ids or list(range(n)), "date": pd.Timestamp(date),
        "edge": edges, "stake_pct": stake_pct, "decimal": 2.0, "won": won,
        "flat_pnl": [1.0 if w else -1.0 for w in won], "close_ev": 0.0})


def test_kelly_daily_cap_keeps_largest_edges_first():
    b = _bets([0.03, 0.09, 0.05, 0.04, 0.06, 0.07], [0.02] * 6, [True] * 6)
    kept, bank, dd = V.simulate_kelly(b, max_daily_pct=0.05)
    assert sorted(kept["edge"]) == [0.07, 0.09]          # 2 x 2% fit in 5%
    assert bank == pytest.approx(104.0)
    assert dd == 0.0


def test_kelly_game_cap_and_compounding():
    b = pd.concat([_bets([0.05, 0.04, 0.03], [0.02] * 3, [False] * 3,
                         game_ids=[1, 1, 1]),
                   _bets([0.05], [0.02], [True], date="2025-01-02")])
    kept, bank, dd = V.simulate_kelly(b)
    assert len(kept) == 3          # the third bet on game 1 passes 4%
    assert bank == pytest.approx((100 - 4) * 1.02)
    assert dd == pytest.approx(0.04)


def test_ratio_ci_brackets_the_point():
    rng = np.random.default_rng(0)
    pnl = rng.choice([-1.0, 0.9], size=400)
    lo, hi = V.ratio_ci(pnl, np.ones(400))
    assert lo < pnl.mean() < hi


def test_summarize_bets_buckets():
    b = _bets([0.03, 0.05, 0.07, 0.12, 0.04], [0.01] * 5,
              [True, False, True, True, False])
    kept, bank, dd = V.simulate_kelly(b)
    s = V.summarize_bets(b, kept, bank, dd)
    assert s["bets"] == 5
    assert {k: v["bets"] for k, v in s["buckets"].items()} == {
        "2.5-4": 1, "4-6": 2, "6-9": 1, "9+": 1}
    assert s["flat_roi"] == pytest.approx(0.2)


def test_ols_slope_recovers_a_line():
    x = np.linspace(-0.1, 0.1, 200)
    y = 0.3 * x + np.random.default_rng(1).normal(0, 0.001, 200)
    r = V.ols_slope(x, y)
    assert r["slope"] == pytest.approx(0.3, abs=0.01)
    assert r["t"] > 10


def test_arm_difference_sign():
    m = pd.DataFrame({"game_id": [1, 2], "flat_pnl": [1.0, 1.0],
                      "close_ev": [0.05, 0.05]})
    c = pd.DataFrame({"game_id": [1, 2], "flat_pnl": [-1.0, -1.0],
                      "close_ev": [-0.02, -0.02]})
    r = V.arm_difference([1, 2, 3], m, c)
    assert r["flat_roi"]["diff"] == pytest.approx(2.0)
    assert r["close_ev"]["diff"] == pytest.approx(0.07)
    assert r["close_ev"]["better"] == "morning"


def test_lgbm_seed_is_restored():
    import models.lgbm as L
    before = dict(L.LGBM_PARAMS)
    with V.lgbm_seed(7):
        assert L.LGBM_PARAMS["random_state"] == 7
    assert L.LGBM_PARAMS == before


def test_walk_forward_matches_lgbm_fold_api():
    """A tiny synthetic run: every validation game is scored, the first
    season is not, and the kept fold model rescoring with the same market
    reproduces the out-of-fold probabilities."""
    rng = np.random.default_rng(3)
    seasons = [20202021, 20212022, 20222023]
    rows = []
    for i, s in enumerate(seasons):
        days = pd.date_range(f"{2020 + i}-10-10", periods=400, freq="12h")
        for d in days:
            rows.append((len(rows), s, d))
    meta = pd.DataFrame(rows, columns=["game_id", "season", "date"])
    n = len(meta)
    p = rng.uniform(0.3, 0.7, n)
    y = (rng.uniform(size=n) < p).astype(int)
    names = ["f1", "f2", "market_home_prob", "market_available"]
    X = np.column_stack([rng.normal(size=n), rng.normal(size=n), p,
                         np.ones(n)])
    oof, fm = V.walk_forward(X, y, meta, names, keep_season=20222023)
    assert np.isnan(oof[meta["season"] == 20202021]).all()
    assert not np.isnan(oof[meta["season"] != 20202021]).any()
    rows3 = np.flatnonzero(meta["season"] == 20222023)
    again = V.rescore(fm, X, names, rows3, meta["game_id"].to_numpy(),
                      pd.Series(p, index=meta["game_id"]))
    assert np.allclose(again, oof[rows3])
    assert math.isfinite(V.per_game_log_loss(y[rows3], oof[rows3]).mean())


# ── Post-hoc diagnostics (V1m, V1mc) ────────────────────────────────

def test_diagnostics_use_the_shared_inplay_rule():
    """V1mc clears the in-play games with the same function the production
    market feature uses (features.market_prices.clear_market)."""
    import features.market_prices as MP
    assert V.clear_market is MP.clear_market
    names = ["f", "market_home_prob", "market_available"]
    X = np.array([[1.0, 0.9, 1.0], [2.0, 0.6, 1.0]])
    y = np.array([1, 0])
    meta = pd.DataFrame({"game_id": [5, 6], "season": [20232024, 20232024]})
    d = V.diagnostic_runs(X, names, y, meta, [5])
    assert d["V1m"][0].tolist() == X.tolist()
    assert d["V1mc"][0][0].tolist() == [1.0, 0.5, 0.0]
    assert d["V1mc"][0][1].tolist() == X[1].tolist()


def test_subset_keeps_rows_aligned():
    y = np.array([1, 0, 1])
    meta = pd.DataFrame({"game_id": [7, 8, 9], "season": 1})
    oofs = {"V0": np.array([0.6, 0.4, 0.7]), "V1": np.array([0.5, 0.5, 0.5])}
    rows = np.array([True, False, True])
    cy, cmeta, coofs, cav = V.subset(y, meta, oofs, np.array([1, 1, 0]) == 1, rows)
    assert cy.tolist() == [1, 1] and cmeta["game_id"].tolist() == [7, 9]
    assert list(cmeta.index) == [0, 1]
    assert coofs["V0"].tolist() == [0.6, 0.7] and cav.tolist() == [True, False]


def test_unibet_mapper_fits_on_training_rows_only():
    """The mapping must not move when validation outcomes change (no
    leakage) and must leave non-Unibet rows alone."""
    rng = np.random.default_rng(5)
    n = 1200
    names = ["f", "market_home_prob", "market_available"]
    p = rng.uniform(0.3, 0.7, n)
    X = np.column_stack([rng.normal(size=n), p, np.ones(n)])
    y = (rng.uniform(size=n) < np.clip(p * 1.2 - 0.05, 0, 1)).astype(int)
    uni = np.arange(n) < 1000                 # last 200 rows: two-way source
    train = np.arange(800)
    t1 = V.unibet_mapper(y, names, uni)(X, train)
    y2 = y.copy()
    y2[800:] = 1 - y2[800:]                   # flip every non-training label
    t2 = V.unibet_mapper(y2, names, uni)(X, train)
    assert np.allclose(t1, t2)
    assert np.allclose(t1[1000:], X[1000:])   # two-way rows untouched
    assert not np.allclose(t1[:1000, 1], X[:1000, 1])
    small = V.unibet_mapper(y, names, uni, min_rows=5000)(X, train)
    assert small is X                         # too few rows: no mapping


def test_walk_forward_applies_the_fold_transform():
    rng = np.random.default_rng(4)
    rows = []
    for i, s in enumerate([20202021, 20212022, 20222023]):
        for d in pd.date_range(f"{2020 + i}-10-10", periods=300, freq="12h"):
            rows.append((len(rows), s, d))
    meta = pd.DataFrame(rows, columns=["game_id", "season", "date"])
    n = len(meta)
    p = rng.uniform(0.3, 0.7, n)
    y = (rng.uniform(size=n) < p).astype(int)
    names = ["f1", "market_home_prob", "market_available"]
    X = np.column_stack([rng.normal(size=n), p, np.ones(n)])
    seen = []

    def tf(Xf, train_idx):
        seen.append(int(train_idx.max()))
        return Xf
    oof, _ = V.walk_forward(X, y, meta, names, fold_transform=tf)
    assert len(seen) == 2
    plain, _ = V.walk_forward(X, y, meta, names)
    assert np.allclose(oof[~np.isnan(oof)], plain[~np.isnan(plain)])
