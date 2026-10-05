"""
Tests for the totals v3 experiment in models/totals.py: Dixon-Coles,
the market-offset rates, the over/under quote sources and cleaning, the
calibration summary, the pre-registered pass rule, and the walk-forward
wiring (database and booster faked, so they run anywhere).

Dixon-Coles → one number (rho) that moves probability between the 0-0 /
1-1 and the 1-0 / 0-1 scores. No-vig → the bookmaker's fee taken out.
"""
import numpy as np
import pandas as pd
import pytest

import models.totals as T
from config.settings import check_db_connection
from models.totals import (MARGIN_WEIGHTS, choose_v3, clean_unibet, dc_tau,
                           env_rates, fit_dc_rho, history_closing_quotes,
                           joint_pmf, market_lambdas, not_worse_than_market,
                           poisson_pmf, prob_over, quote_sources,
                           total_calibration, total_from_joint, total_pmf,
                           v3_passes)
from tests.test_totals import _synthetic

requires_db = pytest.mark.skipif(not check_db_connection(),
                                 reason="database not reachable")


def _draw(joint, rng):
    """One (home, away) regulation score per game from (n, K+1, K+1)."""
    n, k1, _ = joint.shape
    flat = joint.reshape(n, -1)
    cells = (flat.cumsum(axis=1) < rng.random((n, 1))).sum(axis=1)
    return np.divmod(np.minimum(cells, k1 * k1 - 1), k1)


# ── Dixon-Coles ────────────────────────────────────────────────────

class TestDixonColes:
    def test_rho_zero_changes_nothing(self):
        ph = poisson_pmf(np.array([3.0, 2.2]))
        pa = poisson_pmf(np.array([2.8, 3.9]))
        for w in (MARGIN_WEIGHTS, None):
            np.testing.assert_array_equal(joint_pmf(ph, pa, w, 0.0),
                                          joint_pmf(ph, pa, w))
        np.testing.assert_array_equal(dc_tau(ph, pa, 0.0), 1.0)

    def test_hand_computed_factors(self):
        # two-goal support so the means are easy: H ~ [.5, .5] (mean .5),
        # A ~ [.4, .6] (mean .6); rho = .1
        ph, pa = np.array([[0.5, 0.5]]), np.array([[0.4, 0.6]])
        tau = dc_tau(ph, pa, 0.1)[0]
        np.testing.assert_allclose(tau, [[1 - 0.5 * 0.6 * 0.1, 1 + 0.5 * 0.1],
                                         [1 + 0.6 * 0.1, 1 - 0.1]])
        # independent cells .20 .30 / .20 .30 times tau, renormalized
        raw = np.array([[0.20 * 0.97, 0.30 * 1.05], [0.20 * 1.06, 0.30 * 0.9]])
        np.testing.assert_allclose(joint_pmf(ph, pa, None, 0.1)[0],
                                   raw / raw.sum())

    def test_factors_floored_at_zero_and_sum_to_one(self):
        ph = poisson_pmf(np.array([7.5]))
        pa = poisson_pmf(np.array([7.5]))
        assert dc_tau(ph, pa, 0.2)[0, 0, 0] == 0.0       # 1 - 56 * .2 < 0
        j = joint_pmf(ph, pa, MARGIN_WEIGHTS, 0.2)
        np.testing.assert_allclose(j.sum(axis=(1, 2)), 1.0)
        assert (j >= 0).all()

    def test_positive_rho_moves_mass_from_draws_to_one_goal_scores(self):
        ph = poisson_pmf(np.array([2.0]))
        pa = poisson_pmf(np.array([1.8]))
        a, b = joint_pmf(ph, pa, MARGIN_WEIGHTS), joint_pmf(ph, pa,
                                                             MARGIN_WEIGHTS, 0.1)
        assert b[0, 0, 0] < a[0, 0, 0] and b[0, 1, 1] < a[0, 1, 1]
        assert b[0, 1, 0] > a[0, 1, 0] and b[0, 0, 1] > a[0, 0, 1]

    def test_fit_recovers_known_rho(self):
        rng = np.random.default_rng(21)
        n = 60000
        ph = poisson_pmf(rng.uniform(1.2, 2.2, n))
        pa = poisson_pmf(rng.uniform(1.0, 2.0, n))
        h, a = _draw(joint_pmf(ph, pa, MARGIN_WEIGHTS, 0.08), rng)
        assert fit_dc_rho(ph, pa, h, a, MARGIN_WEIGHTS) == pytest.approx(
            0.08, abs=0.025)

    def test_fit_on_scores_without_the_effect_gives_zero(self):
        rng = np.random.default_rng(4)
        n = 60000
        ph = poisson_pmf(rng.uniform(1.2, 2.2, n))
        pa = poisson_pmf(rng.uniform(1.0, 2.0, n))
        h, a = _draw(joint_pmf(ph, pa, MARGIN_WEIGHTS), rng)
        assert abs(fit_dc_rho(ph, pa, h, a, MARGIN_WEIGHTS)) < 0.025

    def test_fit_stays_inside_bounds(self):
        ph = poisson_pmf(np.full(200, 1.5))
        h = np.zeros(200, int)                  # all 0-0: wants rho << 0
        rho = fit_dc_rho(ph, ph, h, h, MARGIN_WEIGHTS)
        assert T.DC_RHO_BOUNDS[0] - 1e-6 <= rho <= T.DC_RHO_BOUNDS[1]


class TestTotalFromJoint:
    def test_matches_a_cell_by_cell_sum(self):
        rng = np.random.default_rng(2)
        joint = rng.random((5, 13, 13))
        joint /= joint.sum(axis=(1, 2), keepdims=True)
        ref = np.zeros((5, 26))
        for h in range(13):
            for a in range(13):
                ref[:, h + a + (h == a)] += joint[:, h, a]
        np.testing.assert_allclose(total_from_joint(joint), ref)

    def test_total_pmf_rho_passes_through(self):
        ph = poisson_pmf(np.array([3.0]))
        pa = poisson_pmf(np.array([2.8]))
        np.testing.assert_allclose(
            total_pmf(ph, pa, MARGIN_WEIGHTS, 0.05),
            total_from_joint(joint_pmf(ph, pa, MARGIN_WEIGHTS, 0.05)))


# ── Market-offset rates ────────────────────────────────────────────

def _p_over_no_push(lh, la, line, w, rho):
    tp = total_pmf(poisson_pmf(lh), poisson_pmf(la), w, rho)
    po, pp = prob_over(tp, line)
    return po / (1.0 - pp)


class TestMarketLambdas:
    @pytest.mark.parametrize("rho", [0.0, 0.05])
    def test_hits_the_market_price_and_keeps_the_split(self, rho):
        fair = np.array([0.45, 0.5, 0.58, 0.52, 0.47])
        line = np.array([5.5, 6.5, 5.5, 6.0, 5.0])        # whole lines push
        bh = np.array([3.0, 3.1, 2.9, 3.2, 2.8])
        ba = np.array([2.7, 2.8, 2.6, 2.9, 3.0])
        lh, la = market_lambdas(fair, line, bh, ba, MARGIN_WEIGHTS, rho)
        np.testing.assert_allclose(
            _p_over_no_push(lh, la, line, MARGIN_WEIGHTS, rho), fair,
            atol=1e-9)
        np.testing.assert_allclose(lh / la, bh / ba)

    def test_higher_market_over_means_higher_rates(self):
        fair = np.array([0.40, 0.50, 0.60])
        lh, _ = market_lambdas(fair, np.full(3, 6.5), np.full(3, 3.0),
                               np.full(3, 2.8))
        assert lh[0] < lh[1] < lh[2]

    def test_each_game_depends_only_on_its_own_price(self):
        fair = np.array([0.45, 0.55, 0.50])
        line, bh, ba = np.full(3, 5.5), np.full(3, 3.0), np.full(3, 2.8)
        lh, la = market_lambdas(fair, line, bh, ba)
        fair2 = fair.copy()
        fair2[1:] = [0.70, 0.30]                 # other games' prices move
        lh2, la2 = market_lambdas(fair2, line, bh, ba)
        assert (lh2[0], la2[0]) == (lh[0], la[0])

    def test_unreachable_price_takes_the_range_end(self):
        lh, la = market_lambdas(np.array([0.999]), np.array([5.5]),
                                np.array([1.0]), np.array([1.0]))
        assert lh[0] == pytest.approx(T.MARKET_SCALE_RANGE[1], rel=1e-6)

    def test_point_in_time_later_outcomes_never_move_a_games_rates(self):
        """The rates use the game's own pre-game price and the trailing
        environment (strictly earlier dates): rewriting every outcome on
        or after a game's date leaves its market rates unchanged."""
        rng = np.random.default_rng(9)
        dates = pd.Series(np.repeat(pd.date_range("2024-10-01", periods=30),
                                    6))
        y_h = rng.poisson(3.0, len(dates)).astype(float)
        y_a = rng.poisson(2.7, len(dates)).astype(float)
        fair = rng.uniform(0.4, 0.6, len(dates))
        line = np.full(len(dates), 6.5)
        day = dates == dates.iloc[90]

        def rates(yh, ya):
            eh = env_rates(dates, yh, 2.95)
            ea = env_rates(dates, ya, 2.70)
            return market_lambdas(fair, line, eh, ea, MARGIN_WEIGHTS)

        before = rates(y_h, y_a)
        later = (dates >= dates.iloc[90]).to_numpy()
        y_h2, y_a2 = y_h.copy(), y_a.copy()
        y_h2[later], y_a2[later] = 9.0, 0.0
        after = rates(y_h2, y_a2)
        for b, a in zip(before, after):
            np.testing.assert_array_equal(b[day.to_numpy()], a[day.to_numpy()])


# ── Quote sources ──────────────────────────────────────────────────

class TestQuoteSources:
    def test_unibet_cleaning_rules(self):
        rows = pd.DataFrame([
            # game, line, over, under, home ml, away ml
            (1, 5.5, -120, 100, 150, 200),       # kept: 4.7% overround
            (2, 6.0, -120, 100, 150, 200),       # not the 5.5 line
            (3, 5.5, -150, 110, 150, 200),       # 12% overround
            (4, 5.5, -900, 600, 150, 200),       # no-vig over ~0.87
            (5, 5.5, -120, 100, -1500, 900),     # in-play moneyline
            (6, 5.5, None, 100, 150, 200),       # one price missing
        ], columns=["game_id", "over_under", "over_price", "under_price",
                    "home_ml", "away_ml"])
        out = clean_unibet(rows)
        assert list(out["game_id"]) == [1]
        assert list(out.columns) == ["game_id", "book_name", "line",
                                     "over_price", "under_price"]
        assert out["book_name"].iloc[0] == "espn_Unibet"
        assert out["line"].iloc[0] == 5.5

    def test_unibet_cleaning_empty(self):
        empty = pd.DataFrame(columns=["game_id", "over_under", "over_price",
                                      "under_price", "home_ml", "away_ml"])
        assert clean_unibet(empty).empty

    def test_history_closing_is_the_last_pregame_snapshot(self):
        start = pd.Timestamp("2025-01-10 00:00")
        t = lambda m: start - pd.Timedelta(minutes=m)      # noqa: E731
        rows = pd.DataFrame([
            # book a: morning 5.5 then closing 6.0; an in-play row after
            (1, "a", "over", -110, 5.5, t(600)), (1, "a", "under", -110, 5.5, t(600)),
            (1, "a", "over", 105, 6.0, t(15)), (1, "a", "under", -125, 6.0, t(15)),
            (1, "a", "over", 300, 6.5, t(-20)), (1, "a", "under", -400, 6.5, t(-20)),
            # book b: last snapshot only has the over side -> earlier pair
            (1, "b", "over", -105, 6.5, t(30)), (1, "b", "under", -115, 6.5, t(30)),
            (1, "b", "over", -110, 6.5, t(10)),
            # book c: sides at different points in one snapshot -> dropped
            (1, "c", "over", -110, 5.5, t(10)), (1, "c", "under", -110, 6.5, t(10)),
        ], columns=["game_id", "book", "side", "price", "point", "snapshot_ts"])
        rows["start_utc"] = start
        q = history_closing_quotes(rows).set_index("book_name")
        assert set(q.index) == {"a", "b"}
        assert (q.loc["a", "line"], q.loc["a", "over_price"],
                q.loc["a", "under_price"]) == (6.0, 105, -125)
        assert (q.loc["b", "over_price"], q.loc["b", "under_price"]) == \
            (-105, -115)

    def test_quote_sources_marks_mixed_games(self):
        q = pd.DataFrame({"game_id": [1, 1, 2, 3, 3],
                          "source": ["x", "x", "y", "x", "y"]})
        assert quote_sources(q).to_dict() == {1: "x", 2: "y", 3: "mixed"}


# ── Calibration and the pass rule ──────────────────────────────────

class TestCalibration:
    def test_hand_computed(self):
        tp = np.zeros((2, 26))
        tp[0, 5], tp[0, 6] = 0.5, 0.5
        tp[1, 6], tp[1, 12] = 0.5, 0.5
        cal = total_calibration(tp, np.array([5, 13]))
        # predicted: t=5 .25, t=6 .5, 11+ .25; observed: t=5 .5, 11+ .5
        assert cal["pred"][5] == 0.25 and cal["pred"][6] == 0.5
        assert cal["pred"][11] == 0.25
        assert cal["obs"][5] == 0.5 and cal["obs"][11] == 0.5
        assert cal["max_gap"] == pytest.approx(0.5)
        # counts: expected .5, 1, .5 vs observed 1, 0, 1
        assert cal["chi2"] == pytest.approx(0.25 / 0.5 + 1.0 / 1.0
                                            + 0.25 / 0.5, rel=1e-6)


def _summary(nll, diff, se, folds, mc_diff, mc_se, n=1000, vs=None):
    return {"nll": nll, "diff": diff, "diff_se": se, "folds_won": folds,
            "market_check": {"n": n, "diff": mc_diff, "diff_se": mc_se},
            "vs": vs or {}}


class TestPassRule:
    def test_each_condition_is_needed(self):
        ok = _summary(2.17, -0.004, 0.001, 4, 0.001, 0.001)
        assert v3_passes(ok)
        assert not v3_passes({**ok, "diff": -0.0019})            # < 2 SE
        assert not v3_passes({**ok, "folds_won": 3})              # folds
        assert not v3_passes(_summary(2.17, -0.004, 0.001, 4,
                                      0.0025, 0.001))             # worse
        assert not v3_passes(_summary(2.17, -0.004, 0.001, 4,
                                      0.0, 0.001, n=150))         # too few

    def test_not_worse_is_the_95_percent_bound(self):
        assert not_worse_than_market({"n": 500, "diff": 0.00196,
                                      "diff_se": 0.001})
        assert not not_worse_than_market({"n": 500, "diff": 0.00197,
                                          "diff_se": 0.001})

    def test_choice(self):
        good = dict(diff=-0.004, se=0.001, folds=5, mc_diff=0.0, mc_se=0.001)
        s = {"T1": _summary(2.1750, good["diff"], good["se"], 5, 0.0, 0.001,
                            vs={"T3": {"diff": 0.0005, "se": 0.0008}}),
             "T2": _summary(2.1790, 0.0, 0.001, 2, 0.0, 0.001),
             "T3": _summary(2.1745, good["diff"], good["se"], 5, 0.0, 0.001)}
        assert choose_v3(s) == "T1"          # simpler, within 1 SE of T3
        s["T1"]["vs"]["T3"]["se"] = 0.0002   # now T3 is clearly better
        assert choose_v3(s) == "T3"
        s["T3"]["market_check"]["diff"] = 0.01
        s["T1"]["market_check"]["diff"] = 0.01
        assert choose_v3(s) is None          # nobody passes the market check
        assert choose_v3({"T0": _summary(2.0, -1, 0.001, 5, 0, 0.001)}) is None


# ── Walk-forward wiring, database and booster faked ────────────────

def _synthetic_c(**kw):
    """_synthetic with variant C's columns (A + the role features)."""
    from features.goalie_role import ROLE_FEATURES
    Xh, Xa, y_h, y_a, meta, _ = _synthetic(**kw)
    rng = np.random.default_rng(1)
    extra = len(ROLE_FEATURES)
    Xh = np.hstack([Xh, rng.normal(0.5, 0.2, (len(meta), extra))])
    Xa = np.hstack([Xa, rng.normal(0.5, 0.2, (len(meta), extra))])
    return Xh, Xa, y_h, y_a, meta, list(T.ATTACK_FEATURES) + ROLE_FEATURES


@pytest.fixture()
def fake_v3(monkeypatch):
    """Fake data load and booster. The 'booster' multiplies each side's
    offset rate by exp(0.2 * (column 0 - 1)); every fit is recorded with
    its training rows, its offsets (init rates) and its column count."""
    state = {"data": _synthetic_c()}
    fits, rho_calls, weight_calls = [], [], []
    real_rho, real_w = T.fit_dc_rho, T.fit_margin_weights

    def fake_fit(Xh, Xa, y_home, y_away, env_home, env_away, train_idx,
                 dates, is_playoff=None):
        fits.append({"train": np.asarray(train_idx).copy(),
                     "off_h": np.asarray(env_home)[train_idx].copy(),
                     "ncol": Xh.shape[1]})
        return {"model": None, "scale": 1.0, "iters": 1}

    def fake_predict(fm, Xh, Xa, env_home, env_away):
        return (np.clip(env_home * np.exp(0.2 * (Xh[:, 0] - 1.0)), *T.LAMBDA_CLIP),
                np.clip(env_away * np.exp(0.2 * (Xa[:, 0] - 1.0)), *T.LAMBDA_CLIP))

    def spy_rho(pmf_h, pmf_a, y_home, y_away, w):
        rho_calls.append(np.asarray(y_home).copy())
        return real_rho(pmf_h, pmf_a, y_home, y_away, w)

    def spy_w(pmf_h, pmf_a, y_home, y_away):
        weight_calls.append(np.asarray(y_home).copy())
        return real_w(pmf_h, pmf_a, y_home, y_away)

    def load(variant="A", with_roles=False):
        assert variant == "C" and with_roles
        return tuple(x.copy() if hasattr(x, "copy") else x
                     for x in state["data"])

    def no_db(*a, **k):
        raise AssertionError("quotes were passed; nothing may load")

    monkeypatch.setattr(T, "load_totals_dataset", load)
    monkeypatch.setattr(T, "fit_totals_fold", fake_fit)
    monkeypatch.setattr(T, "predict_lambdas", fake_predict)
    monkeypatch.setattr(T, "fit_dc_rho", spy_rho)
    monkeypatch.setattr(T, "fit_margin_weights", spy_w)
    monkeypatch.setattr(T, "load_v3_quotes", no_db)
    monkeypatch.setattr(T, "MIN_PRICED_TRAIN", 50)

    def set_data(data):
        state["data"] = data
    return {"set": set_data, "fits": fits, "rho": rho_calls,
            "weights": weight_calls}


NO_QUOTES = pd.DataFrame(columns=["game_id", "book_name", "line",
                                  "over_price", "under_price", "source"])


def _quotes(meta, every=2, seed=0):
    rng = np.random.default_rng(seed)
    g = meta["game_id"].to_numpy()[::every]
    over = rng.choice([-130, -115, -105, 100, 110], len(g))
    under = np.where(over < 0, 100 + rng.integers(0, 20, len(g)), -120)
    return pd.DataFrame({"game_id": g, "book_name": "x", "line": 5.5,
                         "over_price": over, "under_price": under,
                         "source": "test"})


class TestV3Wiring:
    def test_no_prices_means_market_variants_equal_their_fallbacks(
            self, fake_v3):
        res = T.run_totals_v3(market_quotes=NO_QUOTES)
        oof = res["oof"]
        np.testing.assert_array_equal(oof["nll_T1"], oof["nll_T0"])
        np.testing.assert_array_equal(oof["nll_M"], oof["nll_B"])
        assert res["summaries"]["T1"]["market_check"]["n"] == 0
        assert res["chosen"] is None

    def test_unpriced_games_keep_t0_and_priced_games_change(self, fake_v3):
        data = _synthetic_c()
        fake_v3["set"](data)
        quotes = _quotes(data[4])
        res = T.run_totals_v3(market_quotes=quotes)
        oof = res["oof"]
        p = oof["priced"].to_numpy(bool)
        assert p.any() and (~p).any()
        np.testing.assert_array_equal(oof.loc[~p, "nll_T1"],
                                      oof.loc[~p, "nll_T0"])
        assert not np.allclose(oof.loc[p, "nll_T1"], oof.loc[p, "nll_T0"])
        s = res["summaries"]
        for v in T.V3_VARIANTS:
            assert s[v]["market_check"]["n"] == int(p.sum())
            assert set(s[v]["market_by_source"]) == {"test"}
            assert np.isfinite(s[v]["nll"]) and np.isfinite(s[v]["diff_se"])
        # M copies the market at the posted line
        assert s["M"]["market_check"]["diff"] == pytest.approx(0.0, abs=1e-9)
        assert s["T2"]["baseline"] == "B_dc" and s["T1"]["baseline"] == "B"

    def test_shape_parameters_fit_on_training_games_only(self, fake_v3):
        from models.baseline import walk_forward_folds
        data = _synthetic_c()
        fake_v3["set"](data)
        quotes = _quotes(data[4])
        res = T.run_totals_v3(market_quotes=quotes)
        folds = walk_forward_folds(data[4])
        assert len(fake_v3["rho"]) == len(fake_v3["weights"]) == len(folds)
        for got_r, got_w, fold in zip(fake_v3["rho"], fake_v3["weights"],
                                      folds):
            np.testing.assert_array_equal(got_r, data[2][fold.train_idx])
            np.testing.assert_array_equal(got_w, data[2][fold.train_idx])

        # rewrite the last season's outcomes (no fold trains on it): every
        # fold's weights and rho stay the same
        Xh, Xa, y_h, y_a, meta, names = _synthetic_c()
        last = (meta["season"] == meta["season"].max()).to_numpy()
        y_h[last], y_a[last] = 7.0, 0.0
        meta["total"] = (y_h + y_a + (y_h == y_a)).astype(int)
        fake_v3["set"]((Xh, Xa, y_h, y_a, meta, names))
        res2 = T.run_totals_v3(market_quotes=quotes)
        assert res2["rho"] == res["rho"]
        for f1, f2 in zip(res["folds"], res2["folds"]):
            assert f1["margin_weights"] == f2["margin_weights"]

    def test_market_booster_trains_on_priced_training_games_from_market_rates(
            self, fake_v3):
        from models.baseline import walk_forward_folds
        data = _synthetic_c()
        fake_v3["set"](data)
        meta = data[4]
        quotes = _quotes(meta)
        T.run_totals_v3(market_quotes=quotes, variants=("T0", "T1"))
        priced = meta["game_id"].isin(quotes["game_id"]).to_numpy()
        folds = walk_forward_folds(meta)
        fits = fake_v3["fits"]
        assert len(fits) == 2 * len(folds)            # env + market per fold
        for i, fold in enumerate(folds):
            env_fit, mkt_fit = fits[2 * i], fits[2 * i + 1]
            np.testing.assert_array_equal(env_fit["train"], fold.train_idx)
            want = fold.train_idx[priced[fold.train_idx]]
            np.testing.assert_array_equal(mkt_fit["train"], want)
            # offsets are the market rates, not the environment rates
            assert not np.allclose(mkt_fit["off_h"],
                                   env_fit["off_h"][priced[fold.train_idx]])
            assert env_fit["ncol"] == mkt_fit["ncol"] == len(T.ATTACK_FEATURES)

    def test_t3_boosters_use_the_role_columns(self, fake_v3):
        from features.goalie_role import ROLE_FEATURES
        data = _synthetic_c()
        fake_v3["set"](data)
        T.run_totals_v3(market_quotes=_quotes(data[4]), variants=("T3",))
        ncols = {f["ncol"] for f in fake_v3["fits"]}
        assert ncols == {len(T.ATTACK_FEATURES),
                         len(T.ATTACK_FEATURES) + len(ROLE_FEATURES)}

    def test_a_games_score_ignores_later_prices(self, fake_v3):
        """Changing (or deleting) the prices of games on later dates leaves
        every earlier validation game's market-variant score unchanged
        except through training rows (none of which are validation-season
        games): only the moved games' own scores may change."""
        data = _synthetic_c()
        fake_v3["set"](data)
        meta = data[4]
        quotes = _quotes(meta)
        before = T.run_totals_v3(market_quotes=quotes, variants=("T1",))["oof"]
        last_day = meta.loc[meta["season"] == meta["season"].max(), "date"].max()
        late = meta.loc[meta["date"] == last_day, "game_id"]
        q2 = quotes[~quotes["game_id"].isin(late)]
        after = T.run_totals_v3(market_quotes=q2, variants=("T1",))["oof"]
        keep = ~before["game_id"].isin(late)
        np.testing.assert_allclose(after.loc[keep, "nll_T1"],
                                   before.loc[keep, "nll_T1"])

    def test_unknown_variant_refused(self, fake_v3):
        with pytest.raises(ValueError):
            T.run_totals_v3(market_quotes=NO_QUOTES, variants=("T9",))


class TestCommandLineV3:
    def test_v3_flag_runs_the_experiment_only(self, monkeypatch):
        seen = []
        monkeypatch.setattr(T, "run_totals_v3", lambda: seen.append("v3"))
        monkeypatch.setattr(T, "run_totals", lambda **k: pytest.fail(
            "--v3 must not run (or register) the v2 evaluation"))
        T.main(["--v3"])
        assert seen == ["v3"]


@requires_db
class TestV3Data:
    def test_quote_sources_are_clean(self):
        """Read-only: Unibet quotes are all at 5.5 with sane margins,
        odds_history closing quotes are strictly pre-game, and every
        source the pre-registration names is present."""
        q = T.load_v3_quotes()
        assert {"unibet_espn", "odds_history", "draftkings_espn"} <= \
            set(q["source"])
        uni = q[q["source"] == "unibet_espn"]
        assert (uni["line"].astype(float) == 5.5).all()
        m = T.market_over_probs(q)
        assert m["fair_over"].between(0.2, 0.8).all()
