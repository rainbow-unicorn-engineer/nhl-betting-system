"""
Tests for models/totals.py — every PMF number is hand-computed.

The walk-forward and production wiring tests swap the database load and
the LightGBM booster for small deterministic fakes, so they run anywhere
and check exactly one thing each: margin weights come from training games
only, and the drift correction never looks ahead.
"""
import numpy as np
import pandas as pd
import pytest

import models.totals as T
from config.settings import check_db_connection
from models.totals import (MARGIN_WEIGHTS, apply_drift, booster_adjustment,
                           drift_shift, env_rates, expected_total,
                           fit_margin_weights, joint_pmf, market_check,
                           market_over_probs, nll_of_totals, poisson_pmf,
                           prob_over, production_drift_shift, total_pmf)

requires_db = pytest.mark.skipif(not check_db_connection(),
                                 reason="database not reachable")


class TestPMFMachinery:
    def test_poisson_pmf_normalized_and_shaped(self):
        pmf = poisson_pmf(np.array([2.5, 3.5]))
        assert pmf.shape == (2, 13)
        np.testing.assert_allclose(pmf.sum(axis=1), 1.0)
        # mode of Poisson(2.5) is 2
        assert pmf[0].argmax() == 2

    def test_total_pmf_hand_computed_with_ot_shift(self):
        # Independent joint (margin_weights=None):
        # H ~ [0.5, 0.5] over {0,1}; A ~ [0.4, 0.6] over {0,1}
        # (0,0) p=.20 tie -> T=1;  (0,1) p=.30 -> T=1
        # (1,0) p=.20 -> T=1;      (1,1) p=.30 tie -> T=3
        ph = np.array([[0.5, 0.5]])
        pa = np.array([[0.4, 0.6]])
        tp = total_pmf(ph, pa, margin_weights=None)
        assert tp.shape == (1, 4)
        np.testing.assert_allclose(tp[0], [0.0, 0.7, 0.0, 0.3], atol=1e-12)
        np.testing.assert_allclose(tp.sum(axis=1), 1.0)

    def test_no_mass_on_even_regulation_ties(self):
        # With the OT shift, T=0 is impossible (0-0 becomes 1)
        ph = poisson_pmf(np.array([3.0]))
        pa = poisson_pmf(np.array([2.7]))
        tp = total_pmf(ph, pa)
        assert tp[0, 0] == 0.0
        np.testing.assert_allclose(tp.sum(axis=1), 1.0)

    def test_prob_over_and_push(self):
        tp = np.array([[0.0, 0.7, 0.0, 0.3]])
        p_over, p_push = prob_over(tp, [1.5])
        assert p_over[0] == pytest.approx(0.3) and p_push[0] == 0.0
        p_over, p_push = prob_over(tp, [1.0])   # integer line: push at 1
        assert p_over[0] == pytest.approx(0.3)
        assert p_push[0] == pytest.approx(0.7)

    def test_expected_total(self):
        tp = np.array([[0.0, 0.7, 0.0, 0.3]])
        assert expected_total(tp)[0] == pytest.approx(0.7 + 0.9)

    def test_nll_of_totals(self):
        tp = np.array([[0.0, 0.7, 0.0, 0.3]])
        assert nll_of_totals(tp, np.array([3]))[0] == pytest.approx(-np.log(0.3))


class TestMarginReweighting:
    def test_hand_computed_tie_weight(self):
        # Same cells as above; ties weighted x2, other margins x1:
        # .20*2=.40, .30, .20, .30*2=.60 -> sum 1.5 -> /1.5
        # T=1: (.40 + .30 + .20)/1.5 = .6;  T=3: .60/1.5 = .4
        ph = np.array([[0.5, 0.5]])
        pa = np.array([[0.4, 0.6]])
        w = np.array([2.0, 1.0, 1.0, 1.0, 1.0])
        np.testing.assert_allclose(total_pmf(ph, pa, w)[0],
                                   [0.0, 0.6, 0.0, 0.4], atol=1e-12)
        joint = joint_pmf(ph, pa, w)[0]
        np.testing.assert_allclose(joint, [[0.4 / 1.5, 0.3 / 1.5],
                                           [0.2 / 1.5, 0.6 / 1.5]])

    def test_probabilities_still_sum_to_one(self):
        rng = np.random.default_rng(7)
        ph = poisson_pmf(rng.uniform(1.5, 4.5, 50))
        pa = poisson_pmf(rng.uniform(1.5, 4.5, 50))
        for w in (MARGIN_WEIGHTS, rng.uniform(0.2, 3.0, 5), None):
            np.testing.assert_allclose(joint_pmf(ph, pa, w).sum(axis=(1, 2)),
                                       1.0)
            tp = total_pmf(ph, pa, w)
            np.testing.assert_allclose(tp.sum(axis=1), 1.0)
            assert (tp >= 0).all()

    def test_default_is_the_fitted_weights(self):
        ph = poisson_pmf(np.array([3.0]))
        pa = poisson_pmf(np.array([2.8]))
        np.testing.assert_allclose(total_pmf(ph, pa),
                                   total_pmf(ph, pa, MARGIN_WEIGHTS))
        assert not np.allclose(total_pmf(ph, pa), total_pmf(ph, pa, None))

    def test_all_ones_is_independence(self):
        ph = poisson_pmf(np.array([3.0, 2.2]))
        pa = poisson_pmf(np.array([2.8, 3.9]))
        np.testing.assert_allclose(total_pmf(ph, pa, np.ones(5)),
                                   total_pmf(ph, pa, None))

    def test_default_weights_move_mass_toward_ties_and_off_one_goal_games(self):
        ph = poisson_pmf(np.array([3.0]))
        pa = poisson_pmf(np.array([2.8]))
        b = T.margin_buckets(ph.shape[1])
        indep, fitted = joint_pmf(ph, pa, None)[0], joint_pmf(ph, pa)[0]
        assert fitted[b == 0].sum() > indep[b == 0].sum()        # ties up
        assert fitted[b == 1].sum() < indep[b == 1].sum()        # 1-goal down

    @pytest.mark.parametrize("bad", [[1, 1, 1, 1], [1, 1, 0, 1, 1],
                                     [1, np.nan, 1, 1, 1]])
    def test_bad_weights_are_refused(self, bad):
        ph = poisson_pmf(np.array([3.0]))
        with pytest.raises(ValueError):
            total_pmf(ph, ph, np.array(bad, dtype=float))

    def test_fit_recovers_known_weights(self):
        # Scores drawn from a known reweighted joint: the maximum-
        # likelihood fit must give the weights back
        rng = np.random.default_rng(11)
        true_w = np.array([1.15, 0.5, 0.7, 1.3, 1.0])
        lam_h, lam_a = rng.uniform(2.4, 3.6, 40000), rng.uniform(2.2, 3.4, 40000)
        ph, pa = poisson_pmf(lam_h), poisson_pmf(lam_a)
        joint = joint_pmf(ph, pa, true_w).reshape(len(lam_h), -1)
        cells = (joint.cumsum(axis=1) < rng.random((len(lam_h), 1))).sum(axis=1)
        h, a = np.divmod(cells, ph.shape[1])
        fitted = fit_margin_weights(ph, pa, h, a)
        assert fitted[-1] == 1.0
        np.testing.assert_allclose(fitted, true_w, atol=0.05)

    def test_fit_on_independent_scores_gives_ones(self):
        rng = np.random.default_rng(3)
        lam = rng.uniform(2.5, 3.5, 30000)
        h, a = rng.poisson(lam), rng.poisson(lam)
        pm = poisson_pmf(lam)
        np.testing.assert_allclose(fit_margin_weights(pm, pm, h, a),
                                   np.ones(5), atol=0.05)


class TestDriftCorrection:
    DATES = ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03",
             "2024-01-02", "2024-01-03"]
    SEASONS = [1, 1, 1, 1, 2, 2]
    ADJ = [0.1, 0.3, 0.5, 9.0, 1.0, 2.0]

    def test_hand_computed(self):
        got = drift_shift(self.SEASONS, self.DATES, self.ADJ, prior_games=0)
        # season 1: day 1 has no earlier games -> 0; day 2 -> mean(.1,.3);
        # day 3 -> mean(.1,.3,.5). Season 2 starts over at 0.
        np.testing.assert_allclose(got, [0, 0, 0.2, 0.3, 0, 1.0])

    def test_prior_shrinks_toward_zero(self):
        got = drift_shift(self.SEASONS, self.DATES, self.ADJ, prior_games=2)
        # day 3 of season 1: sum .9 over 3 games + 2 pseudo-games of 0
        assert got[3] == pytest.approx(0.9 / 5)
        assert got[2] == pytest.approx(0.4 / 4)

    def test_no_look_ahead_same_day_and_later_games_never_count(self):
        base = drift_shift(self.SEASONS, self.DATES, self.ADJ)
        for i in range(len(self.ADJ)):
            bumped = list(self.ADJ)
            bumped[i] += 100.0
            got = drift_shift(self.SEASONS, self.DATES, bumped)
            same_or_earlier = [j for j in range(len(self.ADJ))
                               if self.SEASONS[j] != self.SEASONS[i]
                               or self.DATES[j] <= self.DATES[i]]
            np.testing.assert_allclose(got[same_or_earlier],
                                       base[same_or_earlier])

    def test_row_order_does_not_matter(self):
        order = [5, 3, 0, 4, 2, 1]
        got = drift_shift(np.take(self.SEASONS, order),
                          np.take(self.DATES, order),
                          np.take(self.ADJ, order))
        np.testing.assert_allclose(got, drift_shift(self.SEASONS, self.DATES,
                                                    self.ADJ)[order])

    def test_booster_adjustment_and_apply(self):
        adj = booster_adjustment([3.0 * np.e], [2.0], [3.0], [2.0 * np.e])
        assert adj[0] == pytest.approx(0.0)        # +1 and -1 on log scale
        lh, la = apply_drift(np.array([3.0]), np.array([2.0]), np.array([0.1]))
        assert lh[0] == pytest.approx(3.0 * np.exp(-0.1))
        assert la[0] == pytest.approx(2.0 * np.exp(-0.1))

    def test_production_shift(self):
        prod = {"drift": {20262027: (-1.0, 75)}}
        assert production_drift_shift(prod, 20262027) == pytest.approx(
            -1.0 / (75 + T.DRIFT_PRIOR_GAMES))
        assert production_drift_shift(prod, 20272028) == 0.0
        assert production_drift_shift(prod, None) == 0.0


class TestMarketCheck:
    def test_market_over_probs_main_line_and_no_vig(self):
        quotes = pd.DataFrame([
            # game 1: two books at 6.5, one at 5.5 -> 6.5 is the main line
            (1, "a", 6.5, -110, -110),      # no-vig .5
            (1, "b", 6.5, -120, 100),       # .54545/(.54545+.5) = .52174
            (1, "c", 5.5, -160, 135),
            # game 2: one book per line; 6.5 is nearer 50% (.457 vs .6)
            (2, "a", 5.5, -150, 150),       # .6/(.6+.4) = .6
            (2, "b", 6.5, 110, -130),       # .47619/(.47619+.56522) = .45726
            (3, "a", None, -110, -110),     # no line: ignored
        ], columns=["game_id", "book_name", "line", "over_price", "under_price"])
        m = market_over_probs(quotes).set_index("game_id")
        assert set(m.index) == {1, 2}
        assert m.loc[1, "line"] == 6.5 and m.loc[1, "n_books"] == 2
        assert m.loc[1, "fair_over"] == pytest.approx((0.5 + 0.521739) / 2,
                                                      abs=1e-5)
        assert m.loc[2, "line"] == 6.5
        assert m.loc[2, "fair_over"] == pytest.approx(0.457256, abs=1e-5)

    def test_market_check_hand_computed(self):
        # model .6 over vs market .5; one over, one under, one push (dropped)
        out = market_check(p_over=[0.6, 0.6, 0.5], p_push=[0, 0, 0.2],
                           market_over=[0.5, 0.5, 0.5],
                           totals=[7, 5, 6], lines=[6.5, 6.5, 6.0],
                           min_games=1)
        assert out["n"] == 2
        assert out["model_log_loss"] == pytest.approx(
            (-np.log(0.6) - np.log(0.4)) / 2)
        assert out["market_log_loss"] == pytest.approx(np.log(2))
        assert out["diff"] == pytest.approx(out["model_log_loss"] - np.log(2))
        assert out["beats_market"] is False

    def test_push_mass_is_conditioned_out(self):
        # P(over)=.4, P(push)=.2 -> P(over | no push) = .5 = the market
        out = market_check([0.4], [0.2], [0.5], [7], [6.0], min_games=1)
        assert out["diff"] == pytest.approx(0.0)

    def test_needs_enough_games_and_confidence(self):
        assert market_check([], [], [], [], [])["n"] == 0
        few = market_check([0.9] * 10, [0] * 10, [0.5] * 10, [7] * 10,
                           [6.5] * 10, min_games=200)
        assert few["beats_market"] is None
        rng = np.random.default_rng(5)
        p = rng.uniform(0.2, 0.8, 3000)
        totals = np.where(rng.random(3000) < p, 7, 5)
        sure = market_check(p, np.zeros(3000), np.full(3000, 0.5), totals,
                            np.full(3000, 6.5))
        assert sure["beats_market"] is True


class TestEnvRates:
    def test_trailing_mean_and_prior_blend(self):
        dates = pd.Series(pd.date_range("2024-01-01", periods=1000, freq="6h"))
        y = np.full(1000, 4.0)
        env = env_rates(dates, y, prior=2.0)
        # First rows: no history -> pure prior
        assert env[0] == pytest.approx(2.0)
        # Late rows: prior weight is fixed, data dominates but blend remains
        assert 3.0 < env[-1] < 4.0
        assert env[-1] > env[100]   # monotone approach toward the data

    def test_point_in_time_no_same_day_leak(self):
        dates = pd.Series(pd.to_datetime(
            ["2024-01-01"] * 5 + ["2024-01-02"] * 5 + ["2024-01-03"] * 5))
        y1 = np.array([3.0] * 5 + [3.0] * 5 + [3.0] * 5)
        y2 = y1.copy()
        y2[10:] = 99.0        # change only Jan-3 outcomes
        env1 = env_rates(dates, y1, prior=3.0)
        env2 = env_rates(dates, y2, prior=3.0)
        # Jan-3 rows' own outcomes must not affect their own env values
        np.testing.assert_allclose(env1[10:], env2[10:])


# ── Walk-forward and production wiring, database and booster faked ──

SYNTH_WEIGHTS = np.array([1.4, 0.6, 0.8, 1.2, 1.0])   # hockey-like shape


def _synthetic(seed=0, n_seasons=3, days=40, per_day=8):
    """load_totals_dataset's shape with random data: seasons of `days`
    dates x `per_day` games, features named like the real ones, scores
    drawn from a margin-reweighted joint (SYNTH_WEIGHTS) so the margin
    fix has something to find."""
    rng = np.random.default_rng(seed)
    meta = pd.concat([pd.DataFrame({
        "season": 20202021 + s * 10001,
        "date": np.repeat(pd.date_range(f"{2020 + s}-10-10", periods=days,
                                        freq="D").values, per_day)})
        for s in range(n_seasons)], ignore_index=True)
    n = len(meta)
    meta.insert(0, "game_id", np.arange(1, n + 1))
    ph, pa = poisson_pmf(np.full(n, 3.0)), poisson_pmf(np.full(n, 2.8))
    joint = joint_pmf(ph, pa, SYNTH_WEIGHTS).reshape(n, -1)
    cells = (joint.cumsum(axis=1) < rng.random((n, 1))).sum(axis=1)
    y_h, y_a = (x.astype(float) for x in np.divmod(cells, ph.shape[1]))
    meta["total"] = (y_h + y_a + (y_h == y_a)).astype(int)
    meta["market_line"] = np.nan
    meta["is_playoff"] = False
    Xh = rng.normal(1.0, 0.3, (n, len(T.ATTACK_FEATURES)))
    Xa = rng.normal(1.0, 0.3, (n, len(T.ATTACK_FEATURES)))
    return Xh, Xa, y_h, y_a, meta, list(T.ATTACK_FEATURES)


@pytest.fixture()
def fake_model(monkeypatch):
    """Replace the data load and the booster: the 'model' multiplies each
    side's environment rate by exp(0.2 * (feature 0 - 1)), so a game's
    rates depend only on its own features. Returns a setter for the data
    and the list of fit_margin_weights calls (their y_home arrays)."""
    state = {"data": _synthetic()}
    weight_calls = []
    real_fit = T.fit_margin_weights

    def spy_fit(pmf_h, pmf_a, y_home, y_away):
        weight_calls.append(np.asarray(y_home).copy())
        return real_fit(pmf_h, pmf_a, y_home, y_away)

    def fake_predict(fm, Xh, Xa, env_home, env_away):
        return (np.clip(env_home * np.exp(0.2 * (Xh[:, 0] - 1.0)), *T.LAMBDA_CLIP),
                np.clip(env_away * np.exp(0.2 * (Xa[:, 0] - 1.0)), *T.LAMBDA_CLIP))

    def no_db():
        raise AssertionError("market_quotes was passed; nothing may load "
                             "from the database")

    monkeypatch.setattr(T, "load_totals_dataset", lambda: tuple(
        x.copy() if hasattr(x, "copy") else x for x in state["data"]))
    monkeypatch.setattr(T, "fit_totals_fold", lambda *a, **k:
                        {"model": None, "scale": 1.0, "iters": 1})
    monkeypatch.setattr(T, "predict_lambdas", fake_predict)
    monkeypatch.setattr(T, "fit_margin_weights", spy_fit)
    monkeypatch.setattr(T, "load_market_quotes", no_db)

    def set_data(data):
        state["data"] = data
    return set_data, weight_calls


NO_QUOTES = pd.DataFrame(columns=["game_id", "book_name", "line",
                                  "over_price", "under_price"])


class TestWalkForwardWiring:
    def test_margin_weights_fitted_on_each_folds_training_games_only(
            self, fake_model):
        from models.baseline import walk_forward_folds
        set_data, calls = fake_model
        data = _synthetic()
        set_data(data)
        res = T.run_totals(register=False, market_quotes=NO_QUOTES)
        y_h, meta = data[2], data[4]
        folds = walk_forward_folds(meta)
        assert len(calls) == len(folds) == 2
        for got, fold in zip(calls, folds):
            np.testing.assert_array_equal(got, y_h[fold.train_idx])
            assert meta["season"].iloc[fold.train_idx].max() < fold.val_season

        # the last season's outcomes are no fold's training data: changing
        # them leaves every fold's weights unchanged
        Xh, Xa, y_h2, y_a2, meta2, names = _synthetic()
        last = (meta2["season"] == meta2["season"].max()).to_numpy()
        y_h2[last], y_a2[last] = 0.0, 0.0
        meta2["total"] = (y_h2 + y_a2 + (y_h2 == y_a2)).astype(int)
        set_data((Xh, Xa, y_h2, y_a2, meta2, names))
        res2 = T.run_totals(register=False, market_quotes=NO_QUOTES)
        for f1, f2 in zip(res["folds"], res2["folds"]):
            assert f1["margin_weights"] == f2["margin_weights"]

    def test_drift_correction_uses_only_earlier_games(self, fake_model):
        set_data, _ = fake_model
        set_data(_synthetic())
        before = T.run_totals(register=False, market_quotes=NO_QUOTES)["oof"]

        # push the model's predictions for the LAST date of each validation
        # season far up: only those games' own scores may change
        Xh, Xa, y_h, y_a, meta, names = _synthetic()
        last_day = (meta.groupby("season")["date"].transform("max")
                    == meta["date"]).to_numpy()
        Xh[last_day, 0] += 3.0
        Xa[last_day, 0] += 3.0
        set_data((Xh, Xa, y_h, y_a, meta, names))
        after = T.run_totals(register=False, market_quotes=NO_QUOTES)["oof"]

        earlier = ~before["game_id"].isin(meta.loc[last_day, "game_id"])
        np.testing.assert_allclose(after.loc[earlier, "nll"],
                                   before.loc[earlier, "nll"])
        assert not np.allclose(after.loc[~earlier, "nll"],
                               before.loc[~earlier, "nll"])

    def test_gate_compares_like_with_like_and_reports_market_check(
            self, fake_model):
        set_data, _ = fake_model
        data = _synthetic()
        set_data(data)
        meta = data[4]
        val = meta[meta["season"] > meta["season"].min()]
        priced = val.iloc[::3]
        quotes = pd.DataFrame({"game_id": priced["game_id"], "book_name": "x",
                               "line": 5.5, "over_price": -110,
                               "under_price": -110})
        res = T.run_totals(register=False, market_quotes=quotes)
        pooled = res["pooled"]
        for k in ("nll", "baseline_nll", "nll_v1", "baseline_nll_v1",
                  "nll_diff_se"):
            assert np.isfinite(pooled[k])
        assert pooled["gate_passed"] == (pooled["nll"] < pooled["baseline_nll"])
        # the baseline gets the margin fix too: on scores with a real
        # margin shape it must beat the independent-joint baseline
        assert pooled["baseline_nll"] < pooled["baseline_nll_v1"]
        mc = pooled["market_check"]
        assert mc["n"] == len(priced)          # a .5 line never pushes
        assert mc["market_log_loss"] == pytest.approx(np.log(2))
        assert sum(f["n_priced"] for f in res["folds"]) == mc["n"]

    def test_no_prices_means_no_market_check(self, fake_model):
        set_data, _ = fake_model
        set_data(_synthetic())
        res = T.run_totals(register=False, market_quotes=NO_QUOTES)
        assert res["pooled"]["market_check"] == {"n": 0, "beats_market": None}


class TestProductionWiring:
    CUTOFF = pd.Timestamp("2022-11-01")      # mid third synthetic season

    def test_weights_and_drift_use_only_games_before_the_cutoff(
            self, fake_model):
        set_data, calls = fake_model
        data = _synthetic()
        set_data(data)
        prod = T.fit_production(self.CUTOFF)
        meta = data[4]
        train = (meta["date"] < self.CUTOFF).to_numpy()
        np.testing.assert_array_equal(calls[-1], data[2][train])
        counts = meta[train].groupby("season").size().to_dict()
        assert {s: c for s, (_, c) in prod["drift"].items()} == counts

        # rewrite every game on or after the cutoff: nothing frozen moves
        Xh, Xa, y_h, y_a, meta2, names = _synthetic()
        Xh[~train, 0] += 5.0
        y_h[~train] = 9.0
        set_data((Xh, Xa, y_h, y_a, meta2, names))
        prod2 = T.fit_production(self.CUTOFF)
        np.testing.assert_allclose(prod2["margin_weights"],
                                   prod["margin_weights"])
        assert prod2["drift"] == prod["drift"]
        assert (prod2["env_home"], prod2["env_away"]) == \
            (prod["env_home"], prod["env_away"])

    def test_scoring_applies_the_slate_seasons_drift(self, fake_model):
        set_data, _ = fake_model
        data = _synthetic()
        set_data(data)
        prod = T.fit_production(self.CUTOFF)
        season = int(data[4]["season"].max())     # has games before cutoff
        X = np.ones((4, len(T.ATTACK_FEATURES)))
        out = T.score_production(prod, X, X, T.ATTACK_FEATURES, season=season)
        plain = T.score_production(prod, X, X, T.ATTACK_FEATURES)
        shift = production_drift_shift(prod, season)
        assert shift != 0.0
        assert out["drift_shift"] == pytest.approx(shift)
        assert plain["drift_shift"] == 0.0
        np.testing.assert_allclose(out["lambda_home"],
                                   plain["lambda_home"] * np.exp(-shift))
        np.testing.assert_allclose(out["pmf_total"].sum(axis=1), 1.0)
        np.testing.assert_allclose(
            out["pmf_total"],
            total_pmf(out["pmf_home"], out["pmf_away"], prod["margin_weights"]))


class TestCommandLine:
    def test_help_runs_nothing(self, monkeypatch, capsys):
        monkeypatch.setattr(T, "run_totals", lambda **k: pytest.fail(
            "--help must not start the evaluation"))
        with pytest.raises(SystemExit) as exit_:
            T.main(["--help"])
        assert exit_.value.code == 0
        assert "--no-register" in capsys.readouterr().out

    @pytest.mark.parametrize("argv,register", [([], True),
                                               (["--no-register"], False)])
    def test_register_flag(self, monkeypatch, argv, register):
        seen = {}
        monkeypatch.setattr(T, "run_totals",
                            lambda register=True: seen.update(r=register))
        T.main(argv)
        assert seen == {"r": register}


@requires_db
class TestDataset:
    def test_regulation_goals_and_shapes(self):
        from models.totals import ATTACK_FEATURES, load_totals_dataset
        Xh, Xa, y_h, y_a, meta, names = load_totals_dataset()
        assert Xh.shape == Xa.shape == (len(meta), len(ATTACK_FEATURES))
        assert names == ATTACK_FEATURES
        # regulation goals: non-negative ints, and reg total <= settlement
        assert (y_h >= 0).all() and (y_a >= 0).all()
        assert ((y_h + y_a) <= meta["total"].to_numpy()).all()
        # in OT/SO games the settlement total is exactly reg total + 1
        # (spot property: no game has settlement > reg + 1)
        assert ((meta["total"].to_numpy() - (y_h + y_a)) <= 1).all()

    def test_is_home_column_constant_per_matrix(self):
        from models.totals import ATTACK_FEATURES, load_totals_dataset
        Xh, Xa, *_ = load_totals_dataset()
        j = ATTACK_FEATURES.index("is_home")
        assert (Xh[:, j] == 1.0).all() and (Xa[:, j] == 0.0).all()

    def test_default_margin_weights_match_a_fit_on_the_history(self):
        """MARGIN_WEIGHTS (used by the checker and the alerts) must be what
        the fit gives on the stored completed games, within the tolerance
        fit_production warns at."""
        from models.totals import ENV_PRIOR_RATE, load_totals_dataset
        _, _, y_h, y_a, meta, _ = load_totals_dataset()
        env_h = env_rates(meta["date"], y_h, ENV_PRIOR_RATE["home"])
        env_a = env_rates(meta["date"], y_a, ENV_PRIOR_RATE["away"])
        w = fit_margin_weights(poisson_pmf(env_h), poisson_pmf(env_a), y_h, y_a)
        np.testing.assert_allclose(w, MARGIN_WEIGHTS,
                                   atol=T.MARGIN_WEIGHT_DRIFT_WARN)

    def test_market_quotes_take_espn_draftkings_closing_prices(self):
        """When raw.historical_odds has over/under price columns (added by
        ingestion/espn_odds.py), DraftKings rows join the market check as
        one more book; Unibet-era rows (some priced in-play) never do.
        Runs in one transaction that is rolled back: the database is left
        exactly as it was, columns included."""
        from sqlalchemy import text
        from config.settings import engine
        with engine.connect() as conn:
            trans = conn.begin()
            try:
                conn.execute(text("""
                    ALTER TABLE raw.historical_odds
                        ADD COLUMN IF NOT EXISTS over_price INTEGER,
                        ADD COLUMN IF NOT EXISTS under_price INTEGER"""))
                pick = """
                    SELECT h.game_id, h.over_under FROM raw.historical_odds h
                    JOIN raw.games g USING (game_id)
                    WHERE h.provider = :p AND h.over_under IS NOT NULL
                      AND g.game_state IN ('FINAL', 'OFF')
                    ORDER BY h.game_id LIMIT 1"""
                dk = conn.execute(text(pick), {"p": "DraftKings"}).one()
                other = conn.execute(text("""
                    SELECT h.game_id FROM raw.historical_odds h
                    WHERE h.provider <> 'DraftKings' AND h.over_under IS NOT NULL
                    ORDER BY h.game_id LIMIT 1""")).scalar()
                conn.execute(text("""
                    UPDATE raw.historical_odds SET over_price = -115,
                           under_price = -105 WHERE game_id = ANY(:g)"""),
                             {"g": [dk.game_id, other]})
                q = T.load_market_quotes(conn)
                row = q[q["game_id"] == dk.game_id]
                assert list(row["book_name"]) == ["espn_DraftKings"]
                assert float(row["line"].iloc[0]) == float(dk.over_under)
                assert (int(row["over_price"].iloc[0]),
                        int(row["under_price"].iloc[0])) == (-115, -105)
                assert other not in set(q["game_id"])
            finally:
                trans.rollback()

    def test_hardened_walk_forward_on_history(self):
        """Registers nothing. The margin fix must help the baseline on the
        real history (measured 2.1815 -> 2.1787); the verdict itself is
        whatever the data says."""
        res = T.run_totals(register=False)
        pooled = res["pooled"]
        assert pooled["baseline_nll"] < pooled["baseline_nll_v1"]
        assert pooled["gate_passed"] == (pooled["nll"] < pooled["baseline_nll"])
        assert len(res["folds"]) >= 2
        assert all(len(f["margin_weights"]) == 5 for f in res["folds"])
