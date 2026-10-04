"""
Tests for experiments/tabpfn/run.py (the TabPFN moneyline trial).

The scoring and decision helpers are pure. The walk-forward plumbing is
tested with small scikit-learn stand-ins for TabPFN, so most of this file
runs on any machine (no torch, no GPU, no database). The point-in-time
tests prove a fold's predictions cannot see its own season's results or
anything later: rewriting or deleting those rows leaves them unchanged.
One smoke test runs the real TabPFN-2 weights when the package and its
weights are available, and skips otherwise.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss

from experiments.tabpfn import run as tp
from models.baseline import walk_forward_folds

NAMES = ["f0", "f1", "f2", "market_home_prob", "market_available"]


def synthetic(n_per_season=300, seasons=(20202021, 20212022, 20222023, 20232024),
              seed=0, unpriced_season=None):
    """Games with a true logit from f0/f1 and a market that knows most of it."""
    rng = np.random.default_rng(seed)
    rows, metas = [], []
    gid = 1
    for k, s in enumerate(seasons):
        start = pd.Timestamp(f"{2020 + k}-10-10")
        for i in range(n_per_season):
            f = rng.normal(size=3)
            true = 0.8 * f[0] - 0.5 * f[1]
            p_true = 1 / (1 + np.exp(-true))
            p_mkt = 1 / (1 + np.exp(-(0.8 * true)))
            avail = 0.0 if s == unpriced_season else 1.0
            rows.append([f[0], f[1], f[2], p_mkt if avail else 0.5, avail, p_true])
            metas.append((gid, s, start + pd.Timedelta(days=i // 8)))
            gid += 1
    arr = np.array(rows)
    y = (rng.uniform(size=len(arr)) < arr[:, -1]).astype(int)
    meta = pd.DataFrame(metas, columns=["game_id", "season", "date"])
    return arr[:, :-1].copy(), y, meta, list(NAMES)


class Recorder:
    """Wraps a scikit-learn model and remembers every row it was fit on."""
    seen = []

    def __init__(self, kind):
        self.kind = kind
        self.model = LogisticRegression(max_iter=1000) if kind == "clf" else LinearRegression()

    def fit(self, X, y):
        Recorder.seen.append(np.array(X, copy=True))
        self.model.fit(X, y)
        if self.kind == "clf":
            self.classes_ = self.model.classes_
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)


def factory_for_seed(seed):
    return lambda kind: Recorder(kind)


# ─────────────────────────────────────────────
# Pure helpers
# ─────────────────────────────────────────────
class TestScoring:
    def test_per_game_log_loss_matches_sklearn(self):
        rng = np.random.default_rng(1)
        p = rng.uniform(0.05, 0.95, 500)
        y = (rng.uniform(size=500) < p).astype(int)
        assert tp.per_game_log_loss(y, p).mean() == pytest.approx(log_loss(y, p))

    def test_paired_mean_and_se(self):
        a = np.array([0.5, 0.6, 0.7, 0.8])
        b = np.array([0.6, 0.6, 0.6, 0.6])
        s = tp.paired(a, b)
        d = a - b
        assert s["mean"] == pytest.approx(d.mean())
        assert s["se"] == pytest.approx(d.std(ddof=1) / 2.0)
        assert s["n"] == 4

    def test_no_vig_two_way(self):
        # -150 / +130: implied 0.6 and 0.4348, normalised to sum to 1
        p = tp.no_vig_two_way(np.array([-150]), np.array([130]))[0]
        assert p == pytest.approx(0.6 / (0.6 + 100 / 230))
        assert tp.no_vig_two_way(np.array([-110]), np.array([-110]))[0] == pytest.approx(0.5)


class TestDecisionRule:
    @staticmethod
    def fold(mean, se):
        return {"n": 1000, "mean": mean, "se": se, "z": mean / se}

    def test_four_of_five_folds_and_market_passes(self):
        folds = [self.fold(-0.003, 0.001)] * 4 + [self.fold(0.001, 0.001)]
        dec = tp.decide(folds, {"n": 500, "mean": -0.0001, "se": 0.001})
        assert dec["folds_beating_lgbm"] == 4 and dec["passed"]

    def test_three_folds_is_not_enough(self):
        folds = [self.fold(-0.003, 0.001)] * 3 + [self.fold(-0.0019, 0.001)] * 2
        dec = tp.decide(folds, {"n": 500, "mean": -0.01, "se": 0.001})
        assert dec["folds_beating_lgbm"] == 3 and not dec["passed"]

    def test_worse_than_market_fails_even_when_beating_lgbm(self):
        folds = [self.fold(-0.01, 0.001)] * 5
        dec = tp.decide(folds, {"n": 500, "mean": 0.0002, "se": 0.001})
        assert dec["beats_lgbm_rule"] and not dec["not_worse_than_market"]
        assert not dec["passed"]

    def test_exactly_two_se_counts(self):
        assert tp.fold_beats(self.fold(-0.002, 0.001))
        assert not tp.fold_beats(self.fold(-0.0019, 0.001))
        assert not tp.fold_beats({"n": 1, "mean": -1.0, "se": float("nan"), "z": 0})


# ─────────────────────────────────────────────
# Variant plumbing with stand-in learners
# ─────────────────────────────────────────────
class TestVariants:
    def test_residual_with_zero_correction_is_the_market(self):
        X, y, meta, names = synthetic()

        class Zero:
            def predict(self, X):
                return np.zeros(len(X))

        raw = tp._raw_logit_m("B", Zero(), X, names)
        expected = np.log(X[:, 3] / (1 - X[:, 3]))
        assert raw == pytest.approx(np.clip(expected, *np.log(np.array([0.02, 0.98]) / (1 - np.array([0.02, 0.98])))))

    def test_residual_target_is_outcome_minus_market(self):
        X, y, meta, names = synthetic()
        Recorder.seen = []
        captured = {}

        class Capture(Recorder):
            def fit(self, X_, y_):
                if self.kind == "reg":
                    captured["X"], captured["y"] = X_.copy(), np.asarray(y_).copy()
                return super().fit(X_, y_)

        folds = walk_forward_folds(meta)
        tp.fit_market_model("B", lambda k: Capture(k), X, y, folds[-1].train_idx,
                            meta["date"], names)
        assert captured["y"] == pytest.approx(
            _labels_for(captured["X"], X, y) - captured["X"][:, 3])

    def test_unpriced_games_use_market_blind_fallback(self):
        X, y, meta, names = synthetic(unpriced_season=20232024)
        folds = walk_forward_folds(meta)
        fold = folds[-1]
        fm = tp.fit_fold("A", factory_for_seed(0), X, y, fold.train_idx, meta["date"], names)
        p = tp.predict(fm, X[fold.val_idx], names)
        # changing the (meaningless) market columns of unpriced games changes nothing
        Xv = X[fold.val_idx].copy()
        Xv[:, 3] = 0.9
        assert tp.predict(fm, Xv, names) == pytest.approx(p)
        assert np.all((p > 0) & (p < 1))

    def test_fallback_never_sees_market_columns(self):
        X, y, meta, names = synthetic()
        Recorder.seen = []
        f, _, blind = tp.fit_blind(factory_for_seed(0), X, y, np.arange(600),
                                   meta["date"], names)
        assert Recorder.seen[0].shape[1] == 3
        assert names.index("market_home_prob") not in blind

    def test_unknown_variant_rejected(self):
        X, y, meta, names = synthetic()
        with pytest.raises(ValueError):
            tp.fit_market_model("Z", factory_for_seed(0), X, y, np.arange(600),
                                meta["date"], names)


def _labels_for(X_fit, X, y):
    """Labels of the rows of X that X_fit's rows came from."""
    lookup = {tuple(np.round(r, 12)): lab for r, lab in zip(X, y)}
    return np.array([lookup[tuple(np.round(r, 12))] for r in X_fit])


# ─────────────────────────────────────────────
# Point in time: no fold can see its own season's results or later
# ─────────────────────────────────────────────
class TestPointInTime:
    variants = [("A", "A", 0), ("B", "B", 0)]

    def baseline_run(self):
        X, y, meta, names = synthetic()
        folds = walk_forward_folds(meta)
        oof = tp.walk_forward(self.variants, factory_for_seed, X, y, meta, names, folds)
        return X, y, meta, names, folds, oof

    def test_learner_only_fits_rows_before_the_validation_season(self):
        X, y, meta, names = synthetic()
        folds = walk_forward_folds(meta)
        for fold in folds:
            Recorder.seen = []
            tp.walk_forward(self.variants, factory_for_seed, X, y, meta, names, [fold])
            train_rows = {tuple(np.round(r, 12)) for r in X[fold.train_idx]}
            blind_rows = {tuple(np.round(r, 12)) for r in X[fold.train_idx][:, :3]}
            for fitted in Recorder.seen:
                pool = train_rows if fitted.shape[1] == X.shape[1] else blind_rows
                assert all(tuple(np.round(r, 12)) in pool for r in fitted)

    def test_rewriting_results_on_or_after_the_season_changes_nothing(self):
        X, y, meta, names, folds, oof = self.baseline_run()
        for fold in folds:
            start = meta.loc[fold.val_idx, "date"].min()
            later = (meta["date"] >= start).to_numpy()
            y2, X2 = y.copy(), X.copy()
            y2[later] = 1 - y2[later]                          # flip every result
            future = later & ~np.isin(np.arange(len(y)), fold.val_idx)
            X2[future, :3] += 5.0                              # rewrite later features
            oof2 = tp.walk_forward(self.variants, factory_for_seed, X2, y2, meta,
                                   names, [fold])
            for key in ("A", "B"):
                assert oof2[key][fold.val_idx] == pytest.approx(oof[key][fold.val_idx])

    def test_deleting_later_seasons_changes_nothing(self):
        X, y, meta, names, folds, oof = self.baseline_run()
        first = folds[0]
        keep = (meta["season"] <= first.val_season).to_numpy()
        meta2 = meta[keep].reset_index(drop=True)
        folds2 = walk_forward_folds(meta2)
        oof2 = tp.walk_forward(self.variants, factory_for_seed, X[keep], y[keep],
                               meta2, names, folds2[:1])
        for key in ("A", "B"):
            assert oof2[key][folds2[0].val_idx] == pytest.approx(oof[key][first.val_idx])


# ─────────────────────────────────────────────
# Report and decision on synthetic data
# ─────────────────────────────────────────────
class TestEvaluate:
    def test_report_shapes_and_ensemble_never_auto_passes_report_only(self):
        X, y, meta, names = synthetic()
        folds = walk_forward_folds(meta)
        oof = tp.walk_forward([("A", "A", 0), ("A_seed1", "A", 1)], factory_for_seed,
                              X, y, meta, names, folds)
        lgbm_p = np.where(np.isnan(oof["A"]), np.nan, X[:, 3])
        oof["C"] = 0.5 * (oof["A"] + lgbm_p)
        r = tp.evaluate(oof, lgbm_p, X, y, meta, names, folds)
        assert set(r["variants"]) == {"A", "A_seed1", "C"}
        assert r["variants"]["A_seed1"]["eligible"] is False
        assert r["variants"]["A_seed1"]["decision"]["passed"] is False
        assert len(r["variants"]["A"]["folds"]) == len(folds)
        assert r["n_scored"] == sum(len(f.val_idx) for f in folds)
        if r["winner"] is None:
            assert r["production_change"].startswith("none")


# ─────────────────────────────────────────────
# Real TabPFN smoke test (skips without the package or its weights)
# ─────────────────────────────────────────────
def test_tabpfn_v2_smoke():
    pytest.importorskip("tabpfn")
    X, y, meta, names = synthetic(n_per_season=150, seasons=(20202021, 20212022))
    try:
        model = tp.make_tabpfn("clf", seed=0)
        model.fit(X[:150], y[:150])
    except Exception as e:  # no cached weights and no network, etc.
        pytest.skip(f"TabPFN-2 weights unavailable: {e}")
    p = model.predict_proba(X[150:])[:, list(model.classes_).index(1)]
    assert np.all((p > 0) & (p < 1))
