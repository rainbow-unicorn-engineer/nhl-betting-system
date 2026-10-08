"""
Tests for models/lgbm.py (Phase 3).

Unit tests cover the pure pieces: the market offset (logit vs unavailable
rows), the time-ordered core/cal split (no shuffling, cal strictly after
core), and temperature scaling direction. The integration test runs the
full walk-forward and asserts the Phase 3 gate: pooled OOF log loss below
the Phase 2 baseline (0.6829) with ECE < 0.02.
"""
import numpy as np
import pandas as pd
import pytest

from models.lgbm import (
    GATE_LOG_LOSS, fit_temperature, market_offset, run_lgbm, time_split,
)


class TestMarketOffset:
    def test_logit_where_available_zero_otherwise(self):
        names = ["foo", "market_home_prob", "market_available"]
        X = np.array([[1.0, 0.6, 1.0],
                      [1.0, 0.5, 0.0],     # unavailable -> 0 regardless
                      [1.0, 0.5, 1.0]])    # available at exactly 0.5 -> 0
        base = market_offset(X, names)
        assert base[0] == pytest.approx(np.log(0.6 / 0.4))
        assert base[1] == 0.0
        assert base[2] == pytest.approx(0.0)

    def test_extreme_probs_are_clipped_finite(self):
        names = ["market_home_prob", "market_available"]
        X = np.array([[0.999, 1.0], [0.001, 1.0]])
        base = market_offset(X, names)
        assert np.isfinite(base).all()


class TestLoadTrainingSet:
    """In-play historical lines leave training and evaluation as 'no
    market', whatever the stored vectors say."""

    @staticmethod
    def _patch(monkeypatch, inplay):
        import contextlib

        import config.settings
        import features.market_prices as MP
        import models.lgbm as L
        names = ["f", "market_home_prob", "market_available"]
        X = np.array([[1.0, 0.95, 1.0], [2.0, 0.55, 1.0], [3.0, 0.5, 0.0]])
        meta = pd.DataFrame({"game_id": [10, 11, 12], "season": 20232024,
                             "date": pd.to_datetime(["2024-04-09"] * 3)})
        monkeypatch.setattr(L, "load_dataset",
                            lambda: (X.copy(), np.array([1, 0, 1]), meta, names))
        monkeypatch.setattr(MP, "load_inplay_game_ids", lambda conn: inplay)
        monkeypatch.setattr(config.settings, "engine", type(
            "E", (), {"connect": lambda self: contextlib.nullcontext()})())
        return L, X

    def test_inplay_games_lose_their_market(self, monkeypatch):
        L, X = self._patch(monkeypatch, [10])
        Xc, y, meta, names = L.load_training_set()
        assert Xc[0].tolist() == [1.0, 0.5, 0.0]
        assert Xc[1:].tolist() == X[1:].tolist()
        assert market_offset(Xc, names)[0] == 0.0

    def test_switch_off_returns_stored_vectors(self, monkeypatch):
        L, X = self._patch(monkeypatch, [10])
        Xs, *_ = L.load_training_set(drop_inplay_market=False)
        assert Xs.tolist() == X.tolist()


class TestTimeSplit:
    def test_cal_tail_is_strictly_later(self):
        dates = pd.Series(pd.date_range("2021-01-01", periods=100))
        idx = np.arange(100)
        np.random.default_rng(0).shuffle(idx)   # order must not matter
        core, cal = time_split(idx, dates, cal_frac=0.2)
        assert len(core) == 80 and len(cal) == 20
        assert dates.iloc[core].max() < dates.iloc[cal].min()
        assert set(core) | set(cal) == set(range(100))


class TestTemperature:
    def test_overconfident_logits_get_squeezed(self):
        rng = np.random.default_rng(7)
        true_logit = rng.normal(0, 1, 20000)
        y = (rng.uniform(size=true_logit.size) <
             1 / (1 + np.exp(-true_logit))).astype(int)
        a, _ = fit_temperature(true_logit * 3.0, y)   # 3x overconfident
        assert 0.25 < a < 0.45                        # ~1/3 recovers truth


# ─────────────────────────────────────────────
# Integration (requires built game vectors incl. market feature)
# ─────────────────────────────────────────────
@pytest.fixture(scope="module")
def result():
    from models.baseline import load_dataset
    try:
        X, y, meta, names = load_dataset()
    except Exception as e:
        pytest.skip(f"game vectors unavailable: {e}")
    if "market_home_prob" not in names or meta["season"].nunique() < 3:
        pytest.skip("vectors lack market feature or enough seasons")
    return run_lgbm(register=False)


class TestIntegration:
    def test_gate_beats_baseline_and_is_calibrated(self, result):
        pooled = result["pooled"]
        assert pooled["log_loss"] < GATE_LOG_LOSS, \
            f"GATE FAILED: {pooled['log_loss']:.4f} >= {GATE_LOG_LOSS}"
        assert pooled["ece"] < 0.02
        assert 0.55 < pooled["auc"] < 0.75

    def test_every_fold_beats_naive(self, result):
        for f in result["folds"]:
            assert f["log_loss"] < f["naive_log_loss"], f

    def test_market_folds_track_the_market(self, result):
        """Boosting from the market must land within noise of the market
        on market-covered folds (not collapse to feature-only quality)."""
        for f in result["folds"]:
            if f["market_log_loss"] and f["market_coverage"] > 0.9:
                assert f["log_loss"] < f["market_log_loss"] + 0.012, f
