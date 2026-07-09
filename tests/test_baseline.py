"""
Tests for models/baseline.py (Phase 2, Task 8).

The critical test (plan-mandated): walk-forward folds may NEVER train on
data at or after the validation period — checked on synthetic folds, and
again on the real dataset's folds. The integration test runs the full
walk-forward and asserts the Phase 2 success gate (pooled OOF log loss
< 0.69) plus registry/artifact side effects.
"""
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from models.baseline import (
    GATE_LOG_LOSS, MODEL_NAME, MODEL_VERSION, PURGE_DAYS,
    expected_calibration_error, load_dataset, run_baseline, walk_forward_folds,
)


def synth_meta():
    """Three seasons, weekly games."""
    rows = []
    gid = 1
    for season, start in ((20202021, "2021-01-01"),
                          (20212022, "2021-10-10"),
                          (20222023, "2022-10-08")):
        for k in range(30):
            rows.append({"game_id": gid, "season": season,
                         "date": pd.Timestamp(start) + timedelta(days=3 * k)})
            gid += 1
    return pd.DataFrame(rows)


class TestWalkForwardFolds:
    def test_never_trains_on_validation_period_or_later(self):
        """THE leakage test: every training game predates the validation
        period by more than the purge gap; no train/val overlap."""
        meta = synth_meta()
        folds = walk_forward_folds(meta)
        assert len(folds) == 2  # 3 seasons -> 2 validation folds
        for fold in folds:
            train, val = meta.iloc[fold.train_idx], meta.iloc[fold.val_idx]
            assert set(fold.train_idx).isdisjoint(fold.val_idx)
            assert (train["season"] < fold.val_season).all()
            assert (val["season"] == fold.val_season).all()
            gap = (val["date"].min() - train["date"].max()).days
            assert gap > PURGE_DAYS

    def test_expanding_window(self):
        folds = walk_forward_folds(synth_meta())
        sizes = [len(f.train_idx) for f in folds]
        assert sizes == sorted(sizes) and sizes[0] < sizes[-1]

    def test_purge_gap_excludes_recent_train_games(self):
        """A prior-season game inside the purge window is dropped."""
        meta = synth_meta()
        val_start = meta.loc[meta["season"] == 20212022, "date"].min()
        extra = pd.DataFrame([{
            "game_id": 999, "season": 20202021,
            "date": val_start - timedelta(days=PURGE_DAYS - 1)}])
        meta2 = pd.concat([meta, extra], ignore_index=True)
        fold = walk_forward_folds(meta2)[0]
        assert 999 not in meta2.iloc[fold.train_idx]["game_id"].values


class TestECE:
    def test_perfectly_calibrated_is_zero(self):
        rng = np.random.default_rng(7)
        prob = rng.uniform(0.05, 0.95, 200_000)
        y = (rng.uniform(size=prob.size) < prob).astype(int)
        assert expected_calibration_error(y, prob) < 0.01

    def test_overconfident_is_penalized(self):
        y = np.array([0, 1] * 100)           # events occur 0% / 100% of the time
        prob = np.array([0.05, 0.95] * 100)  # claims 5% / 95% -> 0.05 off per bin
        assert expected_calibration_error(y, prob) == pytest.approx(0.05, abs=1e-9)
        prob_bad = np.full(200, 0.95)        # 95% claimed vs 50% observed
        assert expected_calibration_error(y, prob_bad) == pytest.approx(0.45, abs=0.01)


# ─────────────────────────────────────────────
# Integration (requires built game vectors)
# ─────────────────────────────────────────────
@pytest.fixture(scope="module")
def dataset():
    try:
        X, y, meta, names = load_dataset()
    except Exception as e:
        pytest.skip(f"game vectors unavailable: {e}")
    if len(y) < 5000 or meta["season"].nunique() < 3:
        pytest.skip("not enough seasons of vectors for a meaningful CV")
    return X, y, meta, names


class TestIntegration:
    def test_real_folds_are_leak_free(self, dataset):
        _, _, meta, _ = dataset
        for fold in walk_forward_folds(meta):
            train, val = meta.iloc[fold.train_idx], meta.iloc[fold.val_idx]
            assert (train["season"] < fold.val_season).all()
            assert (val["date"].min() - train["date"].max()).days > PURGE_DAYS

    def test_gate_and_registration(self, dataset):
        """Run the full walk-forward: the Phase 2 gate must hold."""
        result = run_baseline(register=True)
        pooled = result["pooled"]

        assert pooled["log_loss"] < GATE_LOG_LOSS, \
            f"GATE FAILED: OOF log loss {pooled['log_loss']:.4f} >= {GATE_LOG_LOSS}"
        assert pooled["gate_passed"] is True
        assert 0.5 < pooled["auc"] < 0.75      # signal, but no NHL model is 0.75+
        assert 0.52 < pooled["accuracy"] < 0.65

        # Later folds (more training data) should beat their naive benchmark
        late = result["folds"][1:-1]
        assert all(f["log_loss"] < f["naive_log_loss"] for f in late)

        assert Path(result["artifact"]).exists()
        from config.settings import engine
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT cv_log_loss, cv_auc, artifact_path
                FROM models.model_registry
                WHERE model_name = :n AND version = :v
            """), {"n": MODEL_NAME, "v": MODEL_VERSION}).one()
        assert float(row.cv_log_loss) == pytest.approx(pooled["log_loss"], abs=5e-4)
