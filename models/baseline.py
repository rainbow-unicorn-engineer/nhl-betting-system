"""
models/baseline.py
Baseline logistic regression with walk-forward CV (Phase 2, Task 8).

Purpose: validate that the feature store carries real, leakage-free signal
before any production modeling. The success gate (locked in PHASE2_PLAN /
File 3 §9) is aggregate walk-forward log loss < 0.69 — i.e. the model must
beat the ~0.693 coin-flip baseline out of sample. Calibration matters more
than accuracy for betting, so the run also produces a reliability plot and
ECE from the pooled out-of-fold predictions.

Walk-forward protocol (mandatory — no random splits, ever):
- Expanding window over seasons: train on seasons < N, validate on season N.
- A purge gap excludes from training any game within PURGE_DAYS before the
  validation period's first game. Season folds already have a months-long
  off-season between train and validation, so the purge is a defensive
  invariant (enforced + tested) rather than an active filter — it becomes
  load-bearing if folds ever go intra-season.
- Scaling is fit on each fold's training rows only (no peeking).

Outputs:
- per-fold + pooled out-of-fold metrics: log loss, accuracy, Brier, AUC, ECE
- models/artifacts/baseline_calibration.png (reliability diagram)
- a row in models.model_registry (upserted on model_name+version)
"""
import hashlib
import logging
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

from config.settings import engine

logger = logging.getLogger("nhl.models.baseline")

PURGE_DAYS = 7
MODEL_NAME = "baseline_logreg"
MODEL_VERSION = "v1"
ARTIFACT_DIR = Path(__file__).parent / "artifacts"
GATE_LOG_LOSS = 0.69


def load_dataset() -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, list]:
    """
    Load features.game_vector into (X, y, meta, feature_names).
    Verifies every row shares the same feature-name ordering.
    """
    with engine.connect() as conn:
        rows = pd.read_sql(text("""
            SELECT game_id, season, date, feature_names, feature_vector, home_win
            FROM features.game_vector
            WHERE home_win IS NOT NULL
            ORDER BY date, game_id
        """), conn)
    if rows.empty:
        raise RuntimeError("features.game_vector is empty — run the feature build first")

    names = rows["feature_names"].iloc[0]
    if not all(n == names for n in rows["feature_names"]):
        raise RuntimeError("Inconsistent feature_names across game_vector rows")

    X = np.array(rows["feature_vector"].tolist(), dtype=float)
    y = rows["home_win"].to_numpy(dtype=int)
    meta = rows[["game_id", "season", "date"]].copy()
    meta["date"] = pd.to_datetime(meta["date"])
    return X, y, meta, list(names)


@dataclass
class Fold:
    val_season: int
    train_idx: np.ndarray
    val_idx: np.ndarray


def walk_forward_folds(meta: pd.DataFrame, purge_days: int = PURGE_DAYS) -> List[Fold]:
    """
    Expanding-window season folds. For validation season N, training rows are
    games from earlier seasons whose date is more than `purge_days` before
    the first game of season N. Never yields a fold without training data.
    """
    seasons = sorted(meta["season"].unique())
    folds = []
    for val_season in seasons[1:]:
        val_mask = meta["season"] == val_season
        val_start = meta.loc[val_mask, "date"].min()
        cutoff = val_start - timedelta(days=purge_days)
        train_mask = (meta["season"] < val_season) & (meta["date"] < cutoff)
        folds.append(Fold(
            val_season=val_season,
            train_idx=np.flatnonzero(train_mask.to_numpy()),
            val_idx=np.flatnonzero(val_mask.to_numpy()),
        ))
    return folds


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """ECE with equal-width bins: sum_b (n_b/N) * |acc_b - conf_b|."""
    bins = np.clip((y_prob * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def calibration_plot(y_true, y_prob, path: Path, n_bins: int = 10) -> None:
    """Reliability diagram + prediction histogram from pooled OOF preds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    conf = np.array([y_prob[idx == b].mean() if (idx == b).any() else np.nan
                     for b in range(n_bins)])
    acc = np.array([y_true[idx == b].mean() if (idx == b).any() else np.nan
                    for b in range(n_bins)])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 8), height_ratios=[3, 1], sharex=True)
    ax1.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax1.plot(conf, acc, "o-", label="baseline logistic (OOF)")
    ax1.set_ylabel("Observed home-win rate")
    ax1.set_title("Baseline calibration — pooled walk-forward OOF predictions")
    ax1.legend()
    ax2.hist(y_prob, bins=bins, edgecolor="black")
    ax2.set_xlabel("Predicted P(home win)")
    ax2.set_ylabel("Count")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_baseline(register: bool = True) -> dict:
    """Full walk-forward run. Returns per-fold and pooled metrics."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss,
                                 roc_auc_score)
    from sklearn.preprocessing import StandardScaler

    X, y, meta, names = load_dataset()
    # The baseline validates the feature store itself, so it stays
    # market-blind: market columns (added in Phase 3) are excluded here
    # and belong to models/lgbm.py, which handles the two market regimes.
    market_cols = [i for i, n in enumerate(names)
                   if n in ("market_home_prob", "market_available")]
    if market_cols:
        X = np.delete(X, market_cols, axis=1)
        names = [n for i, n in enumerate(names) if i not in market_cols]
    folds = walk_forward_folds(meta)
    logger.info(f"Dataset: {X.shape[0]} games x {X.shape[1]} features, "
                f"{len(folds)} walk-forward folds")

    fold_metrics = []
    oof_prob = np.full(len(y), np.nan)

    for fold in folds:
        scaler = StandardScaler().fit(X[fold.train_idx])
        clf = LogisticRegression(max_iter=2000)
        clf.fit(scaler.transform(X[fold.train_idx]), y[fold.train_idx])
        prob = clf.predict_proba(scaler.transform(X[fold.val_idx]))[:, 1]
        oof_prob[fold.val_idx] = prob

        yv = y[fold.val_idx]
        # Naive same-fold benchmark: predict the training home-win rate
        naive = np.full_like(prob, y[fold.train_idx].mean())
        m = {
            "val_season": fold.val_season,
            "n_train": len(fold.train_idx),
            "n_val": len(fold.val_idx),
            "log_loss": log_loss(yv, prob),
            "naive_log_loss": log_loss(yv, naive),
            "accuracy": accuracy_score(yv, prob > 0.5),
            "brier": brier_score_loss(yv, prob),
            "auc": roc_auc_score(yv, prob),
        }
        fold_metrics.append(m)
        logger.info(
            f"  fold {m['val_season']}: n_train={m['n_train']:>5} "
            f"log_loss={m['log_loss']:.4f} (naive {m['naive_log_loss']:.4f}) "
            f"acc={m['accuracy']:.3f} brier={m['brier']:.4f} auc={m['auc']:.4f}")

    scored = ~np.isnan(oof_prob)
    pooled = {
        "log_loss": log_loss(y[scored], oof_prob[scored]),
        "accuracy": accuracy_score(y[scored], oof_prob[scored] > 0.5),
        "brier": brier_score_loss(y[scored], oof_prob[scored]),
        "auc": roc_auc_score(y[scored], oof_prob[scored]),
        "ece": expected_calibration_error(y[scored], oof_prob[scored]),
        "n_scored": int(scored.sum()),
        "gate_log_loss": GATE_LOG_LOSS,
        "gate_passed": None,  # set below
    }
    pooled["gate_passed"] = bool(pooled["log_loss"] < GATE_LOG_LOSS)

    logger.info(f"POOLED OOF: log_loss={pooled['log_loss']:.4f} "
                f"acc={pooled['accuracy']:.3f} brier={pooled['brier']:.4f} "
                f"auc={pooled['auc']:.4f} ece={pooled['ece']:.4f} "
                f"({pooled['n_scored']} games) — GATE(<{GATE_LOG_LOSS}) "
                f"{'PASSED' if pooled['gate_passed'] else 'FAILED'}")

    artifact = ARTIFACT_DIR / "baseline_calibration.png"
    calibration_plot(y[scored], oof_prob[scored], artifact)
    logger.info(f"Calibration plot saved to {artifact}")

    if register:
        _register(pooled, meta, names, str(artifact))

    return {"folds": fold_metrics, "pooled": pooled, "artifact": str(artifact)}


def _register(pooled: dict, meta: pd.DataFrame, names: list, artifact: str) -> None:
    """Upsert the run into models.model_registry."""
    feature_hash = hashlib.sha256(",".join(names).encode()).hexdigest()
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO models.model_registry
                (model_name, version, model_type, trained_through,
                 feature_set_hash, cv_log_loss, cv_brier, cv_auc,
                 cv_accuracy, artifact_path, is_active)
            VALUES (:name, :version, 'logistic_regression', :through,
                    :hash, :ll, :brier, :auc, :acc, :artifact, FALSE)
            ON CONFLICT (model_name, version) DO UPDATE SET
                trained_through = EXCLUDED.trained_through,
                feature_set_hash = EXCLUDED.feature_set_hash,
                cv_log_loss = EXCLUDED.cv_log_loss,
                cv_brier = EXCLUDED.cv_brier,
                cv_auc = EXCLUDED.cv_auc,
                cv_accuracy = EXCLUDED.cv_accuracy,
                artifact_path = EXCLUDED.artifact_path
        """), {
            "name": MODEL_NAME, "version": MODEL_VERSION,
            "through": meta["date"].max().date(), "hash": feature_hash,
            "ll": round(pooled["log_loss"], 4),
            "brier": round(pooled["brier"], 4),
            "auc": round(pooled["auc"], 4),
            "acc": round(pooled["accuracy"], 3),
            "artifact": artifact,
        })
    logger.info(f"Registered {MODEL_NAME} {MODEL_VERSION} in models.model_registry")


if __name__ == "__main__":
    run_baseline()
