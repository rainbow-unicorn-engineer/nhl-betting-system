"""
models/lgbm.py
Phase 3 production model: LightGBM boosted FROM the market + calibration.

Same walk-forward protocol as the baseline (expanding season folds, purge
gap, no random splits).

Market-as-offset: instead of leaving market_home_prob as one feature among
109, the booster trains with init_score = logit(market prob) — it learns
residual corrections to the market rather than rediscovering it. Worst
case the trees add nothing and predictions equal the market; the model
can only be dragged below market performance by overfit corrections,
which regularization + the walk-forward gate police. Games without a line
(2024-25) start from logit(0.5)=0 and lean on market_available + the
other features.

Calibration: within each fold's training window the last CAL_FRAC of games
(by date) is a held-out tail used for early stopping and to fit temperature
scaling — a single-parameter monotone squeeze of the logits, the
lowest-variance calibrator there is. Isotonic regression was tried in this
slot first and REJECTED by the evidence: on ~150-1000-game calibration
tails its step function overfits and *worsened* pooled OOF log loss by
+0.02; a final isotonic pass belongs at the strategy layer, fit on pooled
OOF predictions (thousands of games), not per fold. The validation season
never touches booster, early stopping, or calibrator.

Benchmarks reported per fold: naive (train home-win rate) and the market
itself (scored on market_available games). The Phase 2 baseline is the
registry row `baseline_logreg v1` (0.6829); the gate is beating it.
"""
import logging
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

from models.baseline import (
    ARTIFACT_DIR, PURGE_DAYS, calibration_plot, expected_calibration_error,
    load_dataset, walk_forward_folds,
)

logger = logging.getLogger("nhl.models.lgbm")

MODEL_NAME = "lgbm_market"
MODEL_VERSION = "v2"
CAL_FRAC = 0.15           # time-ordered tail of each train window
GATE_LOG_LOSS = 0.6829    # must beat the Phase 2 baseline, not just coin flip
PROB_CLIP = (0.02, 0.98)

LGBM_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "n_estimators": 1000,
    "random_state": 42,
    "verbosity": -1,
}


def market_offset(X: np.ndarray, names: list) -> np.ndarray:
    """Per-row init_score: logit of the market prob, 0 where unavailable."""
    p = X[:, names.index("market_home_prob")]
    avail = X[:, names.index("market_available")] == 1.0
    base = np.zeros(len(X))
    base[avail] = logit(np.clip(p[avail], *PROB_CLIP))
    return base


def time_split(train_idx: np.ndarray, dates, cal_frac: float = CAL_FRAC):
    """Split a fold's training rows by date into (core, cal) index arrays."""
    order = np.argsort(dates.iloc[train_idx].to_numpy())
    ordered = train_idx[order]
    n_core = int(len(ordered) * (1.0 - cal_frac))
    return ordered[:n_core], ordered[n_core:]


def fit_temperature(logits: np.ndarray, y: np.ndarray):
    """Fit temperature scaling p = sigmoid(a*logit + b); returns (a, b)."""
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(logits.reshape(-1, 1), y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def fit_fold(X, y, base, train_idx, dates, names):
    """Fit the two-model fold, each regime split time-wise on its own rows:

    - M (market-offset booster): trained, early-stopped, and temperature-
      scaled ONLY on market-available games — the games it will ever score.
      Without this, a fold whose training window ends in the no-market
      2024-25 season would early-stop and calibrate M on offset-less rows.
    - F (market-blind fallback): trained on all games with the market
      columns removed; scores games without a line.

    Returns a dict consumed by predict_fold."""
    import lightgbm as lgb

    market_cols = [names.index("market_home_prob"),
                   names.index("market_available")]
    blind = np.delete(np.arange(X.shape[1]), market_cols)
    avail_idx = train_idx[X[train_idx, names.index("market_available")] == 1.0]

    m_core, m_cal = time_split(avail_idx, dates)
    m = lgb.LGBMClassifier(**LGBM_PARAMS)
    m.fit(X[m_core], y[m_core],
          init_score=base[m_core],
          eval_set=[(X[m_cal], y[m_cal])],
          eval_init_score=[base[m_cal]],
          eval_metric="binary_logloss",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    m_logits = base[m_cal] + m.booster_.predict(X[m_cal], raw_score=True)

    f_core, f_cal = time_split(train_idx, dates)
    f = lgb.LGBMClassifier(**LGBM_PARAMS)
    f.fit(X[f_core][:, blind], y[f_core],
          eval_set=[(X[f_cal][:, blind], y[f_cal])],
          eval_metric="binary_logloss",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    f_logits = f.booster_.predict(X[f_cal][:, blind], raw_score=True)

    return {"m": m, "temp_m": fit_temperature(m_logits, y[m_cal]),
            "f": f, "temp_f": fit_temperature(f_logits, y[f_cal]),
            "blind": blind,
            "iters": (m.best_iteration_ or LGBM_PARAMS["n_estimators"],
                      f.best_iteration_ or LGBM_PARAMS["n_estimators"])}


def predict_fold(fm: dict, X, base, available) -> np.ndarray:
    """Calibrated P(home win): market-offset model where a line exists,
    market-blind fallback where it doesn't."""
    am, bm = fm["temp_m"]
    af, bf = fm["temp_f"]
    p_m = expit(am * (base + fm["m"].booster_.predict(X, raw_score=True)) + bm)
    p_f = expit(af * fm["f"].booster_.predict(X[:, fm["blind"]], raw_score=True) + bf)
    return np.where(available, p_m, p_f)


def fit_production(cutoff_date=None) -> dict:
    """Train the two-regime scorer (market-offset M + market-blind F) on
    every labeled game strictly before cutoff_date, with the same time-tail
    early stopping + temperature scaling as a walk-forward fold.

    Live use passes no cutoff (all completed games are in the past);
    simulation against a historical date passes that date so the model
    never sees the games it is about to score.
    """
    import pandas as pd

    X, y, meta, names = load_dataset()
    if cutoff_date is not None:
        mask = meta["date"] < pd.Timestamp(cutoff_date)
    else:
        mask = np.ones(len(y), dtype=bool)
    train_idx = np.flatnonzero(mask)
    if len(train_idx) < 500:
        raise RuntimeError(
            f"Only {len(train_idx)} labeled games before {cutoff_date} — "
            f"not enough to train a production model")

    base = market_offset(X, names)
    fm = fit_fold(X, y, base, train_idx, meta["date"], names)
    logger.info(f"Production fit: {len(train_idx)} games through "
                f"{meta['date'].iloc[train_idx].max().date()}, "
                f"iters={fm['iters']}")
    return {"fm": fm, "names": names, "n_train": len(train_idx),
            "trained_through": meta["date"].iloc[train_idx].max()}


def score_production(prod: dict, X_new: np.ndarray, names_new: list) -> np.ndarray:
    """Calibrated P(home win) for new feature vectors from fit_production.
    Refuses to score if the feature ordering differs from training."""
    if list(names_new) != list(prod["names"]):
        raise ValueError("Feature names/order mismatch between production "
                         "model and vectors to score")
    base = market_offset(X_new, names_new)
    avail = X_new[:, names_new.index("market_available")] == 1.0
    return predict_fold(prod["fm"], X_new, base, avail)


def run_lgbm(register: bool = True) -> dict:
    """Full walk-forward run of LightGBM + isotonic. Returns metrics."""
    from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss,
                                 roc_auc_score)

    X, y, meta, names = load_dataset()
    folds = walk_forward_folds(meta)
    base = market_offset(X, names)
    mkt_prob_i = names.index("market_home_prob")
    mkt_avail_i = names.index("market_available")
    logger.info(f"Dataset: {X.shape[0]} games x {X.shape[1]} features, "
                f"{len(folds)} folds (purge {PURGE_DAYS}d)")

    fold_metrics = []
    oof_raw = np.full(len(y), np.nan)
    oof_cal = np.full(len(y), np.nan)

    for fold in folds:
        fm = fit_fold(X, y, base, fold.train_idx, meta["date"], names)

        Xv, bv = X[fold.val_idx], base[fold.val_idx]
        avail_v = X[fold.val_idx, mkt_avail_i] == 1.0
        raw = np.where(
            avail_v,
            expit(bv + fm["m"].booster_.predict(Xv, raw_score=True)),
            expit(fm["f"].booster_.predict(Xv[:, fm["blind"]], raw_score=True)))
        cal = predict_fold(fm, Xv, bv, avail_v)
        oof_raw[fold.val_idx], oof_cal[fold.val_idx] = raw, cal

        yv = y[fold.val_idx]
        naive = np.full_like(cal, y[fold.train_idx].mean())
        has_mkt = X[fold.val_idx, mkt_avail_i] == 1.0
        market_ll = (log_loss(yv[has_mkt], X[fold.val_idx, mkt_prob_i][has_mkt])
                     if has_mkt.sum() > 50 else None)
        m = {
            "val_season": fold.val_season,
            "n_train": len(fold.train_idx), "n_val": len(fold.val_idx),
            "iters": fm["iters"],
            "log_loss_raw": log_loss(yv, raw),
            "log_loss": log_loss(yv, cal),
            "naive_log_loss": log_loss(yv, naive),
            "market_log_loss": market_ll,
            "market_coverage": float(has_mkt.mean()),
            "accuracy": accuracy_score(yv, cal > 0.5),
            "brier": brier_score_loss(yv, cal),
            "auc": roc_auc_score(yv, cal),
            "ece": expected_calibration_error(yv, cal),
        }
        fold_metrics.append(m)
        logger.info(
            f"  fold {m['val_season']}: log_loss={m['log_loss']:.4f} "
            f"(raw {m['log_loss_raw']:.4f}, naive {m['naive_log_loss']:.4f}, "
            f"market {m['market_log_loss'] and round(m['market_log_loss'], 4)}) "
            f"acc={m['accuracy']:.3f} auc={m['auc']:.4f} ece={m['ece']:.4f} "
            f"iters={m['iters']}")

    scored = ~np.isnan(oof_cal)
    ys, ps = y[scored], oof_cal[scored]
    pooled = {
        "log_loss": log_loss(ys, ps),
        "log_loss_raw": log_loss(ys, oof_raw[scored]),
        "accuracy": accuracy_score(ys, ps > 0.5),
        "brier": brier_score_loss(ys, ps),
        "auc": roc_auc_score(ys, ps),
        "ece": expected_calibration_error(ys, ps),
        "n_scored": int(scored.sum()),
        "gate_log_loss": GATE_LOG_LOSS,
    }
    pooled["gate_passed"] = bool(pooled["log_loss"] < GATE_LOG_LOSS)

    logger.info(
        f"POOLED OOF: log_loss={pooled['log_loss']:.4f} "
        f"(raw {pooled['log_loss_raw']:.4f}) acc={pooled['accuracy']:.3f} "
        f"brier={pooled['brier']:.4f} auc={pooled['auc']:.4f} "
        f"ece={pooled['ece']:.4f} — GATE(<{GATE_LOG_LOSS} baseline) "
        f"{'PASSED' if pooled['gate_passed'] else 'FAILED'}")

    artifact = ARTIFACT_DIR / "lgbm_calibration.png"
    calibration_plot(ys, ps, artifact)
    logger.info(f"Calibration plot saved to {artifact}")

    if register:
        _register(pooled, meta, names, str(artifact))

    oof = meta.loc[scored, ["game_id", "season", "date"]].copy()
    oof["prob_home"] = oof_cal[scored]
    oof["prob_home_raw"] = oof_raw[scored]
    return {"folds": fold_metrics, "pooled": pooled, "artifact": str(artifact),
            "oof": oof}


def _register(pooled: dict, meta, names: list, artifact: str) -> None:
    import hashlib
    from sqlalchemy import text
    from config.settings import engine

    feature_hash = hashlib.sha256(",".join(names).encode()).hexdigest()
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO models.model_registry
                (model_name, version, model_type, trained_through,
                 feature_set_hash, cv_log_loss, cv_brier, cv_auc,
                 cv_accuracy, artifact_path, is_active)
            VALUES (:name, :version, 'lightgbm+isotonic', :through,
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
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_lgbm()
