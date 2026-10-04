"""
experiments/tabpfn/run.py
TabPFN trial for the moneyline (win/loss) model: can a pretrained
"foundation model for tables" beat lgbm_market v2 and the market?

TabPFN → a neural network from Prior Labs, published on Hugging Face, that
was pretrained on millions of synthetic tables. It does not "train" on our
data in the usual sense: it reads the training rows as context and predicts
new rows in one forward pass (in-context learning → learning from examples
shown at prediction time, with no weight updates). It is designed for
tables of up to ~10,000 rows, which fits our 950-6,550-game training
windows.

Moneyline → a bet on which team wins the game (overtime and shootout
included). The model's job is P(home win).

===========================================================================
PRE-REGISTRATION (written 2026-10-04 and committed BEFORE any variant was
run on NHL data; nothing below may change after the first run)
===========================================================================

Data and protocol (identical to models/lgbm.py v2):
  - features.game_vector via models.baseline.load_dataset(): the same 109
    features lgbm v2 uses (107 team/goalie/schedule features plus
    market_home_prob and market_available), labeled games 2020-21..2025-26.
  - Folds: models.baseline.walk_forward_folds (expanding seasons, 7-day
    purge → a gap of days dropped between training and test games).
    Validation seasons 2021-22..2025-26 (5 folds).
  - Two regimes, exactly as lgbm v2: a model "M" for games with a market
    line, trained only on lined training games; a market-blind fallback "F"
    (market columns removed) for games without a line, trained on all
    training games. Each regime's training rows are split by date with
    models.lgbm.time_split (last 15% = calibration tail). The learner sees
    the first 85%; one-parameter temperature scaling (→ a single squeeze or
    stretch of the model's confidence) is fit on the tail with
    models.lgbm.fit_temperature. The validation season is never seen by
    the learner or the calibrator.
  - TabPFN settings, fixed: the TabPFN-2 weights (ModelVersion.V2, Prior
    Labs License = Apache 2.0 plus an attribution requirement; the newer
    2.5/2.6/3/3.5 weights are non-commercial and gated, see README),
    n_estimators=8 (TabPFN's internal ensemble of 8 shuffled views; also
    what the package's "auto" resolves to for 109 features), random_state
    42, everything else at the package default. GPU when available, else
    CPU. Installed for this trial: tabpfn 9.1.0, torch 2.14.1+cu130 (CUDA
    13.0 build, RTX 5080).

Variants (all are reported, pass or fail):
  A  "feature": TabPFNClassifier predicts home_win from all 109 features,
     i.e. the outcome with the market probability as a feature. F regime:
     TabPFNClassifier on the 107 market-blind features.
  B  "residual": the market-offset residual. On lined games
     TabPFNRegressor predicts r = home_win - p_market (p_market = no-vig
     market_home_prob) from all 109 features; p = clip(p_market + r_hat,
     0.02, 0.98), then temperature-scaled. F regime: the same market-blind
     classifier as A (B and A differ only on lined games).
  C  "ensemble": the plain average of the calibrated probabilities of
     lgbm v2 and variant A, game by game (chosen before any result: A is
     the primary TabPFN variant).
  Report-only, never eligible to pass: A re-run with random_state 1 and 2,
  to show how much the seed alone moves the numbers.

Reference: lgbm v2 out-of-fold predictions from
models.lgbm.run_lgbm(register=False, plot=False), run in the same process
on the same data.

Metric: per-game log loss (→ a score for how wrong the probabilities were,
lower is better) of the calibrated P(home win). Differences are paired
(same games), with paired SE (→ the size of the random wobble in a
difference; 2 SE or more is unlikely to be luck) = sd(d) / sqrt(n),
d = log loss(variant) - log loss(lgbm v2) per game.

Pass rule (the only way a TabPFN variant changes the production default):
  1. It beats lgbm v2 by at least 2 paired SE in at least 4 of the 5
     folds: mean(d_fold) <= -2 * SE_fold.
  2. AND it is not worse than the market: on priced validation games
     (market_available = 1, all folds pooled) its mean log loss minus the
     no-vig market's is <= 0 (point estimate).
  If several variants pass, the one with the lowest pooled log loss wins.
  If none passes, lgbm v2 stays the production model and nothing else
  changes. Nothing is ever registered in models.model_registry.

Secondary numbers (reported, never part of the rule): pooled log loss,
Brier score, AUC, ECE (→ average calibration miss), per fold and pooled;
for 2024-25, where the stored features carry no line, the closing
Pinnacle no-vig price from raw.odds_history (Odds API, last snapshot
before puck drop) as an outside market bar.

No-vig price → the sportsbook's implied probability with its built-in fee
(vig) removed, by scaling the two sides to sum to 1.

===========================================================================
STATUS
===========================================================================
Not run yet.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from models.lgbm import PROB_CLIP, fit_temperature, time_split

logger = logging.getLogger("nhl.experiments.tabpfn")

SEED = 42
N_ESTIMATORS = 8
REPORT_ONLY_SEEDS = (1, 2)
PASS_SE = 2.0          # must beat lgbm v2 by this many paired SE ...
PASS_FOLDS = 4         # ... in at least this many of the folds
MARKET_COLS = ("market_home_prob", "market_available")
RESULTS_PATH = Path(__file__).parent / "results.json"
OOF_PATH = Path(__file__).resolve().parents[2] / "data" / "experiments" / "tabpfn_oof.csv"
EPS = 1e-15


# ─────────────────────────────────────────────
# Scoring helpers (pure)
# ─────────────────────────────────────────────
def per_game_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Log loss of each game on its own (natural log)."""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired(ll_a: np.ndarray, ll_b: np.ndarray) -> dict:
    """Mean of (a - b) per game, its paired SE and the gap in SE units.
    Negative mean = a is better."""
    d = np.asarray(ll_a, dtype=float) - np.asarray(ll_b, dtype=float)
    n = len(d)
    if n < 2:
        return {"n": n, "mean": float(d.mean()) if n else float("nan"),
                "se": float("nan"), "z": float("nan")}
    se = float(d.std(ddof=1) / np.sqrt(n))
    mean = float(d.mean())
    return {"n": n, "mean": mean, "se": se,
            "z": mean / se if se > 0 else float("nan")}


def fold_beats(stat: dict, k: float = PASS_SE) -> bool:
    """True when the variant beats the reference by >= k paired SE."""
    return bool(np.isfinite(stat["se"]) and stat["mean"] <= -k * stat["se"])


def decide(fold_stats: List[dict], market_stat: dict,
           k: float = PASS_SE, need: int = PASS_FOLDS) -> dict:
    """Apply the pre-registered pass rule to one variant."""
    wins = sum(fold_beats(s, k) for s in fold_stats)
    beats_lgbm = wins >= need
    not_worse_than_market = bool(market_stat["n"] > 0 and market_stat["mean"] <= 0)
    return {"folds_beating_lgbm": wins, "beats_lgbm_rule": beats_lgbm,
            "not_worse_than_market": not_worse_than_market,
            "passed": beats_lgbm and not_worse_than_market}


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    from models.baseline import expected_calibration_error
    return expected_calibration_error(np.asarray(y), np.asarray(p), n_bins)


def summary(y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import brier_score_loss, roc_auc_score
    return {"n": int(len(y)), "log_loss": float(per_game_log_loss(y, p).mean()),
            "brier": float(brier_score_loss(y, p)),
            "auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
            "ece": ece(y, p)}


def no_vig_two_way(home_price: np.ndarray, away_price: np.ndarray) -> np.ndarray:
    """No-vig home probability from two American moneyline prices."""
    def implied(a):
        a = np.asarray(a, dtype=float)
        return np.where(a < 0, -a / (-a + 100.0), 100.0 / (a + 100.0))
    ph, pa = implied(home_price), implied(away_price)
    return ph / (ph + pa)


# ─────────────────────────────────────────────
# Learners
# ─────────────────────────────────────────────
def tabpfn_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover - tabpfn needs torch anyway
        return "cpu"


def make_tabpfn(kind: str, seed: int = SEED, device: Optional[str] = None):
    """A TabPFN-2 classifier ('clf') or regressor ('reg') with the
    pre-registered settings. Imported lazily: the rest of the repo never
    needs torch."""
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.constants import ModelVersion

    device = device or tabpfn_device()
    if device == "cpu":
        # TabPFN refuses >1,000 rows on CPU unless told otherwise
        os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "true")
    cls = TabPFNClassifier if kind == "clf" else TabPFNRegressor
    return cls.create_default_for_version(
        ModelVersion.V2, device=device, n_estimators=N_ESTIMATORS,
        random_state=seed)


Factory = Callable[[str], object]   # kind -> unfitted estimator


def _clf_logit(model, X: np.ndarray) -> np.ndarray:
    p = model.predict_proba(X)
    classes = list(model.classes_)
    p1 = p[:, classes.index(1)]
    return logit(np.clip(p1, *PROB_CLIP))


@dataclass
class FoldModel:
    variant: str
    m: object
    temp_m: tuple
    f: object
    temp_f: tuple
    blind: np.ndarray
    mkt_i: int


def market_prob(X: np.ndarray, names: list) -> np.ndarray:
    return X[:, names.index("market_home_prob")]


def fit_blind(factory: Factory, X, y, train_idx, dates, names):
    """Market-blind fallback F: classifier on all training games, market
    columns removed, temperature on the time tail."""
    blind = np.array([i for i, n in enumerate(names) if n not in MARKET_COLS])
    f_core, f_cal = time_split(train_idx, dates)
    f = factory("clf")
    f.fit(X[f_core][:, blind], y[f_core])
    temp_f = fit_temperature(_clf_logit(f, X[f_cal][:, blind]), y[f_cal])
    return f, temp_f, blind


def fit_market_model(variant: str, factory: Factory, X, y, train_idx, dates, names):
    """Lined-games model M for variant 'A' (feature) or 'B' (residual)."""
    avail_i = names.index("market_available")
    avail_idx = train_idx[X[train_idx, avail_i] == 1.0]
    m_core, m_cal = time_split(avail_idx, dates)
    if variant == "A":
        m = factory("clf")
        m.fit(X[m_core], y[m_core])
    elif variant == "B":
        m = factory("reg")
        resid = y[m_core] - market_prob(X[m_core], names)
        m.fit(X[m_core], resid)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    raw = _raw_logit_m(variant, m, X[m_cal], names)
    return m, fit_temperature(raw, y[m_cal])


def _raw_logit_m(variant: str, m, X: np.ndarray, names: list) -> np.ndarray:
    if variant == "A":
        return _clf_logit(m, X)
    p = np.clip(market_prob(X, names) + m.predict(X), *PROB_CLIP)
    return logit(p)


def fit_fold(variant: str, factory: Factory, X, y, train_idx, dates, names,
             blind_cache: Optional[tuple] = None) -> FoldModel:
    f, temp_f, blind = blind_cache or fit_blind(factory, X, y, train_idx, dates, names)
    m, temp_m = fit_market_model(variant, factory, X, y, train_idx, dates, names)
    return FoldModel(variant, m, temp_m, f, temp_f, blind,
                     names.index("market_available"))


def predict(fm: FoldModel, X: np.ndarray, names: list) -> np.ndarray:
    """Calibrated P(home win): M where a line exists, F where it doesn't."""
    avail = X[:, fm.mkt_i] == 1.0
    out = np.empty(len(X))
    if avail.any():
        a, b = fm.temp_m
        out[avail] = expit(a * _raw_logit_m(fm.variant, fm.m, X[avail], names) + b)
    if (~avail).any():
        a, b = fm.temp_f
        out[~avail] = expit(a * _clf_logit(fm.f, X[~avail][:, fm.blind]) + b)
    return out


def walk_forward(variants, factory_for_seed, X, y, meta, names, folds) -> Dict[str, np.ndarray]:
    """OOF calibrated probabilities per variant key ('A', 'B', 'A_seed1', ...).
    variants: list of (key, variant, seed)."""
    oof = {key: np.full(len(y), np.nan) for key, _, _ in variants}
    for fold in folds:
        t0 = time.time()
        blind_by_seed = {}
        for key, variant, seed in variants:
            factory = factory_for_seed(seed)
            if seed not in blind_by_seed:
                blind_by_seed[seed] = fit_blind(factory, X, y, fold.train_idx,
                                                meta["date"], names)
            fm = fit_fold(variant, factory, X, y, fold.train_idx, meta["date"],
                          names, blind_cache=blind_by_seed[seed])
            oof[key][fold.val_idx] = predict(fm, X[fold.val_idx], names)
        logger.info(f"fold {fold.val_season}: train={len(fold.train_idx)} "
                    f"val={len(fold.val_idx)} ({time.time() - t0:.0f}s)")
    return oof


# ─────────────────────────────────────────────
# Outside market bar for 2024-25 (report only)
# ─────────────────────────────────────────────
def load_pinnacle_close(game_ids) -> pd.Series:
    """Closing Pinnacle no-vig P(home win) per game from raw.odds_history:
    the last h2h snapshot at or before the scheduled start. Read-only."""
    from sqlalchemy import text
    from config.settings import engine
    with engine.connect() as conn:
        rows = pd.read_sql(text("""
            WITH last AS (
                SELECT game_id, MAX(snapshot_ts) AS ts
                FROM raw.odds_history
                WHERE book = 'pinnacle' AND market = 'h2h'
                  AND game_id IS NOT NULL AND snapshot_ts <= commence_time
                GROUP BY game_id)
            SELECT o.game_id,
                   MAX(o.price) FILTER (WHERE o.side = 'home') AS home_price,
                   MAX(o.price) FILTER (WHERE o.side = 'away') AS away_price
            FROM raw.odds_history o
            JOIN last l ON l.game_id = o.game_id AND l.ts = o.snapshot_ts
            WHERE o.book = 'pinnacle' AND o.market = 'h2h'
            GROUP BY o.game_id
        """), conn)
    rows = rows.dropna()
    rows = rows[rows["game_id"].isin(set(game_ids))]
    return pd.Series(no_vig_two_way(rows["home_price"], rows["away_price"]),
                     index=rows["game_id"].to_numpy())


# ─────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────
def evaluate(oof: Dict[str, np.ndarray], lgbm_p: np.ndarray, X, y, meta, names,
             folds, eligible=("A", "B", "C"), pinnacle: Optional[pd.Series] = None) -> dict:
    """Per-fold and pooled metrics, paired tests vs lgbm v2 and the market,
    and the pre-registered decision for each eligible variant."""
    scored = np.zeros(len(y), dtype=bool)
    for f in folds:
        scored[f.val_idx] = True
    avail = X[:, names.index("market_available")] == 1.0
    p_mkt = market_prob(X, names)
    ll_lgbm = per_game_log_loss(y, lgbm_p)
    ll_mkt = per_game_log_loss(y, p_mkt)
    seasons = meta["season"].to_numpy()

    report = {"n_scored": int(scored.sum()), "n_priced": int((scored & avail).sum()),
              "lgbm_v2": {"pooled": summary(y[scored], lgbm_p[scored]),
                          "vs_market": paired(ll_lgbm[scored & avail], ll_mkt[scored & avail])},
              "market": {"pooled_priced": summary(y[scored & avail], p_mkt[scored & avail])},
              "variants": {}}
    fold_rows = []
    for f in folds:
        v = f.val_idx
        pr = v[avail[v]]
        row = {"season": int(f.val_season), "n_train": int(len(f.train_idx)),
               "n_val": int(len(v)), "n_priced": int(len(pr)),
               "lgbm_ll": float(ll_lgbm[v].mean()),
               "market_ll_priced": float(ll_mkt[pr].mean()) if len(pr) else None}
        fold_rows.append(row)
    report["folds"] = fold_rows

    for key, p in oof.items():
        ll = per_game_log_loss(y, p)
        per_fold = []
        for f in folds:
            v = f.val_idx
            pr = v[avail[v]]
            s = paired(ll[v], ll_lgbm[v])
            per_fold.append({"season": int(f.val_season), **summary(y[v], p[v]),
                             "vs_lgbm": s, "beats_lgbm_2se": fold_beats(s),
                             "vs_market_priced": paired(ll[pr], ll_mkt[pr]) if len(pr) else None})
        vs_market = paired(ll[scored & avail], ll_mkt[scored & avail])
        entry = {"pooled": summary(y[scored], p[scored]),
                 "vs_lgbm_pooled": paired(ll[scored], ll_lgbm[scored]),
                 "vs_market_priced_pooled": vs_market,
                 "folds": per_fold,
                 "eligible": key in eligible}
        entry["decision"] = decide([r["vs_lgbm"] for r in per_fold], vs_market)
        if key not in eligible:
            entry["decision"]["passed"] = False
        if pinnacle is not None and len(pinnacle):
            gid = meta["game_id"].to_numpy()
            mask = scored & np.isin(gid, pinnacle.index.to_numpy())
            pin = pinnacle.reindex(gid[mask]).to_numpy()
            entry["vs_pinnacle_close_2024_25"] = paired(ll[mask], per_game_log_loss(y[mask], pin))
        report["variants"][key] = entry

    if pinnacle is not None and len(pinnacle):
        gid = meta["game_id"].to_numpy()
        mask = scored & np.isin(gid, pinnacle.index.to_numpy())
        pin = pinnacle.reindex(gid[mask]).to_numpy()
        report["pinnacle_close_2024_25"] = {
            "n": int(mask.sum()),
            "seasons": sorted({int(s) for s in seasons[mask]}),
            "pinnacle": summary(y[mask], pin),
            "lgbm_v2_vs_pinnacle": paired(ll_lgbm[mask], per_game_log_loss(y[mask], pin))}

    passed = [k for k, e in report["variants"].items() if e["decision"]["passed"]]
    winner = (min(passed, key=lambda k: report["variants"][k]["pooled"]["log_loss"])
              if passed else None)
    report["winner"] = winner
    report["production_change"] = (f"replace lgbm v2 with variant {winner}" if winner
                                   else "none: lgbm v2 stays the production model")
    return report


def run(seeds_report_only=REPORT_ONLY_SEEDS, out: Path = RESULTS_PATH,
        oof_out: Optional[Path] = OOF_PATH) -> dict:
    from models.baseline import load_dataset, walk_forward_folds
    from models.lgbm import run_lgbm

    X, y, meta, names = load_dataset()
    folds = walk_forward_folds(meta)
    device = tabpfn_device()
    logger.info(f"{X.shape[0]} games x {X.shape[1]} features, {len(folds)} folds, "
                f"device={device}")
    if X.shape[1] != 109:
        logger.warning(f"expected the 109 lgbm v2 features, got {X.shape[1]}")

    t0 = time.time()
    lg = run_lgbm(register=False, plot=False)
    lgbm_p = (meta[["game_id"]].merge(lg["oof"][["game_id", "prob_home"]],
                                      on="game_id", how="left")["prob_home"].to_numpy())
    logger.info(f"lgbm v2 reference: pooled {lg['pooled']['log_loss']:.4f} "
                f"({time.time() - t0:.0f}s)")

    variants = [("A", "A", SEED), ("B", "B", SEED)]
    variants += [(f"A_seed{s}", "A", s) for s in seeds_report_only]
    t0 = time.time()
    oof = walk_forward(variants, lambda s: (lambda kind: make_tabpfn(kind, seed=s, device=device)),
                       X, y, meta, names, folds)
    logger.info(f"TabPFN walk-forward done ({time.time() - t0:.0f}s)")
    oof["C"] = 0.5 * (oof["A"] + lgbm_p)

    try:
        season_2425 = meta.loc[meta["season"] == 20242025, "game_id"]
        pinnacle = load_pinnacle_close(season_2425)
    except Exception as e:  # report-only number; never blocks the rule
        logger.warning(f"Pinnacle close unavailable: {e}")
        pinnacle = None

    report = evaluate(oof, lgbm_p, X, y, meta, names, folds, pinnacle=pinnacle)
    report["settings"] = env_versions(device)
    report["lgbm_v2_folds_logged"] = lg["folds"]
    if out:
        out.write_text(json.dumps(report, indent=2, default=_json_default))
        logger.info(f"wrote {out}")
    if oof_out:
        oof_out.parent.mkdir(parents=True, exist_ok=True)
        frame = meta[["game_id", "season", "date"]].copy()
        frame["home_win"] = y
        frame["lgbm_v2"] = lgbm_p
        frame["market"] = np.where(X[:, names.index("market_available")] == 1.0,
                                   market_prob(X, names), np.nan)
        for k, p in oof.items():
            frame[f"tabpfn_{k}"] = p
        frame.dropna(subset=["tabpfn_A"]).to_csv(oof_out, index=False)
    print_report(report)
    return report


def env_versions(device: str) -> dict:
    import importlib.metadata as md
    out = {"device": device, "seed": SEED, "n_estimators": N_ESTIMATORS,
           "weights": "TabPFN-2 (ModelVersion.V2)"}
    for pkg in ("tabpfn", "torch", "lightgbm", "numpy", "scikit-learn"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    try:
        import torch
        out["cuda"] = torch.version.cuda
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return out


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def print_report(r: dict) -> None:
    print(f"\nlgbm v2 pooled log loss {r['lgbm_v2']['pooled']['log_loss']:.4f} "
          f"(n={r['n_scored']}); market on priced games "
          f"{r['market']['pooled_priced']['log_loss']:.4f} (n={r['n_priced']})")
    for key, e in r["variants"].items():
        d, m = e["vs_lgbm_pooled"], e["vs_market_priced_pooled"]
        print(f"\n[{key}] pooled {e['pooled']['log_loss']:.4f}  "
              f"vs lgbm {d['mean']:+.4f} ± {d['se']:.4f} (z {d['z']:+.1f})  "
              f"vs market(priced) {m['mean']:+.4f} ± {m['se']:.4f}  "
              f"ECE {e['pooled']['ece']:.4f}")
        for fr in e["folds"]:
            s = fr["vs_lgbm"]
            print(f"   {fr['season']}: ll {fr['log_loss']:.4f}  vs lgbm "
                  f"{s['mean']:+.4f} ± {s['se']:.4f}  {'BEATS' if fr['beats_lgbm_2se'] else '-'}")
        dec = e["decision"]
        print(f"   decision: folds beating lgbm {dec['folds_beating_lgbm']}/5, "
              f"not worse than market {dec['not_worse_than_market']}, "
              f"{'PASSED' if dec['passed'] else 'not passed'}"
              f"{'' if e['eligible'] else ' (report-only)'}")
        if "vs_pinnacle_close_2024_25" in e:
            s = e["vs_pinnacle_close_2024_25"]
            print(f"   2024-25 vs Pinnacle close: {s['mean']:+.4f} ± {s['se']:.4f} (n={s['n']})")
    print(f"\nProduction change: {r['production_change']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("--no-seeds", action="store_true",
                    help="skip the report-only seed re-runs")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(seeds_report_only=() if args.no_seeds else REPORT_ONLY_SEEDS)
