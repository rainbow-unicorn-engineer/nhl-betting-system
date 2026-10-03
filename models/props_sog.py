"""
models/props_sog.py
Skater shots-on-goal (SOG) props model: a full count distribution of a
skater's shots on goal in a game, given that he plays (toi_seconds > 0),
from which P(SOG > line) is read for the usual prop lines 0.5 .. 4.5.

STATUS: see the STATUS section at the end of this docstring. Registration
is disabled (run_props(register=True) raises): this model has no prices to
be judged against on this database, and nothing reads it yet.

Model M (the candidate), pre-registered before any results were seen:
- One LightGBM regressor, objective poisson, predicting expected SOG.
  It trains with init_score = log(exposure baseline) — the same offset
  trick as the moneyline (market) and totals (league environment)
  models: the trees learn corrections to a sensible starting rate, and
  with no signal they fall back to it.
- Exposure baseline (also baseline B1, below), per player-game:
      shrunk SOG/60 x expected TOI / 60 min x drift ratio
  * shrunk SOG/60: the player's SOG per 60 over his career to date in
    this database (2020-21 onward; every game strictly before the date),
    shrunk toward the league SOG/60 for his position (F or D) over the
    trailing 365 days, with k = 300 minutes of TOI:
        (his SOG + k x prior/60) / (his TOI + k) x 60
    The spec does not name the player's own window; career-to-date was
    chosen before results because it is the reading under which the
    drift ratio below is not a double count (a trailing-365 player rate
    already sits at the current league level).
  * expected TOI: mean TOI over his last 10 games, capped to the range
    (min, max) of his last 20 (the cap cannot bind when both windows are
    full: a mean of 10 of the 20 values lies inside their range).
  * drift ratio: trailing-365-day league SOG/60 (all skaters, dates
    strictly before the game) / the fold's training-window league SOG/60
    (total SOG x 3600 / total TOI over the training rows). It moves a
    career-average rate to today's league level: league SOG/60 fell 6.39
    -> 5.61 from 2021-22 to 2025-26.
- Features (features/player_shots.py, all strictly before the game date):
  rolling SOG/60 and shot-attempt/60 (MoneyPuck attempts incl. missed;
  blocked attempts would count too, but raw.shots on this database has
  none, so this is the unblocked-attempt rate) over the last 5/10/20/40 appearances and season to date, both
  divided by his position's trailing league SOG/60; TOI mean and std over
  the last 5/10; days since his last game and team games missed since;
  F/D; home/away; opponent's shots against per game over its last 10/20
  and his team's shots for per game over its last 10/20 (both divided by
  the trailing league shots per team game); back-to-back; pre-game Elo
  difference (features.elo.compute_elo records each game's rating BEFORE
  its own result — confirmed in code, and the same function fills
  features.matchup). Not used: raw.skater_games pp_toi_seconds,
  sh_toi_seconds, pp_goals, pp_assists, fow, fol — all zero on this
  database (ingestion/nhl_stats.py has never run here).
- Count distribution: negative binomial (NB2: variance = mu + alpha mu^2)
  with mean = the prediction and ONE dispersion alpha per fold, fitted by
  maximum likelihood on that fold's training predictions (alpha >= 0;
  alpha = 0 is Poisson).

Baselines (same rows, same NB-dispersion treatment fitted on their own
training predictions):
- B0 "rolling average": his SOG per game over his last 10 games, any
  season; with fewer than 5 prior games, the league mean SOG per game for
  his position over the trailing 365 days.
- B1 "exposure baseline": exactly the init_score baseline above, no trees.
All three means are clipped to MU_CLIP (a last-10 average of 0 would
otherwise give a zero-mean distribution and infinite surprise).

Evaluation (walk-forward, models.baseline.walk_forward_folds: expanding
season folds, purge gap, a season under 10% of a normal one is not a
fold). Eligible rows — training AND scoring — are player-games with >= 5
prior appearances for that player, regular season and playoffs. Metrics
per fold and pooled: mean negative log-likelihood (NLL → how surprised
the model is by the actual shot count; lower is better) of the observed
SOG under each NB; paired differences M - B1 and M - B0 with paired SE
over player-games; Brier score (→ mean squared error of a probability)
and ECE (→ average gap between predicted and actual frequency, 10 equal-
width bins) of P(SOG > 2.5) and P(SOG > 1.5); for the 2025-26 fold, the
same by position.

GATE (decided before any run): M passes only if pooled NLL(M) < NLL(B1)
by at least 2 paired SE, AND M beats B1 in at least 4 of 5 folds, AND
the pooled ECE of M's P(SOG > 2.5) is <= 0.02. The same comparison vs B0
is reported for information. A pass means "a better forecaster than
simple baselines"; it does NOT mean profitable. That needs prop PRICES
(raw.prop_odds_hist / raw.prop_snapshots do not exist on this database),
and DraftKings props carry about a 6.2% bookmaker margin.

STATUS (2026-10-02, v1, first and only gated run; read-only on the live
database; two further runs reproduced every number): GATE PASSED
as a FORECASTER. Not a betting result: there are no prop prices here.
- Rows: 285,929 played skater games built; 278,541 eligible (>= 5 prior
  appearances); 248,589 scored over 5 folds (2021-22 .. 2025-26;
  2020-21 trains only).
- Pooled NLL: M 1.5783, B1 1.5817, B0 1.6189. M - B1 = -0.00345 (paired
  SE 0.00018 over player-games, so ~19 SE; clustered by game 0.00020, by
  player 0.00021 — information only, the gate uses the row SE).
  M - B0 = -0.04064 (SE 0.00069).
- Per fold, NLL M / B1 / B0 (M - B1, SE):
    2021-22 1.6216 / 1.6247 / 1.6611 (-0.0031, 0.0003)
    2022-23 1.6048 / 1.6077 / 1.6468 (-0.0029, 0.0004)
    2023-24 1.5799 / 1.5838 / 1.6189 (-0.0040, 0.0005)
    2024-25 1.5466 / 1.5505 / 1.5891 (-0.0040, 0.0004)
    2025-26 1.5387 / 1.5420 / 1.5789 (-0.0033, 0.0004)
  M beats B1 in 5 of 5 folds and B0 in 5 of 5.
- P(SOG > 2.5), pooled: Brier M 0.1568, B1 0.1575, B0 0.1628; ECE M
  0.0068 (gate <= 0.02), B1 0.0070, B0 0.0290. P(SOG > 1.5): Brier M
  0.2121, B1 0.2131, B0 0.2210; ECE M 0.0106, B1 0.0146, B0 0.0368.
- 2025-26 by position: forwards (32,958) NLL M 1.5988, B1 1.6013, B0
  1.6390, ECE(>2.5) M 0.0152 vs B1 0.0076; defensemen (16,669) NLL M
  1.4198, B1 1.4246, B0 1.4599, ECE(>2.5) M 0.0073 vs B1 0.0070.
- Fitted dispersion alpha per fold: M 0.042-0.051, B1 0.054-0.062, B0
  0.091-0.102 (shot counts are mildly over-Poisson).
- Weak spot, found after the gate and NOT part of it: M's season-level
  mean drifts. Mean predicted minus actual SOG per game, M / B1: 2021-22
  +0.020 / -0.003, 2022-23 +0.095 / +0.048, 2023-24 +0.078 / +0.053,
  2024-25 +0.026 / +0.048, 2025-26 -0.038 / +0.000. M's per-fold ECE of
  P(> 2.5) (0.008-0.019) is worse than B1's in 4 of 5 folds (2024-25 only
  narrowly; corrected from "3 of 5" by the reproduction review); the pooled
  ECE is low partly because the season biases cancel. This is the same
  failure the totals booster had (models/totals.py, drift correction),
  and its fix — subtract the booster's running same-season mean
  adjustment — is the first thing to try before any pricing use.
- What a pass does and does not mean: M is a better shots forecaster
  than a last-10 average and than the exposure baseline, by a small but
  steady margin over B1 (0.0035 nats a player-game). Whether that beats
  DraftKings props, which carry about a 6.2% margin, is untested: it
  needs stored prop prices (no raw.prop_odds_hist / raw.prop_snapshots
  on this database). Registration stays disabled until that test exists.
"""
import logging

import numpy as np
import pandas as pd

from models.baseline import PURGE_DAYS, expected_calibration_error, walk_forward_folds

logger = logging.getLogger("nhl.models.props_sog")

MODEL_NAME = "props_sog"
MODEL_VERSION = "v1"
# The forecasting gate verdict (STATUS), set by hand. It is NOT a betting
# approval: no price-based check exists yet, and nothing reads it.
GATE_PASSED = True
LINES = (0.5, 1.5, 2.5, 3.5, 4.5)
MIN_PRIOR_GAMES = 5            # eligibility: >= 5 prior appearances
MU_CLIP = (0.02, 15.0)
ALPHA_BOUNDS = (0.0, 5.0)
CAL_FRAC = 0.15                # time-ordered tail for early stopping
GATE_SE = 2.0
GATE_MIN_FOLDS = 4
GATE_ECE = 0.02

LGBM_PARAMS = {
    "objective": "poisson",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 10.0,
    "n_estimators": 2000,
    "random_state": 42,
    "verbosity": -1,
}


# ── Baselines (pure) ───────────────────────────────────────────────

def drift_ratio(league_sog60, train_mean_sog60: float) -> np.ndarray:
    """Trailing league SOG/60 at each row / the training-window mean."""
    if not np.isfinite(train_mean_sog60) or train_mean_sog60 <= 0:
        raise ValueError(f"training-window SOG/60 must be positive, "
                         f"got {train_mean_sog60}")
    return np.asarray(league_sog60, float) / float(train_mean_sog60)


def train_window_sog60(sog, toi_seconds) -> float:
    """League SOG per 60 over a set of rows: total SOG x 3600 / total TOI."""
    return float(np.sum(sog) * 3600.0 / np.sum(toi_seconds))


def exposure_baseline(shrunk_sog60, exp_toi_seconds, ratio) -> np.ndarray:
    """B1 / the offset: shrunk SOG/60 x expected TOI (s) / 3600 x drift."""
    return (np.asarray(shrunk_sog60, float) * np.asarray(exp_toi_seconds, float)
            / 3600.0 * np.asarray(ratio, float))


# ── Negative binomial (pure) ───────────────────────────────────────

def nb_logpmf(y, mu, alpha) -> np.ndarray:
    """log P(Y = y) for NB2 with mean mu and variance mu + alpha mu^2;
    alpha = 0 is the Poisson limit."""
    from scipy.special import gammaln
    y = np.asarray(y, float)
    mu = np.asarray(mu, float)
    if alpha <= 1e-10:
        return y * np.log(mu) - mu - gammaln(y + 1.0)
    r = 1.0 / alpha
    return (gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
            + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu)))


def nb_nll(y, mu, alpha) -> np.ndarray:
    return -nb_logpmf(y, mu, alpha)


def fit_nb_alpha(y, mu) -> float:
    """Maximum-likelihood dispersion alpha in ALPHA_BOUNDS for fixed means."""
    from scipy.optimize import minimize_scalar
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    res = minimize_scalar(lambda a: nb_nll(y, mu, a).mean(),
                          bounds=ALPHA_BOUNDS, method="bounded",
                          options={"xatol": 1e-6})
    a = float(res.x)
    # the bounded search never evaluates the edge itself: check Poisson
    if nb_nll(y, mu, 0.0).mean() <= nb_nll(y, mu, a).mean():
        return 0.0
    return a


def prob_over(mu, alpha, line) -> np.ndarray:
    """P(SOG > line) = 1 - CDF(floor(line))."""
    from scipy.stats import nbinom, poisson
    mu = np.asarray(mu, float)
    k = np.floor(float(line))
    if alpha <= 1e-10:
        return poisson.sf(k, mu)
    r = 1.0 / alpha
    return nbinom.sf(k, r, r / (r + mu))


def count_pmf(mu, alpha, kmax: int = 15) -> np.ndarray:
    """(n, kmax + 1) PMF over 0..kmax (the tail above kmax is left off)."""
    k = np.arange(kmax + 1)
    return np.exp(nb_logpmf(k[None, :], np.asarray(mu, float)[:, None], alpha))


# ── Data ───────────────────────────────────────────────────────────

def load_props_dataset(frame: pd.DataFrame = None) -> pd.DataFrame:
    """Eligible rows (>= MIN_PRIOR_GAMES prior appearances) of the player
    feature frame, sorted by date. frame: an already built
    features.player_shots frame (tests); None builds it from the DB."""
    if frame is None:
        from features.player_shots import load_player_features
        frame = load_player_features()
    df = frame[frame["n_prior"] >= MIN_PRIOR_GAMES].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "game_id", "player_id"]).reset_index(drop=True)


# ── Fitting ────────────────────────────────────────────────────────

def fit_fold(X, y, base, train_idx, dates, params=None) -> dict:
    """Poisson booster with init_score = log(base), early-stopped on the
    time-ordered tail of the training window."""
    import lightgbm as lgb

    from models.lgbm import time_split

    params = dict(LGBM_PARAMS if params is None else params)
    core, cal = time_split(train_idx, dates, CAL_FRAC)
    m = lgb.LGBMRegressor(**params)
    m.fit(X[core], y[core], init_score=np.log(base[core]),
          eval_set=[(X[cal], y[cal])], eval_init_score=[np.log(base[cal])],
          eval_metric="poisson",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return {"model": m, "iters": m.best_iteration_ or params["n_estimators"]}


def predict(fm: dict, X, base) -> np.ndarray:
    """Expected SOG: exp(log(base) + tree adjustment), clipped."""
    raw = fm["model"].booster_.predict(X, raw_score=True)
    return np.clip(np.exp(np.log(base) + raw), *MU_CLIP)


# ── Evaluation ─────────────────────────────────────────────────────

def _paired(a, b) -> tuple:
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def _prob_metrics(y, mu, alpha, line) -> dict:
    p = prob_over(mu, alpha, line)
    hit = (np.asarray(y) > line).astype(float)
    return {"brier": float(np.mean((p - hit) ** 2)),
            "ece": expected_calibration_error(hit, p),
            "mean_p": float(p.mean()), "rate": float(hit.mean())}


def score_block(y, mus: dict, alphas: dict) -> dict:
    """NLL, paired differences vs M, and P(>1.5)/P(>2.5) metrics."""
    out = {"n": len(y), "mean_sog": float(np.mean(y))}
    nll = {k: nb_nll(y, mus[k], alphas[k]) for k in mus}
    for k, mu in mus.items():
        out[f"nll_{k}"] = float(nll[k].mean())
        out[f"mean_mu_{k}"] = float(np.mean(mu))
        for line in (1.5, 2.5):
            pm = _prob_metrics(y, mu, alphas[k], line)
            tag = str(line).replace(".", "")
            out[f"brier{tag}_{k}"] = pm["brier"]
            out[f"ece{tag}_{k}"] = pm["ece"]
            out[f"meanp{tag}_{k}"] = pm["mean_p"]
            out[f"rate{tag}"] = pm["rate"]
    for b in ("B1", "B0"):
        if "M" in nll and b in nll:
            out[f"diff_M_{b}"], out[f"se_M_{b}"] = _paired(nll["M"], nll[b])
    return out


def gate(pooled: dict, folds: list) -> dict:
    """The pre-registered gate (module docstring)."""
    wins_b1 = sum(f["nll_M"] < f["nll_B1"] for f in folds)
    wins_b0 = sum(f["nll_M"] < f["nll_B0"] for f in folds)
    nll_ok = pooled["diff_M_B1"] <= -GATE_SE * pooled["se_M_B1"]
    folds_ok = wins_b1 >= GATE_MIN_FOLDS
    ece_ok = pooled["ece25_M"] <= GATE_ECE
    return {"nll_by_2se": bool(nll_ok), "folds_won_vs_B1": int(wins_b1),
            "folds_ok": bool(folds_ok), "ece25_ok": bool(ece_ok),
            "folds_won_vs_B0": int(wins_b0),
            "b0_nll_by_2se": bool(pooled["diff_M_B0"]
                                  <= -GATE_SE * pooled["se_M_B0"]),
            "passed": bool(nll_ok and folds_ok and ece_ok)}


def run_props(register: bool = False, frame: pd.DataFrame = None,
              params=None) -> dict:
    """Walk-forward evaluation of M, B1 and B0 (module docstring).
    register=True raises: there is no registry entry for this model."""
    if register:
        raise RuntimeError(
            "props_sog registration is disabled: the model has no price-"
            "based evaluation yet (no prop prices on this database). Run "
            "with register=False.")
    from features.player_shots import FEATURES

    df = load_props_dataset(frame)
    X = df[FEATURES].to_numpy(dtype=float)
    y = df["sog"].to_numpy(dtype=float)
    meta = df[["season", "date"]]
    folds = walk_forward_folds(meta)
    logger.info(f"Props SOG dataset: {len(df)} eligible player-games x "
                f"{len(FEATURES)} features, {len(folds)} folds "
                f"(purge {PURGE_DAYS}d)")

    oof = {k: np.full(len(df), np.nan) for k in ("M", "B1", "B0")}
    oof_alpha = {k: np.full(len(df), np.nan) for k in oof}
    fold_metrics = []
    b0_all = np.clip(df["b0_mean"].to_numpy(float), *MU_CLIP)

    for fold in folds:
        tr, val = fold.train_idx, fold.val_idx
        train_mean = train_window_sog60(y[tr], df["toi_seconds"].to_numpy()[tr])
        ratio = drift_ratio(df["league_sog60"].to_numpy(), train_mean)
        base = np.clip(exposure_baseline(df["shrunk_sog60"], df["exp_toi"],
                                         ratio), *MU_CLIP)
        fm = fit_fold(X, y, base, tr, df["date"], params)
        mu = {"M": predict(fm, X[val], base[val]),
              "B1": base[val], "B0": b0_all[val]}
        mu_tr = {"M": predict(fm, X[tr], base[tr]),
                 "B1": base[tr], "B0": b0_all[tr]}
        alphas = {k: fit_nb_alpha(y[tr], mu_tr[k]) for k in mu}
        for k, v in mu.items():
            oof[k][val] = v
            oof_alpha[k][val] = alphas[k]

        m = {"val_season": int(fold.val_season), "n_train": len(tr),
             "iters": int(fm["iters"]), "train_sog60": round(train_mean, 4),
             "mean_drift_ratio": float(ratio[val].mean()),
             **{f"alpha_{k}": round(v, 4) for k, v in alphas.items()},
             **score_block(y[val], mu, alphas)}
        if fold is folds[-1]:
            m["by_position"] = {}
            for pos in ("F", "D"):
                sel = (df["pos_group"].to_numpy()[val] == pos)
                m["by_position"][pos] = score_block(
                    y[val][sel], {k: v[sel] for k, v in mu.items()}, alphas)
        fold_metrics.append(m)
        logger.info(
            f"  fold {m['val_season']}: n={m['n']} NLL M={m['nll_M']:.4f} "
            f"B1={m['nll_B1']:.4f} B0={m['nll_B0']:.4f} | M-B1 "
            f"{m['diff_M_B1']:+.4f} (se {m['se_M_B1']:.4f}) | ECE>2.5 "
            f"M={m['ece25_M']:.4f} | alpha M={alphas['M']:.3f} "
            f"| iters={m['iters']} drift={m['mean_drift_ratio']:.3f}")

    scored = ~np.isnan(oof["M"])
    ys = y[scored]
    pooled = {"n": int(scored.sum()), "mean_sog": float(ys.mean())}
    nll = {}
    # per-row NLL under each row's own fold alpha
    for k in oof:
        a_rows = oof_alpha[k][scored]
        nll[k] = np.empty(len(ys))
        for a in np.unique(a_rows):
            sel = a_rows == a
            nll[k][sel] = nb_nll(ys[sel], oof[k][scored][sel], a)
        pooled[f"nll_{k}"] = float(nll[k].mean())
        for line in (1.5, 2.5):
            p = np.empty(len(ys))
            for a in np.unique(a_rows):
                sel = a_rows == a
                p[sel] = prob_over(oof[k][scored][sel], a, line)
            hit = (ys > line).astype(float)
            tag = str(line).replace(".", "")
            pooled[f"brier{tag}_{k}"] = float(np.mean((p - hit) ** 2))
            pooled[f"ece{tag}_{k}"] = expected_calibration_error(hit, p)
    for b in ("B1", "B0"):
        pooled[f"diff_M_{b}"], pooled[f"se_M_{b}"] = _paired(nll["M"], nll[b])
    pooled["gate"] = gate(pooled, fold_metrics)
    pooled["gate_passed"] = pooled["gate"]["passed"]

    logger.info(
        f"POOLED OOF ({pooled['n']} player-games): NLL M={pooled['nll_M']:.4f} "
        f"B1={pooled['nll_B1']:.4f} B0={pooled['nll_B0']:.4f} | M-B1 "
        f"{pooled['diff_M_B1']:+.5f} (se {pooled['se_M_B1']:.5f}) M-B0 "
        f"{pooled['diff_M_B0']:+.5f} (se {pooled['se_M_B0']:.5f}) | ECE>2.5 "
        f"M={pooled['ece25_M']:.4f} | GATE "
        f"{'PASSED' if pooled['gate_passed'] else 'FAILED'} {pooled['gate']}")

    out = df.loc[scored, ["player_id", "game_id", "season", "date",
                          "pos_group", "sog"]].copy()
    for k in oof:
        out[f"mu_{k}"] = oof[k][scored]
        out[f"alpha_{k}"] = oof_alpha[k][scored]
    return {"folds": fold_metrics, "pooled": pooled, "oof": out}


def _print_report(res: dict) -> None:
    import json

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, float):
            return round(o, 5)
        return o
    print(json.dumps({"folds": clean(res["folds"]),
                      "pooled": clean(res["pooled"])}, indent=1))


def main(argv=None):
    """Command line: --evaluate runs the walk-forward evaluation (reads
    the database, writes nothing) and prints the metrics as JSON."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Walk-forward evaluation of the skater shots-on-goal "
                    "props model vs two baselines (read only; never "
                    "registers)")
    parser.add_argument("--evaluate", action="store_true",
                        help="run the evaluation and print the metrics")
    args = parser.parse_args(argv)
    if not args.evaluate:
        parser.print_help()
        return None
    res = run_props(register=False)
    _print_report(res)
    return res


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
