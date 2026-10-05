"""
models/xg.py
Our own expected-goals (xG) model (PROJECT_CONTEXT "Layer A"), its gate
against MoneyPuck's xG, and a downstream test of whether our xG makes the
moneyline and props models better than MoneyPuck's xG does.

xG → the chance that one shot becomes a goal, judged from where and how
it was taken (features/xg.py has the inputs). Today every xG number in the
feature store is MoneyPuck's (raw.shots.xg_moneypuck): team xG share,
power-play and penalty-kill xG rates, and goalie GSAx (→ goals saved above
expected: the xG a goalie faced minus the goals he let in). This module
asks whether our own model can replace it.

STATUS: see the end of this docstring.

PRE-REGISTRATION (written and committed 2026-10-04, before any variant
was run on real data)
=====================================================================

Data: every unblocked shot attempt in raw.shots (SHOT, GOAL, MISS),
2020-21 .. 2025-26 (683,721 shots; none yet for 2026-27), regular season
and playoffs. Label: is_goal.

Model: one LightGBM binary classifier per variant (XG_PARAMS, fixed
before any run; shot type as a categorical input). No further
calibration step: a log-loss-trained booster is scored as it comes out.

Variants (both reported; the gate and the downstream test use X2 only):
- X1 "core": CORE_FEATURES (location, distance, angle, shot type,
  rebound and rush flags, strength and empty net, period and time,
  overtime, playoffs, score difference, home shooter).
- X2 "core + prior event" (PRIMARY): X1 plus PRIOR_FEATURES, derived
  from the previous unblocked attempt in the same game (seconds since,
  same team, distance the puck moved, angle change per second, was it a
  goal, its distance) and the shooting team's attempts in the previous 10
  seconds.

Walk-forward: models.baseline.walk_forward_folds on the shots' (season,
game date): for each held-out season S (2021-22 .. 2025-26) the model
trains only on shots from earlier seasons (dated more than the 7-day
purge before S starts). Early stopping (100 rounds) uses the last 15% of
the training shots by date (models.lgbm.time_split); S never touches the
fit. 2020-21 has no earlier season: it is never scored by the gate; for
the downstream test only, its shots get 5-fold cross-fitted xG (folds by
game: each game's shots scored by a model trained on the other 80% of
2020-21 games). Those values feed only training rows of the downstream
models (the 2020-21 rolling features and the 2021-22 goalie prior, all
earlier than any scored game).

Comparison (pooled over the held-out shots of 2021-22 .. 2025-26, the
SAME shots for both; per season reported too). MoneyPuck = raw.shots.
xg_moneypuck. Both sets of probabilities are clipped to [1e-6, 1 - 1e-6]
for log loss.
- AUC (→ the chance a random goal got a higher xG than a random
  non-goal; 0.5 is guessing, higher is better).
- Log loss (→ how surprised the probabilities were by what happened;
  lower is better), with the paired difference ours - MoneyPuck per shot
  and its standard error (SE → the size of the random wobble in that
  difference) over shots. The SE clustered by game is reported for
  information only.
- Calibration (→ whether "10%" shots really go in 10% of the time): ECE
  = the shot-weighted average gap |actual goal rate - mean xG| over 10
  equal-count bins (deciles) of each model's own predictions; the
  reliability table by decile is reported for both. The equal-width
  10-bin ECE (models.baseline) is reported for information.

GATE (X2): passes only if ALL of
  (1) AUC(ours) >= AUC(MoneyPuck) - 0.005;
  (2) ECE(ours) <= ECE(MoneyPuck) + 0.005 (decile ECE);
  (3) log loss within 1 SE of MoneyPuck or better: mean(ours - MP) <= 1 SE.

Caveat known in advance: MoneyPuck's model is trained on many past
seasons and may have been refit on some of the held-out seasons, and it
uses inputs raw.shots doesn't keep (e.g. the type and location of the
previous event, including blocked shots, hits and faceoffs). Ours sees
only earlier seasons. The comparison is therefore tilted toward
MoneyPuck; that is accepted, since a model we run live can only ever
train on the past.

DOWNSTREAM TEST (pre-registered with the gate; X2 xG only)
----------------------------------------------------------
Two arms built the same way, in memory, from the same raw tables; they
differ ONLY in which per-shot xG column feeds the existing feature code.
Nothing is written to the database.
- MP arm: xg_moneypuck. OUR arm: X2 walk-forward xG (2021-22 .. 2025-26
  from models trained on earlier seasons, 2020-21 cross-fitted).
- Moneyline: features/team_features.py (xgf_pct, pp_xgf_per60,
  pk_xga_per60) and features/goalie_features.py (GSAx, high-danger save
  share with its xG >= 0.20 cut, xGA per 60, the league GSAx prior) are
  recomputed by their own pure functions with the xG column swapped,
  rounded to the database column scales as stored, assembled by
  features.build_vectors.assemble, and scored by the production
  moneyline model's own fold code (models.lgbm.fit_fold / predict_fold,
  same params, same walk-forward folds). Metric: pooled walk-forward log
  loss of the calibrated P(home win); paired per-game difference OUR - MP
  with its SE.
- Props: the props model doesn't use xG today, so both arms ADD the same
  five player xG features (features.xg.PROPS_XG_FEATURES: his xG per 60
  over his last 10 / 20 games and season to date, his xG per attempt over
  his last 20, the opponent's xG allowed per game over its last 20; all
  relative to trailing league rates, all strictly before the game date)
  to models.props_sog's v2 model (drift-corrected M), same folds, same
  params. Metric: pooled negative-binomial NLL per player-game; paired
  difference OUR - MP with its SE. The current props model without xG
  features (P0) is reported for information.

DOWNSTREAM RULE: our xG replaces MoneyPuck's in the feature store only
if ALL of
  (a) the X2 gate passes;
  (b) moneyline: pooled log loss OUR - MP <= 0 (not worse);
  (c) props: pooled NLL OUR - MP <= +1 SE (not worse beyond noise).
An "improves" claim needs the difference to be at least 2 SE below 0;
(b) and (c) only ask "no worse", because the point of owning the model
is independence from MoneyPuck's xG column (we would still use
MoneyPuck's shot locations). If the rule fails, the feature store keeps
MoneyPuck's xG and this module stays an opt-in experiment. The props
model's production default doesn't change either way (it has no
registry entry and failed its market check).

How to run (read-only; writes nothing to the database):
    python -m models.xg --evaluate      # shot-level gate, X1 and X2
    python -m models.xg --downstream    # gate + moneyline + props test

STATUS (run 2026-10-04, read-only; pre-registration committed in 3dae426
before the first real-data run)
=====================================================================

VERDICT: the gate FAILS (AUC and log loss), so by the downstream rule
our xG is NOT adopted. The feature store keeps MoneyPuck's xG; this
module stays an opt-in experiment. No production default changed.

Check first: the MoneyPuck arm rebuilt in memory (team xG sums, goalie
GSAx and high-danger counts, the league GSAx/60 prior) equals the stored
SQL path exactly for 2021-22 and 2022-23.

Shot level, pooled over 605,110 held-out shots of 2021-22 .. 2025-26
(43,047 goals; no shot lacked a MoneyPuck value):

  variant  AUC ours / MP    log loss ours / MP   diff (SE)           ECE ours / MP  gate
  X1       0.7578 / 0.7865  0.22581 / 0.21605    +0.00976 (0.00025)  0.0054 / 0.0110  FAIL
  X2       0.7608 / 0.7865  0.22465 / 0.21605    +0.00860 (0.00025)  0.0048 / 0.0110  FAIL

(SE clustered by game: 0.00026 for both. ECE = decile ECE; equal-width
ECE X2 0.0045 vs MP 0.0103.) X2 passes calibration (2) and fails AUC (1,
needs >= 0.7815) and log loss (3, it is about 35 SE worse). The prior-
event inputs help (X2 beats X1 by 0.003 AUC, 0.0012 log loss), but not
nearly enough.

X2 by season (AUC ours / MP; log loss diff, SE about 0.00055):
  2021-22 0.7596 / 0.7978  +0.0142    2024-25 0.7665 / 0.7785  +0.0029
  2022-23 0.7559 / 0.7905  +0.0131    2025-26 0.7607 / 0.7764  +0.0039
  2023-24 0.7624 / 0.7901  +0.0088
MoneyPuck's lead is largest in the oldest seasons and shrinks to about
0.012-0.016 AUC in the two newest, consistent with the caveat written in
advance (MoneyPuck's model has probably been fitted on the older seasons
we test on). Even the newest seasons fail the 0.005 bar.

Reliability by decile, X2 (mean xG -> actual goal rate): ours tracks the
diagonal everywhere except the top decile (0.251 -> 0.224); MoneyPuck
under-predicts deciles 4-7 (e.g. 0.061 -> 0.077) and over-predicts the
top decile (0.312 -> 0.254). Ours is the better-calibrated, MoneyPuck's
the sharper (→ better at telling a good chance from a bad one), and
sharpness is what AUC and log loss reward.

Downstream (run anyway, for information; the rule fails at (a)):
- Moneyline, 6,993 games, pooled walk-forward log loss MP 0.66132,
  OUR 0.66311: OUR - MP = +0.00179 (SE 0.00075), about 2.4 SE WORSE.
  By season: 2021-22 +0.0006, 2022-23 -0.0002, 2023-24 -0.0007,
  2024-25 +0.0044 (SE 0.0026), 2025-26 +0.0048 (SE 0.0022). (b) FAILS.
- Props, 248,589 player-games, NLL P0 1.578181, P_MP 1.578261,
  P_OUR 1.578263: OUR - MP = +0.000002 (SE 0.00003). (c) passes. But
  adding the five player xG features makes the props model slightly
  WORSE whichever xG feeds them (P_MP - P0 = +0.00008, SE 0.000035,
  about 2.3 SE): the props model doesn't want xG features at all.
- Decision: a_gate False, b_moneyline_not_worse False,
  c_props_within_1se True -> adopt False.

Fixed after the first run, method unchanged: props_comparison crashed
scoring NLL because props_sog.nb_nll takes one dispersion at a time and
each row carries its own fold's; nb_nll_rows scores each row with its own
fold's value, as run_props itself does. The props arms were not refitted
differently.

The other attempt at this track (branch model/xg-layer-a, file
experiments/2026-10-04-xg-layer-a.md) pre-registered a different bar
(AUC >= 0.77, log loss within +0.002 of MoneyPuck, equal-width ECE <=
0.01, empty-net shots excluded, temperature scaling, a totals value
check). This module's pre-registration is the one that governs. For the
record, X2 fails that bar too (AUC 0.761, log loss +0.0086).

What would close the gap (not tried; each would need its own pre-
registration): the previous NON-shot event (faceoff, hit, giveaway,
blocked shot) with its location and time, from the NHL play-by-play;
shooter handedness (off-wing shots); per-rink location corrections fitted
by us.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from features.xg import ALL_FEATURES, CATEGORICAL, CORE_FEATURES, shot_features
from models.baseline import expected_calibration_error, walk_forward_folds

logger = logging.getLogger("nhl.models.xg")

MODEL_NAME = "xg_lgbm"
MODEL_VERSION = "v1"
GATE_PASSED = False         # STATUS: AUC and log loss fail vs MoneyPuck (2026-10-04)
CAL_FRAC = 0.15
EARLY_STOP = 100
CROSSFIT_FOLDS = 5
P_CLIP = (1e-6, 1 - 1e-6)
GATE_AUC_TOL = 0.005
GATE_ECE_TOL = 0.005
GATE_LL_SE = 1.0
VARIANTS = {"X1": CORE_FEATURES, "X2": ALL_FEATURES}
PRIMARY = "X2"
XG_COL = "xg_ours"

XG_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 500,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "n_estimators": 3000,
    "random_state": 42,
    "n_jobs": 16,
    "verbosity": -1,
}

# Database column scales of the stored feature tables (db/schema.sql), so
# the in-memory arms carry the same rounding as stored vectors
TEAM_SCALE = {"gf_per60": 3, "ga_per60": 3, "xgf_pct": 3, "cf_pct": 3,
              "ff_pct": 3, "sh_pct": 3, "sv_pct": 4, "pdo": 3, "pp_pct": 3,
              "pk_pct": 3, "pp_xgf_per60": 3, "pk_xga_per60": 3,
              "fow_pct": 3, "pim_per60": 2}
GOALIE_SCALE = {"shrunk_sv_pct": 4, "shrunk_gsax": 3, "credibility_z": 3}


# ── Metrics (pure) ─────────────────────────────────────────────────

def clip_p(p) -> np.ndarray:
    return np.clip(np.asarray(p, float), *P_CLIP)


def shot_log_loss(y, p) -> np.ndarray:
    """Per-shot log loss."""
    y = np.asarray(y, float)
    p = clip_p(p)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def decile_bins(p, n_bins: int = 10) -> np.ndarray:
    """Equal-count bins of p (ties broken by order), 0 .. n_bins-1."""
    r = pd.Series(np.asarray(p, float)).rank(method="first").to_numpy()
    return np.minimum((r - 1) * n_bins // len(r), n_bins - 1).astype(int)


def reliability_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Per decile of p: shots, mean predicted, actual goal rate."""
    b = decile_bins(p, n_bins)
    df = pd.DataFrame({"bin": b, "y": np.asarray(y, float), "p": np.asarray(p, float)})
    t = df.groupby("bin").agg(n=("y", "size"), mean_p=("p", "mean"),
                              rate=("y", "mean")).reset_index()
    t["gap"] = t["rate"] - t["mean_p"]
    return t


def ece_deciles(y, p, n_bins: int = 10) -> float:
    """Shot-weighted mean |actual - predicted| over equal-count bins."""
    t = reliability_table(y, p, n_bins)
    return float((t["n"] * t["gap"].abs()).sum() / t["n"].sum())


def paired(a, b, clusters=None) -> dict:
    """mean(a - b), its SE over rows and (optionally) clustered."""
    d = np.asarray(a, float) - np.asarray(b, float)
    out = {"diff": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))),
           "n": int(len(d))}
    if clusters is not None:
        s = pd.Series(d - d.mean()).groupby(np.asarray(clusters)).sum()
        out["se_cluster"] = float(np.sqrt((s ** 2).sum()) / len(d))
    return out


def score_xg(y, p) -> dict:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return {"n": int(len(y)), "goals": int(y.sum()), "mean_p": float(p.mean()),
            "rate": float(y.mean()), "auc": float(roc_auc_score(y, p)),
            "log_loss": float(shot_log_loss(y, p).mean()),
            "ece_decile": ece_deciles(y, p),
            "ece_width10": float(expected_calibration_error(y, p))}


def xg_gate(ours: dict, mp: dict, ll: dict) -> dict:
    """The pre-registered gate (module docstring)."""
    auc_ok = ours["auc"] >= mp["auc"] - GATE_AUC_TOL
    ece_ok = ours["ece_decile"] <= mp["ece_decile"] + GATE_ECE_TOL
    ll_ok = ll["diff"] <= GATE_LL_SE * ll["se"]
    return {"auc_ok": bool(auc_ok), "ece_ok": bool(ece_ok),
            "log_loss_ok": bool(ll_ok),
            "passed": bool(auc_ok and ece_ok and ll_ok)}


def downstream_decision(gate_passed: bool, ml: dict, props: dict) -> dict:
    """The pre-registered downstream rule (module docstring)."""
    a = bool(gate_passed)
    b = ml["diff"] <= 0.0
    c = props["diff"] <= GATE_LL_SE * props["se"]
    return {"a_gate": a, "b_moneyline_not_worse": bool(b),
            "c_props_within_1se": bool(c),
            "moneyline_improves_2se": bool(ml["diff"] <= -2 * ml["se"]),
            "props_improves_2se": bool(props["diff"] <= -2 * props["se"]),
            "adopt": bool(a and b and c)}


# ── Fitting ────────────────────────────────────────────────────────

def fit_xg(X: pd.DataFrame, y: np.ndarray, train_idx: np.ndarray, dates,
           params=None) -> dict:
    """LightGBM fitted on train_idx, early-stopped on its last CAL_FRAC by
    date."""
    import lightgbm as lgb

    from models.lgbm import time_split

    params = dict(XG_PARAMS if params is None else params)
    core, cal = time_split(np.asarray(train_idx), pd.Series(np.asarray(dates)),
                           CAL_FRAC)
    cats = [c for c in CATEGORICAL if c in X.columns]
    m = lgb.LGBMClassifier(**params)
    m.fit(X.iloc[core], y[core], eval_set=[(X.iloc[cal], y[cal])],
          eval_metric="binary_logloss", categorical_feature=cats,
          callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    return {"model": m, "iters": int(m.best_iteration_ or params["n_estimators"]),
            "features": list(X.columns)}


def predict_xg(fm: dict, X: pd.DataFrame) -> np.ndarray:
    return fm["model"].predict_proba(X[fm["features"]])[:, 1]


def walk_forward_xg(shots: pd.DataFrame, feats: pd.DataFrame,
                    features: list, params=None, crossfit_first: bool = False,
                    crossfit_folds: int = CROSSFIT_FOLDS) -> dict:
    """Out-of-sample xG for every shot of every held-out season (model
    trained on earlier seasons only). crossfit_first also fills the first
    season by game-grouped cross-fitting (downstream training rows only).
    Returns {"oof": array (NaN where not scored), "folds": [...]}."""
    meta = shots[["season", "date"]].reset_index(drop=True)
    meta["date"] = pd.to_datetime(meta["date"])
    y = shots["is_goal"].astype(int).to_numpy()
    X = feats[features].reset_index(drop=True)
    dates = meta["date"]
    oof = np.full(len(y), np.nan)
    folds_out = []
    for fold in walk_forward_folds(meta):
        fm = fit_xg(X, y, fold.train_idx, dates, params)
        oof[fold.val_idx] = predict_xg(fm, X.iloc[fold.val_idx])
        folds_out.append({"val_season": int(fold.val_season),
                          "n_train": int(len(fold.train_idx)),
                          "n_val": int(len(fold.val_idx)), "iters": fm["iters"]})
        logger.info(f"  xG fold {fold.val_season}: train {len(fold.train_idx)} "
                    f"val {len(fold.val_idx)} iters {fm['iters']}")
    if crossfit_first:
        first = sorted(meta["season"].unique())[0]
        idx = np.flatnonzero(meta["season"].to_numpy() == first)
        games = shots["game_id"].to_numpy()[idx]
        ug = np.unique(games)
        rng = np.random.default_rng(42)
        fold_of = dict(zip(ug, rng.permutation(len(ug)) % crossfit_folds))
        f = np.array([fold_of[g] for g in games])
        for k in range(crossfit_folds):
            tr, va = idx[f != k], idx[f == k]
            fm = fit_xg(X, y, tr, dates, params)
            oof[va] = predict_xg(fm, X.iloc[va])
        logger.info(f"  xG cross-fit {first}: {len(idx)} shots, "
                    f"{crossfit_folds} game folds")
    return {"oof": oof, "folds": folds_out}


# ── Shot-level evaluation ──────────────────────────────────────────

def evaluate(shots: pd.DataFrame = None, feats: pd.DataFrame = None,
             variants=None, params=None, crossfit_first: bool = False) -> dict:
    """Walk-forward xG for each variant and the pre-registered comparison
    with MoneyPuck. Read-only."""
    if shots is None:
        from features.xg import load_shots
        shots = load_shots()
    shots = shots.reset_index(drop=True)
    if feats is None:
        feats = shot_features(shots)
    variants = variants or list(VARIANTS)
    y = shots["is_goal"].astype(int).to_numpy()
    mp = shots["xg_moneypuck"].to_numpy(float)
    res = {"variants": {}, "n_shots": int(len(shots))}
    for v in variants:
        logger.info(f"xG variant {v} ({len(VARIANTS[v])} features)")
        wf = walk_forward_xg(shots, feats, VARIANTS[v], params,
                             crossfit_first=crossfit_first and v == PRIMARY)
        p = wf["oof"]
        held = np.isin(shots["season"].to_numpy(),
                       [f["val_season"] for f in wf["folds"]])
        sel = held & ~np.isnan(p) & ~np.isnan(mp)
        out = {"folds": wf["folds"], "oof": p,
               "excluded_no_mp": int((held & np.isnan(mp)).sum())}
        out["pooled"] = compare(y[sel], p[sel], mp[sel],
                                shots["game_id"].to_numpy()[sel])
        out["by_season"] = {}
        for s in sorted(np.unique(shots["season"].to_numpy()[sel])):
            ss = sel & (shots["season"].to_numpy() == s)
            out["by_season"][int(s)] = compare(
                y[ss], p[ss], mp[ss], shots["game_id"].to_numpy()[ss],
                tables=False)
        g = out["pooled"]["gate"]
        logger.info(
            f"{v} pooled: AUC {out['pooled']['ours']['auc']:.4f} vs MP "
            f"{out['pooled']['mp']['auc']:.4f}; LL {out['pooled']['ours']['log_loss']:.5f}"
            f" vs {out['pooled']['mp']['log_loss']:.5f} (diff "
            f"{out['pooled']['ll']['diff']:+.5f}, se {out['pooled']['ll']['se']:.5f});"
            f" ECE {out['pooled']['ours']['ece_decile']:.4f} vs "
            f"{out['pooled']['mp']['ece_decile']:.4f} -> gate {g}")
        res["variants"][v] = out
    if PRIMARY in res["variants"]:
        res["gate"] = res["variants"][PRIMARY]["pooled"]["gate"]
    return res


def compare(y, p_ours, p_mp, games, tables: bool = True) -> dict:
    ours, mp = score_xg(y, p_ours), score_xg(y, p_mp)
    ll = paired(shot_log_loss(y, p_ours), shot_log_loss(y, p_mp), games)
    out = {"ours": ours, "mp": mp, "ll": ll, "gate": xg_gate(ours, mp, ll)}
    if tables:
        out["reliability_ours"] = reliability_table(y, p_ours).round(5).to_dict("records")
        out["reliability_mp"] = reliability_table(y, p_mp).round(5).to_dict("records")
    return out


# ── Downstream: moneyline vectors in memory ────────────────────────

def _round_cols(df: pd.DataFrame, scales: dict) -> pd.DataFrame:
    df = df.copy()
    for c, k in scales.items():
        if c in df:
            df[c] = df[c].astype(float).round(k)
    return df


def build_ml_dataset(shots: pd.DataFrame, xg_col: str) -> tuple:
    """(X, y, meta, names) for the moneyline model with the feature store's
    xG inputs taken from shots[xg_col]. Reads with SELECT only; writes
    nothing. Mirrors build_all's order: team rolling, goalie rolling (per
    season, each with its previous-season league prior), vectors."""
    from features import build_vectors as BV
    from features import goalie_features as GF
    from features import team_features as TF
    from features.xg import goalie_xg_sums, team_xg_sums

    base = TF.load_base(None, xg_sums=team_xg_sums(shots, xg_col))
    tr = _round_cols(TF.compute_rolling(base), TEAM_SCALE)
    team_wide = BV.team_wide(tr)

    gsums = goalie_xg_sums(shots, xg_col)
    frames = []
    for s in sorted(base["season"].unique()):
        gb = GF.load_goalie_base(int(s), xg_sums=gsums)
        if gb.empty:
            continue
        sv, g60 = GF.league_priors(int(s), xg_shots=shots, xg_col=xg_col)
        frames.append(GF.compute_goalie_rolling(gb, league_sv=sv,
                                                league_gsax60=g60))
    gr = _round_cols(pd.concat(frames, ignore_index=True), GOALIE_SCALE)
    goalie_wide = BV.goalie_wide(gr)

    games = BV._load_games(None)
    df = BV.assemble(games, team_wide, BV._load_starters(None), goalie_wide,
                     market=BV._load_market(None))
    df = df[df["home_win"].notna()].sort_values(["date", "game_id"]).reset_index(drop=True)
    names = list(BV.FEATURE_NAMES)
    X = df[names].to_numpy(dtype=float)
    y = df["home_win"].astype(int).to_numpy()
    meta = df[["game_id", "season", "date"]].copy()
    meta["date"] = pd.to_datetime(meta["date"])
    return X, y, meta, names


def evaluate_ml(X, y, meta, names) -> dict:
    """models.lgbm's walk-forward, on an in-memory dataset: per-game
    calibrated OOF P(home win) and per-fold log loss."""
    from sklearn.metrics import log_loss

    from models.lgbm import fit_fold, market_offset, predict_fold

    base = market_offset(X, names)
    avail_i = names.index("market_available")
    oof = np.full(len(y), np.nan)
    folds = []
    for fold in walk_forward_folds(meta):
        fm = fit_fold(X, y, base, fold.train_idx, meta["date"], names)
        v = fold.val_idx
        oof[v] = predict_fold(fm, X[v], base[v], X[v, avail_i] == 1.0)
        folds.append({"val_season": int(fold.val_season), "n": int(len(v)),
                      "log_loss": float(log_loss(y[v], oof[v]))})
    return {"oof": oof, "folds": folds}


def ml_comparison(shots: pd.DataFrame, ours_col: str = XG_COL) -> dict:
    """Both arms built and scored identically; paired per-game log loss."""
    arms = {}
    for arm, col in (("MP", "xg_moneypuck"), ("OUR", ours_col)):
        X, y, meta, names = build_ml_dataset(shots, col)
        ev = evaluate_ml(X, y, meta, names)
        arms[arm] = {"y": y, "meta": meta, **ev}
        logger.info(f"moneyline arm {arm}: folds {ev['folds']}")
    a, b = arms["OUR"], arms["MP"]
    if not np.array_equal(a["meta"]["game_id"].to_numpy(), b["meta"]["game_id"].to_numpy()):
        raise RuntimeError("the two arms scored different games")
    sel = ~np.isnan(a["oof"]) & ~np.isnan(b["oof"])
    y = a["y"][sel]
    ll = {k: shot_log_loss(y, arms[k]["oof"][sel]) for k in arms}
    out = {"n": int(sel.sum()),
           "log_loss_MP": float(ll["MP"].mean()),
           "log_loss_OUR": float(ll["OUR"].mean()),
           **paired(ll["OUR"], ll["MP"]),
           "folds": {k: arms[k]["folds"] for k in arms}}
    seasons = a["meta"]["season"].to_numpy()[sel]
    out["by_season"] = {int(s): paired(ll["OUR"][seasons == s], ll["MP"][seasons == s])
                        for s in np.unique(seasons)}
    return out


# ── Downstream: props ──────────────────────────────────────────────

def nb_nll_rows(y, mu, alpha) -> np.ndarray:
    """Per-row negative-binomial NLL when each row carries its own fold's
    dispersion alpha (props_sog.nb_nll takes one alpha at a time)."""
    from models.props_sog import nb_nll
    y, mu, alpha = (np.asarray(v, float) for v in (y, mu, alpha))
    out = np.empty(len(y))
    for a in np.unique(alpha):
        sel = alpha == a
        out[sel] = nb_nll(y[sel], mu[sel], float(a))
    return out


def props_comparison(shots: pd.DataFrame, ours_col: str = XG_COL,
                     frame: pd.DataFrame = None) -> dict:
    """P0 (today's props v2), P_MP and P_OUR (+ PROPS_XG_FEATURES);
    paired per-player-game NLL of the reported model M."""
    from features.player_shots import FEATURES, load_player_features
    from features.xg import PROPS_XG_FEATURES, player_xg_features
    from models.props_sog import run_props

    if frame is None:
        frame = load_player_features()
    runs = {"P0": run_props(frame=frame)}
    for arm, col in (("P_MP", "xg_moneypuck"), ("P_OUR", ours_col)):
        f = player_xg_features(frame, shots, col)
        runs[arm] = run_props(frame=f, features=list(FEATURES) + PROPS_XG_FEATURES)
    nll = {}
    keys = None
    for k, r in runs.items():
        o = r["oof"].sort_values(["date", "game_id", "player_id"]).reset_index(drop=True)
        kk = o[["player_id", "game_id"]].to_numpy()
        if keys is None:
            keys = kk
        elif not np.array_equal(keys, kk):
            raise RuntimeError("props arms scored different player-games")
        nll[k] = nb_nll_rows(o["sog"], o["mu_M"], o["alpha_M"])
    games = keys[:, 1]
    out = {"n": int(len(keys)),
           **{f"nll_{k}": float(v.mean()) for k, v in nll.items()},
           **paired(nll["P_OUR"], nll["P_MP"], games),
           "P_MP_vs_P0": paired(nll["P_MP"], nll["P0"], games),
           "P_OUR_vs_P0": paired(nll["P_OUR"], nll["P0"], games),
           "gates": {k: r["pooled"]["gate"] for k, r in runs.items()},
           "folds": {k: [{"val_season": f["val_season"], "nll_M": f["nll_M"]}
                         for f in r["folds"]] for k, r in runs.items()}}
    return out


# ── Orchestration ──────────────────────────────────────────────────

def run_downstream(shots: pd.DataFrame = None, cache: Path = None) -> dict:
    """Gate (X1 and X2), then the moneyline and props downstream tests."""
    from features.xg import load_shots
    if shots is None:
        shots = load_shots()
    shots = shots.reset_index(drop=True)
    ev = evaluate(shots, crossfit_first=True)
    shots = shots.copy()
    shots[XG_COL] = ev["variants"][PRIMARY]["oof"]
    missing = int(np.isnan(shots[XG_COL]).sum())
    if missing:
        raise RuntimeError(f"{missing} shots have no out-of-sample xG")
    if cache is not None:
        shots[["shot_id", "season", XG_COL]].to_parquet(cache)
    ml = ml_comparison(shots)
    logger.info(f"moneyline OUR - MP: {ml['diff']:+.5f} (se {ml['se']:.5f})")
    props = props_comparison(shots)
    logger.info(f"props OUR - MP: {props['diff']:+.5f} (se {props['se']:.5f})")
    decision = downstream_decision(ev["gate"]["passed"], ml, props)
    logger.info(f"downstream decision: {decision}")
    return {"evaluate": ev, "moneyline": ml, "props": props, "decision": decision}


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return None
    if isinstance(o, (np.floating, float)):
        return round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def main(argv=None):
    import argparse
    import json
    parser = argparse.ArgumentParser(
        description="Our xG model vs MoneyPuck's (read-only; writes nothing "
                    "to the database)")
    parser.add_argument("--evaluate", action="store_true",
                        help="shot-level walk-forward and gate (X1, X2)")
    parser.add_argument("--downstream", action="store_true",
                        help="gate plus the moneyline and props tests")
    parser.add_argument("--out", type=Path, default=None,
                        help="also write the JSON report here")
    args = parser.parse_args(argv)
    if args.downstream:
        res = run_downstream()
    elif args.evaluate:
        res = evaluate()
    else:
        parser.print_help()
        return None
    text = json.dumps(_clean(res), indent=1)
    print(text)
    if args.out:
        args.out.write_text(text)
    return res


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
