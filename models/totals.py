"""
models/totals.py
PMF totals model (Phase 3, Layer D): per-side goal distributions convolved
into a total-goals distribution that can price any over/under line.

Architecture (locked by File 3 / PROJECT_CONTEXT §6):
- One LightGBM Poisson regressor predicts a side's REGULATION goals from
  stacked ATTACK ROWS: every game contributes two rows — (home offense vs
  away defense + away goalie, is_home=1) and the mirror — with role-based
  columns (off_*, def_*, goalie_*). Regulation is the modelable quantity:
  the OT/SO winner's credited extra goal is a rule artifact (exactly one
  more goal iff regulation ties), not team scoring skill.
- Level features, NOT the shared game_vector. The game vector stores
  home-away DIFFERENTIALS — right for win probability, information-
  destroying for totals (a high-vs-high and a low-vs-low matchup both
  have diff = 0). Measured: on diff features the model LOST to a
  featureless league-mean baseline. Levels come straight from
  features.team_rolling / goalie_rolling; starters from features.matchup.
  NaNs (thin early-season windows) are left in place — LightGBM routes
  missing values natively, no zero-imputation lies.
- Environment offset: league scoring drifts season to season (5.84 ->
  6.35 avg totals across the backfill) and NOTHING in the game vector
  encodes it, so a plain booster is stuck at its train-window mean and
  under-predicts rising seasons (measured: -0.2 goals/game bias, and the
  featureless league-mean baseline BEAT the first model build). Fix is
  the same trick as the ML model's market offset: each booster trains
  with init_score = log(trailing-365-day league rate for its side) —
  point-in-time correct, computed only from games strictly before each
  row — and learns residual matchup effects on top.
- Independence assumption (v1): P(H=h, A=a) = P(H=h)P(A=a). NHL goal
  totals are near-Poisson and cross-team correlation is small; a
  Dixon-Coles-style low-score adjustment is a later refinement if the
  calibration evidence demands it.
- Settlement totals: NHL totals settle on the final score INCLUDING the
  OT/SO winner's goal. The total PMF therefore shifts every regulation-tie
  outcome (h == a) up by one goal:
      P(T = t) = sum_{h+a=t, h != a} p_h(h) p_a(a)
               + sum_{h == a, 2h+1 = t} p_h(h) p_a(h)
- Calibration: the environment offset (and only it) sets the level; the
  PMF stays internally coherent — no separate squeeze of P(over) that
  would detach it from the distribution, and no mean-scale correction
  (both variants measured harmful; see fit_totals_fold).

Evaluation (walk-forward, expanding season folds, purge gap):
- Gate: pooled OOF negative log-likelihood of the actual settlement total
  under the model PMF must beat the ENVIRONMENT baseline (the trailing
  league rates alone through the same PMF machinery) — i.e. the features
  must add value beyond knowing how much the league is scoring lately.
- STATUS (2026-07): GATE FAILED, honestly. Pooled OOF NLL 2.1867 vs
  baseline 2.1815; over/under log loss at the DraftKings line 0.7053 vs
  0.693 naive. Public pre-game team features add essentially nothing to
  totals beyond the scoring environment (they only win on the one fold
  with a stable cross-season environment), and the model does NOT beat
  the market total. Consequences: the model registers as inactive, the
  daily job writes PMF predictions for visibility but NO totals
  recommendations, and totals betting stays OFF until this passes with
  (a) confirmed starters (Daily Faceoff, Phase 4), (b) live snapshot O/U
  prices, and (c) boost-from-market-total once 2026-27 snapshot lines
  accumulate — the exact offset trick that made the moneyline model work,
  currently impossible for lack of historical totals prices.
- Market-line evaluation: raw.historical_odds.over_under is a placeholder
  constant (5.5) outside the DraftKings era — the ESPN pickcenter totals
  analogue of the one-sided ML junk (docs/historical_odds.md). Over/under
  log loss at the posted line is only computed on provider='DraftKings'
  rows (2025-26). No O/U *prices* exist historically, so there is no
  payout backtest for totals: the strategy proof starts at paper trading.
"""
import logging

import numpy as np
import pandas as pd
from sqlalchemy import text

from config.settings import engine
from models.baseline import ARTIFACT_DIR, PURGE_DAYS, walk_forward_folds

logger = logging.getLogger("nhl.models.totals")

MODEL_NAME = "poisson_totals"
MODEL_VERSION = "v1"
CAL_FRAC = 0.15
MAX_GOALS = 12                 # per-side PMF support 0..12 (P(13+) ~ 1e-6)
LAMBDA_CLIP = (0.4, 8.0)

# Heavier regularization than the ML model: per-game totals signal in
# public team features is weak, and an under-regularized booster soaked
# up season-identity noise through the level features (measured: it lost
# to its own offset baseline). With weak signal the correct behavior is
# to degenerate gracefully toward the environment offset.
LGBM_PARAMS = {
    "objective": "poisson",
    "learning_rate": 0.02,
    "num_leaves": 7,
    "min_child_samples": 100,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 20.0,
    "n_estimators": 1000,
    "random_state": 42,
    "verbosity": -1,
}


# ── Data ───────────────────────────────────────────────────────────

TEAM_OFF_STATS = ["gf_per60", "sh_pct", "pp_pct", "pp_xgf_per60",
                  "pim_per60", "xgf_pct"]
TEAM_DEF_STATS = ["ga_per60", "sv_pct", "pk_pct", "pk_xga_per60",
                  "pim_per60"]
GOALIE_STATS = ["shrunk_sv_pct", "shrunk_gsax", "credibility_z"]
CONTEXT_FEATURES = ["is_home", "is_playoff", "off_rest", "def_rest",
                    "off_b2b", "def_b2b", "off_travel_km", "def_travel_km",
                    "elo_edge", "game_num", "stage_early", "stage_late",
                    "starter_fallback"]


def attack_feature_names() -> list:
    from features.util import WINDOWS
    names = []
    for w in WINDOWS:
        names += [f"off_{s}_w{w}" for s in TEAM_OFF_STATS]
        names += [f"def_{s}_w{w}" for s in TEAM_DEF_STATS]
        names += [f"off_gp_w{w}", f"def_gp_w{w}"]
    for w in WINDOWS:
        names += [f"goalie_{s}_w{w}" for s in GOALIE_STATS]
    return names + CONTEXT_FEATURES


ATTACK_FEATURES = attack_feature_names()


def _load_games_frame():
    with engine.connect() as conn:
        return pd.read_sql(text("""
            SELECT g.game_id, g.season, g.date, g.home_team, g.away_team,
                   g.game_type, g.home_score, g.away_score, g.is_ot, g.is_so,
                   m.home_rest_days, m.away_rest_days, m.home_b2b, m.away_b2b,
                   m.home_travel_km, m.away_travel_km,
                   m.home_game_num, m.away_game_num, m.season_stage,
                   m.home_elo, m.away_elo,
                   m.home_starter_id, m.away_starter_id,
                   CASE WHEN h.provider = 'DraftKings'
                        THEN h.over_under END AS market_line
            FROM raw.games g
            JOIN features.matchup m USING (game_id)
            LEFT JOIN raw.historical_odds h USING (game_id)
            WHERE g.game_state IN ('FINAL', 'OFF')
            ORDER BY g.date, g.game_id
        """), conn)


def _load_team_levels() -> pd.DataFrame:
    from features.util import WINDOWS
    stats = sorted(set(TEAM_OFF_STATS + TEAM_DEF_STATS))
    with engine.connect() as conn:
        tr = pd.read_sql(text(f"""
            SELECT game_id, team, window_size, games_played,
                   {', '.join(stats)}
            FROM features.team_rolling
        """), conn)
    tr = tr.astype({c: float for c in stats + ["games_played"]})
    wide = tr.pivot(index=["game_id", "team"], columns="window_size",
                    values=stats + ["games_played"])
    wide.columns = [f"{stat}_w{w}" for stat, w in wide.columns]
    return wide.reset_index()


def _load_goalie_levels() -> pd.DataFrame:
    with engine.connect() as conn:
        gr = pd.read_sql(text(f"""
            SELECT game_id, goalie_id, window_size, {', '.join(GOALIE_STATS)}
            FROM features.goalie_rolling
        """), conn)
    gr = gr.astype({c: float for c in GOALIE_STATS})
    wide = gr.pivot(index=["game_id", "goalie_id"], columns="window_size",
                    values=GOALIE_STATS)
    wide.columns = [f"{stat}_w{w}" for stat, w in wide.columns]
    return wide.reset_index()


def build_attack_matrix(games: pd.DataFrame, team_wide: pd.DataFrame,
                        goalie_wide: pd.DataFrame) -> tuple:
    """(X_home_attack, X_away_attack) with ATTACK_FEATURES columns: the
    home matrix describes home offense vs away defense + away goalie;
    the away matrix is the mirror. Pure join/derive, no DB access."""
    from features.util import WINDOWS

    def side_matrix(off, deff, is_home):
        df = games[["game_id"]].copy()
        offw = team_wide.add_prefix("o_")
        defw = team_wide.add_prefix("d_")
        df = games.merge(offw, left_on=["game_id", f"{off}_team"],
                         right_on=["o_game_id", "o_team"], how="left")
        df = df.merge(defw, left_on=["game_id", f"{deff}_team"],
                      right_on=["d_game_id", "d_team"], how="left")
        gw = goalie_wide.add_prefix("g_")
        df = df.merge(gw, left_on=["game_id", f"{deff}_starter_id"],
                      right_on=["g_game_id", "g_goalie_id"], how="left")

        cols = {}
        for w in WINDOWS:
            for s in TEAM_OFF_STATS:
                cols[f"off_{s}_w{w}"] = df[f"o_{s}_w{w}"]
            for s in TEAM_DEF_STATS:
                cols[f"def_{s}_w{w}"] = df[f"d_{s}_w{w}"]
            cols[f"off_gp_w{w}"] = df[f"o_games_played_w{w}"]
            cols[f"def_gp_w{w}"] = df[f"d_games_played_w{w}"]
        for w in WINDOWS:
            for s in GOALIE_STATS:
                cols[f"goalie_{s}_w{w}"] = df[f"g_{s}_w{w}"]
        cols["is_home"] = float(is_home)
        # Playoff scoring is a different regime (lower, tighter) AND the
        # time-ordered calibration tail of every training window is playoff-
        # heavy — without this flag that tail poisons the fitted level.
        cols["is_playoff"] = (df["game_type"] == 3).astype(float)
        cols["off_rest"] = df[f"{off}_rest_days"].astype(float)
        cols["def_rest"] = df[f"{deff}_rest_days"].astype(float)
        cols["off_b2b"] = df[f"{off}_b2b"].fillna(False).astype(float)
        cols["def_b2b"] = df[f"{deff}_b2b"].fillna(False).astype(float)
        cols["off_travel_km"] = df[f"{off}_travel_km"].astype(float)
        cols["def_travel_km"] = df[f"{deff}_travel_km"].astype(float)
        cols["elo_edge"] = (df[f"{off}_elo"].astype(float)
                            - df[f"{deff}_elo"].astype(float))
        cols["game_num"] = df[f"{off}_game_num"].astype(float)
        cols["stage_early"] = (df["season_stage"] == "EARLY").astype(float)
        cols["stage_late"] = (df["season_stage"] == "LATE").astype(float)
        cols["starter_fallback"] = df[f"{deff}_starter_id"].isna().astype(float)
        return pd.DataFrame(cols, index=df.index)[ATTACK_FEATURES]

    xh = side_matrix("home", "away", True)
    xa = side_matrix("away", "home", False)
    return xh.to_numpy(dtype=float), xa.to_numpy(dtype=float)


# Goal-rate features are NON-STATIONARY across seasons (league scoring
# drifted 5.84 -> 6.35): a booster reading raw levels learns residual-vs-
# environment patterns keyed to season identity and extrapolates them
# wrongly into new seasons (measured: fold biases of +0.18/-0.37 goals
# tracking environment drift). These columns are divided by the trailing
# league rate so the features say "vs the league right now".
ENV_NORMALIZED = ["gf_per60", "ga_per60", "pp_xgf_per60", "pk_xga_per60"]


def _env_normalize(X: np.ndarray, env_total: np.ndarray) -> np.ndarray:
    """Divide goal-rate columns by the trailing league per-side rate."""
    X = X.copy()
    rate = env_total / 2.0                    # per-side league rate
    for j, name in enumerate(ATTACK_FEATURES):
        stat = name.split("_w")[0].replace("off_", "").replace("def_", "")
        if stat in ENV_NORMALIZED:
            X[:, j] = X[:, j] / rate
    return X


def load_totals_dataset():
    """Attack matrices + regulation goals + settlement totals + the
    DraftKings-era market line (the only trustworthy historical one)."""
    games = _load_games_frame()
    if games.empty:
        raise RuntimeError("No completed games with matchup rows — "
                           "run the feature build first")
    Xh, Xa = build_attack_matrix(games, _load_team_levels(),
                                 _load_goalie_levels())

    hg = games["home_score"].to_numpy(dtype=float)
    ag = games["away_score"].to_numpy(dtype=float)
    extra = (games["is_ot"] | games["is_so"]).to_numpy()
    home_won = hg > ag
    y_home_reg = hg - (extra & home_won)
    y_away_reg = ag - (extra & ~home_won)

    meta = games[["game_id", "season", "date"]].copy()
    meta["date"] = pd.to_datetime(meta["date"])
    meta["total"] = (hg + ag).astype(int)
    meta["market_line"] = games["market_line"].astype(float)
    meta["is_playoff"] = (games["game_type"] == 3).to_numpy()

    env_total = (env_rates(meta["date"], y_home_reg, ENV_PRIOR_RATE["home"])
                 + env_rates(meta["date"], y_away_reg, ENV_PRIOR_RATE["away"]))
    Xh = _env_normalize(Xh, env_total)
    Xa = _env_normalize(Xa, env_total)
    return Xh, Xa, y_home_reg, y_away_reg, meta, ATTACK_FEATURES


ENV_FAST_DAYS = 120            # tracks the current season's level
ENV_SLOW_DAYS = 365            # stable cross-season prior
ENV_FAST_PRIOR_GAMES = 400     # shrink weight of fast window toward slow
ENV_SLOW_PRIOR_GAMES = 300     # shrink weight of slow window toward constant
ENV_PRIOR_RATE = {"home": 2.95, "away": 2.70}   # long-run regulation means


def _trailing(dates_sorted, y_sorted, window_days):
    """(sum, count) of y over games strictly before each unique date,
    within window_days. Returns per-unique-date arrays + row mapping."""
    udates, first_idx = np.unique(dates_sorted, return_index=True)
    csum = np.concatenate([[0.0], np.cumsum(y_sorted)])
    day_end = np.append(first_idx[1:], len(y_sorted))
    cum_by_day = csum[day_end]
    cnt_by_day = day_end.astype(float)

    lo = np.searchsorted(udates, udates - np.timedelta64(window_days, "D"))
    win_sum = csum[first_idx] - np.where(lo > 0, cum_by_day[lo - 1], 0.0)
    win_cnt = first_idx - np.where(lo > 0, cnt_by_day[lo - 1], 0.0)
    return udates, win_sum, win_cnt


def env_rates(dates: pd.Series, y: np.ndarray, prior: float) -> np.ndarray:
    """Two-tier trailing league scoring rate per row, point-in-time
    correct (same-day games excluded): a fast window that tracks the
    current season's level, shrunk toward a slow window, itself shrunk
    toward the long-run constant. League scoring shifts season to season;
    by mid-season the fast window has absorbed the new level while the
    shrinkage keeps early-season estimates stable."""
    d = pd.to_datetime(dates).to_numpy()
    order = np.argsort(d, kind="stable")
    ds, ys = d[order], y[order]

    udates, slow_sum, slow_cnt = _trailing(ds, ys, ENV_SLOW_DAYS)
    _, fast_sum, fast_cnt = _trailing(ds, ys, ENV_FAST_DAYS)

    slow_rate = ((slow_sum + ENV_SLOW_PRIOR_GAMES * prior)
                 / (slow_cnt + ENV_SLOW_PRIOR_GAMES))
    rate_by_day = ((fast_sum + ENV_FAST_PRIOR_GAMES * slow_rate)
                   / (fast_cnt + ENV_FAST_PRIOR_GAMES))

    day_of_row = np.searchsorted(udates, ds)
    out = np.empty(len(ys))
    out[order] = rate_by_day[day_of_row]
    return out


# ── PMF machinery (pure) ───────────────────────────────────────────

def poisson_pmf(lam: np.ndarray, kmax: int = MAX_GOALS) -> np.ndarray:
    """Row-per-game Poisson PMF over 0..kmax, renormalized after
    truncation. lam: (n,) -> (n, kmax+1)."""
    from scipy.stats import poisson
    lam = np.clip(np.asarray(lam, dtype=float), *LAMBDA_CLIP)
    k = np.arange(kmax + 1)
    pmf = poisson.pmf(k[None, :], lam[:, None])
    return pmf / pmf.sum(axis=1, keepdims=True)


def total_pmf(pmf_h: np.ndarray, pmf_a: np.ndarray) -> np.ndarray:
    """Settlement-total distribution from two per-side regulation PMFs
    (n, K+1) -> (n, 2K+2): convolution with every regulation tie shifted
    up one goal (the OT/SO winner's credited goal)."""
    n, k1 = pmf_h.shape
    joint = pmf_h[:, :, None] * pmf_a[:, None, :]          # (n, K+1, K+1)
    out = np.zeros((n, 2 * k1))
    h_idx, a_idx = np.meshgrid(np.arange(k1), np.arange(k1), indexing="ij")
    t_idx = np.where(h_idx == a_idx, h_idx + a_idx + 1, h_idx + a_idx)
    np.add.at(out, (np.arange(n)[:, None, None],
                    np.broadcast_to(t_idx, joint.shape)), joint)
    return out


def prob_over(tpmf: np.ndarray, line) -> tuple:
    """(P(over), P(push)) at a line. Half-point lines have zero push."""
    totals = np.arange(tpmf.shape[1])
    line = np.asarray(line, dtype=float).reshape(-1, 1)
    p_over = (tpmf * (totals[None, :] > line)).sum(axis=1)
    p_push = (tpmf * (totals[None, :] == line)).sum(axis=1)
    return p_over, p_push


def expected_total(tpmf: np.ndarray) -> np.ndarray:
    return tpmf @ np.arange(tpmf.shape[1])


def nll_of_totals(tpmf: np.ndarray, totals: np.ndarray) -> np.ndarray:
    """Per-game negative log-likelihood of the observed settlement total."""
    p = tpmf[np.arange(len(totals)), np.clip(totals, 0, tpmf.shape[1] - 1)]
    return -np.log(np.clip(p, 1e-12, None))


# ── Fitting ────────────────────────────────────────────────────────

def _time_split(train_idx, dates, cal_frac=CAL_FRAC):
    from models.lgbm import time_split
    return time_split(train_idx, dates, cal_frac)


def fit_totals_fold(Xh, Xa, y_home, y_away, env_home, env_away,
                    train_idx, dates, is_playoff=None) -> dict:
    """One shared Poisson booster over stacked attack rows (home + away
    attacks of every training game), trained with init_score =
    log(environment rate). Early stopping is judged on the regular-season
    rows of the time-ordered tail only — the tail is the END of the train
    window and therefore playoff-heavy, a different scoring regime than
    the slates this model prices."""
    import lightgbm as lgb

    core, cal = _time_split(train_idx, dates)
    if is_playoff is not None:
        reg = cal[~is_playoff[cal]]
        cal = reg if len(reg) >= 100 else cal
    X_core = np.vstack([Xh[core], Xa[core]])
    y_core = np.concatenate([y_home[core], y_away[core]])
    env_core = np.concatenate([env_home[core], env_away[core]])
    X_cal = np.vstack([Xh[cal], Xa[cal]])
    y_cal = np.concatenate([y_home[cal], y_away[cal]])
    env_cal = np.concatenate([env_home[cal], env_away[cal]])

    m = lgb.LGBMRegressor(**LGBM_PARAMS)
    m.fit(X_core, y_core,
          init_score=np.log(env_core),
          eval_set=[(X_cal, y_cal)],
          eval_init_score=[np.log(env_cal)],
          eval_metric="poisson",
          callbacks=[lgb.early_stopping(100, verbose=False)])

    # No mean-scale correction. Both variants were measured: a scale fit
    # on a playoff-mixed tail swings ±0.4 goals/game; on a playoff-
    # filtered tail it helps long-train folds (fold 5: 2.1727 -> 2.1673)
    # but wrecks short-train folds (fold 1: 2.2118 -> 2.2288, the tail
    # locks in a stale environment) and is net-negative pooled (2.1902 vs
    # 2.1867 without). The environment offset owns the level.
    return {"model": m, "scale": 1.0,
            "iters": m.best_iteration_ or LGBM_PARAMS["n_estimators"]}


def predict_lambdas(fm: dict, Xh, Xa, env_home, env_away) -> tuple:
    raw_h = fm["model"].booster_.predict(Xh, raw_score=True)
    raw_a = fm["model"].booster_.predict(Xa, raw_score=True)
    lam_h = np.clip(fm["scale"] * np.exp(np.log(env_home) + raw_h), *LAMBDA_CLIP)
    lam_a = np.clip(fm["scale"] * np.exp(np.log(env_away) + raw_a), *LAMBDA_CLIP)
    return lam_h, lam_a


def fit_production(cutoff_date=None) -> dict:
    """Production totals scorer trained on everything before cutoff_date
    (None = all labeled games). Mirrors models.lgbm.fit_production.
    The environment rate frozen for scoring new games is the trailing rate
    at the training-data horizon — exactly what would be known pre-slate."""
    Xh, Xa, y_h, y_a, meta, names = load_totals_dataset()
    env_h = env_rates(meta["date"], y_h, ENV_PRIOR_RATE["home"])
    env_a = env_rates(meta["date"], y_a, ENV_PRIOR_RATE["away"])
    mask = (meta["date"] < pd.Timestamp(cutoff_date)).to_numpy() \
        if cutoff_date is not None else np.ones(len(meta), dtype=bool)
    train_idx = np.flatnonzero(mask)
    if len(train_idx) < 500:
        raise RuntimeError(f"Only {len(train_idx)} labeled games before "
                           f"{cutoff_date} — not enough for totals model")

    fm = fit_totals_fold(Xh, Xa, y_h, y_a, env_h, env_a,
                         train_idx, meta["date"],
                         is_playoff=meta["is_playoff"].to_numpy())
    # env rate to use for future slates: the trailing window at the
    # training-data horizon
    last = train_idx[np.argsort(meta["date"].iloc[train_idx].to_numpy())][-1]
    logger.info(f"Totals production fit: {len(train_idx)} games, "
                f"scale={fm['scale']:.4f}, iters={fm['iters']}, "
                f"env=({env_h[last]:.3f}, {env_a[last]:.3f})")
    return {"fm": fm, "names": names, "n_train": len(train_idx),
            "env_home": float(env_h[last]), "env_away": float(env_a[last])}


def score_production(prod: dict, Xh_new: np.ndarray, Xa_new: np.ndarray,
                     names_new: list) -> dict:
    """PMFs + expected totals for new RAW attack matrices (environment
    normalization is applied here, with the production env rates)."""
    if list(names_new) != list(prod["names"]):
        raise ValueError("Feature names/order mismatch for totals model")
    n = len(Xh_new)
    env_total = np.full(n, prod["env_home"] + prod["env_away"])
    Xh_new = _env_normalize(Xh_new, env_total)
    Xa_new = _env_normalize(Xa_new, env_total)
    lam_h, lam_a = predict_lambdas(
        prod["fm"], Xh_new, Xa_new,
        np.full(n, prod["env_home"]), np.full(n, prod["env_away"]))
    ph, pa = poisson_pmf(lam_h), poisson_pmf(lam_a)
    tp = total_pmf(ph, pa)
    return {"lambda_home": lam_h, "lambda_away": lam_a,
            "pmf_home": ph, "pmf_away": pa, "pmf_total": tp,
            "expected_total": expected_total(tp)}


# ── Walk-forward validation ────────────────────────────────────────

def run_totals(register: bool = True) -> dict:
    from sklearn.metrics import log_loss

    Xh, Xa, y_h, y_a, meta, names = load_totals_dataset()
    env_h = env_rates(meta["date"], y_h, ENV_PRIOR_RATE["home"])
    env_a = env_rates(meta["date"], y_a, ENV_PRIOR_RATE["away"])
    folds = walk_forward_folds(meta)
    totals = meta["total"].to_numpy()
    logger.info(f"Totals dataset: {len(meta)} games x {len(names)} attack "
                f"features, {len(folds)} folds (purge {PURGE_DAYS}d)")

    oof_nll = np.full(len(meta), np.nan)
    oof_base_nll = np.full(len(meta), np.nan)
    oof_over = np.full(len(meta), np.nan)      # P(over) at DK line
    oof_exp_total = np.full(len(meta), np.nan)
    fold_metrics = []

    for fold in folds:
        fm = fit_totals_fold(Xh, Xa, y_h, y_a, env_h, env_a,
                             fold.train_idx, meta["date"],
                             is_playoff=meta["is_playoff"].to_numpy())
        lam_h, lam_a = predict_lambdas(fm, Xh[fold.val_idx], Xa[fold.val_idx],
                                       env_h[fold.val_idx], env_a[fold.val_idx])
        tp = total_pmf(poisson_pmf(lam_h), poisson_pmf(lam_a))

        # Environment baseline: the trailing league rates alone, through
        # the identical PMF machinery — the bar the features must clear
        tp_base = total_pmf(poisson_pmf(env_h[fold.val_idx]),
                            poisson_pmf(env_a[fold.val_idx]))

        tv = totals[fold.val_idx]
        oof_nll[fold.val_idx] = nll_of_totals(tp, tv)
        oof_base_nll[fold.val_idx] = nll_of_totals(tp_base, tv)
        oof_exp_total[fold.val_idx] = expected_total(tp)

        lines = meta["market_line"].to_numpy()[fold.val_idx]
        lined = np.isfinite(lines)
        m = {
            "val_season": int(meta["season"].iloc[fold.val_idx[0]]),
            "n_val": len(fold.val_idx),
            "iters": fm["iters"], "scale": round(fm["scale"], 4),
            "nll": float(np.mean(oof_nll[fold.val_idx])),
            "baseline_nll": float(np.mean(oof_base_nll[fold.val_idx])),
            "mean_pred_total": float(np.mean(oof_exp_total[fold.val_idx])),
            "mean_actual_total": float(tv.mean()),
            "n_lined": int(lined.sum()),
        }
        if lined.any():
            p_over, p_push = prob_over(tp[lined], lines[lined])
            over_actual = tv[lined] > lines[lined]
            push = tv[lined] == lines[lined]
            oof_over[fold.val_idx[lined]] = p_over
            keep = ~push
            if keep.sum() > 50:
                m["over_log_loss"] = float(log_loss(
                    over_actual[keep], np.clip(p_over[keep], 1e-6, 1 - 1e-6)))
                m["over_rate"] = float(over_actual[keep].mean())
        fold_metrics.append(m)
        logger.info(
            f"  fold {m['val_season']}: nll={m['nll']:.4f} "
            f"(baseline {m['baseline_nll']:.4f}) "
            f"pred_total={m['mean_pred_total']:.2f} vs {m['mean_actual_total']:.2f}"
            + (f" | over_ll={m['over_log_loss']:.4f} (n={m['n_lined']})"
               if "over_log_loss" in m else ""))

    scored = ~np.isnan(oof_nll)
    pooled = {
        "nll": float(np.mean(oof_nll[scored])),
        "baseline_nll": float(np.mean(oof_base_nll[scored])),
        "n_scored": int(scored.sum()),
    }
    pooled["gate_passed"] = pooled["nll"] < pooled["baseline_nll"]

    lined = ~np.isnan(oof_over)
    if lined.any():
        lines = meta["market_line"].to_numpy()
        keep = lined & (totals != lines)
        pooled["over_log_loss"] = float(log_loss(
            (totals > lines)[keep], np.clip(oof_over[keep], 1e-6, 1 - 1e-6)))
        pooled["n_lined"] = int(keep.sum())

    logger.info(
        f"POOLED OOF: nll={pooled['nll']:.4f} vs baseline "
        f"{pooled['baseline_nll']:.4f} — GATE "
        f"{'PASSED' if pooled['gate_passed'] else 'FAILED'}"
        + (f" | over_log_loss={pooled['over_log_loss']:.4f} "
           f"(n={pooled['n_lined']}, naive 0.693)" if "over_log_loss" in pooled
           else ""))

    if register:
        _register(pooled, meta, names)

    oof = meta.loc[scored, ["game_id", "season", "date", "total"]].copy()
    oof["nll"] = oof_nll[scored]
    oof["p_over_line"] = oof_over[scored]
    oof["expected_total"] = oof_exp_total[scored]
    return {"folds": fold_metrics, "pooled": pooled, "oof": oof}


def _register(pooled: dict, meta, names: list) -> None:
    import hashlib
    feature_hash = hashlib.sha256(",".join(names).encode()).hexdigest()
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO models.model_registry
                (model_name, version, model_type, trained_through,
                 feature_set_hash, cv_log_loss, is_active)
            VALUES (:name, :version, 'pmf_totals', :through, :hash, :ll, FALSE)
            ON CONFLICT (model_name, version) DO UPDATE SET
                trained_through = EXCLUDED.trained_through,
                feature_set_hash = EXCLUDED.feature_set_hash,
                cv_log_loss = EXCLUDED.cv_log_loss
        """), {"name": MODEL_NAME, "version": MODEL_VERSION,
               "through": meta["date"].max().date(), "hash": feature_hash,
               "ll": round(pooled.get("over_log_loss", pooled["nll"]), 4)})
    logger.info(f"Registered {MODEL_NAME} {MODEL_VERSION} in models.model_registry")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_totals()
