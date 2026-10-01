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
- Joint score shape (v2, 2026-09): the two sides' Poisson PMFs are
  multiplied, then each (home, away) score cell is reweighted by its
  regulation winning margin — tie, 1, 2, 3, 4+ goals — and the joint
  renormalized (joint_pmf, MARGIN_WEIGHTS). v1 assumed the two scores
  independent, which gets hockey's joint shape badly wrong: regulation
  ties 22.3% actual vs 16.7% independent, one-goal wins 17.7% vs 30.4%,
  three-goal wins 23.5% vs 14.9% (most likely the pulled goalie turning
  one-goal games into ties or empty-net two-goal games). The weights are
  fitted by maximum likelihood on training games only. Per side, the
  goals stay near-Poisson (variance/mean 1.02-1.03); a negative binomial
  (→ a Poisson with extra scatter) measured worse and is not used.
- New-season drift correction (v2, 2026-09): the booster's inputs shift
  from season to season and it reads the shift as a scoring change —
  its mean log-adjustment is ~0 on the seasons it trained on but moves
  on the next one: -0.015, +0.021, -0.011, -0.037, -0.039 for 2021-22
  to 2025-26 (-0.037 is ~0.2 goals a game too low, so it leaned under at
  the DraftKings line: mean P(over) 0.441 vs 0.508 actual). Each game's
  rates are divided by exp(the booster's mean adjustment over the same
  season's games on EARLIER dates), shrunk toward 0 by
  DRIFT_PRIOR_GAMES while few have been played (drift_shift). The
  adjustment is a prediction, not an outcome, and same-day games never
  count, so it is known before puck drop. DRIFT_PRIOR_GAMES barely
  matters (10 to 50 all gave 2.1801). Measured alternative, worse:
  blanking the drift-prone inputs (shooting %, save %, Elo gap) instead
  (2.1834 alone, 2.1809 with the correction).
- Settlement totals: NHL totals settle on the final score INCLUDING the
  OT/SO winner's goal. The total PMF therefore shifts every regulation-tie
  outcome (h == a) up by one goal, over the (reweighted) joint J:
      P(T = t) = sum_{h+a=t, h != a} J(h, a) + sum_{h == a, 2h+1 = t} J(h, h)
- Calibration: the environment offset (and only it) sets the level; the
  PMF stays internally coherent — no separate squeeze of P(over) that
  would detach it from the distribution, and no mean-scale correction
  (both variants measured harmful; see fit_totals_fold).

Evaluation (walk-forward, expanding season folds, purge gap):
- Gate (hardened 2026-09): pooled OOF negative log-likelihood (NLL → how
  surprised the model is by the actual final total; lower is better)
  must beat the ENVIRONMENT baseline — the trailing league rates alone
  through the same PMF machinery — WITH THE SAME margin weights, fitted
  per fold on its training games. The baseline has no booster, so the
  drift correction has nothing to remove there. The gate therefore asks
  only whether the team stats add information beyond how much the league
  is scoring lately. (Before hardening, a structural fix given to the
  model alone could "pass": the model with both fixes scores 2.1801
  against the unfixed baseline's 2.1815, a win owed to the margin fix,
  not the team stats.)
- Market check (reported with the gate; market_check): over/under log
  loss of the model vs the no-vig market P(over) (bookmaker margin
  removed) at the market's main line, same games, pushes dropped. It
  needs O/U PRICES (load_market_quotes): the live snapshots
  (raw.odds_snapshots, from 2026-27) and, once ingestion/espn_odds.py
  stores them, ESPN's DraftKings closing prices (most of 2025-26). With
  neither it reports "no prices". Beating the environment is not
  beating the bookmakers: totals betting needs this check too.
- STATUS (2026-09-29, v2): GATE FAILED, honestly. Pooled OOF NLL 2.1801
  vs the hardened baseline 2.1787 (+0.0014, paired SE 0.0011) over 6,993
  games. Per season, model vs hardened baseline: 2021-22 2.2035 vs
  2.1986, 2022-23 2.1599 vs 2.1605, 2023-24 2.1762 vs 2.1763, 2024-25
  2.2034 vs 2.2016, 2025-26 2.1575 vs 2.1563 — ahead in 2 of 5 seasons,
  by 0.0006 and 0.0001. Before the fixes (v1, 2026-07): 2.1867 vs 2.1815.
  The fixes make the numbers more accurate — the margin fix alone takes
  the baseline from 2.1815 to 2.1787 and the model from 2.1867 to
  2.1817, the drift correction then takes the model to 2.1801; together
  they move the model's mean predicted total in 2025-26 from 5.87 to
  6.19 (actual 6.23) and its over/under log loss at the DraftKings line
  from 0.7051 to 0.6958 (n=1,011, 2025-26; a coin flip scores 0.6931) —
  but they do not give the team stats an edge. A one-off market check
  against 161 sampled 2025-26 games with DraftKings closing O/U prices
  from ESPN's summary API (not stored): model 0.6978 vs no-vig market
  0.6904 (model worse by 0.0074 ± 0.0125, too few games to be
  conclusive).
- Why it fails: the public team stats carry almost no totals signal (OOF
  rank correlation of predicted vs actual total 0.02-0.06 by season;
  even the DraftKings line manages only 0.084), and season-level scoring
  shifts dominate the error. Audited for leakage/join bugs on challenge
  (2026-07-17): feature NaN rates are 0.0% (goalie) / 1.2% (team, thin
  early windows), goalie features have real variance (std 0.40
  normalized), and removing the goalie features slightly IMPROVES OOF
  NLL (2.1834 vs 2.1867) while removing the team features worsens it
  (2.1881) — the shrunk historical-form goalie features are net noise
  for totals (Buhlmann k=66: most goalies sit near league average most
  of the time).
- Confirmed starters will NOT make this pass by themselves (corrected
  2026-09; the 2026-07 note said they would help). The historical test
  already used each game's ACTUAL starting goalie: features/
  build_vectors.py takes the starter from raw.goalie_games.is_starter and
  writes it to features.matchup, which this model reads. So the test
  knew every starter perfectly, and the goalie features were still net
  noise. Daily Faceoff starters only keep live scoring as good as this
  test; they add nothing the test lacked.
- Consequences: the model registers as inactive, the daily job writes PMF
  predictions for the bet checker and the alerts but NO totals
  recommendations, and GATE_PASSED stays False. The realistic path is
  boost-from-market-total (the offset trick that made the moneyline
  model work) once 2026-27 snapshot O/U prices accumulate, judged by the
  market check, not by the environment gate alone.
- Market-line evaluation: raw.historical_odds.over_under is a placeholder
  constant (5.5) outside the DraftKings era — the ESPN pickcenter totals
  analogue of the one-sided ML junk (docs/historical_odds.md). Over/under
  log loss at the posted line is only computed on provider='DraftKings'
  rows (2025-26). No O/U *prices* are stored historically (ESPN's summary
  carries DraftKings closing O/U prices for most of 2025-26, but
  ingestion/espn_odds.py doesn't keep them), so there is no payout
  backtest for totals: the strategy proof starts at paper trading.
"""
import logging

import numpy as np
import pandas as pd
from sqlalchemy import text

from config.settings import engine
from models.baseline import ARTIFACT_DIR, PURGE_DAYS, walk_forward_folds

logger = logging.getLogger("nhl.models.totals")

MODEL_NAME = "poisson_totals"
MODEL_VERSION = "v2"           # v2: margin-reweighted joint + drift correction
# The walk-forward gate verdict (STATUS above), set by hand. Two readers:
# the bet checker (betting/checker.py), which won't give a totals leg a
# BET verdict while it is False, and the daily recommendation job
# (betting/recommend.py), which keeps totals predictions-only while it is
# False and logs an ERROR when it is True, because no totals betting path
# exists yet (the job writes no totals picks and settlement grades
# moneyline picks only). So flipping it does NOT start totals betting on
# its own: that also needs a totals pick writer and totals settlement.
GATE_PASSED = False
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


# ── Goalie experiment variants (2026-10, pre-registered) ──────────
#
# Evaluation-only switch; production (fit_production / score_production)
# always uses variant A.
#   A  v2 as committed (ATTACK_FEATURES)
#   B  A minus every goalie_* column (the shrunk goalie-form features)
#   C  A plus the defending goalie's starter-role features
#      (features/goalie_role.py ROLE_FEATURES), appended after A's columns
#   D  C minus every goalie_* column
VARIANTS = ("A", "B", "C", "D")


def variant_feature_names(variant: str = "A") -> list:
    """Attack-row column names of an experiment variant, in order."""
    from features.goalie_role import ROLE_FEATURES
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    names = list(ATTACK_FEATURES)
    if variant in ("C", "D"):
        names += ROLE_FEATURES
    if variant in ("B", "D"):
        names = [n for n in names if not n.startswith("goalie_")]
    return names


def apply_variant(Xh, Xa, role_h, role_a, variant: str = "A") -> tuple:
    """(Xh, Xa, names) for a variant from the (environment-normalized) A
    matrices and the defending goalie's role features (DataFrames with
    ROLE_FEATURES columns, row-aligned; ignored for A and B). Pure."""
    from features.goalie_role import ROLE_FEATURES
    names = variant_feature_names(variant)
    if variant in ("C", "D"):
        Xh = np.hstack([Xh, role_h[ROLE_FEATURES].to_numpy(dtype=float)])
        Xa = np.hstack([Xa, role_a[ROLE_FEATURES].to_numpy(dtype=float)])
        full = list(ATTACK_FEATURES) + ROLE_FEATURES
    else:
        full = list(ATTACK_FEATURES)
    keep = [full.index(n) for n in names]
    return Xh[:, keep], Xa[:, keep], names


def load_totals_dataset(variant: str = "A", with_roles: bool = False):
    """Attack matrices + regulation goals + settlement totals + the
    DraftKings-era market line (the only trustworthy historical one).
    variant: see VARIANTS (default A = production v2). with_roles (or a
    C/D variant): also compute the starter-role features in memory and
    add meta["backup_start"] — True when either team's starter had
    role_is_primary == 0."""
    games = _load_games_frame()
    if games.empty:
        raise RuntimeError("No completed games with matchup rows — "
                           "run the feature build first")
    Xh, Xa = build_attack_matrix(games, _load_team_levels(),
                                 _load_goalie_levels())
    variant_feature_names(variant)                 # validates the name

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
    if variant == "A" and not with_roles:
        return Xh, Xa, y_home_reg, y_away_reg, meta, ATTACK_FEATURES

    from features.goalie_role import defending_role_frame, load_appearances
    with engine.connect() as conn:
        app = load_appearances(conn)
    role_h, role_a = defending_role_frame(games, app)
    # role_h describes the away starter, role_a the home starter
    meta["backup_start"] = ((role_h["role_is_primary"] == 0)
                            | (role_a["role_is_primary"] == 0)).to_numpy()
    Xh, Xa, names = apply_variant(Xh, Xa, role_h, role_a, variant)
    return Xh, Xa, y_home_reg, y_away_reg, meta, names


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


# ── Joint score shape: regulation-margin reweighting ───────────────
#
# Two independent Poissons get the joint shape of hockey scores wrong:
# regulation ties happen 22.3% of the time (independence says 16.7%),
# one-goal regulation wins 17.7% (it says 30.4%) and three-goal wins
# 23.5% (it says 14.9%) — the likely cause is the late pulled goalie,
# which turns a one-goal game into a tie or an empty-net two-goal game.
# So each (home, away) score cell is reweighted by its regulation
# margin bucket — tie, 1, 2, 3, 4+ goals — and the joint renormalized.
# The 4+ weight is fixed at 1 (only the ratios matter). Weights are
# fitted by maximum likelihood on training games only
# (fit_margin_weights); the walk-forward refits them inside every fold.

MARGIN_CAP = 4                 # margins of 4+ goals share one bucket
# Default weights for callers that only hold the per-side PMFs (the bet
# checker, the arbitrage/middle alerts): fit_margin_weights on the
# trailing-environment PMFs of every completed game, 2020-21 through
# 2025-26 (7,945 games; fitted 2026-09-29) — all of them training seasons
# for the live 2026-27 season. The walk-forward fold fits ranged tie
# 1.11-1.16, 1-goal 0.50-0.55, 2-goal 0.69-0.73, 3-goal 1.21-1.36. The
# daily job's own scores use the weights fit_production fits on its
# training window, and it warns when one moves more than
# MARGIN_WEIGHT_DRIFT_WARN away from these, so they can be refreshed.
MARGIN_WEIGHTS = np.array([1.1540, 0.5080, 0.7228, 1.3429, 1.0])
MARGIN_WEIGHT_DRIFT_WARN = 0.05


def margin_buckets(k1: int) -> np.ndarray:
    """(k1, k1) regulation-margin bucket of each (home, away) cell:
    min(|h - a|, MARGIN_CAP), so 0 = tie."""
    h, a = np.meshgrid(np.arange(k1), np.arange(k1), indexing="ij")
    return np.minimum(np.abs(h - a), MARGIN_CAP)


def _check_weights(w) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    if w.shape != (MARGIN_CAP + 1,) or not np.all(np.isfinite(w)) \
            or not np.all(w > 0):
        raise ValueError(f"margin weights must be {MARGIN_CAP + 1} positive "
                         f"finite numbers (tie, 1, 2, 3, 4+), got {w}")
    return w


def joint_pmf(pmf_h: np.ndarray, pmf_a: np.ndarray,
              margin_weights=MARGIN_WEIGHTS) -> np.ndarray:
    """(n, K+1, K+1) joint regulation-score distribution from per-side
    PMFs: their product, each cell times its margin bucket's weight,
    renormalized per game to sum to 1. margin_weights=None keeps the
    plain independent product."""
    joint = pmf_h[:, :, None] * pmf_a[:, None, :]
    if margin_weights is None:
        return joint
    w = _check_weights(margin_weights)
    joint = joint * w[margin_buckets(pmf_h.shape[1])][None]
    return joint / joint.sum(axis=(1, 2), keepdims=True)


def fit_margin_weights(pmf_h: np.ndarray, pmf_a: np.ndarray,
                       y_home, y_away) -> np.ndarray:
    """Maximum-likelihood margin weights (tie, 1, 2, 3; 4+ fixed at 1)
    from training games only: per-side PMFs (n, K+1) and the observed
    regulation goals. Under the reweighted joint a game's likelihood is
    base(h, a) * w[bucket] / sum_b(w_b * mass_b), where mass_b is the
    game's independent probability of margin bucket b, so only the
    bucket masses matter. The negative log-likelihood is convex in the
    log-weights (a unique optimum): at it, the average predicted share of
    each bucket equals the observed share."""
    from scipy.optimize import minimize

    k1 = pmf_h.shape[1]
    buckets = margin_buckets(k1)
    joint = pmf_h[:, :, None] * pmf_a[:, None, :]
    mass = np.stack([joint[:, buckets == b].sum(axis=1)
                     for b in range(MARGIN_CAP + 1)], axis=1)      # (n, 5)
    h = np.clip(np.asarray(y_home).astype(int), 0, k1 - 1)
    a = np.clip(np.asarray(y_away).astype(int), 0, k1 - 1)
    observed = np.bincount(np.minimum(np.abs(h - a), MARGIN_CAP),
                           minlength=MARGIN_CAP + 1) / len(h)

    def nll_and_grad(log_w):
        lw = np.append(log_w, 0.0)
        weighted = mass * np.exp(lw)[None, :]
        z = weighted.sum(axis=1)
        nll = -observed @ lw + np.mean(np.log(z))
        grad = -observed + (weighted / z[:, None]).mean(axis=0)
        return nll, grad[:-1]

    res = minimize(nll_and_grad, np.zeros(MARGIN_CAP), jac=True,
                   method="L-BFGS-B")
    return np.exp(np.append(res.x, 0.0))


def total_pmf(pmf_h: np.ndarray, pmf_a: np.ndarray,
              margin_weights=MARGIN_WEIGHTS) -> np.ndarray:
    """Settlement-total distribution from two per-side regulation PMFs
    (n, K+1) -> (n, 2K+2): the joint (margin-reweighted by default; see
    joint_pmf) summed along each total, with every regulation tie shifted
    up one goal (the OT/SO winner's credited goal)."""
    n, k1 = pmf_h.shape
    joint = joint_pmf(pmf_h, pmf_a, margin_weights)        # (n, K+1, K+1)
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


# ── New-season drift correction (point-in-time) ────────────────────
#
# The booster's average adjustment to the environment rate is ~0 on the
# seasons it trained on but moves on each new season (-0.037 and -0.039
# on the log scale for 2024-25 and 2025-26, ~0.2 goals a game): its
# inputs shift from season to season and it reads the shift as a scoring
# change. The environment offset is meant to own the scoring level, so
# the level the booster adds on top is removed: every game's rates are
# divided by exp(the booster's mean adjustment over the same season's
# games on EARLIER dates), shrunk toward 0 by DRIFT_PRIOR_GAMES
# pseudo-games while few games have been played. The adjustment is a
# prediction, not an outcome, and only games already played count, so
# the correction is known before each game.

DRIFT_PRIOR_GAMES = 25


def booster_adjustment(lam_h, lam_a, env_h, env_a) -> np.ndarray:
    """Per game, the booster's mean log-adjustment of the two sides'
    rates relative to the environment rates."""
    return 0.5 * (np.log(np.asarray(lam_h) / np.asarray(env_h))
                  + np.log(np.asarray(lam_a) / np.asarray(env_a)))


def drift_shift(seasons, dates, adj,
                prior_games: float = DRIFT_PRIOR_GAMES) -> np.ndarray:
    """Per game: sum of `adj` over the same season's games on strictly
    earlier dates / (their count + prior_games). Same-day and later games
    never count, so the value is known before puck drop. 0 on a season's
    first date."""
    df = pd.DataFrame({"season": np.asarray(seasons),
                       "date": pd.to_datetime(np.asarray(dates)),
                       "adj": np.asarray(adj, dtype=float)})
    day = df.groupby(["season", "date"], sort=True)["adj"].agg(["sum", "count"])
    earlier = day.groupby(level="season").cumsum() - day
    shift = (earlier["sum"] / (earlier["count"] + prior_games)).fillna(0.0)
    keys = pd.MultiIndex.from_frame(df[["season", "date"]])
    return shift.reindex(keys).to_numpy(dtype=float)


def apply_drift(lam_h, lam_a, shift) -> tuple:
    """Rates with the booster's season drift removed, clipped as usual."""
    k = np.exp(-np.asarray(shift, dtype=float))
    return (np.clip(np.asarray(lam_h) * k, *LAMBDA_CLIP),
            np.clip(np.asarray(lam_a) * k, *LAMBDA_CLIP))


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
    Everything frozen for scoring new games is what would be known
    pre-slate: the trailing environment rate at the training-data
    horizon, margin weights fitted on the training games, and each
    season's running booster adjustment over its training games (the
    drift correction; score_production looks up the slate's season)."""
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
    weights = fit_margin_weights(poisson_pmf(env_h[train_idx]),
                                 poisson_pmf(env_a[train_idx]),
                                 y_h[train_idx], y_a[train_idx])
    moved = np.abs(weights - MARGIN_WEIGHTS).max()
    if moved > MARGIN_WEIGHT_DRIFT_WARN:
        logger.warning(f"Totals margin weights fitted on the training games "
                       f"{np.round(weights, 4).tolist()} differ from "
                       f"MARGIN_WEIGHTS {MARGIN_WEIGHTS.tolist()} by up to "
                       f"{moved:.3f}: the bet checker and the alerts still "
                       f"use MARGIN_WEIGHTS, so refresh it in models/totals.py")

    lam_h, lam_a = predict_lambdas(fm, Xh[train_idx], Xa[train_idx],
                                   env_h[train_idx], env_a[train_idx])
    adj = pd.Series(booster_adjustment(lam_h, lam_a, env_h[train_idx],
                                       env_a[train_idx]))
    by_season = adj.groupby(meta["season"].to_numpy()[train_idx])
    drift = {int(k): (float(v["sum"]), int(v["count"]))
             for k, v in by_season.agg(["sum", "count"]).iterrows()}

    # env rate to use for future slates: the trailing window at the
    # training-data horizon
    last = train_idx[np.argsort(meta["date"].iloc[train_idx].to_numpy())][-1]
    logger.info(f"Totals production fit: {len(train_idx)} games, "
                f"scale={fm['scale']:.4f}, iters={fm['iters']}, "
                f"env=({env_h[last]:.3f}, {env_a[last]:.3f}), "
                f"margin weights {np.round(weights, 3).tolist()}")
    return {"fm": fm, "names": names, "n_train": len(train_idx),
            "env_home": float(env_h[last]), "env_away": float(env_a[last]),
            "margin_weights": weights, "drift": drift}


def production_drift_shift(prod: dict, season=None) -> float:
    """The drift correction for a slate in `season`: the booster's mean
    adjustment over that season's training games (all before the cutoff),
    shrunk like drift_shift. 0 for a season with no training games yet,
    or when season is None."""
    total, count = prod.get("drift", {}).get(
        int(season) if season is not None else None, (0.0, 0))
    return total / (count + DRIFT_PRIOR_GAMES) if count else 0.0


def score_production(prod: dict, Xh_new: np.ndarray, Xa_new: np.ndarray,
                     names_new: list, season=None) -> dict:
    """PMFs + expected totals for new RAW attack matrices (environment
    normalization is applied here, with the production env rates). The
    rates get the drift correction of the slate's `season` (none when
    season is None) and the total PMF uses the production margin
    weights. pmf_home/pmf_away are the per-side Poisson PMFs that get
    stored; total_pmf of them with MARGIN_WEIGHTS (what the checker and
    the alerts compute) matches pmf_total up to the weights' refresh."""
    if list(names_new) != list(prod["names"]):
        raise ValueError("Feature names/order mismatch for totals model")
    n = len(Xh_new)
    env_total = np.full(n, prod["env_home"] + prod["env_away"])
    Xh_new = _env_normalize(Xh_new, env_total)
    Xa_new = _env_normalize(Xa_new, env_total)
    lam_h, lam_a = predict_lambdas(
        prod["fm"], Xh_new, Xa_new,
        np.full(n, prod["env_home"]), np.full(n, prod["env_away"]))
    shift = production_drift_shift(prod, season)
    lam_h, lam_a = apply_drift(lam_h, lam_a, np.full(n, shift))
    ph, pa = poisson_pmf(lam_h), poisson_pmf(lam_a)
    tp = total_pmf(ph, pa, prod.get("margin_weights", MARGIN_WEIGHTS))
    return {"lambda_home": lam_h, "lambda_away": lam_a,
            "pmf_home": ph, "pmf_away": pa, "pmf_total": tp,
            "expected_total": expected_total(tp), "drift_shift": shift}


# ── Market check (over/under prices) ───────────────────────────────
#
# Beating the environment baseline says the team stats add information;
# it says nothing about beating the bookmakers. The market check asks
# that: over/under log loss of the model's P(over) against the market's
# no-vig P(over) (bookmaker margin removed), on the same games, at the
# market's main line. It needs O/U PRICES: the live snapshots
# (raw.odds_snapshots, market 'total', from 2026-27) and ESPN's
# DraftKings closing prices once ingestion/espn_odds.py stores them
# (load_market_quotes); with neither it reports "no prices". A push
# refunds the bet, so both sides are compared as P(over | no push) and
# pushes are dropped.

MARKET_CHECK_MIN_GAMES = 200


def market_over_probs(quotes: pd.DataFrame) -> pd.DataFrame:
    """Per game, the market's main line and no-vig P(over) there. quotes:
    game_id, book_name, line, over_price, under_price — one row per book
    (its last pre-game quote). Main line: the one most books quote; on a
    tie, the one whose fair P(over) is nearest 50%. Fair P(over): median
    across those books of each book's no-vig over probability."""
    from features.util import american_implied_prob

    cols = ["game_id", "line", "fair_over", "n_books"]
    q = quotes.dropna(subset=["line", "over_price", "under_price"])
    if q.empty:
        return pd.DataFrame(columns=cols)
    po = q["over_price"].astype(float).map(american_implied_prob)
    pu = q["under_price"].astype(float).map(american_implied_prob)
    q = q.assign(line=q["line"].astype(float), novig=po / (po + pu))
    by_line = (q.groupby(["game_id", "line"])["novig"]
               .agg(["median", "count"]).reset_index())
    by_line["balance"] = (by_line["median"] - 0.5).abs()
    best = (by_line.sort_values(["game_id", "count", "balance"],
                                ascending=[True, False, True])
            .drop_duplicates("game_id"))
    return (best.rename(columns={"median": "fair_over", "count": "n_books"})
            [cols].reset_index(drop=True))


def market_check(p_over, p_push, market_over, totals, lines,
                 min_games: int = MARKET_CHECK_MIN_GAMES) -> dict:
    """Over/under log loss, model vs the no-vig market, on the same
    games (pushes dropped; the model's P(over) taken given no push).
    beats_market: True only when the model's log loss is lower with 95%
    confidence (the paired difference's upper bound is below 0); None
    when fewer than min_games games qualify."""
    p_over, p_push = np.asarray(p_over, float), np.asarray(p_push, float)
    market_over = np.asarray(market_over, float)
    totals, lines = np.asarray(totals, float), np.asarray(lines, float)
    keep = totals != lines
    n = int(keep.sum())
    if n == 0:
        return {"n": 0, "beats_market": None}
    over = (totals > lines)[keep]
    model = np.clip(p_over[keep] / np.clip(1.0 - p_push[keep], 1e-12, None),
                    1e-6, 1 - 1e-6)
    market = np.clip(market_over[keep], 1e-6, 1 - 1e-6)

    def ll(p):
        return -(over * np.log(p) + (~over) * np.log(1.0 - p))

    diff = ll(model) - ll(market)
    se = float(diff.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    out = {"n": n, "model_log_loss": float(ll(model).mean()),
           "market_log_loss": float(ll(market).mean()),
           "diff": float(diff.mean()), "diff_se": se}
    out["beats_market"] = (None if n < min_games
                           else bool(out["diff"] + 1.96 * se < 0))
    return out


_SNAPSHOT_QUOTES_SQL = """
    SELECT DISTINCT ON (o.game_id, o.book_name)
           o.game_id, o.book_name, o.line, o.over_price, o.under_price
    FROM raw.odds_snapshots o
    JOIN raw.games g USING (game_id)
    WHERE o.market_type = 'total' AND o.line IS NOT NULL
      AND o.over_price IS NOT NULL AND o.under_price IS NOT NULL
      AND g.game_state IN ('FINAL', 'OFF')
      AND g.start_time_utc IS NOT NULL
      AND o.captured_at < (g.start_time_utc AT TIME ZONE 'UTC')
    ORDER BY o.game_id, o.book_name, o.captured_at DESC
"""

# ESPN's DraftKings closing over/under price, once ingestion/espn_odds.py
# stores it (raw.historical_odds.over_price / under_price go with the
# closing line over_under). DraftKings rows only: ESPN's older Unibet-era
# prices include in-play quotes.
_ESPN_QUOTES_SQL = """
    SELECT h.game_id, 'espn_' || h.provider AS book_name,
           h.over_under AS line, h.over_price, h.under_price
    FROM raw.historical_odds h
    JOIN raw.games g USING (game_id)
    WHERE h.provider = 'DraftKings' AND h.over_under IS NOT NULL
      AND h.over_price IS NOT NULL AND h.under_price IS NOT NULL
      AND g.game_state IN ('FINAL', 'OFF')
"""


def load_market_quotes(conn=None) -> pd.DataFrame:
    """Over/under quotes with both prices for completed games, one row per
    book: each snapshot book's last quote taken strictly before puck drop
    (raw.games.start_time_utc; the same window settlement uses for a
    close — games with no start time are left out, their quotes can't be
    shown to be pre-game), plus ESPN's DraftKings closing quote as one
    more book when raw.historical_odds has the price columns. conn: read
    inside that connection's transaction."""
    if conn is None:
        with engine.connect() as c:
            return load_market_quotes(c)
    cols = ["game_id", "book_name", "line", "over_price", "under_price"]
    parts = [pd.read_sql(text(_SNAPSHOT_QUOTES_SQL), conn)]
    stored = {r[0] for r in conn.execute(text("""
        SELECT column_name FROM information_schema.columns
        WHERE table_schema = 'raw' AND table_name = 'historical_odds'
    """))}
    if {"over_price", "under_price"} <= stored:
        parts.append(pd.read_sql(text(_ESPN_QUOTES_SQL), conn))
    parts = [f[cols] for f in parts if not f.empty]
    return (pd.concat(parts, ignore_index=True) if parts
            else pd.DataFrame(columns=cols))


# ── Walk-forward validation ────────────────────────────────────────

def run_totals(register: bool = True, market_quotes=None,
               variant: str = "A", with_roles: bool = False) -> dict:
    """Walk-forward evaluation with the hardened gate (module docstring,
    Evaluation). market_quotes: game_id, book_name, line, over_price,
    under_price rows for the market check; None loads them from the
    database (load_market_quotes). variant: the goalie experiment's
    feature set (VARIANTS; only A may be registered). with_roles: also
    report the NLL on backup starts (meta["backup_start"])."""
    from sklearn.metrics import log_loss

    if register and variant != "A":
        raise ValueError(f"variant {variant} is an experiment; only the "
                         f"production variant A may be registered")
    if variant == "A" and not with_roles:
        Xh, Xa, y_h, y_a, meta, names = load_totals_dataset()
    else:
        Xh, Xa, y_h, y_a, meta, names = load_totals_dataset(
            variant=variant, with_roles=True)
    backup = (meta["backup_start"].to_numpy(dtype=bool)
              if "backup_start" in meta else None)
    env_h = env_rates(meta["date"], y_h, ENV_PRIOR_RATE["home"])
    env_a = env_rates(meta["date"], y_a, ENV_PRIOR_RATE["away"])
    folds = walk_forward_folds(meta)
    totals = meta["total"].to_numpy()
    seasons, dates = meta["season"].to_numpy(), meta["date"].to_numpy()
    logger.info(f"Totals dataset: {len(meta)} games x {len(names)} attack "
                f"features, {len(folds)} folds (purge {PURGE_DAYS}d)")

    if market_quotes is None:
        market_quotes = load_market_quotes()
    mkt = market_over_probs(market_quotes).set_index("game_id")
    mkt_line = meta["game_id"].map(mkt["line"]).to_numpy(dtype=float)
    mkt_fair = meta["game_id"].map(mkt["fair_over"]).to_numpy(dtype=float)

    oof_nll = np.full(len(meta), np.nan)
    oof_base_nll = np.full(len(meta), np.nan)
    oof_nll_v1 = np.full(len(meta), np.nan)        # before the fixes
    oof_base_nll_v1 = np.full(len(meta), np.nan)
    oof_over = np.full(len(meta), np.nan)      # P(over) at DK line
    oof_exp_total = np.full(len(meta), np.nan)
    oof_mkt_over = np.full(len(meta), np.nan)  # at the market's line
    oof_mkt_push = np.full(len(meta), np.nan)
    fold_metrics = []

    for fold in folds:
        tr, val = fold.train_idx, fold.val_idx
        fm = fit_totals_fold(Xh, Xa, y_h, y_a, env_h, env_a,
                             tr, meta["date"],
                             is_playoff=meta["is_playoff"].to_numpy())
        lam_h, lam_a = predict_lambdas(fm, Xh[val], Xa[val],
                                       env_h[val], env_a[val])
        # Drift correction: the booster's running adjustment over this
        # season's earlier games (predictions only) comes off each game
        adj = booster_adjustment(lam_h, lam_a, env_h[val], env_a[val])
        shift = drift_shift(seasons[val], dates[val], adj)
        lam_hc, lam_ac = apply_drift(lam_h, lam_a, shift)
        # Margin weights fitted on the training games only, and given to
        # the model AND the baseline: the gate then asks only whether the
        # team stats add information, not which side got a structural fix
        weights = fit_margin_weights(poisson_pmf(env_h[tr]),
                                     poisson_pmf(env_a[tr]), y_h[tr], y_a[tr])
        tp = total_pmf(poisson_pmf(lam_hc), poisson_pmf(lam_ac), weights)

        # Environment baseline: the trailing league rates alone, through
        # the identical PMF machinery — the bar the features must clear
        pe_h, pe_a = poisson_pmf(env_h[val]), poisson_pmf(env_a[val])
        tp_base = total_pmf(pe_h, pe_a, weights)

        tv = totals[val]
        oof_nll[val] = nll_of_totals(tp, tv)
        oof_base_nll[val] = nll_of_totals(tp_base, tv)
        oof_exp_total[val] = expected_total(tp)
        # v1 for comparison: independent joint, no drift correction
        oof_nll_v1[val] = nll_of_totals(
            total_pmf(poisson_pmf(lam_h), poisson_pmf(lam_a), None), tv)
        oof_base_nll_v1[val] = nll_of_totals(total_pmf(pe_h, pe_a, None), tv)

        priced = np.isfinite(mkt_line[val])
        if priced.any():
            po, pp = prob_over(tp[priced], mkt_line[val][priced])
            oof_mkt_over[val[priced]], oof_mkt_push[val[priced]] = po, pp

        lines = meta["market_line"].to_numpy()[val]
        lined = np.isfinite(lines)
        m = {
            "val_season": int(meta["season"].iloc[val[0]]),
            "n_val": len(val),
            "iters": fm["iters"], "scale": round(fm["scale"], 4),
            "nll": float(np.mean(oof_nll[val])),
            "baseline_nll": float(np.mean(oof_base_nll[val])),
            "nll_v1": float(np.mean(oof_nll_v1[val])),
            "baseline_nll_v1": float(np.mean(oof_base_nll_v1[val])),
            "margin_weights": [round(float(w), 3) for w in weights],
            "mean_booster_adj": float(adj.mean()),
            "mean_pred_total": float(np.mean(oof_exp_total[val])),
            "mean_actual_total": float(tv.mean()),
            "n_lined": int(lined.sum()),
            "n_priced": int(priced.sum()),
        }
        if backup is not None:
            b = backup[val]
            m["n_backup"] = int(b.sum())
            if b.any():
                m["backup_nll"] = float(np.mean(oof_nll[val][b]))
                m["backup_baseline_nll"] = float(np.mean(oof_base_nll[val][b]))
        if lined.any():
            p_over, p_push = prob_over(tp[lined], lines[lined])
            over_actual = tv[lined] > lines[lined]
            push = tv[lined] == lines[lined]
            oof_over[val[lined]] = p_over
            keep = ~push
            if keep.sum() > 50:
                m["over_log_loss"] = float(log_loss(
                    over_actual[keep], np.clip(p_over[keep], 1e-6, 1 - 1e-6)))
                m["over_rate"] = float(over_actual[keep].mean())
        fold_metrics.append(m)
        logger.info(
            f"  fold {m['val_season']}: nll={m['nll']:.4f} "
            f"(baseline {m['baseline_nll']:.4f}; before the fixes "
            f"{m['nll_v1']:.4f} vs {m['baseline_nll_v1']:.4f}) "
            f"pred_total={m['mean_pred_total']:.2f} vs {m['mean_actual_total']:.2f}"
            + (f" | over_ll={m['over_log_loss']:.4f} (n={m['n_lined']})"
               if "over_log_loss" in m else ""))

    scored = ~np.isnan(oof_nll)
    pooled = {
        "nll": float(np.mean(oof_nll[scored])),
        "baseline_nll": float(np.mean(oof_base_nll[scored])),
        "nll_v1": float(np.mean(oof_nll_v1[scored])),
        "baseline_nll_v1": float(np.mean(oof_base_nll_v1[scored])),
        "n_scored": int(scored.sum()),
    }
    diff = oof_nll[scored] - oof_base_nll[scored]
    pooled["nll_diff_se"] = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    pooled["gate_passed"] = pooled["nll"] < pooled["baseline_nll"]
    pooled["variant"] = variant
    pooled["folds_won"] = int(sum(f["nll"] < f["baseline_nll"]
                                  for f in fold_metrics))
    if backup is not None:
        b = scored & backup
        pooled["n_backup"] = int(b.sum())
        if b.any():
            bd = oof_nll[b] - oof_base_nll[b]
            pooled["backup_nll"] = float(np.mean(oof_nll[b]))
            pooled["backup_baseline_nll"] = float(np.mean(oof_base_nll[b]))
            pooled["backup_diff_se"] = (float(bd.std(ddof=1) / np.sqrt(len(bd)))
                                        if len(bd) > 1 else float("nan"))

    lined = ~np.isnan(oof_over)
    if lined.any():
        lines = meta["market_line"].to_numpy()
        keep = lined & (totals != lines)
        pooled["over_log_loss"] = float(log_loss(
            (totals > lines)[keep], np.clip(oof_over[keep], 1e-6, 1 - 1e-6)))
        pooled["n_lined"] = int(keep.sum())

    priced = ~np.isnan(oof_mkt_over)
    pooled["market_check"] = market_check(
        oof_mkt_over[priced], oof_mkt_push[priced], mkt_fair[priced],
        totals[priced], mkt_line[priced])

    logger.info(
        f"POOLED OOF: nll={pooled['nll']:.4f} vs baseline "
        f"{pooled['baseline_nll']:.4f} (both margin-reweighted) — GATE "
        f"{'PASSED' if pooled['gate_passed'] else 'FAILED'} | before the "
        f"fixes {pooled['nll_v1']:.4f} vs {pooled['baseline_nll_v1']:.4f}"
        + (f" | over_log_loss={pooled['over_log_loss']:.4f} "
           f"(n={pooled['n_lined']}, naive 0.693)" if "over_log_loss" in pooled
           else ""))
    mc = pooled["market_check"]
    if mc["n"] == 0:
        logger.info("Market check: no over/under prices for any scored game "
                    "yet (live snapshots or ESPN's DraftKings closing "
                    "prices), so no comparison with the market — totals "
                    "betting also needs this check")
    else:
        verdict = {None: f"too few games to judge (< {MARKET_CHECK_MIN_GAMES})",
                   True: "BEATS the market",
                   False: "does not beat the market"}[mc["beats_market"]]
        logger.info(f"Market check: over/under log loss {mc['model_log_loss']:.4f} "
                    f"vs the no-vig market {mc['market_log_loss']:.4f} over "
                    f"{mc['n']} games (difference {mc['diff']:+.4f} ± "
                    f"{1.96 * mc['diff_se']:.4f}) — {verdict}")

    if register:
        _register(pooled, meta, names)

    oof = meta.loc[scored, ["game_id", "season", "date", "total"]].copy()
    oof["nll"] = oof_nll[scored]
    oof["baseline_nll"] = oof_base_nll[scored]
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


def main(argv=None) -> dict:
    """Command line: run the walk-forward evaluation and register the
    model (inactive), or with --no-register only report. --help runs
    nothing."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Walk-forward evaluation of the totals model (hardened "
                    "gate + market check); registers it, inactive, in "
                    "models.model_registry")
    parser.add_argument("--no-register", action="store_true",
                        help="evaluate and report only; write nothing")
    args = parser.parse_args(argv)
    return run_totals(register=not args.no_register)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
