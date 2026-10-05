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
- Starter-role experiment (2026-10-01, pre-registered; run_totals
  variant=A|B|C|D, features/goalie_role.py). Pass rule fixed in advance:
  pooled NLL below the baseline by >= 2 paired SEs AND ahead in >= 4 of 5
  folds. Baseline 2.1787 throughout. A (v2) 2.1801 (+0.0014, SE 0.0011,
  2/5 folds); B (no goalie_*) 2.1793 (+0.0006, SE 0.0010, 1/5); C (v2 +
  defending goalie's start shares, primary flag, rest, back-to-back,
  carried save%, gap to the other goalie) 2.1790 (+0.0003, SE 0.0010,
  2/5); D (C minus goalie_*) 2.1791 (+0.0004, SE 0.0010, 3/5). NONE
  passes. O/U log loss at the DraftKings line (n=1,011): A 0.6958, B
  0.6949, C 0.6941, D 0.6945 (coin flip 0.6931). On games where either
  starter was not his team's season-to-date leader (4,466 scored games;
  the definition counts ties and early-season splits, so it is broad) all
  four variants are still behind the baseline (2.1788-2.1795 vs 2.1781).
  Attack rows facing a non-leader goalie do score ~0.12 more regulation
  goals on average, but that difference did not turn into a lower
  out-of-fold NLL for any variant.
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

v3 experiment: market offset + Dixon-Coles (2026-10-04, PRE-REGISTERED
before any variant ran; code run_totals_v3, opt-in)
- Why: v2 has no edge over the environment and its over/under log loss at
  the DraftKings line is worse than the no-vig market (0.6959 vs 0.6882,
  n=1,010). The moneyline model only became useful when it was boosted
  FROM the market (models/lgbm.py). Two-way over/under prices now exist for
  every past season, so the same trick can be tried for totals.
- T0 was reproduced first, unchanged: 2.1801 vs 2.1787 (the v2 numbers).
- Market prices (→ one fair P(over) per game at the market's main line,
  market_over_probs over the union of these quotes; no-vig → the book's
  fee taken out; closing → the last price before puck drop):
  (a) raw.odds_snapshots, each book's last pre-game quote (live, 2026-27);
  (b) raw.odds_history (2024-25, 10 books incl. Pinnacle): each book's
      LAST snapshot strictly before puck drop, over and under from that one
      snapshot; consensus = median no-vig P(over) across the books at the
      line most of them quote;
  (c) ESPN's DraftKings closing over/under (2025-26);
  (d) ESPN's Unibet closing over/under (2020-21 to 2023-24), kept only
      when the line is 5.5, the overround (→ the two sides' implied
      probabilities summed, minus 1: the book's fee) is 3.5%-6.5%, the
      no-vig P(over) is 0.25-0.80, and both of the game's Unibet moneyline
      prices are under 1000 in size. Those are in-play markers: 2023-24
      has lines from 2.0 to 13.0, moneylines of +/-1000 to 10000, and a
      175-game block with 6.5%-9.8% margins whose log loss is too good for
      a pre-game price (0.6545 vs 0.6654 for a constant). Data check done
      before this pre-registration (prices vs outcomes only, no model):
      Unibet's 5.5 is a real line, not the old placeholder. Its no-vig
      P(over 5.5) is calibrated in every bin from 0.40 to 0.70 (0.576
      predicted vs 0.584 actual in the 0.55-0.60 bin) and beats a constant
      by 0.003-0.008 log loss in 2020-21 to 2022-23. At 5.5 overtime cannot
      change the result (a regulation tie has an even total). Kept: 926 /
      1,378 / 1,359 / 1,069 games.
- Variants:
  T0  v2 as committed (environment offset, margin weights, drift
      correction, goalie variant A features).
  T1  market offset. For a priced game, the market rates are the
      environment rates (the home/away split stays the environment's)
      times one common factor, solved so that the model's own PMF
      machinery (the fold's margin weights; T2/T3 also rho) gives
      P(over line | no push) = the no-vig market P(over) (market_lambdas).
      A second booster (variant A features, same params) is trained ONLY on
      priced training games with init_score = log(market rate), the
      models/lgbm.py pattern, early-stopped on the priced regular-season
      tail. Its season drift relative to the market rates is removed with
      v2's drift_shift rule. Unpriced games: T0 unchanged.
  T2  T1 + a Dixon-Coles rho (→ one number that moves probability between
      the 0-0 / 1-1 and the 1-0 / 0-1 regulation scores, i.e. low-score
      dependence between the two teams). Joint = independent x margin
      weight x tau(h, a; rate_h, rate_a, rho), renormalized. rho is fitted
      by maximum likelihood on each fold's training games with the
      environment PMFs and that fold's margin weights (fitted first),
      bounded to [-0.2, 0.2]. It is used everywhere the PMF is: the market
      inversion, both boosters' scoring, the baseline.
  T3  T2 + goalie-role variant C features (features/goalie_role.py) in
      both boosters.
  M   reference only, never adoptable: the market rates through the PMF
      with no booster (environment baseline where unpriced). It copies the
      market at the posted line, so the market check cannot fail it; it
      shows what the market alone is worth to the NLL.
- Metrics, all pooled out-of-fold over the 5 walk-forward folds:
  - NLL vs the MATCHED environment baseline: the environment rates
    through the variant's own joint machinery (T0, T1: margin weights,
    i.e. 2.1787; T2, T3: margin weights + rho). A structural fix goes to
    both sides, as in the hardened gate. Paired SE and folds won.
  - NLL vs T0 (paired difference and SE).
  - Calibration: predicted vs actual regulation tie rate; the total-goals
    distribution, mean predicted P(total = t) vs the observed share for
    t = 0..10 and 11+, with the largest gap and a chi-square (→ the sum
    over those buckets of (observed - expected)^2 / expected; smaller is
    better calibrated); push rates P(total = 5, 6, 7) (→ a push is a tie
    with a whole-number line: the stake is refunded).
  - Market check (market_check): over/under log loss at the main line vs
    the no-vig P(over), pushes dropped, pooled over all priced scored
    games, also reported per source.
- Pass rule, fixed now. A T1-T3 variant passes when ALL hold:
  (1) pooled NLL below its matched baseline by >= 2 paired SEs;
  (2) below that baseline in >= 4 of the 5 folds;
  (3) the market check is NOT WORSE at 95%: diff - 1.96 SE <= 0 (diff =
      model minus market log loss) over >= 200 priced games.
  Several pass: the lowest pooled NLL, unless a simpler passing variant
  (lower number) is within 1 paired SE of it; then the simpler one.
- Consequences, fixed now. The passing variant becomes the default:
  MODEL_VERSION "v3", with the production scorer switched to it. GATE_PASSED
  flips to True only if that variant ALSO BEATS the market: diff + 1.96 SE
  < 0 over the pooled priced games. Reason: GATE_PASSED lets the bet
  checker give totals legs a BET verdict, and a model that is merely "not
  worse" than the market has no proven edge, so it would bet noise. If no
  variant passes, v2 stays the default and GATE_PASSED stays False.
- STATUS (2026-10-04, v3 experiment): NO VARIANT PASSES. v2 stays the
  default, MODEL_VERSION stays "v2", GATE_PASSED stays False. Run once,
  as pre-registered (python -m models.totals --v3, read-only, ~20 s).
  Data: 7,979 games in the 5 validation seasons' dataset, 7,140 priced
  (Unibet 4,732 after cleaning — exactly the 926 / 1,378 / 1,359 / 1,069
  counted before the pre-registration; odds_history 1,398 games from
  13,969 book quotes; DraftKings 1,010); 6,214 priced validation games,
  6,143 without a push at the line. T0 reproduced v2 exactly.
  Pooled NLL (6,993 scored games), vs the matched baseline, folds won:
    T0  2.1801  vs B    2.1787  +0.0014 ± 0.0011  2/5
    T1  2.1770  vs B    2.1787  -0.0017 ± 0.0016  3/5
    T2  2.1771  vs B_dc 2.1787  -0.0016 ± 0.0016  3/5
    T3  2.1764  vs B_dc 2.1787  -0.0023 ± 0.0016  3/5
    M   2.1740  vs B    2.1787  -0.0047 ± 0.0012  5/5 (reference only)
  vs T0: T1 -0.0031 ± 0.0012, T2 -0.0030 ± 0.0012, T3 -0.0037 ± 0.0012,
  M -0.0061 ± 0.0015. vs M: T1 +0.0030 ± 0.0010, T2 +0.0031 ± 0.0010,
  T3 +0.0025 ± 0.0009 (the boosters make the market worse).
  Per fold (2021-22..2025-26), T1 / T3 / M / B: 2.1913 / 2.1902 /
  2.1877 / 2.1986; 2.1592 / 2.1584 / 2.1544 / 2.1605; 2.1772 / 2.1767 /
  2.1762 / 2.1763; 2.2040 / 2.2051 / 2.2011 / 2.2016; 2.1532 / 2.1516 /
  2.1504 / 2.1563. Fitted rho per fold: -0.0005, +0.0083, -0.0018,
  +0.0055, -0.0060 — no low-score dependence left once the margin
  weights are in, so T2 is T1 to 4 decimals.
  Market check (over/under log loss at the main line vs no-vig, n=6,143):
    T0 0.6871  T1 0.6844  T2 0.6844  T3 0.6840  B 0.6852  vs market
    0.6815; diffs +0.0056 ± 0.0013, +0.0028 ± 0.0008, +0.0029 ± 0.0009,
    +0.0024 ± 0.0008, +0.0037 ± 0.0012. Every variant is WORSE than the
    market at 95%. By source, T3 minus market: Unibet +0.0024 ± 0.0010
    (n=3,806), 2024-25 consensus +0.0036 ± 0.0022 (n=1,327), DraftKings
    2025-26 +0.0010 ± 0.0015 (n=1,010). The plain environment baseline
    B is level with the 2024-25 ten-book consensus (+0.0002 ± 0.0024)
    but 0.0069 ± 0.0028 behind DraftKings 2025-26 and 0.0041 ± 0.0016
    behind Unibet.
  Calibration (pooled): regulation tie rate predicted 0.218 (T0-T3) /
  0.222 (M, B) vs actual 0.2225. Total-goals distribution chi-square
  (12 buckets) T0 33.8, T1 38.9, T2 38.9, T3 39.2, M 37.4, B 31.6 —
  all clearly off, the same way: too much mass on 11+ goals (predicted
  0.054-0.057 vs 0.0465) and 2 goals, too little on 6, 7 and 8.
  Push rates at totals 5 / 6 / 7: T3 0.217 / 0.104 / 0.197 vs actual
  0.219 / 0.112 / 0.205 (6 and 7 are under-predicted by every variant).
  Rule outcome: T1-T3 fail (1) (about 1-1.4 SE, not 2), (2) (3 of 5
  folds) and (3) (worse than the market). M passes (1) and (2) but is
  not adoptable: it IS the market.
  What it means: starting from the market total helps the NLL (M beats
  the environment in every season), but no booster on top of it — team
  stats, goalie roles or rho — adds anything the closing price lacks;
  each one makes the market's over/under probability worse. What would
  flip GATE_PASSED: a new pre-registered variant that beats its baseline
  by >= 2 SE in >= 4/5 folds AND beats the no-vig market's over/under
  log loss with 95% confidence (diff + 1.96 SE < 0) on >= 200 priced
  games. None of T1-T3 is close (their best upper bound is +0.0040).
  Not pre-registered here and left untested: the home/away split from
  the moneyline instead of the environment's (an older draft of this
  experiment did that); it changes the regulation-tie and push shape,
  not the expected total, so it is unlikely to close the market gap.
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


def dc_tau(pmf_h: np.ndarray, pmf_a: np.ndarray, rho: float) -> np.ndarray:
    """(n, K+1, K+1) Dixon-Coles factors (v3 experiment, variant T2/T3):
    1 everywhere except the four low scores, with lh, la the two sides'
    rates (each PMF's mean):
        tau(0,0) = 1 - lh*la*rho   tau(0,1) = 1 + lh*rho
        tau(1,0) = 1 + la*rho      tau(1,1) = 1 - rho
    (home goals first). rho > 0 moves probability from 0-0 and 1-1 to
    1-0 and 0-1; rho < 0 the other way. Factors are floored at 0."""
    n, k1 = pmf_h.shape
    tau = np.ones((n, k1, k1))
    if rho == 0.0:
        return tau
    k = np.arange(k1)
    lh, la = pmf_h @ k, pmf_a @ k
    tau[:, 0, 0] = 1.0 - lh * la * rho
    tau[:, 0, 1] = 1.0 + lh * rho
    tau[:, 1, 0] = 1.0 + la * rho
    tau[:, 1, 1] = 1.0 - rho
    return np.clip(tau, 0.0, None)


def joint_pmf(pmf_h: np.ndarray, pmf_a: np.ndarray,
              margin_weights=MARGIN_WEIGHTS, rho: float = 0.0) -> np.ndarray:
    """(n, K+1, K+1) joint regulation-score distribution from per-side
    PMFs: their product, each cell times its margin bucket's weight,
    renormalized per game to sum to 1. margin_weights=None keeps the
    plain independent product. rho != 0 also multiplies in the
    Dixon-Coles factors (dc_tau) before renormalizing."""
    joint = pmf_h[:, :, None] * pmf_a[:, None, :]
    if margin_weights is None and rho == 0.0:
        return joint
    if margin_weights is not None:
        w = _check_weights(margin_weights)
        joint = joint * w[margin_buckets(pmf_h.shape[1])][None]
    if rho != 0.0:
        joint = joint * dc_tau(pmf_h, pmf_a, float(rho))
    return joint / joint.sum(axis=(1, 2), keepdims=True)


DC_RHO_BOUNDS = (-0.2, 0.2)


def fit_dc_rho(pmf_h: np.ndarray, pmf_a: np.ndarray, y_home, y_away,
               margin_weights=MARGIN_WEIGHTS) -> float:
    """Maximum-likelihood Dixon-Coles rho from training games only, given
    the per-side PMFs and the (already fitted) margin weights. Only the
    four low-score cells change, so a game's log-likelihood moves by
    log tau(observed cell) - log(sum of joint x tau)."""
    from scipy.optimize import minimize_scalar

    k1 = pmf_h.shape[1]
    joint = joint_pmf(pmf_h, pmf_a, margin_weights)
    k = np.arange(k1)
    lh, la = pmf_h @ k, pmf_a @ k
    h = np.clip(np.asarray(y_home).astype(int), 0, k1 - 1)
    a = np.clip(np.asarray(y_away).astype(int), 0, k1 - 1)
    cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
    mass = np.stack([joint[:, i, j] for i, j in cells], axis=1)     # (n, 4)
    obs = np.stack([(h == i) & (a == j) for i, j in cells], axis=1)

    def taus(rho):
        t = np.stack([1.0 - lh * la * rho, 1.0 + lh * rho,
                      1.0 + la * rho, np.full_like(lh, 1.0 - rho)], axis=1)
        return np.clip(t, 1e-12, None)

    def nll(rho):
        t = taus(rho)
        z = 1.0 + (mass * (t - 1.0)).sum(axis=1)
        log_obs = np.where(obs, np.log(t), 0.0).sum(axis=1)
        return -np.mean(log_obs - np.log(z))

    res = minimize_scalar(nll, bounds=DC_RHO_BOUNDS, method="bounded",
                          options={"xatol": 1e-6})
    return float(res.x)


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


def total_from_joint(joint: np.ndarray) -> np.ndarray:
    """(n, K+1, K+1) regulation joint -> (n, 2K+2) settlement-total PMF:
    each cell's mass goes to h + a, and a regulation tie (h == a) to
    h + a + 1 (the OT/SO winner's credited goal)."""
    n, k1, _ = joint.shape
    h_idx, a_idx = np.meshgrid(np.arange(k1), np.arange(k1), indexing="ij")
    t_idx = np.where(h_idx == a_idx, h_idx + a_idx + 1, h_idx + a_idx)
    to_total = np.zeros((k1 * k1, 2 * k1))
    to_total[np.arange(k1 * k1), t_idx.ravel()] = 1.0
    return joint.reshape(n, k1 * k1) @ to_total


def total_pmf(pmf_h: np.ndarray, pmf_a: np.ndarray,
              margin_weights=MARGIN_WEIGHTS, rho: float = 0.0) -> np.ndarray:
    """Settlement-total distribution from two per-side regulation PMFs
    (n, K+1) -> (n, 2K+2): the joint (margin-reweighted by default, and
    Dixon-Coles adjusted when rho != 0; see joint_pmf) summed along each
    total, with every regulation tie shifted up one goal (the OT/SO
    winner's credited goal)."""
    return total_from_joint(joint_pmf(pmf_h, pmf_a, margin_weights, rho))


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


# ── v3 experiment: market offset + Dixon-Coles (pre-registered) ────
#
# Module docstring, "v3 experiment", has the variants, metrics and the
# pass rule. Everything here is evaluation only and writes nothing.

V3_VARIANTS = ("T0", "T1", "T2", "T3")
V3_REFERENCES = ("M", "B", "B_dc")      # market alone, the two baselines
MARKET_SCALE_RANGE = (0.25, 4.0)        # bisection bracket for the factor
MARKET_BISECT_STEPS = 40
MIN_PRICED_TRAIN = 200                  # fewer: priced games use M's rates
GATE_SE = 2.0                           # pass rule (1)
GATE_FOLDS = 4                          # pass rule (2)
CAL_MAX_TOTAL = 11                      # calibration buckets 0..10, 11+

# Unibet cleaning (docstring, source d)
UNIBET_LINE = 5.5
UNIBET_OVERROUND = (0.035, 0.065)
UNIBET_FAIR = (0.25, 0.80)
UNIBET_ML_MAX = 1000


def market_lambdas(fair_over, line, base_h, base_a,
                   margin_weights=MARGIN_WEIGHTS, rho: float = 0.0,
                   steps: int = MARKET_BISECT_STEPS) -> tuple:
    """Per game, the rates (base_h * c, base_a * c) whose total PMF
    (margin weights, rho) gives P(over line | no push) = fair_over: the
    market's expected scoring put through this model's own machinery, with
    the home/away split of the base rates. c is found by bisection on
    log c inside MARKET_SCALE_RANGE (P(over) rises with c); a price outside
    what the range can reach gets the nearest end. Pure."""
    fair = np.asarray(fair_over, dtype=float)
    line = np.asarray(line, dtype=float)
    base_h = np.asarray(base_h, dtype=float)
    base_a = np.asarray(base_a, dtype=float)
    lo = np.full(len(fair), np.log(MARKET_SCALE_RANGE[0]))
    hi = np.full(len(fair), np.log(MARKET_SCALE_RANGE[1]))
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        k = np.exp(mid)
        tp = total_pmf(poisson_pmf(base_h * k), poisson_pmf(base_a * k),
                       margin_weights, rho)
        po, pp = prob_over(tp, line)
        p = po / np.clip(1.0 - pp, 1e-12, None)
        up = p < fair
        lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
    k = np.exp(0.5 * (lo + hi))
    return (np.clip(base_h * k, *LAMBDA_CLIP),
            np.clip(base_a * k, *LAMBDA_CLIP))


def clean_unibet(rows: pd.DataFrame) -> pd.DataFrame:
    """Unibet closing over/under rows that pass the pre-registered checks
    (line 5.5, overround 3.5-6.5%, no-vig P(over) 0.25-0.80, both
    moneyline prices under 1000 in size), as quotes. rows: game_id,
    over_under, over_price, under_price, home_ml, away_ml. Pure."""
    from features.util import american_implied_prob
    cols = ["game_id", "book_name", "line", "over_price", "under_price"]
    r = rows.dropna(subset=["over_under", "over_price", "under_price"])
    if r.empty:
        return pd.DataFrame(columns=cols)
    po = r["over_price"].astype(float).map(american_implied_prob)
    pu = r["under_price"].astype(float).map(american_implied_prob)
    over_round = po + pu - 1.0
    fair = po / (po + pu)
    ml_ok = ((r["home_ml"].astype(float).abs() < UNIBET_ML_MAX)
             & (r["away_ml"].astype(float).abs() < UNIBET_ML_MAX))
    ok = ((r["over_under"].astype(float) == UNIBET_LINE)
          & over_round.between(*UNIBET_OVERROUND)
          & fair.between(*UNIBET_FAIR) & ml_ok)
    out = r[ok].rename(columns={"over_under": "line"})
    return out.assign(book_name="espn_Unibet")[cols].reset_index(drop=True)


def history_closing_quotes(rows: pd.DataFrame) -> pd.DataFrame:
    """Each book's closing over/under quote from raw.odds_history rows
    (game_id, book, side, price, point, snapshot_ts, start_utc): the last
    snapshot strictly before puck drop that has both sides at the same
    point. Pure."""
    cols = ["game_id", "book_name", "line", "over_price", "under_price"]
    r = rows[rows["snapshot_ts"] < rows["start_utc"]]
    r = r.dropna(subset=["price", "point"])
    if r.empty:
        return pd.DataFrame(columns=cols)
    key = ["game_id", "book", "snapshot_ts"]
    over = r[r["side"] == "over"].drop_duplicates(key, keep="last")
    under = r[r["side"] == "under"].drop_duplicates(key, keep="last")
    both = over.merge(under, on=key, suffixes=("_o", "_u"))
    both = both[both["point_o"].astype(float) == both["point_u"].astype(float)]
    last = (both.sort_values("snapshot_ts")
            .drop_duplicates(["game_id", "book"], keep="last"))
    return pd.DataFrame({
        "game_id": last["game_id"].to_numpy(),
        "book_name": last["book"].to_numpy(),
        "line": last["point_o"].astype(float).to_numpy(),
        "over_price": last["price_o"].astype(int).to_numpy(),
        "under_price": last["price_u"].astype(int).to_numpy()})


_HISTORY_TOTALS_SQL = """
    SELECT o.game_id, o.book, o.side, o.price, o.point, o.snapshot_ts,
           (g.start_time_utc AT TIME ZONE 'UTC') AS start_utc
    FROM raw.odds_history o
    JOIN raw.games g USING (game_id)
    WHERE o.market = 'totals' AND g.game_state IN ('FINAL', 'OFF')
      AND g.start_time_utc IS NOT NULL
"""

_UNIBET_SQL = """
    SELECT h.game_id, h.over_under, h.over_price, h.under_price,
           h.home_ml, h.away_ml
    FROM raw.historical_odds h
    JOIN raw.games g USING (game_id)
    WHERE h.provider = 'Unibet' AND g.game_state IN ('FINAL', 'OFF')
"""


def load_v3_quotes(conn=None) -> pd.DataFrame:
    """Every pre-registered over/under source as quote rows (game_id,
    book_name, line, over_price, under_price, source): live snapshots,
    ESPN DraftKings closing, raw.odds_history closing, cleaned Unibet.
    Read-only."""
    if conn is None:
        with engine.connect() as c:
            return load_v3_quotes(c)
    parts = []
    snap = load_market_quotes(conn)
    if not snap.empty:
        parts.append(snap.assign(source=np.where(
            snap["book_name"].astype(str).str.startswith("espn_"),
            "draftkings_espn", "snapshot")))
    hist = pd.read_sql(text(_HISTORY_TOTALS_SQL), conn)
    if not hist.empty:
        hist["snapshot_ts"] = pd.to_datetime(hist["snapshot_ts"])
        hist["start_utc"] = pd.to_datetime(hist["start_utc"])
        parts.append(history_closing_quotes(hist).assign(source="odds_history"))
    uni = pd.read_sql(text(_UNIBET_SQL), conn)
    parts.append(clean_unibet(uni).assign(source="unibet_espn"))
    parts = [p for p in parts if not p.empty]
    cols = ["game_id", "book_name", "line", "over_price", "under_price",
            "source"]
    return (pd.concat(parts, ignore_index=True)[cols] if parts
            else pd.DataFrame(columns=cols))


def quote_sources(quotes: pd.DataFrame) -> pd.Series:
    """game_id -> its quote source ('mixed' when several)."""
    if quotes.empty or "source" not in quotes:
        return pd.Series(dtype=object)
    return quotes.groupby("game_id")["source"].agg(
        lambda s: s.iloc[0] if s.nunique() == 1 else "mixed")


def total_calibration(tpmf: np.ndarray, totals: np.ndarray,
                      max_total: int = CAL_MAX_TOTAL) -> dict:
    """Mean predicted P(total = t) vs the observed share, t = 0..max-1 and
    max+ (one bucket), with the largest gap and the chi-square
    sum((obs - exp)^2 / exp) over counts. Pure."""
    t = np.clip(np.asarray(totals).astype(int), 0, max_total)
    pred = np.hstack([tpmf[:, :max_total],
                      tpmf[:, max_total:].sum(axis=1, keepdims=True)])
    expected = pred.sum(axis=0)
    observed = np.bincount(t, minlength=max_total + 1).astype(float)
    n = len(t)
    chi2 = float(np.sum((observed - expected) ** 2 / np.clip(expected, 1e-9,
                                                             None)))
    return {"pred": (expected / n).round(4).tolist(),
            "obs": (observed / n).round(4).tolist(),
            "max_gap": float(np.max(np.abs(observed - expected)) / n),
            "chi2": chi2}


def not_worse_than_market(mc: dict) -> bool:
    """Pass rule (3): enough priced games and the model's log loss is not
    worse than the market's at 95% (diff - 1.96 SE <= 0)."""
    return (mc.get("n", 0) >= MARKET_CHECK_MIN_GAMES
            and np.isfinite(mc.get("diff_se", np.nan))
            and mc["diff"] - 1.96 * mc["diff_se"] <= 0)


def v3_passes(s: dict) -> bool:
    """The pre-registered pass rule (1)-(3) on one variant's summary."""
    return (s["diff"] <= -GATE_SE * s["diff_se"]
            and s["folds_won"] >= GATE_FOLDS
            and not_worse_than_market(s["market_check"]))


def choose_v3(summaries: dict):
    """The variant the pre-registered rule adopts (None when none of
    T1-T3 passes): the lowest pooled NLL, unless a simpler passing
    variant is within 1 paired SE of it."""
    passing = [v for v in V3_VARIANTS[1:] if v in summaries
               and v3_passes(summaries[v])]
    if not passing:
        return None
    best = min(passing, key=lambda v: summaries[v]["nll"])
    for v in passing:                          # in order: simplest first
        if v == best:
            return v
        d = summaries[v]["nll"] - summaries[best]["nll"]
        if d <= summaries[v]["vs"].get(best, {}).get("se", np.inf):
            return v
    return best


def _paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    return {"diff": float(d.mean()),
            "se": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1
            else float("nan")}


def run_totals_v3(market_quotes=None, variants=V3_VARIANTS) -> dict:
    """The pre-registered v3 walk-forward (module docstring). Registers and
    writes nothing. market_quotes: rows like load_v3_quotes (None loads
    them). Returns {"summaries", "folds", "chosen", "rho", "oof"}."""
    for v in variants:
        if v not in V3_VARIANTS:
            raise ValueError(f"unknown v3 variant {v!r}")
    Xh_c, Xa_c, y_h, y_a, meta, names_c = load_totals_dataset(
        variant="C", with_roles=True)
    n_a = len(ATTACK_FEATURES)
    if list(names_c[:n_a]) != list(ATTACK_FEATURES):
        raise RuntimeError("variant C columns must start with variant A's")
    feats = {"A": (Xh_c[:, :n_a], Xa_c[:, :n_a]), "C": (Xh_c, Xa_c)}

    env_h = env_rates(meta["date"], y_h, ENV_PRIOR_RATE["home"])
    env_a = env_rates(meta["date"], y_a, ENV_PRIOR_RATE["away"])
    folds = walk_forward_folds(meta)
    totals = meta["total"].to_numpy()
    seasons, dates = meta["season"].to_numpy(), meta["date"].to_numpy()
    is_po = meta["is_playoff"].to_numpy(dtype=bool)
    n = len(meta)

    if market_quotes is None:
        market_quotes = load_v3_quotes()
    mkt = market_over_probs(market_quotes).set_index("game_id")
    line = meta["game_id"].map(mkt["line"]).to_numpy(dtype=float)
    fair = meta["game_id"].map(mkt["fair_over"]).to_numpy(dtype=float)
    priced = np.isfinite(line) & np.isfinite(fair)
    source = meta["game_id"].map(quote_sources(market_quotes)).to_numpy(
        dtype=object)
    logger.info(f"v3 dataset: {n} games, {priced.sum()} priced, "
                f"{len(folds)} folds")

    names = list(variants) + list(V3_REFERENCES)
    nll = {v: np.full(n, np.nan) for v in names}
    tie = {v: np.full(n, np.nan) for v in names}
    p_over = {v: np.full(n, np.nan) for v in names}
    p_push = {v: np.full(n, np.nan) for v in names}
    tpmfs = {v: np.full((n, 2 * (MAX_GOALS + 1)), np.nan) for v in names}
    fold_rows = []
    rhos = {}
    need_rho = any(v in ("T2", "T3") for v in variants)

    for fold in folds:
        tr, val = fold.train_idx, fold.val_idx
        pe_tr = (poisson_pmf(env_h[tr]), poisson_pmf(env_a[tr]))
        w = fit_margin_weights(*pe_tr, y_h[tr], y_a[tr])
        rho = fit_dc_rho(*pe_tr, y_h[tr], y_a[tr], w) if need_rho else 0.0
        rhos[int(fold.val_season)] = rho

        def env_model(fs):
            Xh, Xa = feats[fs]
            fm = fit_totals_fold(Xh, Xa, y_h, y_a, env_h, env_a, tr,
                                 meta["date"], is_playoff=is_po)
            lh, la = predict_lambdas(fm, Xh[val], Xa[val],
                                     env_h[val], env_a[val])
            adj = booster_adjustment(lh, la, env_h[val], env_a[val])
            return apply_drift(lh, la, drift_shift(seasons[val], dates[val],
                                                   adj))

        tr_p, val_p = tr[priced[tr]], val[priced[val]]
        mrates = {}

        def market_rates(r):
            if r not in mrates:
                idx = np.concatenate([tr_p, val_p])
                mh, ma = np.full(n, np.nan), np.full(n, np.nan)
                if len(idx):
                    mh[idx], ma[idx] = market_lambdas(
                        fair[idx], line[idx], env_h[idx], env_a[idx], w, r)
                mrates[r] = (mh, ma)
            return mrates[r]

        def market_model(fs, r):
            mh, ma = market_rates(r)
            if len(val_p) == 0:
                return np.array([]), np.array([])
            if len(tr_p) < MIN_PRICED_TRAIN:
                return mh[val_p], ma[val_p]
            Xh, Xa = feats[fs]
            fm = fit_totals_fold(Xh, Xa, y_h, y_a, mh, ma, tr_p,
                                 meta["date"], is_playoff=is_po)
            lh, la = predict_lambdas(fm, Xh[val_p], Xa[val_p],
                                     mh[val_p], ma[val_p])
            adj = booster_adjustment(lh, la, mh[val_p], ma[val_p])
            return apply_drift(lh, la, drift_shift(seasons[val_p],
                                                   dates[val_p], adj))

        pos = {g: i for i, g in enumerate(val)}
        loc_p = np.array([pos[g] for g in val_p], dtype=int)

        def combine(fallback, market):
            lh, la = fallback[0].copy(), fallback[1].copy()
            if len(loc_p):
                lh[loc_p], la[loc_p] = market
            return lh, la

        env_val = (env_h[val], env_a[val])
        rates = {"B": (env_val, 0.0), "B_dc": (env_val, rho)}
        lam_t0 = env_model("A")
        rates["M"] = (combine(env_val, tuple(x[val_p] for x in
                                             market_rates(0.0))), 0.0)
        if "T0" in variants:
            rates["T0"] = (lam_t0, 0.0)
        if "T1" in variants:
            rates["T1"] = (combine(lam_t0, market_model("A", 0.0)), 0.0)
        if "T2" in variants:
            rates["T2"] = (combine(lam_t0, market_model("A", rho)), rho)
        if "T3" in variants:
            rates["T3"] = (combine(env_model("C"), market_model("C", rho)),
                           rho)

        row = {"val_season": int(fold.val_season), "n_val": len(val),
               "n_priced": int(len(val_p)), "n_priced_train": int(len(tr_p)),
               "margin_weights": [round(float(x), 3) for x in w],
               "rho": round(rho, 4)}
        for v, ((lh, la), r) in rates.items():
            joint = joint_pmf(poisson_pmf(lh), poisson_pmf(la), w, r)
            tp = total_from_joint(joint)
            tpmfs[v][val] = tp
            nll[v][val] = nll_of_totals(tp, totals[val])
            tie[v][val] = np.trace(joint, axis1=1, axis2=2)
            if len(val_p):
                po, pp = prob_over(tp[loc_p], line[val_p])
                p_over[v][val_p], p_push[v][val_p] = po, pp
            row[v] = float(np.mean(nll[v][val]))
        fold_rows.append(row)
        logger.info(f"  fold {row['val_season']}: rho={rho:+.4f} priced "
                    f"{row['n_priced']}/{row['n_val']} | " + " ".join(
                        f"{v}={row[v]:.4f}" for v in names if v in row))

    scored = ~np.isnan(nll["B"])
    reg_tie = (y_h == y_a)
    summaries = {}
    for v in names:
        base = "B_dc" if v in ("T2", "T3") else "B"
        s = {"nll": float(np.mean(nll[v][scored])),
             "baseline": base,
             "baseline_nll": float(np.mean(nll[base][scored])),
             "n_scored": int(scored.sum())}
        pb = _paired(nll[v][scored], nll[base][scored])
        s["diff"], s["diff_se"] = pb["diff"], pb["se"]
        s["folds_won"] = int(sum(
            np.mean(nll[v][f.val_idx]) < np.mean(nll[base][f.val_idx])
            for f in folds))
        s["vs"] = {u: _paired(nll[v][scored], nll[u][scored])
                   for u in names if u != v}
        s["tie_pred"] = float(np.mean(tie[v][scored]))
        s["tie_actual"] = float(np.mean(reg_tie[scored]))
        s["calibration"] = total_calibration(tpmfs[v][scored], totals[scored])
        s["push_pred"] = {k: float(np.mean(tpmfs[v][scored][:, k]))
                          for k in (5, 6, 7)}
        s["push_actual"] = {k: float(np.mean(totals[scored] == k))
                            for k in (5, 6, 7)}
        pr = scored & priced
        s["market_check"] = market_check(p_over[v][pr], p_push[v][pr],
                                         fair[pr], totals[pr], line[pr])
        s["market_by_source"] = {
            str(src): market_check(p_over[v][m], p_push[v][m], fair[m],
                                   totals[m], line[m])
            for src in sorted(set(source[pr]))
            for m in [pr & (source == src)]}
        s["passes"] = v in V3_VARIANTS and v3_passes(s)
        s["beats_market"] = bool(s["market_check"].get("beats_market"))
        summaries[v] = s

    chosen = choose_v3({v: summaries[v] for v in variants})
    for v in names:
        s = summaries[v]
        mc = s["market_check"]
        logger.info(
            f"{v:5s} nll={s['nll']:.4f} vs {s['baseline']} "
            f"{s['baseline_nll']:.4f} ({s['diff']:+.4f} ± {s['diff_se']:.4f},"
            f" {s['folds_won']}/{len(folds)} folds) vs T0 "
            f"{s['vs'].get('T0', {'diff': 0.0})['diff']:+.4f} | tie "
            f"{s['tie_pred']:.3f}/{s['tie_actual']:.3f} chi2 "
            f"{s['calibration']['chi2']:.1f} | market "
            + (f"{mc['model_log_loss']:.4f} vs {mc['market_log_loss']:.4f} "
               f"({mc['diff']:+.4f} ± {mc['diff_se']:.4f}, n={mc['n']})"
               if mc.get("n") else "none")
            + (" PASSES" if s["passes"] else ""))
    logger.info(f"v3 adopted by the pre-registered rule: {chosen}")

    oof = meta.loc[scored, ["game_id", "season", "date", "total"]].copy()
    for v in names:
        oof[f"nll_{v}"] = nll[v][scored]
    oof["priced"] = priced[scored]
    oof["source"] = source[scored]
    return {"summaries": summaries, "folds": fold_rows, "chosen": chosen,
            "rho": rhos, "oof": oof}


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
    parser.add_argument("--v3", action="store_true",
                        help="run the pre-registered v3 experiment (market "
                             "offset + Dixon-Coles); writes nothing")
    args = parser.parse_args(argv)
    if args.v3:
        return run_totals_v3()
    return run_totals(register=not args.no_register)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
