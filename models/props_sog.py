"""
models/props_sog.py
Skater shots-on-goal (SOG) props model: a full count distribution of a
skater's shots on goal in a game, given that he plays (toi_seconds > 0),
from which P(SOG > line) is read for the usual prop lines 0.5 .. 4.5.

STATUS: see the STATUS section at the end of this docstring. Registration
is disabled (run_props(register=True) raises), and nothing reads it. The
price check (STATUS, "Market check") found that it does NOT beat the prop
market.

v3 PRE-REGISTRATION (2026-10-04). Written and committed before any v3
variant was run. The market parts are in models/props_market_check.py.
Disclosed: before writing this, ONE diagnostic of the existing baseline
B1 was run (no booster, no new feature). It split B1's mean error per
validation season into a rate part (B1's SOG/60 x the ACTUAL TOI, minus
actual SOG) and an expected-TOI part. The TOI part is ~0 in every season
(|.| <= 0.003 SOG a game). The rate part is all of it: -0.004, +0.049,
+0.050, +0.048, -0.000 for 2021-22 .. 2025-26. In 2024-25 B1 predicts
5.86 SOG/60 at the actual TOI against an actual 5.69. The trailing-365-
day league rate it rests on averaged 5.88 that season (the season before
ran at 6.09), and the career-to-date player rates sat 3.6% above it. So
the root cause is the LEVEL, in two parts. First, a trailing-365-day
league rate lags a falling league. Second, career-to-date rates were
earned in higher-shooting seasons, and the ratio to the training-window
mean that re-levels them assumes each player's career has the training
window's mix. P3's formula below follows from this reading. No P-variant
result had been seen when it was written.

Variants. Each one adds to the one before. All share the same rows,
folds, eligibility (>= 5 prior appearances), LightGBM params, early
stopping, NB dispersion fit, and the v2 in-season drift correction (M =
M2 in every variant, relative to that variant's own offset):
- P0 = v2 exactly (FEATURES, offset B1). It must reproduce v2's numbers
  (pooled NLL M 1.57818, B1 1.58173). The 2026-27 games loaded since are
  not a fold and come after every fold's rows, and every feature looks
  backward only, so the reproduction should be exact.
- P1 = P0 + power-play features (features/player_shots.py FEATURES_PP).
  Each one uses only his appearances strictly before the game:
    pp_toi_l5/l10/l20: mean power-play (PP → his team has more skaters
      on the ice after an opponent's penalty) minutes per appearance
      over his last 5/10/20 appearances;
    pp_share_l5/l10/l20: his PP seconds / his team's PP seconds over
      those same games. A game's team PP seconds = the sum of the team's
      skaters' PP seconds / 5 (five skaters are on the ice in a 5-on-4);
    pk_toi_l10/l20: mean penalty-kill (PK → his team is the one a
      skater short) minutes per appearance;
    pp_sog60_l20_rel, pp_sog60_season_rel: his shots on goal at power-
      play strength per 60 PP minutes, over his last 20 appearances and
      season to date, counting only games with shot data, divided by his
      position's trailing league SOG/60. A PP shot is a raw.shots event
      SHOT or GOAL whose strength (written from the shooter's side) gives
      his team more skaters than the opponent, with the opponent at 4 or
      fewer: 5v4, 5v3, 4v3, 6v4, 6v3;
    nonpp_sog60_l20_rel: the rest of his SOG per 60 of his non-PP
      minutes, same rules;
    team_pp_l10: his team's PP minutes per game over its last 10 games;
    opp_pk_l10: the opponent's PK minutes per game over its last 10
      games (team PK seconds = the skaters' PK seconds / 4). This measures
      how often the opponent takes penalties.
  Unknown stats (stats_filled_at NULL) are missing, never zero.
- P2 = P1 + usage proxies (FEATURES_USAGE):
    es_toi_l10: mean even-strength minutes (TOI - PP - PK), last 10;
    es_toi_rank_pct, toi_rank_pct: his rank on es_toi_l10 / toi_mean_l10
      among his team's skaters of his position group (F or D) dressed
      for this game, (rank - 1) / (n - 1): 0 = the most, 1 = the least;
    pp_rank: his rank on pp_toi_l10 among all his team's skaters dressed
      for this game (1 = the most; ranks 1-5 are roughly the first
      power-play unit);
    fo_l20: faceoffs taken per appearance over his last 20 (centres take
      most of them); fo_win_l20: faceoffs won / taken over his last 20.
  The ranks use who is dressed for this game. That is known at warm-ups
  before puck drop, and it is the same information the model already
  conditions on (it predicts given that he plays). They also use only
  the pre-game values of every teammate.
- P3 = P2 with the offset replaced by B3 (the baseline fix). Features
  are P2's. B3 = index x L_env x expected TOI / 3600, with expected TOI
  unchanged, where:
    L_env(d, pos): this season's league SOG/60 for his position (F or
    D), from the season's games dated strictly before d, shrunk toward
    last year's level:
        L_env = (S_sd + K_env x L_prev / 3600) / (T_sd + K_env) x 3600
      S_sd, T_sd = SOG and TOI of every played player-game of that
      position in the same season dated before d. L_prev = the
      position's trailing-365-day rate on the season's first date
      (league_pos_sog60 there). K_env = 100,000 minutes, about a fifth of
      a season's forward minutes.
    index (relative and time-decayed): over his earlier appearances j
    (any season),
        w_j   = 0.5 ^ ((d - d_j) / 365 days)
        index = (sum_j w_j sog_j + K_p)
                / (sum_j w_j toi_j x L_env(d_j, pos) / 3600 + K_p)
      K_p = 300 minutes x L_env(d, pos) / 60, which is v2's shrinkage
      (k = 300 TOI minutes) in expected-shot units. An index of 1 means
      his position's league rate. Each past game is judged against the
      league level of its own date, so a change in the league's level
      cannot leak into the player's rate. No training-window ratio is
      needed, so B3 is the same in every fold.
GATE (as v1/v2, unchanged; each variant's M against B1, the v2 offset):
pooled NLL(M) - NLL(B1) <= -2 paired SE; M beats B1 in >= 4 of 5 folds;
pooled ECE of P(SOG > 2.5) <= 0.02.
ADOPTION (decided before any run): start with A = P0. For k = 1, 2, 3 in
that order, Pk replaces A if (a) Pk passes the gate, (b) pooled NLL(Pk) -
NLL(A) <= -2 paired SE over the same player-games, and (c) Pk's NLL is
lower than A's in >= 3 of 5 folds. The final A becomes the default
(DEFAULT_VARIANT). If P0 stays, v3 = v2. Reported for information, not
part of the rule: per-fold mean bias, per-fold ECE, and B3 against B1.
MARKET: models/props_market_check.py is re-run for the adopted variant
on every shots-on-goal price row now loaded. Registration stays disabled
(run_props(register=True) raises) unless that check passes.

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
- In-season drift correction (v2; M2 in the code, now the default M —
  STATUS has the decision): for a player-game g on date D in season S,
  r = log(M1's prediction) - log(exposure baseline) is the booster's
  adjustment, and
      c(g) = exp( sum of r over S's eligible player-games dated strictly
                  before D / (their count n + DRIFT_PRIOR_ROWS) )
  i.e. their mean shrunk toward 0 with weight n / (n + 2000); c = 1 on a
  season's first date. M2 = M1 / c. It is applied when scoring
  validation rows and, within each training season from the in-sample
  booster predictions, before fitting the NB alpha. Same booster,
  features, params, folds and eligibility as v1 (M1). The adjustment is
  a prediction, not an outcome, and same-day rows never count, so c is
  known before puck drop. Copied from models/totals.py (drift_shift).

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
simple baselines"; it does NOT mean profitable. That needs a check
against prop PRICES, which this module does not have yet, and DraftKings
props carry about a 6.2% bookmaker margin.

STATUS (2026-10-03, v2 = M2, the drift-corrected booster; read-only on
the live database): GATE PASSED as a FORECASTER, and M2 replaced v1 under
the pre-registered decision rule below. Not a betting result: no price-
based check exists yet.
- v2 decision rule (fixed before the run): M2 replaces v1 only if (a) M2
  passes the v1 gate vs B1, (b) pooled NLL(M2) <= NLL(v1), and (c) the
  mean over folds of |mean predicted - actual SOG| is lower for M2. The
  same run re-scored v1 from the same boosters and reproduced its numbers
  exactly (pooled NLL 1.57827, B1 1.58173). All three held, so v2 is the
  default (DRIFT_CORRECT = True; run_props(drift_correct_m=False) gives
  v1, and every run reports both as M1 and M2):
  (a) pooled NLL M2 1.57818 vs B1 1.58173: M2 - B1 = -0.00355 (paired SE
      0.00018, ~20 SE); M2 beats B1 in 5 of 5 folds and B0 in 5 of 5;
      pooled ECE(> 2.5) 0.0087 (gate <= 0.02). PASS.
  (b) M2 - v1 = -0.000094 (paired SE 0.000052, -1.8 SE): not worse, but
      also not a clear improvement. PASS.
  (c) mean |fold bias| v1 0.0511 -> M2 0.0499. PASS, narrowly.
- v2 per fold, NLL M2 / v1 / B1 (M2 - v1, SE) and mean predicted minus
  actual SOG M2 / v1 / B1:
    2021-22 1.62150 / 1.62159 / 1.62470 (-0.00009, 0.00002)  +0.017 / +0.020 / -0.003
    2022-23 1.60374 / 1.60479 / 1.60771 (-0.00105, 0.00008)  +0.068 / +0.095 / +0.048
    2023-24 1.57988 / 1.57989 / 1.58384 (-0.00001, 0.00001)  +0.077 / +0.078 / +0.053
    2024-25 1.54769 / 1.54657 / 1.55053 (+0.00112, 0.00016)  +0.071 / +0.026 / +0.048
    2025-26 1.53823 / 1.53868 / 1.54199 (-0.00045, 0.00019)  +0.016 / -0.038 / +0.000
  M2 - B1 per fold: -0.0032, -0.0040, -0.0040, -0.0028, -0.0038.
- v2 calibration, ECE of P(> 2.5) per fold M2 / v1 / B1: 0.0078 / 0.0076
  / 0.0140, 0.0132 / 0.0189 / 0.0150, 0.0131 / 0.0131 / 0.0099, 0.0141 /
  0.0089 / 0.0086, 0.0061 / 0.0120 / 0.0055 — M2 is worse than B1 in 3
  of 5 folds (v1: 4 of 5). ECE of P(> 1.5) per fold M2 / v1 / B1: 0.0121
  / 0.0122 / 0.0188, 0.0186 / 0.0250 / 0.0223, 0.0195 / 0.0196 / 0.0208,
  0.0167 / 0.0107 / 0.0139, 0.0048 / 0.0128 / 0.0082. Pooled: ECE(> 2.5)
  M2 0.0087, v1 0.0068, B1 0.0070; ECE(> 1.5) M2 0.0136, v1 0.0106, B1
  0.0146; Brier(> 2.5) M2 0.15681, v1 0.15681, B1 0.15753; Brier(> 1.5)
  M2 0.21207, v1 0.21210, B1 0.21312. Pooled mean predicted minus actual
  SOG: M2 +0.050, v1 +0.036, B1 +0.029 (the pooled ECEs are worse for M2
  because v1's season biases of opposite sign cancelled when pooled).
  2025-26 by position, M2: forwards NLL 1.5982 (v1 1.5988), ECE(> 2.5)
  0.0078 (v1 0.0152, B1 0.0076); defensemen NLL 1.4196 (v1 1.4198),
  ECE(> 2.5) 0.0050 (v1 0.0073, B1 0.0070). Fitted alpha per fold, M2:
  0.042-0.054. A second run reproduced every v2 number.
- What the correction does and does not fix: it removes the booster's own
  level, so M2's level moves toward B1's. In 2025-26 that took M from
  -0.038 to +0.016; in 2024-25 it took M from +0.026 to +0.071, toward
  B1's own +0.048. B1 over-predicts by ~+0.05 SOG a game in 2022-23
  through 2024-25, and no booster-side correction can remove that.
  Second, c works on the log scale: in 2023-24 the booster's mean log
  adjustment is -0.004 (c ~ 1, no change) while its mean prediction sits
  +0.025 above B1's. Much of the remaining level error is the exposure
  baseline's (drift ratio and career-to-date rate); that is the next
  thing to examine, not part of this change.
- v1 history (2026-10-02, the first gated run; the numbers below are v1,
  "M" = v1; two further runs reproduced every number):
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
  failure the totals booster had (models/totals.py, drift correction);
  its fix, tried 2026-10-03 as M2, is v2 above.
- What a pass does and does not mean: M is a better shots forecaster
  than a last-10 average and than the exposure baseline, by a small but
  steady margin over B1 (0.0035 nats a player-game in v1, 0.0036 in v2).
  Whether that beats DraftKings props, which carry about a 6.2% margin,
  is untested: it needs a check against stored prop prices, which this
  module does not run yet. Registration stays disabled until that test
  exists.
- Market check (2026-10-03; models/props_market_check.py, pre-registered
  there before any result; read-only; `python -m models.props_market_check`).
  VERDICT: v2 does NOT beat the market, for either book or pooled.
  Model = this module's 2025-26 out-of-fold validation fold (49,627
  player-games, booster trained on 228,914 earlier rows; the same run
  reproduced the v2 numbers above: fold NLL 1.53823, pooled 1.57818).
  Prices = raw.prop_odds_hist player_shots_on_goal two-sided pre-game
  pairs: the stored current pair when last_updated < event_start, else
  the opening pair. DraftKings 1,354 rows / 75 games (2026-01-14 ..
  05-08; 20 regular-season, 55 playoff games): 975 current, 379 opening;
  1,331 matched (unmatched: 1 no player id, 19 did not play, 3 < 5 prior
  appearances). ESPN BET 5,769 rows / 410 games (2025-10-07 .. 12-01):
  171 current, 5,595 opening (81 of them carry a current pair stamped at
  or after puck drop, never used), 3 with no two-sided pre-game pair;
  5,733 matched (15 no player id, 18 < 5 prior). No pushes (all lines
  N.5). The 27 late-playoff games (2026-05-09 .. 06-14) are not loaded.
  PRIMARY: per prop, log loss(model P(over)) - log loss(no-vig market
  P(over)); SE clustered by game; beats only if mean + 1.96 SE < 0 and
  n >= 300.
    DraftKings 1,331 props: +0.00428 (SE 0.00547; upper +0.01500) NO
    ESPN BET   5,733 props: +0.00526 (SE 0.00167; upper +0.00853) NO
    pooled     7,064 props: +0.00508 (SE 0.00170; upper +0.00840) NO
  The market is the better forecaster; on ESPN BET (openings) and pooled
  it is better by about 3 SE. Brier model / market: DraftKings 0.24716 /
  0.24536, ESPN BET 0.24564 / 0.24315, pooled 0.24593 / 0.24357. ECE (10
  bins) model / market: 0.0486 / 0.0360, 0.0224 / 0.0130, pooled 0.0271 /
  0.0081. Mean P(over) model / market / actual over rate: DraftKings
  0.455 / 0.497 / 0.461, ESPN BET 0.519 / 0.514 / 0.516.
  SECONDARY (information only, flat 1 unit at the quoted prices, ROI with
  a game-clustered bootstrap 95% interval, 2,000 resamples, seed 7):
    DraftKings T=0.04 481 bets, hit 0.536, ROI +0.016 [-0.082, +0.117];
               T=0.06 334 bets, hit 0.560, ROI +0.062 [-0.046, +0.176]
    ESPN BET   T=0.04 1,550 bets, hit 0.518, ROI -0.022 [-0.071, +0.024];
               T=0.06 911 bets, hit 0.535, ROI +0.016 [-0.046, +0.081]
    pooled     T=0.04 2,031 bets, hit 0.522, ROI -0.013 [-0.055, +0.030];
               T=0.06 1,245 bets, hit 0.541, ROI +0.028 [-0.030, +0.085]
  Every interval includes 0. Pooled T=0.04 by line: 0.5 14 bets ROI
  -0.147, 1.5 1,034 +0.005, 2.5 794 -0.028, 3.5+ 189 -0.046; by
  position: F 1,526 -0.015, D 505 -0.007 (all intervals include 0).
  DraftKings bets are 88% unders at T=0.04 (425 of 481): its no-vig over
  probability averages 0.497 (0.528 with the vig left in) against an over
  rate of 0.461 in this mostly-
  playoff sample, which is a small and unrepresentative sample (75
  games), not an edge. DraftKings "N+" milestones (over only, no no-vig
  possible; information only): 1,337 matched; T=0.04 40 bets, ROI +0.012
  [-0.334, +0.302]; T=0.06 18 bets, ROI -0.321 [-0.724, +0.168].
  GATE_PASSED (the forecasting gate) and the disabled registration are
  unchanged: the model is a better forecaster than its baselines, and a
  worse one than the prop market.
"""
import logging

import numpy as np
import pandas as pd

from models.baseline import PURGE_DAYS, expected_calibration_error, walk_forward_folds

logger = logging.getLogger("nhl.models.props_sog")

MODEL_NAME = "props_sog"
MODEL_VERSION = "v2"           # v2: v1 + in-season drift correction (M2)
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
# In-season drift correction (M2; STATUS): shrinkage pseudo-count, in
# player-games, of the booster's running same-season mean adjustment.
DRIFT_PRIOR_ROWS = 2000
# Whether the reported model M is the drift-corrected M2 (v2) rather than
# v1 (M1); set by the pre-registered decision rule recorded in STATUS.
DRIFT_CORRECT = True

# v3 variants (pre-registration above): features and the offset the
# booster starts from. B1 = the v2 exposure baseline, B3 = the fixed one.
from features.player_shots import FEATURES, FEATURES_PP, FEATURES_USAGE  # noqa: E402

VARIANTS = {
    "P0": {"features": list(FEATURES), "offset": "B1"},
    "P1": {"features": list(FEATURES) + FEATURES_PP, "offset": "B1"},
    "P2": {"features": list(FEATURES) + FEATURES_PP + FEATURES_USAGE,
           "offset": "B1"},
    "P3": {"features": list(FEATURES) + FEATURES_PP + FEATURES_USAGE,
           "offset": "B3"},
}
ADOPT_ORDER = ("P1", "P2", "P3")
ADOPT_SE = 2.0                 # Pk must beat the adopted one by >= 2 SE
ADOPT_MIN_FOLDS = 3            # ... and in >= 3 of 5 folds
# The variant run_props reports as M by default, set by the v3 adoption
# rule (STATUS).
DEFAULT_VARIANT = "P0"

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


# ── In-season drift correction (point-in-time; M2) ─────────────────
#
# The same fix as the totals booster's (models/totals.py, drift_shift):
# the booster's inputs shift from season to season and it reads the shift
# as a change in shooting level, so its mean log-adjustment to the
# exposure baseline is ~0 on the seasons it trained on but moves on the
# next one. The baseline's drift ratio is meant to own the league level,
# so the level the booster adds on top is divided out: every player-game
# is divided by c = exp(the booster's mean adjustment over the same
# season's player-games on strictly EARLIER dates, shrunk toward 0 with
# weight n / (n + DRIFT_PRIOR_ROWS)). The adjustment is a prediction, not
# an outcome, and same-day rows never count, so c is known before puck
# drop.

def booster_adjustment(mu, base) -> np.ndarray:
    """Per player-game, r = log(M's prediction) - log(exposure baseline)."""
    return np.log(np.asarray(mu, float)) - np.log(np.asarray(base, float))


def drift_shift(seasons, dates, adj,
                prior_rows: float = DRIFT_PRIOR_ROWS) -> np.ndarray:
    """Per row, log c: the mean of `adj` over the same season's rows on
    strictly earlier dates, shrunk toward 0 with weight n / (n +
    prior_rows), n = their count; i.e. sum / (n + prior_rows). 0 on a
    season's first date."""
    df = pd.DataFrame({"season": np.asarray(seasons),
                       "date": pd.to_datetime(np.asarray(dates)),
                       "adj": np.asarray(adj, dtype=float)})
    day = df.groupby(["season", "date"], sort=True)["adj"].agg(["sum", "count"])
    earlier = day.groupby(level="season").cumsum() - day
    shift = (earlier["sum"] / (earlier["count"] + prior_rows)).fillna(0.0)
    keys = pd.MultiIndex.from_frame(df[["season", "date"]])
    return shift.reindex(keys).to_numpy(dtype=float)


def apply_drift(mu, shift) -> np.ndarray:
    """mu / c with c = exp(shift), clipped as usual."""
    return np.clip(np.asarray(mu, float) * np.exp(-np.asarray(shift, float)),
                   *MU_CLIP)


def drift_correct(mu, base, seasons, dates,
                  prior_rows: float = DRIFT_PRIOR_ROWS) -> tuple:
    """(corrected means, log c per row) for one set of rows. Seasons are
    handled separately, so a multi-season training window works too."""
    shift = drift_shift(seasons, dates, booster_adjustment(mu, base),
                        prior_rows)
    return apply_drift(mu, shift), shift


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
        out[f"bias_{k}"] = float(np.mean(mu) - np.mean(y))
        for line in (1.5, 2.5):
            pm = _prob_metrics(y, mu, alphas[k], line)
            tag = str(line).replace(".", "")
            out[f"brier{tag}_{k}"] = pm["brier"]
            out[f"ece{tag}_{k}"] = pm["ece"]
            out[f"meanp{tag}_{k}"] = pm["mean_p"]
            out[f"rate{tag}"] = pm["rate"]
    _paired_diffs(out, nll)
    return out


def _paired_diffs(out: dict, nll: dict) -> None:
    """Paired NLL differences (and SEs) of each model key vs B1 and B0,
    M2 - M1 when both variants are present, and B3 - B1."""
    for m in ("M", "M1", "M2"):
        for b in ("B1", "B0"):
            if m in nll and b in nll:
                out[f"diff_{m}_{b}"], out[f"se_{m}_{b}"] = _paired(nll[m], nll[b])
    if "M1" in nll and "M2" in nll:
        out["diff_M2_M1"], out["se_M2_M1"] = _paired(nll["M2"], nll["M1"])
    if "B3" in nll and "B1" in nll:
        out["diff_B3_B1"], out["se_B3_B1"] = _paired(nll["B3"], nll["B1"])


def gate(pooled: dict, folds: list, key: str = "M") -> dict:
    """The pre-registered gate (module docstring) for model `key`."""
    wins_b1 = sum(f[f"nll_{key}"] < f["nll_B1"] for f in folds)
    wins_b0 = sum(f[f"nll_{key}"] < f["nll_B0"] for f in folds)
    nll_ok = pooled[f"diff_{key}_B1"] <= -GATE_SE * pooled[f"se_{key}_B1"]
    folds_ok = wins_b1 >= GATE_MIN_FOLDS
    ece_ok = pooled[f"ece25_{key}"] <= GATE_ECE
    return {"nll_by_2se": bool(nll_ok), "folds_won_vs_B1": int(wins_b1),
            "folds_ok": bool(folds_ok), "ece25_ok": bool(ece_ok),
            "folds_won_vs_B0": int(wins_b0),
            "b0_nll_by_2se": bool(pooled[f"diff_{key}_B0"]
                                  <= -GATE_SE * pooled[f"se_{key}_B0"]),
            "passed": bool(nll_ok and folds_ok and ece_ok)}


def drift_decision(pooled: dict, folds: list) -> dict:
    """The pre-registered v2 decision rule (STATUS): M2 replaces M1 only if
    (a) M2 passes the v1 gate vs B1, (b) pooled NLL(M2) <= NLL(M1), and
    (c) the mean over folds of |mean predicted - actual SOG| is lower."""
    g2 = gate(pooled, folds, "M2")
    abs_bias = {k: float(np.mean([abs(f[f"bias_{k}"]) for f in folds]))
                for k in ("M1", "M2")}
    a = g2["passed"]
    b = pooled["nll_M2"] <= pooled["nll_M1"]
    c = abs_bias["M2"] < abs_bias["M1"]
    return {"a_m2_passes_gate": bool(a), "b_nll_not_worse": bool(b),
            "c_bias_lower": bool(c),
            "mean_abs_bias_M1": abs_bias["M1"],
            "mean_abs_bias_M2": abs_bias["M2"],
            "adopt_m2": bool(a and b and c)}


def run_props(register: bool = False, frame: pd.DataFrame = None,
              params=None, drift_correct_m: bool | None = None,
              variant: str | None = None) -> dict:
    """Walk-forward evaluation of M, B1, B0 and B3 (module docstring).
    variant (default DEFAULT_VARIANT): one of VARIANTS, i.e. the booster's
    features and its offset (B1 or B3); the gate is always against B1.
    Both booster variants are scored from the same fitted booster: M1 (v1,
    no correction) and M2 (in-season drift correction, relative to the
    variant's offset); "M" — the model the gate and the output speak for —
    is M2 when drift_correct_m (default DRIFT_CORRECT), else M1.
    register=True raises: there is no registry entry for this model."""
    if register:
        raise RuntimeError(
            "props_sog registration is disabled: the model has no price-"
            "based evaluation yet (it has not been checked against prop "
            "prices). Run "
            "with register=False.")
    if variant is None:
        variant = DEFAULT_VARIANT
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r} (have {sorted(VARIANTS)})")
    spec = VARIANTS[variant]
    feats = spec["features"]

    df = load_props_dataset(frame)
    X = df[feats].to_numpy(dtype=float)
    y = df["sog"].to_numpy(dtype=float)
    meta = df[["season", "date"]]
    folds = walk_forward_folds(meta)
    logger.info(f"Props SOG dataset ({variant}, offset {spec['offset']}): "
                f"{len(df)} eligible player-games x {len(feats)} features, "
                f"{len(folds)} folds (purge {PURGE_DAYS}d)")

    if drift_correct_m is None:
        drift_correct_m = DRIFT_CORRECT
    active = "M2" if drift_correct_m else "M1"
    oof = {k: np.full(len(df), np.nan)
           for k in ("M", "M1", "M2", "B1", "B0", "B3")}
    oof_alpha = {k: np.full(len(df), np.nan) for k in oof}
    oof_shift = np.full(len(df), np.nan)
    fold_metrics = []
    b0_all = np.clip(df["b0_mean"].to_numpy(float), *MU_CLIP)
    b3_all = np.clip(df["b3_mean"].to_numpy(float), *MU_CLIP)
    seasons = df["season"].to_numpy()
    dates = df["date"].to_numpy()

    for fold in folds:
        tr, val = fold.train_idx, fold.val_idx
        train_mean = train_window_sog60(y[tr], df["toi_seconds"].to_numpy()[tr])
        ratio = drift_ratio(df["league_sog60"].to_numpy(), train_mean)
        base = np.clip(exposure_baseline(df["shrunk_sog60"], df["exp_toi"],
                                         ratio), *MU_CLIP)
        off = base if spec["offset"] == "B1" else b3_all
        if not np.isfinite(off[np.concatenate([tr, val])]).all():
            raise ValueError(f"non-finite {spec['offset']} offset in fold "
                             f"{fold.val_season}")
        fm = fit_fold(X, y, off, tr, df["date"], params)
        m1_val = predict(fm, X[val], off[val])
        m1_tr = predict(fm, X[tr], off[tr])
        # M2: the booster's running same-season adjustment (earlier dates
        # only) divided out — on the validation season, and within each
        # training season from the in-sample predictions (for the alpha)
        m2_val, shift_val = drift_correct(m1_val, off[val], seasons[val],
                                          dates[val])
        m2_tr, _ = drift_correct(m1_tr, off[tr], seasons[tr], dates[tr])
        oof_shift[val] = shift_val
        mu = {"M1": m1_val, "M2": m2_val, "B1": base[val], "B0": b0_all[val],
              "B3": b3_all[val]}
        mu_tr = {"M1": m1_tr, "M2": m2_tr, "B1": base[tr], "B0": b0_all[tr],
                 "B3": b3_all[tr]}
        alphas = {k: fit_nb_alpha(y[tr], mu_tr[k]) for k in mu}
        mu["M"], alphas["M"] = mu[active], alphas[active]
        for k, v in mu.items():
            oof[k][val] = v
            oof_alpha[k][val] = alphas[k]

        m = {"val_season": int(fold.val_season), "n_train": len(tr),
             "iters": int(fm["iters"]), "train_sog60": round(train_mean, 4),
             "mean_drift_ratio": float(ratio[val].mean()),
             "mean_log_c": float(shift_val.mean()),
             "last_log_c": float(shift_val[-1]),
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
            f"| iters={m['iters']} drift={m['mean_drift_ratio']:.3f} | "
            f"M1={m['nll_M1']:.5f} M2={m['nll_M2']:.5f} bias M1 "
            f"{m['bias_M1']:+.3f} M2 {m['bias_M2']:+.3f}")

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
        pooled[f"bias_{k}"] = float(oof[k][scored].mean() - ys.mean())
        for line in (1.5, 2.5):
            p = np.empty(len(ys))
            for a in np.unique(a_rows):
                sel = a_rows == a
                p[sel] = prob_over(oof[k][scored][sel], a, line)
            hit = (ys > line).astype(float)
            tag = str(line).replace(".", "")
            pooled[f"brier{tag}_{k}"] = float(np.mean((p - hit) ** 2))
            pooled[f"ece{tag}_{k}"] = expected_calibration_error(hit, p)
    _paired_diffs(pooled, nll)
    pooled["model"] = active
    pooled["variant"] = variant
    pooled["gate"] = gate(pooled, fold_metrics)
    pooled["gate_passed"] = pooled["gate"]["passed"]
    pooled["gate_M1"] = gate(pooled, fold_metrics, "M1")
    pooled["gate_M2"] = gate(pooled, fold_metrics, "M2")
    pooled["drift_decision"] = drift_decision(pooled, fold_metrics)

    logger.info(
        f"POOLED OOF ({pooled['n']} player-games): NLL M={pooled['nll_M']:.4f} "
        f"B1={pooled['nll_B1']:.4f} B0={pooled['nll_B0']:.4f} | M-B1 "
        f"{pooled['diff_M_B1']:+.5f} (se {pooled['se_M_B1']:.5f}) M-B0 "
        f"{pooled['diff_M_B0']:+.5f} (se {pooled['se_M_B0']:.5f}) | ECE>2.5 "
        f"M={pooled['ece25_M']:.4f} | GATE "
        f"{'PASSED' if pooled['gate_passed'] else 'FAILED'} {pooled['gate']} "
        f"(M = {active}) | M2-M1 {pooled['diff_M2_M1']:+.5f} "
        f"(se {pooled['se_M2_M1']:.5f}) | decision {pooled['drift_decision']}")

    out = df.loc[scored, ["player_id", "game_id", "season", "date",
                          "pos_group", "sog"]].copy()
    for k in oof:
        out[f"mu_{k}"] = oof[k][scored]
        out[f"alpha_{k}"] = oof_alpha[k][scored]
    out["log_c"] = oof_shift[scored]
    return {"folds": fold_metrics, "pooled": pooled, "oof": out,
            "variant": variant}


# ── v3: comparing variants and the adoption rule ───────────────────

def row_nll(res: dict, key: str = "M") -> pd.Series:
    """Per scored player-game, the NLL of the actual SOG under `key`'s
    mean and its own fold alpha, indexed by (player_id, game_id)."""
    o = res["oof"]
    nll = nb_nll_rows(o["sog"], o[f"mu_{key}"], o[f"alpha_{key}"])
    return pd.Series(nll, index=pd.MultiIndex.from_frame(
        o[["player_id", "game_id"]]), name=key).sort_index()


def nb_nll_rows(y, mu, alpha) -> np.ndarray:
    """NB NLL with a per-row alpha (each fold has its own)."""
    y, mu, alpha = (np.asarray(y, float), np.asarray(mu, float),
                    np.asarray(alpha, float))
    out = np.empty(len(y))
    for a in np.unique(alpha):
        sel = alpha == a
        out[sel] = nb_nll(y[sel], mu[sel], a)
    return out


def compare_variants(res_new: dict, res_old: dict) -> dict:
    """Paired comparison of two runs' M on the same player-games: pooled
    NLL(new) - NLL(old) with its paired SE, and the folds in which new's
    NLL is lower."""
    a, b = row_nll(res_new), row_nll(res_old)
    if not a.index.equals(b.index):
        raise ValueError("the two runs scored different player-games")
    diff, se = _paired(a.to_numpy(), b.to_numpy())
    fo = {int(f["val_season"]): f["nll_M"] for f in res_old["folds"]}
    fn = {int(f["val_season"]): f["nll_M"] for f in res_new["folds"]}
    if set(fo) != set(fn):
        raise ValueError("the two runs have different folds")
    wins = sum(fn[s] < fo[s] for s in fn)
    return {"diff": diff, "se": se, "folds_won": int(wins),
            "n_folds": len(fn),
            "fold_diffs": {s: fn[s] - fo[s] for s in sorted(fn)}}


def adoption(results: dict, start: str = "P0",
             order=ADOPT_ORDER) -> dict:
    """The pre-registered v3 adoption rule. results: variant -> run_props
    result. A starts at `start`; each Pk in `order` replaces A if (a) Pk
    passes the gate vs B1, (b) NLL(Pk) - NLL(A) <= -ADOPT_SE paired SE
    and (c) Pk's NLL is lower than A's in >= ADOPT_MIN_FOLDS folds."""
    adopted = start
    steps = []
    for k in order:
        if k not in results:
            continue
        cmp_ = compare_variants(results[k], results[adopted])
        a = bool(results[k]["pooled"]["gate"]["passed"])
        b = bool(cmp_["diff"] <= -ADOPT_SE * cmp_["se"])
        c = bool(cmp_["folds_won"] >= ADOPT_MIN_FOLDS)
        steps.append({"variant": k, "against": adopted, "gate_passed": a,
                      "nll_by_2se": b, "folds_ok": c,
                      **{f"cmp_{x}": v for x, v in cmp_.items()},
                      "adopted": a and b and c})
        if a and b and c:
            adopted = k
    return {"adopted": adopted, "steps": steps}


def run_v3(frame: pd.DataFrame = None, params=None,
           variants=("P0",) + ADOPT_ORDER) -> dict:
    """Every v3 variant on one feature frame, then the adoption rule.
    Read only (run_props with register=False)."""
    if frame is None:
        from features.player_shots import load_player_features
        frame = load_player_features()
    results = {v: run_props(register=False, frame=frame, params=params,
                            variant=v) for v in variants}
    return {"results": results, "decision": adoption(results)}


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
    parser.add_argument("--variant", choices=sorted(VARIANTS), default=None,
                        help=f"v3 variant to evaluate (default "
                             f"{DEFAULT_VARIANT})")
    parser.add_argument("--v3", action="store_true",
                        help="run every v3 variant and the adoption rule")
    args = parser.parse_args(argv)
    if args.v3:
        out = run_v3()
        for v, res in out["results"].items():
            print(f"== {v}")
            _print_report(res)
        import json
        print(json.dumps(out["decision"], indent=1, default=float))
        return out
    if not args.evaluate:
        parser.print_help()
        return None
    res = run_props(register=False, variant=args.variant)
    _print_report(res)
    return res


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
