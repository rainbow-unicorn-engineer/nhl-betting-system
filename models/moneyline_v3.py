"""
models/moneyline_v3.py
Moneyline model v3 experiment, priced backtests and the bet-timing study.

Terms (each explained once):
- Moneyline → a bet on who wins, overtime and shootout included.
- Log loss → a score for how wrong the probabilities were; lower is
  better. Pooled = averaged over every validation game of every fold.
- Walk-forward → train only on earlier seasons, score the next one, never
  the other way round (models.baseline.walk_forward_folds).
- Paired SE → the random wobble of a difference between two models scored
  on the same games: the standard deviation of the per-game differences
  divided by the square root of the number of games. A gap of 2 SE or
  more is unlikely to be luck.
- No-vig price → a book's probability with its built-in fee (the vig)
  taken out. Consensus no-vig close → the median over books of the
  closing no-vig home probability (features/market_prices.py).
- Market offset → the LightGBM booster starts from logit(market
  probability) and only learns a correction to it (models/lgbm.py).
- Edge → our probability minus the no-vig market probability for a side,
  in percentage points. Kelly → the bet size that grows a bankroll
  fastest if our probabilities are right; quarter-Kelly bets a quarter of
  it. Flat stake → one unit on every bet, whatever the edge.
- ROI → profit divided by the amount staked.
- CLV (closing-line value) → whether our price was better than the
  market's closing price. Closing EV of a bet → p_close × decimal − 1,
  where p_close is the consensus no-vig closing probability of the side
  we bet and decimal is the decimal odds we took: the profit per unit
  staked if the closing market is right. It is positive when the price
  beat the close.
- Game-clustered bootstrap → resample whole games with replacement many
  times and recompute the number; the middle 95% of those values is the
  confidence interval (CI).

=====================================================================
PRE-REGISTRATION (written 2026-10-04, committed before any variant ran)
=====================================================================

Data. features.game_vector as stored (seasons 2020-21..2025-26; its two
power-play columns are a constant 99 because the vectors were built
before raw.skater_games power-play time was filled). Folds:
models.baseline.walk_forward_folds (validation seasons 2021-22..2025-26).
Every variant is fitted with models.lgbm.fit_fold / predict_fold
unchanged: same LightGBM parameters, seed 42, the same time-ordered 15%
calibration tail, early stopping and temperature scaling; market-offset
booster M where a market exists, market-blind fallback F otherwise.

Variants.
  V0  lgbm v2 exactly as now. Market = raw.historical_odds normalised
      (Unibet three-way 2020-21..2023-24, DraftKings 2025-26 from
      December); 2024-25 has no market, so F scores it.
  V1  V0, plus the Odds API consensus no-vig CLOSE as the market for
      every game that has one (all of 2024-25): market_home_prob =
      nv_consensus, market_available = 1. Other seasons unchanged.
  V1p Diagnostic only, never eligible for adoption: V1 with Pinnacle's
      no-vig close where Pinnacle quoted (consensus otherwise).
  V2  V1 + 12 power-play features (features/power_play.py): for windows
      20 and 82 games (82 = season to date), home minus away of
        pp_toi_share  = sum(PP TOI) / sum(team TOI)
        pk_toi_share  = sum(PK TOI) / sum(team TOI)
        pp_gf_per60   = 3600 * sum(PP goals) / sum(PP TOI)
        pk_ga_per60   = 3600 * sum(PP goals against) / sum(PK TOI)
        pp_xgf_per60  = 3600 * sum(PP xG for) / sum(PP TOI)
        pk_xga_per60  = 3600 * sum(PP xG against) / sum(PK TOI)
      Team PP TOI = sum of its skaters' PP TOI / 5, PK TOI = sum of
      short-handed TOI / 4, team TOI = summed goalie TOI, PP goals from
      raw.team_games, PP xG = MoneyPuck shots at 5v4/5v3/4v3 (the
      features/team_features.py conventions). Rolled within team and
      season over games strictly before the game (shift, then roll).
      NaN when either side has no prior game or a zero denominator
      (LightGBM handles missing values natively).
  V3  V2 + 7 goalie-role features: home starter minus away starter of
      each features.goalie_role.ROLE_FEATURES (the actual starter, the
      same proxy the stored vectors use).

Primary metric. Pooled walk-forward log loss over every validation game
of every fold, compared with V0 by the paired SE of per-game differences.

Market test (the real bar). Priced games = 2024-25 games with an Odds
API close plus 2025-26 games with a DraftKings close. On those games:
d_mkt = mean(model log loss - market log loss), market = consensus
no-vig close (DraftKings no-vig in 2025-26), with its paired SE. Also
reported per season and against Pinnacle (2024-25).

Adoption rule (fixed in advance).
  A variant Vk (k = 1, 2, 3) is ELIGIBLE when both hold:
    (a) it beats V0 pooled by at least 2 paired SE (mean difference
        <= -2 SE), and
    (b) it is not worse than the market at 95%: d_mkt < 1.645 * SE_mkt
        (no one-sided 95% evidence that the market beats it).
  Among eligible variants: V1 is chosen; V2 replaces V1 only if V2 also
  beats V1 pooled by at least 2 paired SE; V3 replaces V2 (or V1, if V2
  was not chosen) only if it beats the chosen one by at least 2 paired
  SE. No eligible variant → V0 stays.
  Production: the chosen variant becomes models/lgbm.py's default only
  if the live pick job can rebuild its inputs. V1 needs nothing new (live
  picks already use a live consensus market). V2/V3 would also need the
  live slate builder to compute the same features; until that is built,
  a chosen V2/V3 is reported as chosen but the default stays at V1 (or
  V0) and the gap is listed as follow-up work.
  "Beats the market" (a separate claim, not needed for adoption):
  d_mkt + 1.96 * SE_mkt < 0.
  Secondary, reported only: log loss on the games where V0 has a market;
  per-season log loss; calibration (ECE); seeds 1 and 2 as a robustness
  check (the rule uses seed 42).

Priced backtests (descriptive; they change no default, and the edge
threshold stays 2.5%). For every variant, from its walk-forward
probabilities:
  2024-25  fair = consensus no-vig close; prices = (i) best of the 10
           books at the close, (ii) best of the 6 licensed US books,
           (iii) each bettor-relevant book alone (draftkings, fanduel,
           betmgm, williamhill_us, betrivers, espnbet, pinnacle).
  2025-26  fair and price = DraftKings close.
  Decision = betting.engine.evaluate_market (edge >= 2.5% and positive
  Kelly at the price). Quarter-Kelly with the locked caps (2% a bet, 10%
  a day, 4% a game): each day's candidates sorted by edge, largest
  first, staked from the bankroll at the start of the day and kept while
  they fit the caps; the bankroll compounds day by day. Flat-stake view:
  1 unit per bet. Reported: bets, hit rate, flat ROI and Kelly ROI on
  stake with 95% game-clustered bootstrap CIs (10,000 resamples, seed
  20261004), end bankroll, maximum drawdown, mean closing EV vs the
  consensus no-vig close; by edge bucket 2.5-4, 4-6, 6-9 and 9+ points.
  Exchanges: raw.odds_history holds no Kalshi or Polymarket prices (the
  history was bought with 10 sportsbooks), so an exchange-only backtest
  has no data; that is stated, not simulated. A clearly labelled
  hypothetical shows what the taker fees would do: buy at the consensus
  no-vig price plus 1 cent, Kalshi 0.07 * p * (1 - p) and Polymarket
  0.0695 * p * (1 - p) per contract.

Bet-timing study (2024-25 games with a 10:00 Central morning snapshot,
889 games). Model used: the adopted default (V0 if none is adopted, which
then uses the morning consensus as its market input in the same way).
Morning arm: the same fold models scored with the MORNING consensus
no-vig as the market input; fair = morning consensus; price = best of 10
books at the morning snapshot. Close arm: as the 2024-25 backtest,
restricted to the same games.
  H1  The model's morning edge predicts the line's move: ordinary least
      squares of move = (close consensus - morning consensus, home) on
      e = (model morning home prob - morning consensus home), all games.
      Supported if the slope > 0 with t >= 2.
  H2  Morning bets (the engine's decisions) beat the close: mean closing
      EV of the morning price > 0 by at least 2 SE (game-clustered),
      also reported as the no-vig move toward our side
      (p_close - p_morning for the side bet).
  H3  Which slot pays better: flat ROI per bet and mean closing EV,
      morning arm minus close arm on the same games, with a 95%
      game-clustered bootstrap CI. A slot is "better" only if its CI
      excludes 0. Expected to be underpowered (about 900 games).
  Caveat fixed in advance: both arms use the actual starting goalie,
  which is often unconfirmed at 10:00, so the morning arm is slightly
  flattered.

STATUS: pre-registered, not yet run.
"""
