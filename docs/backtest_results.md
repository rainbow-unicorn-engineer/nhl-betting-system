# Moneyline backtest results (current, reproducible)

**Run date:** 2026-10-04 · **Code:** `models/moneyline_v3.py` (pre-registration and full numbers in its docstring) · **Raw output:** `models/artifacts/moneyline_v3_results.json` · **Re-run:** `.venv\Scripts\python -m models.moneyline_v3` (read-only: it writes only the JSON file)

This page replaces the betting half of [phase3_results.md](phase3_results.md). Its headline, "+26.6% flat ROI on bets with a 6-9% edge", came from 46 bets in one season and does not reproduce: the same bucket is +5.0% on 65 bets in 2025-26 today, with an interval from -20.5% to +30.8%.

## Terms

- **Moneyline** → a bet on who wins, overtime and shootout included.
- **No-vig price** → a sportsbook's probability with its built-in fee (the vig) taken out. **Consensus no-vig close** → the median of that over the books, at the last snapshot before the game.
- **Log loss** → a score for how wrong the probabilities were; lower is better.
- **Paired SE** → the random wobble of a difference between two models scored on the same games. A gap of 2 SE or more is unlikely to be luck.
- **Edge** → our probability minus the no-vig market probability, in percentage points.
- **Flat stake** → one unit on every bet. **Quarter-Kelly** → a stake sized by the edge and the price, a quarter of the size that grows a bankroll fastest if our probabilities are right; capped at 2% of the bankroll a bet, 10% a day, 4% a game.
- **ROI** → profit divided by the amount staked.
- **95% interval** (game-clustered bootstrap) → resample whole games 10,000 times and keep the middle 95% of the results. If it contains 0, the result could be luck.
- **CLV / closing EV** → whether the price we took was better than the market's final price. Closing EV = (closing no-vig probability of our side) x (decimal odds we took) - 1. Positive means we beat the close.
- **Pinnacle** → a low-margin book that professional bettors use; its closing price is the usual "sharpest" benchmark.

## The data

| Season | Prices | Games |
|---|---|---|
| 2024-25 | The Odds API historical close (median 14 minutes before puck drop) at 10 books, including Pinnacle; plus a 10:00 Central morning snapshot for 889 games | 1,398 |
| 2025-26 | DraftKings close from ESPN | 1,014 |
| 2020-21 .. 2023-24 | Unibet three-way regulation lines from ESPN: used as a model input only, never as a payable price | — |

There are **no Kalshi or Polymarket prices** in any of this: the history covers 10 sportsbooks. So there is no real exchange backtest; the exchange rows below are a labelled what-if.

## 1. Does the model beat the market? No.

Walk-forward (→ train on earlier seasons, score the next) over 6,993 games; market comparison on the 2,412 priced games.

| Model | What it adds | Log loss | vs V0 (SE) | vs market on priced games (SE) |
|---|---|---:|---:|---:|
| V0 | the current model | 0.66163 | — | +0.0037 (0.0025) |
| V1 | real 2024-25 consensus market as its starting point | 0.66223 | +0.0006 (0.0010) | +0.0055 (0.0020) |
| V2 | V1 + power-play form | 0.66228 | +0.0007 (0.0011) | +0.0070 (0.0019) |
| V3 | V2 + goalie-role inputs | 0.66207 | +0.0004 (0.0011) | +0.0074 (0.0019) |

Positive = worse. The market's own log loss on those games is 0.6666 (Pinnacle alone 0.6576 in 2024-25).

- **Rule fixed in advance:** a variant replaces V0 only if it beats V0 by 2 SE and is not worse than the market at 95%. None does, under any of three random seeds. **The production model is unchanged.**
- **The model is at best equal to the market, never better.** In 2024-25 V1 ties the consensus close and Pinnacle (-0.0010, SE 0.0017). In 2025-26 V0 ties DraftKings (+0.00004), and V1 is clearly worse (+0.0144, SE 0.0042). Why V1 drifts in 2025-26 is not known (a guess: corrections learned on Unibet seasons do not carry over).
- Two after-the-fact checks (mapping the Unibet lines onto a two-way scale; removing 106 Unibet rows that look captured during the game) did not help. They were never eligible for adoption.

## 2. Priced backtests

Every bet the engine would place at the locked 2.5-point threshold, settled at the real price.

| Model, season, prices | Bets | Flat ROI (95% interval) | Quarter-Kelly ROI | Mean closing EV |
|---|---:|---:|---:|---:|
| V0, 2024-25, best of 10 books | 1,049 | +0.6% (-6.0%, +6.9%) | +1.0% | -1.6% |
| V0, 2024-25, best of 6 licensed US books | 1,049 | -0.4% (-6.9%, +5.8%) | +0.4% | -2.6% |
| V0, 2024-25, one US book alone | 958-1,033 | -0.5% to -2.0% | -0.1% to -1.9% | -3.8% to -4.9% |
| V0, 2024-25, Pinnacle alone | 1,046 | -0.4% (-6.7%, +6.1%) | +0.5% | -2.4% |
| V0, 2025-26, DraftKings | 401 | -1.3% (-10.7%, +8.5%) | -0.1% | -4.2% |
| V1, 2024-25, best of 10 books | 585 | +4.4% (-3.6%, +12.1%) | +5.6% | -1.8% |
| V1, 2025-26, DraftKings | 727 | **-10.0% (-16.6%, -3.4%)** | -9.1% | -4.2% |

What this says:

- **No result shows a real edge.** Every interval for V0 contains 0. V1's +4.4% in 2024-25 is followed by a loss that is clearly not luck in 2025-26.
- **Closing EV is negative everywhere.** Betting near the close at these prices pays the books' margin: about 1.6 points with the best of 10 books, about 4 points at one US book. Line shopping (→ taking the best price across several books) is worth roughly 2-3 points of ROI on its own.
- Maximum drawdown (→ the largest drop from a bankroll peak) for V0 quarter-Kelly: 29% in 2024-25, 31% in 2025-26.

### By claimed edge (V0, flat stake)

| Edge (points) | 2024-25 best of 10: bets, ROI (interval) | 2025-26 DraftKings: bets, ROI (interval) |
|---|---|---|
| 2.5-4 | 239, -12.5% (-25.1%, 0.0%) | 195, -2.9% (-16.9%, +10.6%) |
| 4-6 | 253, -2.3% (-14.8%, +10.0%) | 133, -1.5% (-17.9%, +15.7%) |
| 6-9 | 256, +16.0% (+2.6%, +29.2%) | 65, +5.0% (-20.5%, +30.8%) |
| 9+ | 301, +0.3% (-12.1%, +12.8%) | 8, -6.9% |

The 6-9 bucket is positive in both seasons, but the 9+ bucket is not, and with 4 buckets x 2 seasons x 4 models looked at, one interval that excludes 0 is about what chance alone would give. **This is not evidence for raising the threshold.** It is a question for 2026-27 paper trading, judged on the bets placed then.

### Exchanges (what-if only)

No exchange prices exist historically. As a labelled what-if, buy at the consensus no-vig price plus 1 cent and pay the taker fee (→ the fee for buying at the price on offer): Kalshi 0.07 x p x (1 - p) per contract, Polymarket US 0.0695 x p x (1 - p) since 2026-10-01, where p is the contract price.

| V0, 2024-25 | Bets | Flat ROI |
|---|---:|---:|
| No fee | 1,049 | +0.1% |
| Kalshi fee | 1,025 | -3.1% |
| Polymarket fee | 1,025 | -3.1% |

The fee is about 1.75 cents on a 50-cent contract: a 3-point ROI hurdle on a coin-flip game. `betting/engine.py` now charges it when it picks the best price, sizes the Kelly stake, computes EV and settles paper bets.

## 3. Bet timing: morning or close? (2024-25, 889 games, V0)

Questions fixed in advance; all three come out "not shown".

| Question | Result | Verdict |
|---|---|---|
| H1: does the model's morning edge predict where the line moves? | slope 0.039, t = 1.90 (needed t >= 2) | Not supported. A hint, not evidence |
| H2: do morning bets beat the closing line? | closing EV -1.65% (SE 0.18%); the line moved toward our side by only 0.17 points (SE 0.09) | No: morning prices are worse than the no-vig close |
| H3: which slot pays better? | morning minus close: flat ROI -1.1% (-3.4%, +1.1%); closing EV +0.16% (-0.21%, +0.52%) | Neither |

Morning arm: 361 bets, +5.2% flat (-4.4%, +15.0%). Close arm: 359 bets, +6.4% (-3.5%, +16.3%). Caveat fixed in advance: both arms used the actual starting goalie, which is often not confirmed at 10:00, so the morning arm is slightly flattered.

## Bottom line

The moneyline model is about as good as the closing market and no better. At the prices a bettor can actually get, the backtests do not show a profit that can be told apart from luck, and the closing-line value is negative. Nothing here justifies real stakes. What would change that: a model that beats the no-vig close in log loss by 2 SE on unseen games, or positive CLV measured on 2026-27 paper bets.
