# Moneyline backtest results (current, reproducible)

**Run date:** 2026-10-04, updated 2026-10-07 after an independent review (clean-game log loss added, see "The in-play leak") and 2026-10-08 after a second review (the production model's own market test and backtests added; they replace the experiment's V0 figures in the headline rows) · **Code:** `models/moneyline_v3.py` (pre-registration and full numbers in its docstring) · **Raw output:** `models/artifacts/moneyline_v3_results.json` · **Re-run:** `.venv\Scripts\python -m models.moneyline_v3` (read-only: it writes only the JSON file); the production model's figures: `.venv\Scripts\python -m models.moneyline_v3 --production` (read-only, writes nothing)

This page replaces the betting half of [phase3_results.md](phase3_results.md). Its headline, "+26.6% flat ROI on bets with a 6-9% edge", came from 46 bets in one season and does not reproduce: for the production model the same bucket is -11.4% on 75 bets in 2025-26, with an interval from -33.3% to +11.5%.

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
- **In play** → a price taken after the game has started, so it already reflects the score.
- **Leak** → information the model could not have had before the game (here, the game's own result) getting into its inputs by mistake. It makes a model look better than it is.

## The data

| Season | Prices | Games |
|---|---|---|
| 2024-25 | The Odds API historical close (median 14 minutes before puck drop) at 10 books, including Pinnacle; plus a 10:00 Central morning snapshot for 889 games | 1,398 |
| 2025-26 | DraftKings close from ESPN | 1,014 |
| 2020-21 .. 2023-24 | Unibet three-way regulation lines from ESPN: used as a model input only, never as a payable price | — |

There are **no Kalshi or Polymarket prices** in any of this: the history covers 10 sportsbooks. So there is no real exchange backtest; the exchange rows below are a labelled what-if.

## The in-play leak (found 2026-10-07)

The model's main input is the market's own price for each game. For the older seasons that price is the "closing" line ESPN stored. For **106 games in late 2023-24, ESPN's Unibet line was not taken before the game but during it**: prices like -10000 on a team already winning late, or over/under lines such as 2.0 or 13.0 that only make sense once goals have been scored. A model fed those prices is partly being told who won. On those 106 games the model's log loss was 0.45 (0.14 on the 34 priced at 1,000 or more), against about 0.66 everywhere else.

What that did: the experiment's models used those prices twice. They were **scored** on the 106 games with the leaked price, and every season fitted after 2023-24 (the 2024-25 and 2025-26 folds) was **trained** on them. So the experiment's headline log loss (0.66163 for V0, the current model) was about 0.003 too good.

There are two ways to take the leak out, and they are not the same:

- **Clean games** (the "clean games" column below): score every model again without the 106 games. This removes them from the **scoring only**; the models were still trained on them. V0: 0.66484.
- **Re-train without them** (what production now does): treat those games as having no market, for training and for scoring. This is the fully honest figure. V0 re-trained this way: **0.66479** on all 6,993 games, **0.66520** on the 6,887 clean ones.

No decision changes, and no variant comes near 2 SE either way. The re-trained model looks slightly worse against the market than the experiment's V0 did (see section 1). Its 2024-25 predictions do not change at all: V0 has no market input in 2024-25, so that season is scored by the market-blind fallback model (a model that does not use the price), which the masking does not touch. Its 2025-26 predictions do change, because that fold's training seasons include 2023-24.

What was fixed:

- A shared rule now finds those rows (`features/market_prices.py`, `inplay_mask`): a moneyline of 1,000 or more on either side; both sides priced as underdogs at once; an over/under line under 5 or at 8 and above; or any Unibet line from 2024-04-08 to the end of 2023-24, the stretch where 42 of 97 lines break the other rules. It flags those 106 Unibet games and no DraftKings line.
- The production model (`models/lgbm.py`) and the stored-feature builder (`features/build_vectors.py`) treat those games as having **no market** (`market_available = 0`), for training and for scoring. The production model's own walk-forward log loss goes from 0.6616 to **0.6648** (2023-24: 0.6453 to 0.6593; 2025-26: 0.6887 to 0.6904). It still passes its gate (beat the 0.6829 baseline).
- The simulation fallback in `betting/recommend.py` never prices a game off such a line.

## 1. Does the model beat the market? No.

Walk-forward (→ train on earlier seasons, score the next) over 6,993 games; market comparison on the 2,412 priced games. "All games" is the pre-registered figure and includes the leak. "Clean games" leaves the 106 in-play games out of the **scoring only**: V0 to V3 were still trained on them. The first row is the production model re-trained without them, the fully honest figure; the other rows are the pre-registered experiment.

| Model | What it adds | Log loss, all games | Log loss, clean games | vs V0, clean (SE) | vs market on priced games (SE) |
|---|---|---:|---:|---:|---:|
| **Production model** | V0 re-trained with the in-play lines treated as no market (added 2026-10-08, not pre-registered) | **0.66479** | **0.66520** | +0.0004 (0.0003) | **+0.0047 (0.0025)** |
| V0 (experiment) | the current model, trained with the in-play lines | 0.66163 | 0.66484 | — | +0.0037 (0.0025) |
| V1 | real 2024-25 consensus market as its starting point | 0.66223 | 0.66545 | +0.0006 (0.0010) | +0.0055 (0.0020) |
| V2 | V1 + power-play form | 0.66228 | 0.66547 | +0.0006 (0.0011) | +0.0070 (0.0019) |
| V3 | V2 + goalie-role inputs | 0.66207 | 0.66525 | +0.0004 (0.0012) | +0.0074 (0.0019) |

Positive = worse. The market's own log loss on those games is 0.6666 (Pinnacle alone 0.6576 in 2024-25).

- **Rule fixed in advance:** a variant replaces V0 only if it beats V0 by 2 SE and is not worse than the market at 95%. None does, under any of three random seeds. **No variant replaces V0**; the only change to the production model is the in-play masking above.
- **The model is at best equal to the market, never better.** The production model is 0.0047 worse than the no-vig close (SE 0.0025): just under 2 SE, so "equal at best" is the generous reading. In 2024-25 it is 0.0064 worse (SE 0.0040), and in 2025-26 0.0024 worse than DraftKings (SE 0.0023). The experiment's V0, trained with the leaked prices, had tied DraftKings in 2025-26 (+0.00004); that tie does not survive the re-train. In 2024-25 V1 ties the consensus close and Pinnacle (-0.0010, SE 0.0017); in 2025-26 V1 is clearly worse (+0.0144, SE 0.0042). Why V1 drifts in 2025-26 is not known (a guess: corrections learned on Unibet seasons do not carry over).
- Two after-the-fact checks were never eligible for adoption. Mapping the Unibet lines onto a two-way scale (V1m) changed nothing. Removing the 106 in-play lines from V1's input (V1mc) looks worse on all games (+0.0022 vs V1, SE 0.0008), but only because the other models are still reading the result off those 106 prices. On the clean games it is slightly **better** than V1 (-0.0007, SE 0.0005) and the closest of all to the market (+0.0042 vs the close, SE 0.0019; V1 +0.0055). None of these gaps reaches 2 SE. Its backtests repeat V1's pattern: +9.5% on 356 bets in 2024-25 (interval +1.6% to +17.4%), then -8.6% on 741 bets in 2025-26 (-15.3% to -2.1%), with negative closing EV in both. A good season followed by a clearly bad one is not an edge.

## 2. Priced backtests

Every bet the engine would place at the locked 2.5-point threshold, settled at the real price. "Production model" is V0 re-trained without the in-play lines. Its 2024-25 rows are identical to the experiment's V0 (that season is scored by the market-blind fallback model, see above); its 2025-26 row is not, so both are shown. V1 rows are the experiment's.

| Model, season, prices | Bets | Flat ROI (95% interval) | Quarter-Kelly ROI | Mean closing EV |
|---|---:|---:|---:|---:|
| Production model, 2024-25, best of 10 books | 1,049 | +0.6% (-6.0%, +6.9%) | +1.0% | -1.6% |
| Production model, 2024-25, best of 6 licensed US books | 1,049 | -0.4% (-6.9%, +5.8%) | +0.4% | -2.6% |
| Production model, 2024-25, one US book alone | 958-1,033 | -0.5% to -2.0% | -0.1% to -1.9% | -3.8% to -4.9% |
| Production model, 2024-25, Pinnacle alone | 1,046 | -0.4% (-6.7%, +6.1%) | +0.5% | -2.4% |
| **Production model, 2025-26, DraftKings** | **487** | **-3.5% (-12.0%, +5.1%)** | -6.0% | -4.2% |
| V0 (experiment, trained with the in-play lines), 2025-26, DraftKings | 401 | -1.3% (-10.7%, +8.5%) | -0.1% | -4.2% |
| V1 (experiment), 2024-25, best of 10 books | 585 | +4.4% (-3.6%, +12.1%) | +5.6% | -1.8% |
| V1 (experiment), 2025-26, DraftKings | 727 | **-10.0% (-16.6%, -3.4%)** | -9.1% | -4.2% |

What this says:

- **No result shows a real edge.** Every interval for the production model (and for the experiment's V0) contains 0. V1's +4.4% in 2024-25 is followed by a loss that is clearly not luck in 2025-26.
- **Closing EV is negative everywhere.** Betting near the close at these prices pays the books' margin: about 1.6 points with the best of 10 books, about 4 points at one US book. Line shopping (→ taking the best price across several books) is worth roughly 2-3 points of ROI on its own.
- Maximum drawdown (→ the largest drop from a bankroll peak) for the production model at quarter-Kelly: 29% in 2024-25, 35% in 2025-26.

### By claimed edge (production model, flat stake)

| Edge (points) | 2024-25 best of 10: bets, ROI (interval) | 2025-26 DraftKings: bets, ROI (interval) |
|---|---|---|
| 2.5-4 | 239, -12.5% (-25.1%, 0.0%) | 225, +2.8% (-10.4%, +15.7%) |
| 4-6 | 253, -2.3% (-14.8%, +10.0%) | 168, -11.5% (-24.9%, +1.8%) |
| 6-9 | 256, +16.0% (+2.6%, +29.2%) | 75, -11.4% (-33.3%, +11.5%) |
| 9+ | 301, +0.3% (-12.1%, +12.8%) | 19, +25.1% (-19.1%, +67.1%) |

The 6-9 bucket is +16.0% in 2024-25 but -11.4% in 2025-26, and no bucket is positive in both seasons. (The experiment's leaky-trained V0 had the 2025-26 6-9 bucket at +5.0% on 65 bets; that does not survive the re-train either.) With 4 buckets x 2 seasons x 4 models looked at, one interval that excludes 0 is about what chance alone would give. **This is not evidence for raising the threshold.** It is a question for 2026-27 paper trading, judged on the bets placed then.

### Exchanges (what-if only)

No exchange prices exist historically. As a labelled what-if, buy at the consensus no-vig price plus 1 cent and pay the taker fee (→ the fee for buying at the price on offer): Kalshi 0.07 x p x (1 - p) per contract, Polymarket US 0.0695 x p x (1 - p) since 2026-10-01, where p is the contract price.

| Production model, 2024-25 | Bets | Flat ROI |
|---|---:|---:|
| No fee | 1,049 | +0.1% |
| Kalshi fee | 1,025 | -3.1% |
| Polymarket fee | 1,025 | -3.1% |

The fee is about 1.75 cents on a 50-cent contract: a 3-point ROI hurdle on a coin-flip game. `betting/engine.py` now charges it when it picks the best price, sizes the Kelly stake, computes EV and settles paper bets.

## 3. Bet timing: morning or close? (2024-25, 889 games)

Questions fixed in advance; all three come out "not shown" for the pre-registered model, the experiment's V0 (trained with the in-play lines). The table and arm figures below are that run.

| Question | Result | Verdict |
|---|---|---|
| H1: does the model's morning edge predict where the line moves? | slope 0.039, t = 1.90 (needed t >= 2) | Not supported. A hint, not evidence |
| H2: do morning bets beat the closing line? | closing EV -1.65% (SE 0.18%); the line moved toward our side by only 0.17 points (SE 0.09) | No: morning prices are worse than the no-vig close |
| H3: which slot pays better? | morning minus close: flat ROI -1.1% (-3.4%, +1.1%); closing EV +0.16% (-0.21%, +0.52%) | Neither |

Morning arm: 361 bets, +5.2% flat (-4.4%, +15.0%). Close arm: 359 bets, +6.4% (-3.5%, +16.3%).

**The production model, re-run after the fact (2026-10-08).** The timing study uses the 2024-25 fold's market-starting model, which was trained on 2023-24, so the re-train changes it. H1 now comes out at slope 0.040, t = 2.54, over the t >= 2 bar. That is a re-run after the results were known, not the pre-registered test, so it counts as a hint to check on 2026-27 paper bets, not as evidence. H2 is unchanged (closing EV -1.65%, SE 0.16%: morning prices are still worse than the no-vig close) and H3 still finds neither slot better (flat ROI +0.3%, interval -1.5% to +2.2%). Morning arm 440 bets, +5.7% (-3.1%, +14.4%); close arm 445 bets, +5.4% (-3.4%, +14.0%).

How the close arm was built, which the first version of this page left out: the plan said "as the 2024-25 backtest, on the same games", but V0's 2024-25 backtest is scored by the market-blind fallback model (V0 has no 2024-25 market input). The close arm instead gives the same market-starting model the closing price as its input, just as the morning arm gives it the morning price. Both arms are then the same model with a different starting price, so H3 measures the timing and not two different models. That is why the close arm (359 bets, +6.4%) does not look like the V0 2024-25 backtest above (1,049 bets, +0.6%). Caveat fixed in advance: both arms used the actual starting goalie, which is often not confirmed at 10:00, so the morning arm is slightly flattered.

## Bottom line

The moneyline model is about as good as the closing market and no better. At the prices a bettor can actually get, the backtests do not show a profit that can be told apart from luck, and the closing-line value is negative. Nothing here justifies real stakes. What would change that: a model that beats the no-vig close in log loss by 2 SE on unseen games, or positive CLV measured on 2026-27 paper bets.
