# Why the totals (over/under) model failed its gate

> Research snapshot, 2026-09-28. Produced by read-only investigation (live requests to free endpoints, the
> official docs, and a disposable copy of the database). File and line references point at the code as it was
> that day, before later fixes; see README.md and PROJECT_CONTEXT.md for the current state.

Re-runs the totals walk-forward test, explains what the Poisson model assumes and where hockey breaks it, and tests candidate fixes (margin reweighting, drift removal, negative binomial, the market line).

## Summary
I reran the totals (over/under) test on the throwaway clone and got the recorded failure: an average error score of 2.1867 for the model against 2.1815 for the baseline, where lower is better. At the DraftKings line the model's log loss was 0.7051, which is worse than always calling it 50/50 (0.6931).

"The Poisson failed" means one specific thing. The model adds team stats on top of how much the whole league is scoring lately. Those team stats did not predict better than the league scoring rate alone, and on average they did slightly worse. Three causes, all measured:
- **The team stats barely track the actual score.** The correlation between predicted and actual totals is 0.02 to 0.06 (0 means no relationship, 1 means perfect).
- **The model drifts too low in each new season.** Its predictions fall about 3.7% below the league rate, roughly 0.2 goals a game. That made it lean under at the DraftKings line: it gave the over a 44% average chance while overs actually won 50.8% of the time.
- **It treats the two teams' scores as unrelated, which they aren't.** Regulation ties happen 22.3% of the time; the model expects 16.7%. One-goal regulation wins happen 17.7% of the time; the model expects 30.4%.

One of the planned fixes will not help. The historical test already knew who actually started in goal, and the goalie stats still made predictions slightly worse. Confirmed starters from Daily Faceoff will keep live results in line with the test, but they will not make it pass.

Of the fixes I tested, correcting how the two scores relate gave the only measurable gain, and it was small. Allowing for games with more scatter than normal (the "negative binomial" idea) made things worse. A test using the DraftKings line alone was inconclusive. Nothing beat a coin flip at the DraftKings line.
## Details
All work was read-only in C:\Working\nhl-betting-system. Everything ran on the throwaway clone (port 55432) with register=False, and nothing was written to models/artifacts.

## 1. What the check was
- **The main check (verified, models/totals.py:44-48 and line 567).** Across all past seasons, each scored by a model trained only on earlier seasons, the model's average error score must beat a baseline. The error score is negative log-likelihood (NLL): how surprised the model was by the actual final total. The final total includes the extra OT or shootout goal. The baseline is the league's recent scoring rate alone, run through the same calculation. In plain terms: the team stats have to add something beyond "how much is the league scoring lately".
- **A second number reported alongside it (verified, lines 71-76 and 569-575).** Over/under log loss at the DraftKings line, against 0.6931, which is the score for always saying 50/50. Only the 2025-26 season has real lines.
- **The data behind that second number is thin (verified in the clone).** The DraftKings lines are 5.5 in 415 games, 6.5 in 592, and 7.5 in 4. No historical over/under prices exist (the juice, meaning the odds on each side). Before 2025-26 the stored line is a placeholder 5.5.

## 2. Results (verified: rerun on the clone, matches the recorded numbers)
Overall: model 2.1867, baseline 2.1815, gate not passed. Over/under log loss 0.7051 over 1,011 games. The doc says 0.7053 over 1,015; the small gap is probably a few rows that differ between the clone and the original run.

| Season scored | Model NLL | Baseline NLL | Model's avg predicted total | Actual avg total |
|---|---|---|---|---|
| 2021-22 | 2.2118 | 2.2039 | 5.95 | 6.29 |
| 2022-23 | 2.1643 | 2.1647 | 6.41 | 6.35 |
| 2023-24 | 2.1778 | 2.1773 | 6.15 | 6.19 |
| 2024-25 | 2.2070 | 2.1990 | 5.86 | 6.09 |
| 2025-26 | 2.1727 | 2.1626 | 5.87 | 6.23 |

The model beat the baseline in only 1 of 5 seasons (2022-23), by 0.0004.

**Reading the error score:** an NLL of 2.18 means the model gave the exact final total about an 11% chance on average. A gap of 0.005 means it gave the actual result about 0.5% less probability than the baseline did. That's small, but it is consistent.

## 3. What Poisson regression is
A Poisson model says goals arrive at random at a steady rate. You only predict the rate (the expected number of goals), and the full spread of possible scores follows from it: the chances of 0, 1, 2, 3 goals and so on.

This model makes three assumptions:
- **The spread follows from the average.** Scores should scatter by an amount set by the average. Too much scatter is called "overdispersion".
- **The two teams' scores are unrelated.** Home and away goals are independent of each other.
- **Team stats set the rate.** A LightGBM machine-learning model nudges the league rate up or down for each game using team stats (models/totals.py:399-443).

How these hold up for hockey (verified, checks.py):
- **Scatter per team: close to Poisson.** For one team's goals, the variance-to-mean ratio is 1.026 (home) and 1.019 (away). Poisson predicts exactly 1, so that's close.
- **Unrelated scores: false.** The home and away goal counts are slightly negatively correlated, at -0.098. As a result, the combined score scatters less than Poisson assumes (ratio 0.91 to 0.94 in every season).
- **Ties:** 22.3% of games are tied after regulation; Poisson expects 16.7%. The gap is biggest at 2-2 (7.0% vs 5.1%) and 3-3 (7.2% vs 5.0%).
- **Winning margins in regulation (verified directly in raw.games):**

| Margin | Actual | Poisson expects |
|---|---|---|
| 1 goal | 17.7% | 30.4% |
| 2 goals | 19.5% | 23.2% |
| 3 goals | 23.5% | 14.9% |

- **Likely cause (my inference, not verified):** pulling the goalie late. A trailing team that pulls its goalie either ties the game or gives up an empty-net goal. The data can't confirm this: raw.shots has no empty-net marker, and goalie_id is never blank on goals.

## 4. Why it lost
1. **Team stats carry almost no signal for totals (verified).** The correlation of predicted vs actual totals is 0.015 to 0.058 by season. Even the DraftKings line correlates only 0.084 with the actual total (a rank correlation), so the game-to-game total is mostly luck, and even the bookmaker's line explains under 1% of it.
2. **The model drifts in each new season (verified).**
   - On the seasons it trained on, the model's average adjustment is about zero (+0.0025 and -0.0018). On the new season it drops to -0.037 and -0.039. That is about 3.7% fewer goals, or roughly 0.2 a game.
   - Regulation goals, predicted vs actual: 2024-25 model 5.70, league rate 5.88, actual 5.88. 2025-26 model 5.70, league rate 5.92, actual 5.99.
   - This drift explains the worse-than-coin-flip over/under result. The model's average chance for the over was 0.441; overs actually hit 0.508.
   - The model's biggest single input is the Elo rating gap between the two teams, at 19% to 33% of the model's total feature-importance score. The Elo gap is a "who is better" difference, not a pace measure. My inference: it mostly decides which team scores, not how many goals there are.
3. **Scoring levels shift between seasons (verified).** In 2021-22 scoring rose all year and the league-rate tracker lagged behind. In April its regulation rate was 6.09 against an actual 6.40. It is still the hardest thing to beat.
4. **Diff features and the playoff-heavy tuning window (not re-tested).** Both are fixes already made in the current code, per its own notes. Diff features means using only home-minus-away differences (lines 13-17). The early-stopping check now uses only regular-season games from the most recent stretch of training data (lines 409-412).
5. **Confirmed starters are not the missing piece (verified, and this contradicts the code's own STATUS note).**
   - features/build_vectors.py:23-28 sets each historical game's starter to the goalie who actually started (raw.goalie_games.is_starter). So the backtest already knew every starter perfectly.
   - Even so, the recorded tests (lines 53-58) found goalie stats slightly harmful: dropping them improved the score from 2.1867 to 2.1834.
   - Daily Faceoff will only stop live results from falling behind the backtest.
6. **The market isn't used (verified, but inconclusive).**
   - Games with a 5.5 line averaged 6.05 goals (overs won 53.0%). Games with a 6.5 line averaged 6.44 goals (overs won 49.5%).
   - The line tracks the actual total better than the model does: rank correlation 0.084 vs 0.023. The model's predictions do track the line (0.357).
   - I let the line nudge the league rate, fitted on the first half of 2025-26 (502 games) and tested on the second half (509 games). The error score changed by +0.0004, with a 95% range of -0.0073 to +0.0088, which is noise.
   - When I also added the model's adjustment, it was given a negative weight (-0.115). In other words, it was working against the right answer.
   - With only two line values and no over/under prices, most of the market's information is missing.

## 5. Experiments
All runs used the same season-by-season test. Numbers are overall NLL, with the change against the baseline's 2.1815. Lower is better.

| Variant | Overall NLL | vs baseline | O/U log loss, 2025-26 (coin flip 0.6931) |
|---|---|---|---|
| Baseline (league rate) | 2.1815 | — | 0.6971 |
| Current model | 2.1867 | +0.0052 | 0.7051 |
| (a) Model with drift removed | 2.1845 | +0.0030 | 0.6980 |
| (b) Baseline + tie-only fix | 2.1808 | -0.0007 | 0.6962 |
| (c) Baseline + margin fix | 2.1787 | -0.0028 | 0.6950 |
| (d) Model + margin fix | 2.1817 | +0.0002 | 0.6993 |
| (e) Drift removed + margin fix | 2.1800 | -0.0015 | 0.6957 |
| (f) Negative binomial | 2.1834 | +0.0019 | 0.6972 |

What each variant is:
- **(a)** Subtracts the model's running average adjustment within the season. This only uses games already played, so it would be known before each game.
- **(b)** Ties only, the Dixon-Coles-style fix (Dixon-Coles is a classic soccer model that corrects how often certain low scores happen).
- **(c)** Reweights the chance of each score by winning margin. The weights were fitted on the training seasons each time and came out steady across folds: ties x1.11 to 1.16, 1-goal margins x0.50 to 0.55, 2-goal x0.69 to 0.73, 3-goal x1.21 to 1.36, all relative to 4 or more.
- **(f)** Allows for more scatter than normal. It is worse because per-team scatter is already close to Poisson and combined totals scatter less, not more.

What the table shows:
- **The margin fix is the only real gain (c), and it is small.** It improved 4 of 5 seasons; 2024-25 got worse. On a per-season level it is consistent, but it is not proven.
- **(e) scores below the baseline, but that does not pass the gate honestly.** It uses a structural fix the baseline didn't get. The fair comparison is (e) 2.1800 against (c) 2.1787, and on that comparison the team stats still add nothing.


## Recommendations
Fixes, ranked. The first four need no new data.

1. **Correct how the two scores relate (the margin fix).**
   - *What:* reweight each possible score by winning margin: ties, 1, 2, 3, and 4 or more goals. That is four settings, fitted on the training seasons each time.
   - *Where:* models/totals.py, in total_pmf. The bet checker and the arbitrage/middle alerts use the same score distributions, so they benefit too.
   - *Data:* what the repo already has.
   - *Effort:* about half a day, plus tests.
   - *Expected gain:* -0.0028 NLL, and over/under log loss 0.6971 → 0.6950.
   - *Caveat:* this makes the numbers more accurate but does not give the team stats an edge.

2. **Remove the new-season drift.**
   - *What:* subtract the model's running average adjustment. Alternatively, drop or rescale the inputs that shift from season to season, such as save %, shooting %, and possibly the Elo gap, then re-test.
   - *Data:* none new.
   - *Effort:* a few hours.
   - *Expected gain:* +0.0052 → +0.0030 against the baseline. It narrows the gap but doesn't close it.

3. **Fix the gate so it can't be gamed.**
   - *What:* give the baseline every structural fix the model gets. The current code would "pass" with fixes 1 and 2 combined (2.1800 < 2.1815) even though the team stats still add nothing (2.1800 vs 2.1787).
   - *Also:* set a betting check against the market's own probability with the bookmaker's margin removed, plus closing-line value (whether bets beat the final pre-game line). Coin flip and the league rate are not enough for betting decisions.
   - *Effort:* small.

4. **Correct the plan's expectations about starters.**
   - *What:* the historical test already used the actual starters (build_vectors.py:23-28), so Daily Faceoff will not make the totals check pass. Update the STATUS note in models/totals.py and PROJECT_CONTEXT §9.
   - *Effort:* documentation only.

5. **Use the market's over/under line and prices as the starting point (the moneyline trick).**
   - *Why:* this is the biggest likely gain. The line alone already tracks the actual total better than the model does (0.084 vs 0.023). With the over/under prices, you can work out the market's real expected total.
   - *Data:* over and under prices from the live Odds API snapshots (raw.odds_snapshots has line, over_price and under_price columns, but the clone has 0 rows). Alternatively, a paid historical Odds API backfill.
   - *Effort:* about 1 to 2 days of code, mostly reusing the moneyline pattern. Proving it needs roughly a season of 2026-27 snapshots, unless you buy the history.
   - *Evidence so far:* inconclusive, based on the line alone over 509 games.

6. **Improve the league-rate tracker.** In 2021-22 it lagged rising scoring by about 0.3 goals.
   - *What:* tune the 120-day and 365-day windows and how quickly it moves off its long-run average, validated season by season.
   - *Effort:* a few hours.
   - *Risk:* overfitting to only five seasons.

7. **New inputs that could actually affect goal totals.**
   - *What:* pace from both teams' shots and expected goals (xG), penalty volume from both teams, how aggressively each coach pulls the goalie, and injuries to top scorers (ESPN's injury feed).
   - *Effort:* medium to large, with an uncertain payoff given how weak the signal is.

**Don't do:** switch to a negative binomial model. It measured worse (+0.0019), because hockey totals scatter less than Poisson assumes, not more.

**Realistic path:** fixes 1 to 4 now, then paper trading with fix 5 once the 2026-27 over/under snapshots come in. Keep totals betting off until a gate based on the market's own over/under probability passes.
