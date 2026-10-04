# Player props: feasibility, data, and methods inventory

> Research snapshot, 2026-09-28. Produced by read-only investigation (live requests to free endpoints, the
> official docs, and a disposable copy of the database). File and line references point at the code as it was
> that day, before later fixes; see README.md and PROJECT_CONTEXT.md for the current state.

Inventories the modelling methods in the repo (Poisson, Dixon-Coles, xG, Elo, Kelly, CLV), what player data exists, which free sources fill the gaps, what props odds cost, and a phased plan.

## Summary
Yes, a props model can be built. Most of the player data is already in the database, and the gaps can be filled from free sources. What can't be done yet is proving it makes money, because the repo has no props prices at all. I found one free source that may cover last season.

How the five methods you asked about stand:
- **Kelly sizing:** built and in use.
- **CLV tracking:** built, but it has no data yet. The throwaway copy of the database shows 0 picks and 0 paper bets.
- **Poisson:** built, but only as the over/under totals model, and that model failed its quality check.
- **Elo:** plain team Elo. The goalie adjustment exists, but as separate goalie features rather than inside Elo.
- **xG:** taken from MoneyPuck. No model is built here from shot coordinates, though the coordinates are stored.
- **Dixon-Coles:** not in the repo.

Two new findings:
1. ESPN's free API still serves DraftKings player-prop prices for 2025-26 games. For one January 2026 game it returned 375 lines, covering shots on goal, points, assists, power-play points, blocked shots and saves, with opening and later prices. This could give a free first backtest, but it has caveats (below).
2. League shot volume is falling: skater shots on goal per 60 minutes went from 6.39 in 2021-22 to 5.61 in 2025-26, and goalie saves per start went from 27.6 to 24.2. That is the same season-to-season drift that sank the totals model, so props models have to adjust for it.

Correction to the earlier message: the database holds shots and player games only from 2020-21 on, not since 2007.
## Details
Tags: [V] = checked this session, with file:line or a live response. [I] = my inference or general industry knowledge, not verified here. All database figures come from the throwaway copy (port 55432). The repo was not modified.

== 1. INVENTORY OF THE METHODS YOU NAMED ==

**Poisson: YES, totals only. [V]**
- A Poisson distribution is the standard way to give odds for 0, 1, 2, 3... of something happening, such as goals or shots.
- Where it lives: `models/totals.py:105` (the model is set to "poisson") and `models/totals.py:349-356` (`poisson_pmf`). The on/off switch is `GATE_PASSED = False` at `models/totals.py:94`. The model is registered as poisson_totals v1 and marked inactive.
- Why it failed (`models/totals.py:48-76`):
  - It had to beat a simple baseline that knows only how much the whole league has been scoring lately. It lost: error score 2.1867 against the baseline's 2.1815 (lower is better).
  - At the DraftKings over/under line, its log loss was 0.7053. Log loss is a standard error score for probability predictions, again lower is better, and a coin flip scores 0.693. So it did worse than guessing 50/50.
- Plain-English cause: the Poisson shape is fine. The inputs (team form and goalie form) carry almost no information about which games go high or low; the rank correlation between predicted and actual totals was only 0.03-0.04. What dominates is season-wide swings in scoring.
  - The goalie-form features were net noise: removing them improved the score to 2.1834.
  - It had no confirmed starting goalies and no historical totals prices to learn from.
- The documented fix (same place): confirmed starters, live over/under prices, and starting the model from the market's total once 2026-27 lines build up. That last one is the trick that made the moneyline model work.

**Dixon-Coles: NO. [V]**
- It is a small correction to Poisson for low-scoring results like 0-0 and 1-1, from soccer modelling.
- It appears only as a possible "later refinement" at `models/totals.py:30-33`.
- [I] It would not fix the totals failure, because the problem is the inputs, not the distribution's shape.

**xG: from MoneyPuck, not built here. [V]**
- xG (expected goals) is the chance that a given shot becomes a goal.
- Loaded at `ingestion/moneypuck.py:97`: `out["xg_moneypuck"] = df["xGoal"]`.
- Used for goalie goals-saved-above-expected (`features/goalie_features.py:20-25` and `:96`) and for team xG share and power-play xG (`features/team_features.py:57-58`, `:173`, `:183-184`).
- The planned in-house model ("Layer A — LightGBM per-shot goal probability", `PROJECT_CONTEXT.md:109`) has no code.
- The ingredients are stored: x/y coordinates, distance, angle, shot type, rebound, rush and strength (`db/schema.sql:87-98`), across 683,721 shots.
- The MoneyPuck file is downloaded from a mirror site (`ingestion/moneypuck.py:36`; flagged as an open issue at `README.md:61`).

**Elo: plain team Elo, not goalie-adjusted. [V]**
- Elo is a running team-strength rating that rises with wins and falls with losses.
- Settings (`features/elo.py:6-18`, `:35-38`): K=20, home ice +50 points, a 1/3 pull back toward 1500 each season, and wins only (no margin of victory). The update is at `features/elo.py:89-95`.
- It feeds the moneyline model as `elo_diff` (`features/build_vectors.py:70`, `:217`) and the totals model as `elo_edge` (`models/totals.py:237`).
- Goalie quality enters separately, as the starters' shrunk save % and goals-saved-above-expected differences (`features/build_vectors.py:57-68`). "Shrunk" means pulled toward league average when there is little data.

**Kelly: YES. [V]**
- Kelly is a formula that sizes each bet from the edge and the odds.
- Code: full Kelly at `betting/engine.py:40-45`, quarter-Kelly at `:23`, a 2% per-bet cap at `:24`, a 10% per-day cap at `:25`, applied at `:74-77`. The parlay checker uses it at `betting/checker.py:240-241`.
- Backtest (`README.md:48`): 363 bets, +1.4% ROI with quarter-Kelly stakes and −6.0% with flat stakes, so no demonstrated edge.
- Gap: `PROJECT_CONTEXT.md:129` sets a rule of at most 3 correlated bets per game, but no code enforces a per-game cap. The only cap enforced is the daily one (`betting/recommend.py:595`). This matters once props allow several bets per game.

**CLV: YES, built, no data yet. [V]**
- CLV (closing-line value) asks whether the price you took beat the market's final price before puck drop.
- Code: design at `betting/settle.py:15-27`, the calculation at `:142-146`, the report at `:221-260`, and the closing snapshot run by `pipeline.py close` (`pipeline.py:167`). It covers moneyline only.
- The database copy has 0 picks, 0 paper bets and 0 odds snapshots.

== 2. CAN WE DO PROPS? ==

**Data already in the database [V]:**
- **Player games (`raw.skater_games`):** 285,941 player-games across 7,945 games and 1,576 skaters, 2020-21 through 2025-26, regular season and playoffs.
  - Filled in: ice time, goals, assists, points, shots on goal, hits, blocks, penalty minutes, plus/minus.
  - Always empty or zero, because the collection code never writes them (`ingestion/nhl_api.py:236-249`): power-play ice time, penalty-kill ice time, power-play goals and assists, faceoffs, and on-ice shot and xG counts.
- **Goalie games (`raw.goalie_games`):** 31,778 rows covering 15,890 starts by 243 goalies, with saves, shots against, goals against, a starter flag and strength splits. The xG columns there are empty; those numbers are computed from the shots table instead.
- **Shots (`raw.shots`, from MoneyPuck):** 683,721 unblocked shot attempts with shooter, goalie, coordinates and xG. Blocked attempts aren't included (`ingestion/moneypuck.py:69`).
  - The boxscore shots-on-goal count matches MoneyPuck's on-goal events in 99.4% of player-games, so MoneyPuck's richer per-shooter data can be used safely.
- **Empty tables:** shifts, rosters, starting goalies and odds snapshots all have 0 rows. There is no props odds data anywhere.

**Free sources that fill the gaps (one request each) [V]:**
- **Power-play ice time.** The NHL stats API's per-game ice-time report (`api.nhle.com/stats/rest/en/skater/timeonice?isGame=true…`) splits each player's ice time into even strength, power play and penalty kill. Example: 258 seconds of power-play time for one player in game 2025020001.
- **Power-play points.** The NHL stats API's per-game power-play report gives power-play goals, assists, points, shots and ice time, which is what the power-play points prop needs.
- **Linemates and units.** The NHL shift charts (`api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId=…`) returned 856 shifts for one game. That gives historical linemates and power-play units.
- **Injuries.** ESPN's injury endpoint (`site.api.espn.com/apis/site/v2/sports/hockey/nhl/injuries`) listed 31 teams and 102 players, each with a status, date and comment. [I] It shows current injuries only.
- **Line combinations.** Daily Faceoff's line-combinations page carries forward lines 1-4, defence pairs 1-3, power-play units 1-2, penalty-kill units, goalies, injured reserve and a per-player injury status. It shows the current lineup only. The repo scrapes only starting goalies from Daily Faceoff.
- **ESPN's free props feed:**
  - MTL@BUF on 2026-01-15: 375 DraftKings lines with opening and later prices, covering points, assists, power-play points, shots on goal (SOG), blocked shots, saves and goalscorer markets.
  - Caveat: the two sides of each over/under are not labelled "over" and "under" in the fields returned. That has to be solved before the data is usable.
  - Caveat: the last update was at 21:31Z, 2.5 hours before the 00:00Z puck drop. That is a pre-game price, not a true closing price.
  - CAR@BUF on 2025-01-15: 238 lines from ESPN BET, with no SOG lines. Every update is timestamped after puck drop, so only the opening prices can be trusted.
  - A 2023-24 game had no props.
  - I sampled only one game per season.

**The Odds API (from its docs) [V]:**
- NHL props market keys: `player_points`, `player_power_play_points`, `player_assists`, `player_blocked_shots`, `player_shots_on_goal`, `player_goals`, `player_total_saves`, `player_goal_scorer_first` / `_last` / `_anytime`, plus `_alternate` versions of the over/under markets.
- Props come only through `/events/{eventId}/odds`, one game at a time. Cost is markets returned × regions. Listing events is free.
- Historical odds are paid plans only; the free tier's "Historical Odds" is struck through on the pricing page.
  - Cost: 10 × markets × regions per game per snapshot.
  - Props history starts 2023-05-03, with snapshots every 5 minutes.
  - Plans: 20K credits for $30/month, 100K for $59, 5M for $119.
- My arithmetic on the credits:
  - **Collecting going forward:** 4 markets = 4 credits per game per snapshot. That is about 800 credits a month for one daily snapshot, or about 1,600 with a pre-game closing snapshot for CLV.
  - **What's already committed:** the free 500 credits already go about 300-480 to moneyline (`PROJECT_CONTEXT.md:186`). The current feed also asks for two regions (`ingestion/odds_api.py:111`), which would double the props cost.
  - **Backfill of past prices:** about 52,000 credits per season for closing prices, or about 157,000 for the three seasons available (2023-24 to 2025-26).

**Modelling approach, per market:**
- The basic recipe for every market:
  1. Project the player's ice time.
  2. Multiply by his rate per 60 minutes, pulled toward his position's average so a hot streak doesn't fool it.
  3. Adjust for the opponent and for the league's current trend.
  4. Turn the expected count into a probability of going over the line.
- **Shots on goal (the best first target):** use even-strength and power-play ice time separately. Shot attempts per 60 × the share that hit the net gives a more stable rate than shots on goal alone. Adjust for how many shots the opponent allows.
  - A plain Poisson is adequate: within a player's season, the variance is only 1.07 times the average [V]. Poisson assumes the spread equals the average, and a ratio near 1 means it fits.
  - Negative binomial (Poisson with extra spread allowed) barely helped: average log-likelihood −1.5526 vs −1.5487, where closer to zero is better [V].
- **Points, goals and assists:**
  - Start from the team's expected goals (reuse each side's goal expectation from the totals model), then apply the player's share of team scoring while he is on the ice and his share of ice time.
  - Goals: use his individual xG per 60, pulled strongly toward average.
  - Poisson fits (ratio about 0.97) [V]. Anytime goalscorer probability = 1 − e^(−expected goals).
  - Assists are the noisiest and depend on linemates and power-play unit.
- **Power-play points:** power-play ice time × power-play points per 60 × the opponent's penalty rate and penalty-kill quality. This needs the power-play data from the stats API first.
- **Blocked shots:** mostly defencemen. Ice time × blocks per 60 × the opponent's shot volume (ratio 1.05) [V]. [I] Blocks are counted by each arena's scorekeepers, and some rinks count more than others, so adjust by venue.
- **Goalie saves:** expected shots against (opponent offence, own team's defence, league trend) × save %, using the repo's shrunk goalie ratings. Also model the risk of being pulled early and whether he starts at all.
  - The spread is wider than Poisson allows (ratio 1.55) [V], so use negative binomial or simulation.
  - It needs confirmed starters, and Daily Faceoff is already wired in.

**Quick test on the database copy [V]:**
- The scratch script (`props_feas.py`) used a shrunk per-60 rate × average ice time over the last 10 games, with Poisson, on 46,154 player-games from the 2025-26 regular season.
- Every line ranked players in the right order: the predicted buckets rose steadily with actual results. All lines scored better on Brier than just using the base rate (Brier is an error score for yes/no probabilities, lower is better).
- Points over 0.5: predicted 35.7%, actual 35.0%. Blocks over 1.5: predicted 20.1%, actual 19.4%.
- Shots on goal over 1.5: predicted 47.7%, actual 42.8%. It ran high because league shot rates fell (6.39 → 5.61 per 60, and saves per start 27.6 → 24.2). A trailing league-rate adjustment is required.
- This shows the approach is workable. It does not show we can beat the bookmaker: the books clear this bar too.

== 3. ARE PROPS REALLY EASIER TO BEAT? ==

**Why they may be softer [I]:**
- Books post hundreds of lines per slate and less sharp money shapes them.
- Many lines come from simple projections or shared data feeds.
- They are slow to react to news such as line promotions, power-play unit changes and goalie confirmations.

**Caveats:**
- **Higher margin [V, one game].** The overround is the bookmaker's built-in cut: add up the implied probabilities of both sides and subtract 100%.
  - DraftKings props on 2026-01-15 had a median overround of about 6.2% (points, assists, SOG, power-play points, blocks). DraftKings NHL moneylines across 1,014 games of 2025-26 had a median of 4.3%. So you need a bigger edge just to break even.
  - The repo's props edge threshold of 4% (`PROJECT_CONTEXT.md:127`) may be too low.
  - 191 of the 375 entries had no paired opposite side (goalscorer and milestone markets). [I] Those usually carry even bigger margins.
- **Low limits cap profit [I].** Props limits are often a few hundred dollars, so even a real edge earns little in absolute terms.
- **Account limiting [I].** Books limit winning prop bettors fastest. In the two-bettor structure, bettor A's sportsbook accounts are the scarce resource. On Kalshi, props are "thinner than NBA/NFL" (`docs/texas_execution_options.md:22-25`) [V].
- **Correlation.** Props in the same game move together: they share the same game total and the same power-play chances. Stacking them at quarter-Kelly each is riskier than it looks, and the per-game cap isn't enforced in code.
- **Noise.** Counts of 0-3 per game mean hundreds of bets are needed to separate skill from luck. CLV is the faster signal, but it needs prop snapshots before the game and at the close, which cost credits.


## Recommendations
Phased plan. Effort is in focused working days.

**Phase 0 — check the free ESPN props feed (0.5 day, $0). This is the first step.**
- Write a read-only scratch script that pulls ESPN DraftKings props for about 20 games spread across 2025-26.
- Work out how to tell the over from the under in each pair.
- Measure coverage: the share of games with props, which markets, and how long before puck drop the last update lands.
- Match the lines to actual results in `raw.skater_games` and `raw.goalie_games`.
- If coverage is good, this becomes a free backtest set: about 1,300 games of shots on goal and saves at pre-game prices.

**Phase 1 — fill the data gaps (2-3 days, free sources).**
- From the NHL stats API, collect per-game power-play and penalty-kill ice time plus power-play goals, assists and points. Pull it by date range, not game by game.
- Also store the boxscore's shift count and power-play goals.
- Optionally, collect shift charts to get historical linemates and power-play units.
- Archive the ESPN 2025-26 props into a new props-odds table keyed by player. The existing `raw.odds_snapshots` table has no player column.

**Phase 2 — shots-on-goal and goalie-saves models (3-5 days).**
- Build rates that use only games played before each prediction, pulled toward position average.
- Add an ice-time projection, opponent adjustments, and a trailing league-rate adjustment. That last one is required, since league shots are falling.
- Use Poisson for shots and negative binomial or simulation for saves.
- Validate season by season, the same way the existing models are tested.
- Gate: the model must match or beat the DraftKings pre-game price on log loss, and show positive simulated ROI after the roughly 6% margin. Record a failure honestly, as was done for totals.

**Phase 3 — points, goals, assists, power-play points and blocks (2-3 days).**
- Reuse the same machinery, plus the totals model's per-team goal expectations.

**Phase 4 — live capture and CLV (2-3 days plus about $30/month).**
- The free 500 Odds API credits are already used by moneyline. Either buy the 20K plan or run a separate key.
- Request props with regions=us only; two regions doubles the cost.
- Take a morning snapshot plus a pre-game close snapshot per game.
- Add daily ESPN injuries and Daily Faceoff line combinations.
- Enforce the per-game correlated-bet cap in code before props go live.
- Raise the props edge threshold above 4% to account for the higher margin.

**Phase 5 — paper-trade props through 2026-27.**
- Use CLV as the main measure. Decide on real money only after about 500 bets.

**Optional, if Phase 2 looks promising.** Buy one month of historical Odds API data to test against true closing lines for 2023-24 to 2025-26:
- About $59 on the 100K plan covers roughly 2 seasons with one closing snapshot and 4 markets.
- About $119 on the 5M plan covers all 3 seasons.
- Spend about 40 credits sampling one game first to confirm the books actually offered NHL props on those dates.

**Not worth doing now:**
- An in-house xG model: MoneyPuck's is fine. The real risk is the mirror site it's downloaded from.
- Dixon-Coles: the totals problem is the inputs, not the distribution.
- Goalie-adjusted Elo: goalie quality already enters as separate features, and the moneyline model starts from the market's price.

**Total:** about 2-3 weeks of focused work to reach paper trading. Whether props are really easier to beat for this system stays unproven until paper-trading CLV comes in.
