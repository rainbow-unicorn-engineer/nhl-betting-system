# NHL Betting System

Prices every NHL game with a calibrated probability model, compares that price with the sportsbooks, sizes any edge with quarter-Kelly staking, and grades every pick by closing-line value. It runs in paper-trading mode: it writes recommendations and settles them as paper bets, and it places nothing. The first paper-trading season is 2026-27. So far the settlement code has run only on simulated past slates.

**Closing-line value (CLV)** is the project's main measure of success. It asks whether the price a bet was taken at beat the market's final price before puck drop. Short-run profit is mostly luck; beating the closing line consistently is the standard evidence of a real edge.

The design choice behind everything else is calibration over accuracy. Research puts a rough ceiling of about 62% on single-game NHL prediction accuracy, and the best public models land near it. Any edge therefore has to come from stating probabilities more precisely than the market does, measured honestly.

[PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) holds the locked decisions and the running log of lessons learned. Where it and the code disagree, this README says so and describes what the code does.

## Terms

- **Moneyline**: a bet on who wins, overtime and shootout included.
- **Puck line**: hockey's point spread, almost always ±1.5 goals.
- **Totals (over/under)**: a bet on whether combined goals land above or below a number.
- **Vig**: the bookmaker's built-in margin. The **no-vig** price removes it, so the two sides' probabilities add up to 100%.
- **+EV**: positive expected value, meaning a bet that makes money on average at that price.
- **Kelly**: a formula that sizes each bet from the edge and the odds to grow the bankroll fastest. **Quarter-Kelly** bets a quarter of that to cut the swings.
- **Flat stakes**: the same amount on every bet, used to judge the picks apart from the staking rule.
- **Arbitrage**: prices at two books far enough apart that backing both sides locks in a profit whatever happens.
- **Middle**: an over at a low total at one book and an under at a higher total at another, so both win if the score lands between them.
- **Walk-forward validation**: train on earlier seasons and test on the next, never the reverse, the way the model is used live. A **purge gap** leaves the days just before each test period out of training, so nothing near the boundary leaks across.
- **Log loss**: the standard score for probability forecasts. Lower is better, and 0.693 is a coin flip.
- **ECE (expected calibration error)**: how far stated probabilities drift from actual win rates.
- **Temperature scaling**: a one-number adjustment that makes a model's probabilities more or less extreme until they match actual win rates.
- **Gate**: a pass mark set before a phase starts. A model that misses its gate isn't used for betting.
- **Snapshot**: one stored copy of every book's current prices, taken by a scheduled run.
- **Closing price (the close)**: the last price before puck drop. Here it is the last stored snapshot before puck drop, so it is only as late as the last run before the game.
- **In-play**: a game already under way. Books keep quoting it, but those prices are for a different bet, and the system never stores them.
- **Credits**: The Odds API's billing unit. Each request costs one credit per market per region.
- **Region**: The Odds API's name for a group of bookmakers it bills together, such as `us`, `us2`, `us_ex` (the exchanges) or `eu`. Naming up to 10 books with `bookmakers=` bills as one region, whichever regions they come from.
- **Exchange**: a venue where bettors trade contracts with each other instead of betting against a book, such as Kalshi, Polymarket and Novig. The exchange charges a trading fee on top of the quoted price.
- **Opening line**: a book's first price on a game. The closing line is its last before puck drop.
- **Player prop**: a bet on one player's own numbers, such as "over 2.5 shots on goal".
- **Power play**: time when one team has an extra skater because the other team took a penalty. The shorthanded team is on the **penalty kill**.
- **Void**: a bet the book cancels and refunds in full, usually because the game was postponed or cancelled.

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Data foundation: PostgreSQL schema; NHL API, MoneyPuck, and Odds API ingestion; pipeline commands | Done: 6 seasons, 7,945 games, 683k shots |
| 2 | Feature store (team form, goalie quality, rest and travel, Elo ratings) and a baseline model | Done: walk-forward log loss 0.6829, under its 0.69 gate |
| 3 | Moneyline model, betting engine, backtest, dashboard, daily recommendations, totals model, bet checker, arbitrage and middle alerts | Done. The totals model failed its gate, and its 2026-09-29 rebuild (v2) fails the stricter gate too: it is more accurate, but the team stats still add nothing beyond the league's scoring rate. Totals betting is off |
| 4 | Live-season operations | In progress. Done: paper settlement, the CLV ledger, confirmed starters, locked picks with a pre-game closing snapshot, per-game bet limits, named-bookmaker odds requests (Kalshi and Polymarket included, at half the credits), the free NHL odds feed stored beside The Odds API, ESPN injuries, power-play stats, ESPN opening and over/under prices, and props-line collection (history from ESPN, live from The Odds API on a second machine). Remaining: cloud migration, a props model, a daily recommendation digest, an Odds API historical backfill |
| 5 | Other sports and same-game parlays | Not started |

Where the evidence stands, from [docs/phase3_results.md](docs/phase3_results.md):

| Measure | Result |
|---|---|
| Moneyline model log loss, walk-forward | 0.6607. In every season that has a market line, the market scored better, by 0.001 to 0.009 |
| Calibration error (ECE) | 0.0146, inside the 0.02 target |
| Backtest at the 2.5% edge threshold | 363 bets: +1.4% ROI with quarter-Kelly stakes, −6.0% with flat stakes. No demonstrated edge |
| Backtest, bets with a claimed edge of 2.5–4% | −16.8% flat-stake ROI on 198 bets. Small disagreements with the market are noise |
| Backtest, bets with a claimed edge of 6–9% | +26.6% flat-stake ROI, but on only 46 bets |

The backtest covers 2025-26 only, the one season with real DraftKings two-way prices (1,018 games). That was also the model's weakest season, and the maximum drawdown was 17.9%. The results point to raising the moneyline threshold to about 5–6 percentage points. That has to be confirmed by paper trading in 2026-27 before it is locked, because a threshold chosen on the season it was measured on proves nothing. Paper bets store their edge, so paper trading can run at 2.5 and the higher threshold can be judged afterwards by filtering.

## Known issues

A code review on 2026-09-28 found problems with odds matching, paper-trading CLV, the season setting, the install, and the Odds API key in the logs. Those are fixed. A second pass the same day fixed the season filter for September loads, postponed and cancelled games, the daily cap under overlapping runs, consensus CLV, the test suite reaching the live database, `--help` running jobs, the status count, and snapshots on days without games, and added the Windows setup. A third pass fixed the test opt-in that could still reach the live database, the close job's credit use on the free plan, console windows popping up from the Windows tasks, picks made a day ahead being voided, postponed games never getting a new pick, an empty backfill list, and the 2019-20 bubble playoffs, which no load stored. The bet checker's tests now run without a database. The MoneyPuck shots file comes from peter-tanner.com, which turned out to be MoneyPuck's own download link (moneypuck.com links it and says it is updated nightly), not an unchecked mirror. These are still open:

1. **The 2024-25 historical odds have no loader.** ESPN no longer serves that season's lines, and the Kaggle mirror that filled them on the live copy was loaded by hand. On a fresh install, 2024-25 games have no market line. See [docs/historical_odds.md](docs/historical_odds.md).
2. **Exchange prices leave out the trading fee.** The default `ODDS_BOOKMAKERS` includes Kalshi, Polymarket and Novig. The Odds API's documentation doesn't say how their prices are formed or whether fees are in them, so treat them as fee-free quotes. With `BETTABLE_BOOKS` unset, a pick's best price can come from an exchange, and its edge is then overstated by the fee (Kalshi's taker fee is 0.07 × p × (1 − p) per contract, where p is the contract price; see `betting/promo.py`). Set `BETTABLE_BOOKS` to the books you actually use.
3. **The named-bookmaker request hasn't run against the live API yet.** It was built and tested without a key. On the first real run, check the log: a full snapshot should report "this call cost 3" and a close "1", and `SELECT DISTINCT book_name FROM raw.odds_snapshots` should list `kalshi` and `polymarket`.
4. **The NHL's free odds feed is unproven as a closing line.** Its "last updated" stamp doesn't track prices, its FanDuel is the Canadian book, it prices only the next game date, and at 1 a.m. Central on 2026-10-01 it still listed the previous night's finished games while that day's eight games had no prices. It is stored for comparison only and never prices or grades a pick. See [The NHL's free odds feed](#the-nhls-free-odds-feed).
5. **Power-play features change once power-play stats are loaded.** The box-score load never filled power-play ice time, so the power-play and penalty-kill rate features (`pp_xgf_per60`, `pk_xga_per60`) have always been a constant 99, which no model could use. The daily run now fills the current season, so from the next feature rebuild those features hold real values for this season and the constant for older ones, until those are filled too. A model trained on that mix learns from a feature that means different things in different seasons. On the test copy, filling a quarter of 2025-26 left the moneyline and baseline walk-forward scores unchanged to four decimals for 2025-26, and moved them by about 0.001 on the first 8 games of 2026-27. Fill every season, rebuild the features and retrain before trusting new model numbers (see [Upgrading an existing database](#upgrading-an-existing-database)).
6. **`test_baseline`'s database test fails once a new season has finished games.** It requires every walk-forward fold but the first and last to beat a coin flip. With a few finished 2026-27 games in the database, 2026-27 becomes the last fold and 2025-26 a middle one, where the baseline scores 0.6949 against the coin flip's 0.6932. The baseline isn't used for picks.
7. **Logs written before 2026-09-28 may contain the Odds API key.** A failed request used to log the full request URL, key included. Delete those logs, or rotate the key.

## How it works

Five layers, each replaceable without touching the others:

```
DATA        ingestion/   ->  raw.*        NHL API, MoneyPuck shots, odds, starters, injuries, props lines
FEATURES    features/    ->  features.*   point-in-time team, goalie, schedule, and Elo features
MODELS      models/      ->  models.*     calibrated probabilities
STRATEGY    betting/     ->  betting.*    edges, stakes, paper bets, CLV
INTERFACE   dashboard/                    Streamlit control room
```

### Data sources

| Source | What it provides | Access |
|---|---|---|
| NHL API (`api-web.nhle.com`, through `nhl-api-py`) | Teams, schedule, results, box scores, player and team game stats | Free, undocumented, can change without notice |
| NHL stats API (`api.nhle.com/stats/rest`) | Per game and player: power-play and penalty-kill ice time, power-play goals and assists, faceoffs won and lost. A month of games per request | Free, undocumented |
| NHL odds feed (`api-web.nhle.com` partner-game and schedule endpoints) | DraftKings (US feed) and FanDuel Canada (Canadian feed) moneyline, puck line, over/under and 3-way prices, plus moneylines from several partner books on the schedule, for the next game date only | Free, undocumented, no history |
| MoneyPuck | Every unblocked shot attempt since 2007-08, with expected goals (xG, the chance a given shot becomes a goal) | Free zipped CSVs for non-commercial use, from MoneyPuck's download link at peter-tanner.com; credit MoneyPuck.com wherever its data is shown |
| The Odds API | Moneyline, puck line and totals prices from ten named books by default: Kalshi, Polymarket, Pinnacle, DraftKings, FanDuel, BetMGM, BetRivers, theScore Bet (`espnbet`), Hard Rock and Novig. Player props, one game per request | API key; the free plan is 500 credits a month per key |
| ESPN summary API | One reference line per past game: the closing moneyline, puck line and total, and, where ESPN has them, the opening moneylines and total and the over/under and puck-line prices, opening and closing. The book varies by season | Free, no key, undocumented |
| ESPN injury list | Every listed player's status (day-to-day, out, injured reserve, suspended), injury and expected return | Free, no key; the current list only, so it is saved daily |
| ESPN core API | Past player-prop prices for 2025-26: DraftKings on some dates and in the playoffs, ESPN BET opening prices in October and November 2025 | Free, no key, undocumented |
| Daily Faceoff | Projected starting goalies, each tagged Confirmed or a softer status | Scraped from the page's embedded data, one request per run; can break without notice |

Research notes behind the 2026-09-29 feeds work are in [docs/research/](docs/research/).

### Machine roles

Each machine has one role, its own `.env`, its own Odds API key, and its own database. Each key has its own 500 free credits a month, so the two jobs never share a budget.

| Role | Machine | Scheduled runs | What its key pays for |
|---|---|---|---|
| Picks | The Mac ([ops/launchd/](ops/launchd/)) | `daily`, `close --due` every 15 minutes, optional midday `odds` | Moneyline snapshots and closes: at most about 228 credits a month, about 321 with the midday run |
| Props | The Windows PC ([ops/windows/](ops/windows/), `-Role props`) | `refresh` at 9:00, `props` at 10:00, `props --due` every 15 minutes | Player-props lines: at most 308 to 464 credits a month for one market (February to January) |

A pick can be graded only against closing snapshots taken on the machine that made it, which is why the picks machine takes its own closes. The props machine makes no picks; its `refresh` run loads the schedule (every props line is tied to a game), box scores, power-play stats and the injury list, all free.

### The Odds API: ten named books

- Requests name their books with `bookmakers=` instead of asking for whole regions. The API bills every group of up to 10 books as one region, so a full snapshot (three markets) costs 3 credits and a moneyline close 1, half the old 6 and 2 for `us,us2`.
- The default ten (`ODDS_BOOKMAKERS`) start with `kalshi` and `polymarket`, the two venues the owner can use from Texas, then `pinnacle`, `draftkings`, `fanduel`, `betmgm`, `betrivers`, `espnbet`, `hardrockbet` and `novig`. The offshore and sweepstakes books that `us,us2` returned (Bovada, BetOnline, MyBookie, LowVig, BetUS, Fliff and others) are gone, so the median fair price now comes from these ten.
- Kalshi, Polymarket and Novig are exchanges; see [Known issues](#known-issues) on their fees. Pinnacle's prices come from its public website and may lag, per The Odds API's bookmaker list.
- An eleventh book logs a warning, because 11 to 20 books bill as two regions. `ODDS_BOOKMAKERS=` (set to nothing) goes back to whole regions, from `ODDS_REGIONS` (default `us,us2`).
- Each request logs the book selection, the expected cost, and then what the call really cost, from the API's `x-requests-last` header.

### The NHL's free odds feed

The NHL's own site API carries betting prices, free and with no key: the US partner feed is DraftKings (moneyline, puck line, over/under, and a 3-way regulation line with a draw price), the Canadian feed is FanDuel Canada, and the schedule lists moneylines from partner books (DraftKings, FanDuel, Tipsport, Sportradar and others).

- **Stored apart.** Each snapshot goes into `raw.nhl_feed_snapshots`, never `raw.odds_snapshots`, so it never prices or grades a pick. The `daily`, `odds` and `close` runs take one right after each Odds API snapshot (three free requests), and `python pipeline.py nhl-odds` takes one by hand. Started and finished games are skipped.
- **Limits.** Only the next game date has prices, and a played game loses them, so there is no history. The feed's "last updated" stamp is not a price time: on opening day it stayed a month old while DraftKings' prices moved in 10 of 20 price series. Its FanDuel is the Canadian book, while The Odds API's `fanduel` is the US one, so FanDuel gaps can be real differences between two books. Veikkaus is left out by default (`NHL_FEED_EXCLUDE_BOOKS`), because its favourite contradicted every other book in 2 of 3 games checked. A price whose two sides' implied probabilities don't add up like a real price is dropped with a warning.
- **Comparing.** `python pipeline.py compare-feeds [--date YYYY-MM-DD] [--detail]` pairs each feed DraftKings or FanDuel price with the same book's Odds API price for the same game and market taken within 10 minutes, and reports the gap in implied-probability points and in cents (American odds on a scale where -105 and +105 are 10 apart), and how often each price series changed between snapshots. DraftKings is the clean test. The question it answers is whether the feed is fresh enough to add free closing snapshots. Until a season of pairs says yes, it is a reference only.

### ESPN: prices, injuries and past props

- **Opening and over/under prices.** The ESPN summary the backfill already downloads always carried each game's opening lines and the over/under and puck-line prices; the old loader kept only the closing moneyline. It now stores them all (`raw.historical_odds`), and the daily top-up does so for new games. `python -m ingestion.espn_odds --refresh --season 20252026` fills rows stored before, about one request a game; it never overwrites a stored closing line and never mixes in prices from a different book. On a 74-game 2025-26 DraftKings sample, 71 games have closing over/under prices and 70 opening ones, and in 18 the opening total was on a different line from the close. Older Unibet seasons have closing over/under prices but no opening ones, except 2023-24, which has everything; Unibet's moneylines are 3-way and some of its prices were taken in play, so a backtest should use `provider = 'DraftKings'` rows only. ESPN serves no lines for 2024-25 or for October to late November 2025.
- **Injuries.** `python pipeline.py injuries` saves ESPN's current injury list into `raw.injuries`, one snapshot a day, because ESPN keeps no history. Names are matched to `raw.players` the way the Daily Faceoff goalies are (full name, then first initial and surname, then surname on the same team; a goalie never matches a skater). On 2026-10-01 the list had 117 players, 93 matched; the other 24 are prospects with no NHL games. Nothing uses it yet; it is the start of an injury history for future features and for ruling out injured goalies.
- **Past props.** `python -m ingestion.espn_props --season 20252026 [--limit N]` backfills past player-prop prices into `raw.prop_odds_hist`, about 4 to 6 requests a game, so the whole season takes 1 to 1.5 hours; it is resumable. DraftKings props cover only certain 2025-26 dates and the playoffs, last updated 2 to 5 hours before puck drop on the dates checked. ESPN BET covers October and November 2025, but its last update came hours after puck drop, so only its opening prices are kept. A price updated after puck drop is never stored as a pre-game price.

### Power-play stats

`python pipeline.py nhl-stats` fills each skater's power-play and penalty-kill ice time, power-play goals and assists, and faceoffs won and lost in `raw.skater_games`, from the NHL stats API. The box-score load always left those columns at zero. The daily run (and the props machine's `refresh`) fills newly finished games, usually 3 requests; `--season 20252026` fills a whole season in about 27 requests and a minute. It only updates rows the box-score load already stored, and `stats_filled_at` records which rows have real values. A week of December 2025 filled 1,980 of 1,980 player-games: 1,266 with power-play time, 1,168 with penalty-kill time, and no row with more power-play goals than goals or more special-teams time than total time.

### Player props on the props machine

`python pipeline.py props` takes one snapshot of every game starting in the next 24 hours into `raw.prop_snapshots`; `props --due`, every 15 minutes, takes one more just before each puck drop (games starting within 16 minutes with no prop snapshot in the last 16). Listing the games is free. Each game then costs 1 credit per market returned, and nothing when no book has posted props yet. The default market is shots on goal (`PROPS_MARKETS=player_shots_on_goal`); a morning and a pre-game snapshot of every game is at most 308 (February) to 464 (January) credits a month on the 2026-27 schedule, inside the free 500. Four markets need the paid 20K plan. Each line is matched to a game in `raw.games` (so the schedule must be current) and each player to `raw.players`; an unmatched name is stored with no player id and logged. No props are bet: there is no props model yet.

### Models

- **Moneyline (`lgbm_market`).** A LightGBM model, a gradient-boosted decision-tree library, that starts from the market's own probability and learns corrections to it instead of rediscovering the market from scratch. Games without a line fall back to a market-blind version. Temperature scaling calibrates the output. The recommendation job retrains it from the database on every run.
- **Totals (`poisson_totals` v2).** Predicts each side's regulation goals, then combines the two into a total-goals distribution that can price any over/under line. Version 2 (2026-09-29) fixes two measured faults. It reweights the combined score distribution by winning margin, because real games end tied in regulation more often (22% against the 17% that independent scores imply) and by one goal less often. And it removes the model's drift within a season, using only games played before the one being priced. The walk-forward error score (NLL, negative log-likelihood: lower is better) improved from 2.1867 to 2.1801, and over/under log loss at the DraftKings line from 0.7051 to 0.6958 over 1,011 games (0.693 is a coin flip). But the gate now gives the baseline, the league's recent scoring rate, the same margin fix, and the baseline still wins, 2.1787, with the model ahead in only 2 of 5 seasons. So the team stats add nothing, `GATE_PASSED` stays False, and the predictions are stored for the bet checker and the alerts but never bet. The checker and alerts get the margin fix automatically. A new market check compares the model with the no-vig over/under price; 24 priced games are too few to judge (it needs 200). The likely route to a totals edge is to start from the market's own total, once 2026-27 over/under snapshots build up.
- **Baseline (`baseline_logreg`).** Logistic regression kept as a check that the features carry real signal without look-ahead.

### The daily chain

`python pipeline.py daily` runs these steps in order on the picks machine. Steps marked non-fatal log an error and let the chain continue. An exception in any other step ends the run, and that day gets no settlement, starters, or recommendations.

1. Waits up to 3 minutes for the network, because scheduled jobs can fire the moment a laptop wakes. If the network is still down, the run stops here.
2. Refreshes teams, the schedule from 3 days back to 7 days ahead of today, box-score game logs, and team game stats. The schedule refresh stores each game's start time and schedule state (on schedule, postponed, suspended, or cancelled) and moves a postponed game to its new date.
3. Takes a full odds snapshot (3 credits with the default books). When no game starts in the next 24 hours it skips the request and costs nothing.
4. Takes a free snapshot of the NHL's odds feed (non-fatal).
5. Tops up ESPN reference lines, with their opening and over/under prices, for newly finished games (non-fatal).
6. Fills power-play, penalty-kill and faceoff stats for newly finished games (non-fatal).
7. Refreshes MoneyPuck shots for `CURRENT_SEASON` (non-fatal). It downloads a fresh file when the cached one is missing or more than 20 hours old, then reloads the season's shots. It does nothing until the season has a finished game.
8. Rebuilds features for `CURRENT_SEASON`.
9. Settles finished paper bets and rebuilds the bankroll and CLV ledger (non-fatal).
10. Pulls starting goalies from Daily Faceoff (non-fatal).
11. Saves ESPN's injury list (non-fatal).
12. Scores the day's slate and writes recommendations for games that don't have one yet (non-fatal).

"Today" is the local date: in `LOCAL_TIMEZONE` when that is set, otherwise in the machine's time zone. `CURRENT_SEASON` is the season containing that date (see [Configuration](#configuration)).

`python pipeline.py odds` is the midday run: a full odds snapshot, a free NHL-feed snapshot, starting goalies, recommendations for games that still have no pick, then an arbitrage and middle scan. `python pipeline.py close` takes a moneyline-only snapshot (1 credit) and a free NHL-feed snapshot, and does nothing else. It supplies the closing price that settlement grades each pick against. `python pipeline.py close --due`, the scheduled form, takes those snapshots only when a game starts within 16 minutes and no moneyline snapshot is less than 16 minutes old, and otherwise exits at no cost. Run every 15 minutes, that is one close per start time, in the last run before puck drop. [Snapshot schedule](#snapshot-schedule) says when to run each one.

`python pipeline.py refresh` is the props machine's daily run: steps 1 and 2, then the power-play stats and the injury list. It makes no Odds API request and no picks.

### Betting rules

- A bet needs the model's win probability to beat the market's fair probability by at least the threshold, measured in percentage points: 2.5 by default, so a 55% model price against a 52% market qualifies.
- The fair probability comes from each book's latest price with its margin removed, then the median across every book in `ODDS_BOOKMAKERS` (ten named books by default, Kalshi, Polymarket and Pinnacle among them). The bet is priced at whichever book pays best, and it must still be +EV at that price. `BETTABLE_BOOKS` limits that best price to the books you can use. When it is unset, the best price can come from any of the ten, exchanges included, whose quotes leave out their trading fee.
- The stake is a quarter of full Kelly, capped at 2% of the bankroll per bet and 10% per day. The daily cap counts the stakes of picks already issued for that date, except picks marked `SKIPPED` and voided picks. When it is reached, the weakest remaining edges are skipped. The cap is checked again when the picks are written, under a database lock, so two runs that overlap can't together go past 10%. Stakes are sized from the fixed `BANKROLL` setting and don't compound with paper profit and loss.
- **Per-game limits.** Bets on the same game tend to win or lose together, so at most 3 bets go on one game (`MAX_BETS_PER_GAME`) and at most 4% of the bankroll is staked on it in all (`MAX_GAME_STAKE_PCT`), counting every market and the picks already issued, except `SKIPPED` and voided ones. When a game is full, its weaker edges are skipped, each with its reason in the log. The limits are checked again under the same database lock. While only moneylines are bet (one pick per game), they can't bind yet; they are in place for when totals or props are.
- Only games that haven't started are scored. A game the NHL API marks as live, postponed, suspended or cancelled, or whose start time has passed, is left out.
- Each game gets at most one moneyline pick. The pick is locked at the book, price, and snapshot time it was issued at, and later runs never re-price or delete it. They only add picks for games that still have none, within what is left of the day's budget. A pick marked `SKIPPED` by hand releases its stake from the budget but still blocks a new pick for that game. A pick settled `VOID` doesn't block one: when a postponed game is played on its new date, it can get a new pick at the new date's prices.
- A pick still pending when its game finishes is settled as a paper bet at its locked book and price. The closing price is the same book's last snapshot taken after the pick and before puck drop. If that book has no such snapshot, CLV compares the median no-vig probability of every book's last quote in the same window with the no-vig fair probability the pick was issued at, so neither side of the comparison carries the book's margin. If no snapshot qualifies, CLV is left blank, not set to zero.
- **The void rule.** A pick is settled `VOID` instead, the way a book would void the bet, when its game is postponed or cancelled, or when the game finally starts more than 3 hours earlier or later than the start time it had when the pick was written (a postponed game played on its new date). Each pick stores that start time (`scheduled_start`), so a pick made a day ahead with `betting.recommend --date` is not voided when its game starts on time. A pick written before that column existed has no stored start, and falls back to the old rule: void when the game starts more than 36 hours after the pick was priced. A void has no profit or loss and no CLV. It isn't counted as a bet, a stake, a win or a loss in the bankroll ledger or the CLV report, and it releases its stake from the day's budget.

The edge and staking rules live in [betting/engine.py](betting/engine.py) and are unit-tested, and the live job and the backtest both call them. The daily cap is applied in `betting/recommend.py` and `betting/backtest.py`, and settlement is in `betting/settle.py`. The bet checker measures against the offered price with the margin left in, so it reads lower than the daily job for the same bet.

The code departs from PROJECT_CONTEXT §7 in four places, and the code is what runs. It removes the margin by proportional rescaling, not the power method. It accepts quotes up to 18 hours old, not 5 minutes. The totals and props thresholds are unused, because neither market is bet. And when a pick's own book has no closing quote, CLV compares two no-vig probabilities instead of §7's two implied ones, because the consensus close has no single price with a margin in it.

## What this deliberately does not do

- **No real money yet.** Real stakes wait for 500+ paper bets with average CLV above 1 percentage point and a significance test on the claimed edge. A full backtest season produced 363 bets at a 2.5-point threshold and only 46 in the 6–9 point band, so at a higher threshold that gate is several seasons away.
- **No automatic bet placement.** Recommendations are for a person to act on. The dashboard is read-only; there is no approve, skip, or "I placed this" step yet.
- **No totals betting** until the totals model passes its gate. Each recommendation run logs that totals are predictions-only.
- **No props betting.** Props lines are collected, live on the props machine and from ESPN for 2025-26, for a props model that doesn't exist yet.
- **No random train/test splits and no look-ahead.** Validation is walk-forward with a 7-day purge gap, and every feature is built only from games that finished before puck drop.
- **No proxy betting.** Each bettor places only their own bets with their own money. Execution research is in [docs/texas_execution_options.md](docs/texas_execution_options.md). The promo-hedging calculator keeps each bettor's stakes and P&L separate and never models one person betting for another.

## Quick start

Requires Python 3.11+, Docker, and optionally a key from [the-odds-api.com](https://the-odds-api.com). The live copy runs on macOS; Windows equivalents are in the comments.

```bash
git clone https://github.com/rainbow-unicorn-engineer/nhl-betting-system.git
cd nhl-betting-system
python3 -m venv .venv                       # Windows: py -m venv .venv
source .venv/bin/activate                   # Windows: .venv\Scripts\Activate.ps1
python -m pip install -e ".[ml,dashboard,dev]"
cp .env.example .env                        # Windows: Copy-Item .env.example .env
# Edit .env now: set POSTGRES_PASSWORD and ODDS_API_KEY before the next line
docker compose up -d                        # PostgreSQL 16; the schema and password are fixed on first start
python pipeline.py setup                    # checks the database, nhlpy, and the Odds API key; seeds the venues
```

If PowerShell refuses to run `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

Also set `LOCAL_TIMEZONE` in `.env` if this machine's clock isn't on your time zone, and `BETTABLE_BOOKS` if you can bet only at some books. Both are described under [Configuration](#configuration).

Set the password before the first `docker compose up`. The committed default is baked into the database on first start. `docker-compose.yml` publishes port 5432 on this machine only (`127.0.0.1`), so other computers on the network can't reach the database. To change the password of a database that already exists, run `docker compose exec db psql -U nhl -d nhl_betting -c "ALTER USER nhl PASSWORD '<new>'"`, then update `.env`.

Without a real Odds API key, no odds request is made: each snapshot logs an error saying the key is not set, and the rest of the pipeline still runs. The `your_key_here` placeholder from `.env.example` counts as not set. No picks are made, because a game with no price is never bet. Each machine has its own `.env`, so a Mac and a PC can each use their own key, with 500 free credits a month each.

### Load history (once)

```bash
python -m config.migrate --seed-venues      # arena coordinates and time zones; the same command on macOS and Windows
python pipeline.py backfill                 # 2020-21 through the current season: games, game logs, MoneyPuck shots; about 20 minutes a season; safe to re-run
python -m ingestion.espn_odds               # free historical reference lines with their prices; resumable
python -m ingestion.nhl_stats --season 20252026   # power-play stats; repeat for each season, about a minute each
python pipeline.py features                 # every season
python -m models.lgbm                       # walk-forward evaluation; registers the moneyline model
python -m models.totals                     # registers the totals model
python pipeline.py status
```

- **The venue seed** adds arena coordinates and time zones, which the travel features need. It applies `db/seed_venues.sql` through Python, read as UTF-8, so it is one command on every system and "Montréal" arrives intact. It is safe to run before the backfill and safe to re-run. `python pipeline.py setup` already runs it when no team has coordinates yet, so on a new database this line only confirms it.
- **Run `python -m models.lgbm` before the first recommendation job.** The job retrains its model every run but can only save predictions once `lgbm_market` has a registry row. Without it, the job scores the slate, fails when it saves, logs a non-fatal error, and writes nothing. `python -m models.totals` is needed only for the totals predictions; without it, moneyline picks are still written.
- **ESPN no longer serves 2024-25 lines.** The original setup filled that season from a Kaggle mirror by hand, and there is no loader for it in the repo. See [docs/historical_odds.md](docs/historical_odds.md).
- **Power-play stats before features.** Fill every loaded season (`20202021` through the current one) before building features, so the power-play features are real in every season, not only some (see [Known issues](#known-issues)).
- **Past props (optional).** `python -m ingestion.espn_props --season 20252026` loads ESPN's past props, 1 to 1.5 hours, resumable.

### Every day

These runs are automatic on each machine (see [Scheduling](#scheduling)). By hand, on a new machine or to catch up:

```bash
# Picks machine (the Mac)
python pipeline.py daily                    # morning: picks are issued at this snapshot's prices
python pipeline.py odds                     # optional, midday: confirmed starters and alerts
python pipeline.py close --due              # every 15 minutes: a close only when a game is about to start
python pipeline.py close                    # by hand: a closing snapshot right now (1 credit)
python pipeline.py compare-feeds            # the free NHL feed against The Odds API, today's games
streamlit run dashboard/app.py              # the control room

# Props machine (the Windows PC)
python pipeline.py refresh                  # morning: schedule, box scores, power-play stats, injuries; no credits
python pipeline.py props                    # a props snapshot of every game in the next 24 hours
python pipeline.py props --due              # every 15 minutes: one pre-game props snapshot per game
```

### Before each season

The season rolls over on July 1 without any change to the code.

1. Run `python pipeline.py setup`. Check that it prints the new season, your time zone, and today's date, and that the Odds API key is OK. If `NHL_SEASON` is still set in `.env` from a playoff run that went past July 1, remove it.
2. Build the new season's features now with `python pipeline.py features --season 20262027`, or let the next `daily` run do it.
3. Check that each machine's scheduled jobs are loaded: `daily`, `close`, and `odds` if you use it, on the picks machine; `refresh`, `props` and `props-due` on the props machine (see [Scheduling](#scheduling)). Each machine's `.env` needs its own `ODDS_API_KEY`.
4. Decide `EDGE_MIN_ML` for the season (see Status).
5. Before using the promo calculator, re-check the Kalshi and Polymarket fees hard-coded in `betting/promo.py`.

### Upgrading an existing database

`db/schema.sql` runs only when the Docker volume is first created, so an older database can be missing tables and columns added since. Nothing needs doing by hand for the schema: the next pipeline command creates any missing table with its indexes and adds any missing column, and so do the dashboard and the module commands that use them. It checks first, so an up-to-date database is left alone. No data is moved. `python -m config.migrate` does the same on demand.

- **Tables added on 2026-09-29:** `raw.nhl_feed_snapshots` (the free NHL odds feed), `raw.injuries` (ESPN's injury list), `raw.prop_odds_hist` and `raw.prop_odds_fetches` (ESPN's past props and the backfill's resume log), and `raw.prop_snapshots` (live props lines).
- **Columns added on 2026-09-29:** fourteen on `raw.historical_odds` (the opening moneylines and total, the over/under and puck-line prices opening and closing, the opening puck line, ESPN's event id, and `prices_fetched_at`), and `raw.skater_games.stats_filled_at`.
- **Columns added on 2026-09-28:** `raw.games.start_time_utc` (puck drop), `raw.games.schedule_state` (postponed, suspended, or cancelled), `betting.recommendations.priced_at` (when a pick's price was captured), `betting.recommendations.scheduled_start` (the game's start time when the pick was written), and, on an older database where settlement has never run, `betting.placed_bets.is_paper`.

After pulling the 2026-09-29 changes, on each machine:

1. Run `python pipeline.py setup`. It creates the new tables and columns.
2. Run `python -m models.totals` once. The totals model is now version 2, and until `poisson_totals v2` is registered, each recommendation run logs a non-fatal "not in registry" error and stores no totals predictions. Moneyline picks are unaffected.
3. Fill the power-play stats for every season before the next `daily` run, then rebuild the features and re-run the walk-forward evaluations, because those features change (see [Known issues](#known-issues)). About 6 minutes for the stats:

   ```bash
   for S in 20202021 20212022 20222023 20232024 20242025 20252026 20262027; do python -m ingestion.nhl_stats --season $S; done
   python pipeline.py features
   python -m models.lgbm
   python -m models.totals
   ```

   In PowerShell: `foreach ($S in 20202021,20212022,20222023,20232024,20242025,20252026,20262027) { python -m ingestion.nhl_stats --season $S }`.
4. Fill the ESPN prices for the games already stored: `python -m ingestion.espn_odds --refresh --season 20252026` first (the DraftKings season a backtest can use), then the other seasons if wanted. About one request a game; resumable.
5. On the picks machine, check `ODDS_BOOKMAKERS` and `BETTABLE_BOOKS` in `.env` (see [Configuration](#configuration)); without them the defaults apply. On the Windows PC, register the props role: `.\ops\windows\register-tasks.ps1 -Role props`, which also removes any picks tasks registered there before.

Games already stored have no start time or schedule state until their schedule is loaded again. The next `daily` run fills the window from 3 days back to 7 days ahead. To fill a whole season at once, run `python -m ingestion.nhl_api season 20262027` (the current season). Older seasons can stay blank. An odds snapshot matches a game with no start time by its Eastern date instead, and settlement logs a warning that the game's closing window has no puck-drop limit.

Picks written before the 2026-09-28 upgrade have no `priced_at`, so their closing window has no lower limit, and the snapshot they were priced from can still grade them at zero CLV. They have no `scheduled_start` either, so the void rule checks them with the old 36-hour limit (see [Betting rules](#betting-rules)).

On a machine that already runs the pipeline, also reinstall with `python -m pip install -e ".[ml,dashboard,dev]"` to pick up the dependencies added on 2026-09-28 (`tzdata`, `matplotlib`, `nhl-api-py` 3.2.0 or later, and LightGBM held below 5), install the `close` job that runs `close --due` every 15 minutes in place of any fixed-time close (see [Scheduling](#scheduling)), and set `LOCAL_TIMEZONE`, plus `BETTABLE_BOOKS` if wanted, in `.env`.

## Looking at the results

The dashboard shows pending picks and the next few days' games with start times in your local time, the model registry, the backtest, and the paper bankroll. `python -m betting.settle --report` prints CLV and ROI, and how many settled bets have a CLV. For anything else, query PostgreSQL from the repo folder:

```bash
docker compose exec db psql -U nhl -d nhl_betting
```

```sql
-- Today's picks (the container clock is UTC, so name your time zone; priced_at is UTC too)
SELECT g.away_team || '@' || g.home_team AS game, r.side, r.best_book, r.best_price,
       r.edge_pct, r.recommended_stake, r.status, r.priced_at
FROM betting.recommendations r JOIN raw.games g USING (game_id)
WHERE g.date = (now() AT TIME ZONE 'America/Chicago')::date
ORDER BY r.edge_pct DESC;

-- Settled paper bets, newest first. clv is blank when no snapshot came between the pick and puck drop
SELECT g.date, g.away_team || '@' || g.home_team AS game, p.placed_price,
       p.closing_line, p.clv, p.result, p.pnl
FROM betting.placed_bets p
JOIN betting.recommendations r USING (rec_id)
JOIN raw.games g ON g.game_id = r.game_id
WHERE p.is_paper
ORDER BY g.date DESC
LIMIT 50;
```

```sql
-- The latest injury list, by team
SELECT team_abbrev, player_name, position, status, injury_type, return_date
FROM raw.injuries
WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM raw.injuries)
ORDER BY team_abbrev, status;

-- Shots-on-goal props stored on the props machine, newest snapshot first
SELECT captured_at, game_id, book, player_name, line, over_price, under_price
FROM raw.prop_snapshots
WHERE market = 'player_shots_on_goal'
ORDER BY captured_at DESC, player_name
LIMIT 50;
```

`betting.bankroll_log` holds one row per settled day and is rebuilt from `placed_bets` on every settle run. `models.predictions` holds every scored game, including the totals distributions. Arbitrage and middle alerts appear only as warning lines in the run log, plus an optional macOS notification. They aren't stored or shown on the dashboard.

## Commands

**Pipeline** (`python pipeline.py <command>`):

| Command | What it does |
|---|---|
| `setup` | Checks the database connection (adding any missing tables and columns), `nhlpy`, and the Odds API key, applies the venue seed when no team has coordinates yet, and prints the season, the time zone, and today's local date |
| `status` | Row counts for the main tables (the new feeds included), upcoming games (any not yet final), and games per season |
| `backfill` | Loads every season from `BACKFILL_FIRST_SEASON` through the current one from the NHL API and MoneyPuck |
| `features [--season YYYYYYYY]` | Builds the feature store for one season or all |
| `daily` | The picks machine's daily chain described above |
| `odds` | Full odds snapshot (3 credits), a free NHL-feed snapshot, starters, recommendations for games without a pick, and alerts |
| `close [--due]` | Moneyline-only odds snapshot (1 credit) for the closing price, then a free NHL-feed snapshot. No picks, no alerts. With `--due`, only when a game starts within 16 minutes and no moneyline snapshot is under 16 minutes old; otherwise it logs why and exits at no cost |
| `recommend` | Scores today's slate into `betting.recommendations` |
| `starters` | Starting goalies from Daily Faceoff |
| `settle` | Settles paper bets and rebuilds the bankroll and CLV ledger |
| `refresh` | The props machine's daily run: schedule, box scores, power-play stats and the injury list. No Odds API request, no picks |
| `props [--due] [--markets M]` | A player-props snapshot from The Odds API: every game in the next 24 hours, or with `--due` only games starting within 16 minutes that have no prop snapshot in the last 16. 1 credit a game per market returned |
| `nhl-odds` | A free snapshot of the NHL's odds feed, right now |
| `compare-feeds [--date YYYY-MM-DD] [--detail]` | The NHL feed against The Odds API for one date's games: price gaps and how often prices changed. Reads only |
| `injuries` | Saves ESPN's injury list for today |
| `nhl-stats [--season YYYYYYYY]` | Fills power-play, penalty-kill and faceoff stats: the current season's unfilled dates, or one whole season |

`python pipeline.py --help` lists the commands, and `<command> --help` prints a command's options and runs nothing. An unknown command or option exits with code 2.

**Module tools** (`python -m <module>`):

| Module | Use |
|---|---|
| `betting.checker --leg "MTL@BUF ml away -125" --leg "SJS@WSH total over 6.5 -110"` | Is this bet or parlay +EV? Uses stored model probabilities; add `--date`, `--price` for a boosted parlay, `--bankroll` |
| `betting.recommend --date 2026-01-15 --simulate --dry-run` | Replays a past slate as if it were upcoming, writing nothing. Also takes `--bankroll` and `--edge-min` |
| `betting.settle --report` | CLV and ROI report, with a count of settled bets that have a CLV |
| `betting.backtest` | Payout backtest on DraftKings-era prices. Uses the 2.5 threshold in `betting/engine.py`. Redraws `models/artifacts/lgbm_calibration.png` |
| `betting.alerts` | Arbitrage and middle scan over the freshest quotes; results go to the log |
| `betting.promo free_bet --amount 100 --bonus-odds 400 --hedge-venue kalshi --contract-price 0.80` | Hedge stakes for a sportsbook promo, per bettor. `--contract-price` is the price of the contract that pays if the bonus leg loses. Details: [docs/promo_hedging_calculator.md](docs/promo_hedging_calculator.md) |
| `models.baseline`, `models.lgbm`, `models.totals [--no-register]` | Walk-forward evaluation and registry entry for each model. `models.baseline` and `models.lgbm` redraw their calibration plots in `models/artifacts/` (`baseline_calibration.png`, `lgbm_calibration.png`, both tracked by git); `models.totals` draws none, takes about a minute, registers `poisson_totals v2` (inactive), and with `--no-register` only reports |
| `ingestion.nhl_api teams\|daily\|season <YYYYYYYY>\|backfill-all` | NHL API loads, piece by piece. `season` also fills in start times for that whole season |
| `ingestion.moneypuck <start_year> [--download]` | One season of MoneyPuck shots, such as `2025` for 2025-26. `--download` fetches a fresh copy; without it the CSV must already be in `DATA_DIR` |
| `ingestion.espn_odds [season] [--refresh] [--limit N]` | Historical reference lines with their prices; all seasons when no season is given. `--refresh` re-fetches stored rows whose prices were never parsed and fills them |
| `ingestion.espn_props --season S [--limit N] [--refresh]` | Past player-prop prices from ESPN into `raw.prop_odds_hist`; resumable |
| `ingestion.espn_injuries [--date YYYY-MM-DD]` | ESPN's injury list; the date must be within a day of today, because ESPN has no history |
| `ingestion.nhl_stats --season S \| --from YYYY-MM-DD [--to ...] \| --missing` | Power-play, penalty-kill and faceoff stats for a season, a date range, or the current season's unfilled dates. Exits 1 if any request window failed |
| `ingestion.nhl_odds snapshot \| compare [--date D] [--detail]` | The NHL feed: one snapshot, or the comparison report |
| `ingestion.props_odds [--due] [--markets M]` | One props snapshot, as `pipeline.py props` |
| `ingestion.dailyfaceoff [--date YYYY-MM-DD]` | Starters for one date |
| `ingestion.odds_api [--markets h2h,spreads,totals]` | One odds snapshot: 3 credits for the default markets, 1 for `h2h`, with the default books. Skipped, at no cost, when no game starts in the next 24 hours |
| `config.migrate [--seed-venues]` | Creates any tables and adds any columns a database made from an older `db/schema.sql` is missing. Pipeline commands do this on their own. `--seed-venues` also applies `db/seed_venues.sql` |

`--help` prints the options and runs nothing for `pipeline.py` and each of its commands, `betting.checker`, `betting.recommend`, `betting.settle`, `betting.alerts`, `betting.promo`, `ingestion.odds_api`, `ingestion.espn_odds`, `ingestion.espn_props`, `ingestion.espn_injuries`, `ingestion.nhl_stats`, `ingestion.nhl_odds`, `ingestion.props_odds`, `ingestion.dailyfaceoff`, `models.totals`, and `config.migrate`, so it never spends credits or writes to the database. `models.baseline`, `models.lgbm`, and `betting.backtest` take no options and start their full run whatever you pass them.

## Configuration

Settings come from `.env`. `.env.example` sets the first four rows and lists the optional ones commented out, with their defaults, except `BANKROLL`, `EDGE_MIN_ML`, `MAX_ODDS_AGE_HOURS` and the alert settings.

| Variable | Default | Purpose |
|---|---|---|
| `POSTGRES_HOST`, `_PORT`, `_DB`, `_USER`, `_PASSWORD` | `localhost`, `5432`, `nhl_betting`, `nhl`, `nhl_dev_2026` | Database connection. Only the password also reaches the Docker container; `docker-compose.yml` fixes the database name, user, and port, so change those in both files |
| `ODDS_API_KEY` | none | Live odds snapshots. Each machine has its own `.env`, so each can use its own key and its own monthly credits |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `DATA_DIR` | `<repo>/data`; `.env.example` sets `./data` | Cache for MoneyPuck CSV downloads. A relative path is relative to the repo, whatever folder a command starts in |
| `LOCAL_TIMEZONE` | this machine's time zone | Your time zone as an IANA name, such as `America/Chicago`. It decides "today": the slate date, the schedule refresh window, and the season. The dashboard shows start times in it. Set it wherever the clock runs on UTC, such as a container or a cloud server. An unknown name, or a region folder such as `America` on its own, logs a warning and falls back to the machine's zone |
| `BETTABLE_BOOKS` | unset: every book | Comma-separated Odds API bookmaker keys you can bet at, such as `kalshi,polymarket`; case doesn't matter. Only these books can supply a pick's price, while the fair probability still uses every book. Each recommendation run logs which books it used |
| `ODDS_BOOKMAKERS` | `kalshi,polymarket,pinnacle,draftkings,fanduel,betmgm,betrivers,espnbet,hardrockbet,novig` | The books every odds snapshot asks for, as Odds API keys from any region; case, repeats and blanks are ignored, and a malformed key is dropped with an error. Up to 10 bill as one region (3 credits a full snapshot, 1 a close); 11 to 20 bill as two and log a warning. Set to nothing (`ODDS_BOOKMAKERS=`) to ask for whole regions instead |
| `ODDS_REGIONS` | `us,us2` | Used only when `ODDS_BOOKMAKERS` is set to nothing: 2 regions, so 6 credits a full snapshot and 2 a close |
| `MAX_BETS_PER_GAME` | `3` | At most this many bets on one game, counting every market (`SKIPPED` and voided picks don't count). A whole number, 1 or more; a bad value logs an error and the default is used |
| `MAX_GAME_STAKE_PCT` | `0.04` | The most staked on one game, all bets together, as a share of `BANKROLL`. Above 0 and at most 1; a bad value logs an error and the default is used |
| `NHL_FEED_EXCLUDE_BOOKS` | `veikkaus` | NHL-feed book names (lower case, no spaces) left out of `raw.nhl_feed_snapshots`, comma-separated. Set to nothing to keep every book |
| `PROPS_MARKETS` | `player_shots_on_goal` | Comma-separated Odds API prop markets for `props`; each costs 1 credit a game per snapshot. `--markets` overrides it |
| `PROPS_BOOKMAKERS` | the `ODDS_BOOKMAKERS` default | Books for props, as for `ODDS_BOOKMAKERS`. Set to nothing to use `PROPS_REGIONS` |
| `PROPS_REGIONS` | `us` | Used only when `PROPS_BOOKMAKERS` is set to nothing |
| `PROPS_CLOSE_LEAD_MINUTES` | `16` | `props --due` snapshots a game when it starts within this many minutes |
| `PROPS_CLOSE_MIN_GAP_MINUTES` | `16` | ... and it has no prop snapshot younger than this (0 allowed) |
| `NHL_SEASON` | the season containing today's local date | Pins the season as eight digits, such as `20252026`. The default rolls over on July 1, so set this only to finish a playoff run that goes past July 1, then remove it. A value that isn't eight digits with the second year one after the first logs an error and the default is used |
| `BACKFILL_FIRST_SEASON` | `20202021` | The first season `backfill` loads. It loads every season from this one through the current one. A malformed value logs an error and `20202021` is used. A first season later than the current one (or an `NHL_SEASON` pinned before it) logs an error, and only the current season is loaded |
| `BANKROLL` | `1000` | Fixed bankroll that stakes are sized against, and the paper ledger's starting balance. Changing it restates the whole paper ledger on the next settle run |
| `EDGE_MIN_ML` | `0.025` | Moneyline edge threshold for the recommendation job only. The checker and backtest keep the 2.5 in `betting/engine.py` |
| `MAX_ODDS_AGE_HOURS` | `18` | Odds older than this are ignored when scoring a slate |
| `CLOSE_LEAD_MINUTES` | `16` | `close --due` snapshots only when a game starts within this many minutes. A value that isn't a number above 0 logs an error and the default is used |
| `CLOSE_MIN_GAP_MINUTES` | `16` | ... and no moneyline snapshot is younger than this. A value that isn't a number of 0 or more logs an error and the default is used. With the 15-minute cycle, 16 and 16 take one close per start time and fit the free plan (see [Snapshot schedule](#snapshot-schedule)) |
| `ALERTS_MAX_AGE_MINUTES` | `30` | Only quotes this fresh count for arbitrage and middles |
| `MIN_ARB_PROFIT` | `0.001` | Smallest locked-in arbitrage profit worth alerting |
| `ALERTS_NOTIFY` | unset | `1` sends a macOS notification for each arbitrage alert |

The season is `CURRENT_SEASON` in `config/settings.py`: `NHL_SEASON` when that is set, otherwise the season containing today's local date. `python pipeline.py setup` prints the season, the time zone, and today's local date. For `BETTABLE_BOOKS`, the keys the API has returned so far are in `raw.odds_snapshots`: `SELECT DISTINCT book_name FROM raw.odds_snapshots ORDER BY 1`.

### Odds API quota

A full snapshot (`daily`, `odds`, or `python -m ingestion.odds_api`) requests three markets (moneyline, puck line, totals). The Odds API bills markets times regions, and up to 10 named books count as one region, so with the default `ODDS_BOOKMAKERS` a full snapshot costs 3 credits and a moneyline-only `close` costs 1. Asking for the regions `us,us2` instead (`ODDS_BOOKMAKERS=` set to nothing) doubles both, to 6 and 2.

A request is free only when the API lists no NHL events at all, which in practice means the off-season. In season it also lists the coming days' games, so a request on a day without games still costs full price. So every snapshot command first checks `raw.games`: when no game starts in the next 24 hours, it logs why and makes no request. That keeps off-days free as long as the schedule is current, which the daily run takes care of.

Each machine reads its own `.env`, so the Mac and the PC each use their own key, each with its own 500 free credits a month (see [Machine roles](#machine-roles)). Machines that share one key share its 500 credits.

| Plan | Credits a month, per key | What that allows in season |
|---|---|---|
| Free | 500 | Picks machine: the daily run plus `close --due` at its defaults, at most about 228 in any month of the 2026-27 schedule, and the midday `odds` run (about 93 more in a full month) fits too. Props machine: one props market at a morning and a pre-game snapshot, 308 to 464 a month |
| 20K ($30 a month) | 20,000 | The whole [snapshot schedule](#snapshot-schedule) with an `odds` run every 15 minutes through each game day, or four props markets |

After each request the log shows the credits remaining, the credits used, and what that call cost. The log never shows the API key.

## Snapshot schedule

A pick is locked at the price of the snapshot it was issued from, so grading it needs a later snapshot taken before its puck drop. The daily and midday times are defaults for a machine on Central time. The close job has no times: it reads each game's start time from the database.

| Run | When | Credits | What it is for |
|---|---|---|---|
| `daily` | Morning, 9:00 | 3, or 0 when no game starts in the next 24 hours | Refreshes everything and issues the day's picks at this snapshot's prices |
| `odds` (optional) | Midday, 13:00, once most starting goalies are confirmed | 3, or 0 the same way | Confirmed starters, picks for games that still have none, and arbitrage and middle alerts. It never changes a pick already issued |
| `close --due` | Every 15 minutes | 1 when due, otherwise 0 | Moneyline only. It snapshots when a game starts within 16 minutes and no moneyline snapshot is under 16 minutes old, so every start time, afternoon games included, gets one close, in the last run before its puck drop |

The credits are for the default `ODDS_BOOKMAKERS`. Each of these runs also takes a free NHL-feed snapshot right after its Odds API snapshot.

Why 16 and 16: any 16 minutes hold one of the 15-minute runs, so every start time has a run in its last 16 minutes, and the next run, 15 minutes later, finds a snapshot less than 16 minutes old and skips. The extra minute absorbs a run that starts a little late. When two start times are less than 16 minutes apart (22 times in 2026-27), the later game can share the earlier one's close, taken up to about 30 minutes before its puck drop.

What the close job costs depends on how many different start times each slate has. Replaying it against the 2026-27 schedule (1,344 regular-season games over 185 game days, 4.3 different start times a game day on average), with a run every 15 minutes, for each of the 15 minutes the cycle could start on:

| Setting | Closes a game day | Daily run plus closes, per calendar month (UTC) | Close before puck drop, median | Games with a close in their last 16 minutes |
|---|---|---|---|---|
| Default: `CLOSE_LEAD_MINUTES=16`, `CLOSE_MIN_GAP_MINUTES=16` | 4.2 (about 4 credits) | Oct 222, Nov 196, Dec 204, Jan 222, Feb 168, Mar 227 on average; never more than 228 | 9 minutes | 98.7%; every game has one in its last 30 |
| The old default, `CLOSE_LEAD_MINUTES=40`, `CLOSE_MIN_GAP_MINUTES=25` | 6.4 (about 6 credits) | Oct 286, Nov 254, Dec 269, Jan 294, Feb 220, Mar 292 on average; up to 329 | 8 minutes | 66%; every game has one in its last 30 |

These figures were replayed at the old cost of 6 credits a full snapshot and 2 a close, and are halved here for the default named books, which bill as one region instead of two. The daily run is 3 credits a game day, about 93 in a month with a game every day; the rest is the close job. At the defaults the two stay under the free plan's 500 in every month, whatever minute the cycle starts on, and the midday `odds` run (about 93 more) fits as well. Watch the credits-remaining figure in the log: when the credits run out, every snapshot fails for the rest of the month, including the morning one the picks come from. The old fixed-time closes (17:30 and 20:30) used about 60 credits a month at today's cost, but missed afternoon games.

- Each game's close is the last snapshot before its own puck drop. A close that runs after a game has started skips that game.
- `python pipeline.py close` without `--due` takes a snapshot right away, for 1 credit.
- A game with no snapshot between its pick and its puck drop settles with a blank CLV. A Mac that sleeps through the evening takes no closes, because launchd skips interval runs while the Mac is asleep.

## Scheduling

The live copy runs on macOS under launchd. Don't add cron entries as well, because the crontab is intentionally empty.

Templates for three agents are in [ops/launchd/](ops/launchd/), with install steps in its README: `com.nhlbetting.daily`, `com.nhlbetting.close` (`close --due` every 15 minutes), and the optional `com.nhlbetting.odds`. Each job logs to the repo's `logs/` folder, which git ignores. The live Mac already has `com.nhlbetting.daily` and `com.nhlbetting.odds` in `~/Library/LaunchAgents`, with their own times and log paths. Nothing replaces those files automatically: add the `close` agent by hand (unloading any earlier fixed-time close first), and compare the other two with the templates before replacing them.

- **Check the agents** with `launchctl list | grep com.nhlbetting`. The middle column is the last exit code.
- **Docker Desktop must be running**, or every run fails at the database check.
- **A Mac that was asleep** at the daily or midday time runs that job once when it wakes. The run waits up to 3 minutes for the network, and a late close skips the games already under way.
- **Pause them for the off-season** with `launchctl unload ~/Library/LaunchAgents/com.nhlbetting.odds.plist`, and the same for `.daily` and `.close`.

The Mac keeps the picks role, so it needs no props job: the new free steps run inside `daily`, `odds` and `close`.

**On Windows**, [ops/windows/register-tasks.ps1](ops/windows/) registers one role's jobs in Task Scheduler, the scheduler built into Windows. `-Role` is required. `-Role props`, the Windows PC's role, registers `refresh` at 9:00, `props` at 10:00 and `props --due` every 15 minutes. `-Role picks` registers the Mac's three jobs instead: `daily` at 9:00, `odds` at 13:00 with `-IncludeOdds`, and `close --due` every 15 minutes. Registering one role removes the other role's tasks. Each task starts in the repo folder, appends to `logs\<task>.log`, and runs hidden through `conhost.exe --headless`, so no console window pops up (Windows 10 21H2 or later, or Windows 11; `-VisibleConsole` for older Windows). It also takes `-RepoPath`, `-PythonPath`, `-DailyTime`, `-PropsTime` and `-Unregister`; its README has the details.

## Tests

```bash
pytest
```

**A plain `pytest` never touches a database.** `tests/conftest.py` runs before any test imports the settings. Unless you opt in, it points the database settings at a closed port, so every database test skips, whatever `.env` names. A line at the top of the output says which way it went, and why. Opt in only for a disposable database, one of two ways:

- **Its name ends in `_test`**, set as `POSTGRES_DB` in the environment or in `.env`.
- **`NHL_ALLOW_DB_TESTS=1` together with `POSTGRES_HOST`, `POSTGRES_PORT` and `POSTGRES_DB`**, all three set in the environment for that run, pointing at the copy. The flag alone does nothing, and so do overrides that name the same database as `.env` (`localhost` and `127.0.0.1` count as the same server): the tests still skip, and the line at the top says why. So the flag on its own can never send the suite to the database `.env` names.

The suite has 697 tests. Without a database, 612 pass and 85 skip, in about a minute. Nine files need no database at all: `test_betting_engine`, `test_promo`, `test_setup`, `test_feature_utils`, `test_odds_api`, `test_moneypuck`, `test_pipeline`, `test_migrate`, and `test_db_guard`; most other files have pure tests too. The new feed modules' tests (`test_nhl_odds`, `test_espn_odds`, `test_espn_injuries`, `test_espn_props`, `test_nhl_stats`, `test_props_odds`) run on trimmed copies of real API responses in `tests/fixtures/` and never touch the network.

### Running the database tests on a copy

Most database tests need real history (finished 2020-21 games, the 2026-01-15 slate with its features and reference lines), so the practical test database is a copy of yours. Inside the Docker container, from the repo folder:

```bash
docker compose exec db createdb -U nhl nhl_betting_test
docker compose exec db sh -c "pg_dump -U nhl nhl_betting | psql -q -U nhl -d nhl_betting_test"
```

The pipe runs inside the container, so the command is the same in every shell. For an empty test database instead, load the schema the container already holds, then the venues. The history then has to be loaded with `backfill`, which takes hours:

```bash
docker compose exec db createdb -U nhl nhl_betting_test
docker compose exec db psql -U nhl -d nhl_betting_test -f /docker-entrypoint-initdb.d/01_schema.sql
#   Git Bash rewrites paths that start with "/": prefix that line with MSYS_NO_PATHCONV=1
```

Then name the copy for the test run only. Because the name ends in `_test`, no opt-in variable is needed:

```bash
POSTGRES_DB=nhl_betting_test python -m config.migrate --seed-venues    # empty database only
POSTGRES_DB=nhl_betting_test pytest                                     # macOS, Linux, Git Bash
$env:POSTGRES_DB = "nhl_betting_test"; pytest; Remove-Item Env:POSTGRES_DB   # PowerShell
```

A copy with any other name, or one on another server or port, needs the flag and all three settings:

```bash
NHL_ALLOW_DB_TESTS=1 POSTGRES_HOST=127.0.0.1 POSTGRES_PORT=55432 POSTGRES_DB=nhl_betting_clone POSTGRES_PASSWORD=<its password> pytest
```

With a database, 696 of the 697 pass against a copy of the live data, in about a minute and a quarter; the one failure is `test_baseline`'s, once 2026-27 has finished games (see [Known issues](#known-issues)). The end-to-end recommendation test needs `lgbm_market` registered (`python -m models.lgbm`), and `TestSlateVectors` needs the stored features rebuilt after any power-play stats load (`python pipeline.py features`), because it compares a live build with the stored one. What the database tests write, all in the database they run against:

- **`test_settle`** adds synthetic picks (books `testbook`, `otherbook`, `voidbook`) and odds snapshots on 2020-21 games, then runs settlement, which settles or voids every due pick in that database, not only its own. It rebuilds `betting.bankroll_log` afterwards. It temporarily changes a few 2020-21 games' start times, and one game's schedule state, game state and score (to a postponed game that hasn't been played), and puts them back.
- **`test_recommend`** issues and deletes picks for 2026-01-15 games, deleting only the ones it created. The end-to-end test overwrites the moneyline and totals predictions of the 2026-01-15 games and deletes only the rows it added. One test temporarily marks a 2026-01-15 game as upcoming and postponed, then restores it.
- **Feature tests** (`test_build_all`, `test_build_vectors`, `test_team_features`, `test_goalie_features`, `test_schedule_features`, `test_elo`) rebuild `features.*` tables: 2020-21 and 2021-22 rolling features and matchups, every game vector, and every Elo rating.
- **`test_baseline`** re-registers `baseline_logreg` in `models.model_registry`.
- **`test_nhl_api`**, **`test_checker`**, and **`test_dailyfaceoff`** insert synthetic rows and delete them: games with ids 9999020001 to 9999020004 (the last three in January 2031), predictions, and a 2026-01-15 starting goalie (replacing, then deleting, any real row for that team and date).
- **The feed tests** insert synthetic games and delete them with everything they wrote: `test_nhl_stats` (game 9999020101), `test_props_odds` (9999020201), `test_espn_odds` (9999020301), `test_espn_props` (9999020302), and `test_nhl_odds` (9999030001 and 9999030002, on 2031-02-15). `test_espn_injuries` writes and deletes injury snapshots dated 2031-01-14 and 2031-01-15. They also create the new tables and columns if the database lacks them.
- **`test_recommend`**'s per-game limit test issues and deletes a totals pick and moneyline picks on 2026-01-15 games; **`test_totals`** reads ESPN DraftKings prices inside a transaction it rolls back.

`test_baseline` and `test_lgbm` also redraw `models/artifacts/baseline_calibration.png` and `lgbm_calibration.png`, which git tracks: restore them with `git checkout -- models/artifacts/` unless the model changed. Don't run the database tests while a scheduled job is writing to the same database.

## Repository layout

```
config/settings.py     Settings from .env, database engine, local date and season
config/migrate.py      Creates the tables and adds the columns an older database is missing; applies the venue seed
db/                    schema.sql (applied by Docker on first start), seed_venues.sql
ingestion/             nhl_api, moneypuck, odds_api, espn_odds, dailyfaceoff; nhl_odds (free NHL feed),
                       nhl_stats (power-play stats), espn_injuries, espn_props, props_odds (live props)
features/              team, goalie, schedule, and Elo builders; build_all orchestrates
models/                baseline, lgbm (moneyline), totals; artifacts/ holds calibration plots
betting/               engine, recommend, settle, backtest, checker, alerts, promo
dashboard/app.py       Streamlit control room: Today, Model, Backtest, Bankroll and Check a bet tabs
ops/launchd/           launchd job templates for the Mac (the picks role): daily, odds, close
ops/windows/           Task Scheduler registration script for Windows: -Role props or -Role picks
pipeline.py            Master command-line entry point
tests/                 Test suite; conftest.py keeps it off the live database; fixtures/ holds trimmed API responses
PROJECT_CONTEXT.md     Locked decisions, status, and lessons learned
docs/                  Phase results, odds audit, execution research, design documents
docs/PHASE2_PLAN.md    The Phase 2 build plan, kept for reference
docs/research/         Research notes: data feeds, the totals model, props feasibility
```

Further reading:

- [docs/phase2_results.md](docs/phase2_results.md) and [docs/phase3_results.md](docs/phase3_results.md): model results and what they mean
- [docs/historical_odds.md](docs/historical_odds.md): where the historical lines come from and their quirks
- [features/README.md](features/README.md) and [models/README.md](models/README.md): layer notes. `models/README.md` predates Phase 3 and still lists LightGBM with isotonic calibration as next; the shipped model uses temperature scaling
- `docs/File1`–`File3` and the status report (.docx): the original repository research and system design

PROJECT_CONTEXT §10, "How to Resume Work", is out of date. Use the Quick start here instead.

## Stack

Python 3.11+, PostgreSQL 16 in Docker, SQLAlchemy 2 with the psycopg2 driver, pandas, LightGBM, scikit-learn, matplotlib, and Streamlit. The `ml` extra also installs XGBoost and Optuna, and the `dashboard` extra installs Plotly; no module imports any of them yet.

## Licensing

The repo has no license file. `nhl-api-py`, the main data dependency, was GPL-3.0 through version 3.1.x and has been Apache-2.0 since 3.2.0; a fresh install gets 3.3.0. `pyproject.toml` requires 3.2.0 or later, so every install gets an Apache-2.0 version. MoneyPuck data is free for non-commercial use only and must be credited wherever it is shown. Methods borrowed from the reviewed open-source models were reimplemented from scratch rather than copied. Get a legal review before redistributing anything.
