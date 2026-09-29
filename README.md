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
- **Void**: a bet the book cancels and refunds in full, usually because the game was postponed or cancelled.

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Data foundation: PostgreSQL schema; NHL API, MoneyPuck, and Odds API ingestion; pipeline commands | Done: 6 seasons, 7,945 games, 683k shots |
| 2 | Feature store (team form, goalie quality, rest and travel, Elo ratings) and a baseline model | Done: walk-forward log loss 0.6829, under its 0.69 gate |
| 3 | Moneyline model, betting engine, backtest, dashboard, daily recommendations, totals model, bet checker, arbitrage and middle alerts | Done. The totals model failed its gate: it beat neither a simple scoring-environment baseline nor the market's over/under line. Totals betting is off |
| 4 | Live-season operations | In progress. Done: paper settlement, the CLV ledger, confirmed starters, locked picks with a pre-game closing snapshot. Remaining: cloud migration, player props, a daily recommendation digest, an Odds API historical backfill |
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

A code review on 2026-09-28 found problems with odds matching, paper-trading CLV, the season setting, the install, and the Odds API key in the logs. Those are fixed. A second pass the same day fixed the season filter for September loads, postponed and cancelled games, the daily cap under overlapping runs, consensus CLV, the test suite reaching the live database, `--help` running jobs, the status count, and snapshots on days without games, and added the Windows setup. A third pass fixed the test opt-in that could still reach the live database, the close job's credit use on the free plan, console windows popping up from the Windows tasks, picks made a day ahead being voided, postponed games never getting a new pick, an empty backfill list, and the 2019-20 bubble playoffs, which no load stored. These are still open:

1. **The bet checker's tests need a database.** Without one, the 9 `TestLegMath` and `TestParlayMath` tests in `tests/test_checker.py` fail with a connection error, because the checker opens a connection even when its lookups are patched.
2. **The 2024-25 historical odds have no loader.** ESPN no longer serves that season's lines, and the Kaggle mirror that filled them on the live copy was loaded by hand. On a fresh install, 2024-25 games have no market line. See [docs/historical_odds.md](docs/historical_odds.md).
3. **The MoneyPuck download comes from a mirror.** `ingestion/moneypuck.py` fetches the shots file from peter-tanner.com, not moneypuck.com, and how often that mirror updates during the season hasn't been checked. If it falls behind, the daily refresh reloads the same shots without an error, and shot-based features (expected goals, shot-attempt shares, goalie goals saved above expected) stop updating.
4. **Logs written before 2026-09-28 may contain the Odds API key.** A failed request used to log the full request URL, key included. Delete those logs, or rotate the key.

## How it works

Five layers, each replaceable without touching the others:

```
DATA        ingestion/   ->  raw.*        NHL API, MoneyPuck shots, odds, starting goalies
FEATURES    features/    ->  features.*   point-in-time team, goalie, schedule, and Elo features
MODELS      models/      ->  models.*     calibrated probabilities
STRATEGY    betting/     ->  betting.*    edges, stakes, paper bets, CLV
INTERFACE   dashboard/                    Streamlit control room
```

### Data sources

| Source | What it provides | Access |
|---|---|---|
| NHL API (`api-web.nhle.com`, through `nhl-api-py`) | Teams, schedule, results, box scores, player and team game stats | Free, undocumented, can change without notice |
| MoneyPuck | Every unblocked shot attempt since 2007-08, with expected goals (xG, the chance a given shot becomes a goal) | Free zipped CSVs for non-commercial use, downloaded here from a mirror; credit MoneyPuck.com wherever its data is shown |
| The Odds API | Moneyline, puck line, and totals prices from about 20 US-facing books | API key; the free plan is 500 credits a month |
| ESPN summary API | One reference line per past game, used for the market feature and backtests. The provider varies by season | Free, no key, undocumented |
| Daily Faceoff | Projected starting goalies, each tagged Confirmed or a softer status | Scraped from the page's embedded data, one request per run; can break without notice |

### Models

- **Moneyline (`lgbm_market`).** A LightGBM model, a gradient-boosted decision-tree library, that starts from the market's own probability and learns corrections to it instead of rediscovering the market from scratch. Games without a line fall back to a market-blind version. Temperature scaling calibrates the output. The recommendation job retrains it from the database on every run.
- **Totals (`poisson_totals`).** Predicts each side's regulation goals, then combines the two into a total-goals distribution that can price any over/under line. It failed its gate, so its predictions are stored for the bet checker and the alerts but never bet.
- **Baseline (`baseline_logreg`).** Logistic regression kept as a check that the features carry real signal without look-ahead.

### The daily chain

`python pipeline.py daily` runs these steps in order. Steps marked non-fatal log an error and let the chain continue. An exception in any other step ends the run, and that day gets no settlement, starters, or recommendations.

1. Waits up to 3 minutes for the network, because scheduled jobs can fire the moment a laptop wakes. If the network is still down, the run stops here.
2. Refreshes teams, the schedule from 3 days back to 7 days ahead of today, box-score game logs, and team game stats. The schedule refresh stores each game's start time and schedule state (on schedule, postponed, suspended, or cancelled) and moves a postponed game to its new date.
3. Takes a full odds snapshot (6 credits). When no game starts in the next 24 hours it skips the request and costs nothing.
4. Tops up ESPN reference lines for newly finished games (non-fatal).
5. Refreshes MoneyPuck shots for `CURRENT_SEASON` (non-fatal). It downloads a fresh file when the cached one is missing or more than 20 hours old, then reloads the season's shots. It does nothing until the season has a finished game.
6. Rebuilds features for `CURRENT_SEASON`.
7. Settles finished paper bets and rebuilds the bankroll and CLV ledger (non-fatal).
8. Pulls starting goalies from Daily Faceoff (non-fatal).
9. Scores the day's slate and writes recommendations for games that don't have one yet (non-fatal).

"Today" is the local date: in `LOCAL_TIMEZONE` when that is set, otherwise in the machine's time zone. `CURRENT_SEASON` is the season containing that date (see [Configuration](#configuration)).

`python pipeline.py odds` is the midday run: a full odds snapshot, starting goalies, recommendations for games that still have no pick, then an arbitrage and middle scan. `python pipeline.py close` takes a moneyline-only snapshot (2 credits) and does nothing else. It supplies the closing price that settlement grades each pick against. `python pipeline.py close --due`, the scheduled form, takes that snapshot only when a game starts within 16 minutes and no moneyline snapshot is less than 16 minutes old, and otherwise exits at no cost. Run every 15 minutes, that is one close per start time, in the last run before puck drop. [Snapshot schedule](#snapshot-schedule) says when to run each one.

### Betting rules

- A bet needs the model's win probability to beat the market's fair probability by at least the threshold, measured in percentage points: 2.5 by default, so a 55% model price against a 52% market qualifies.
- The fair probability comes from each book's latest price with its margin removed, then the median across every book. The bet is priced at whichever book pays best, and it must still be +EV at that price. `BETTABLE_BOOKS` limits that best price to the books you can use. When it is unset, the best price can come from any book in the `us` and `us2` regions, including offshore books such as Bovada and BetOnline and sweepstakes books such as Fliff.
- The stake is a quarter of full Kelly, capped at 2% of the bankroll per bet and 10% per day. The daily cap counts the stakes of picks already issued for that date, except picks marked `SKIPPED` and voided picks. When it is reached, the weakest remaining edges are skipped. The cap is checked again when the picks are written, under a database lock, so two runs that overlap can't together go past 10%. Stakes are sized from the fixed `BANKROLL` setting and don't compound with paper profit and loss.
- Only games that haven't started are scored. A game the NHL API marks as live, postponed, suspended or cancelled, or whose start time has passed, is left out.
- Each game gets at most one moneyline pick. The pick is locked at the book, price, and snapshot time it was issued at, and later runs never re-price or delete it. They only add picks for games that still have none, within what is left of the day's budget. A pick marked `SKIPPED` by hand releases its stake from the budget but still blocks a new pick for that game. A pick settled `VOID` doesn't block one: when a postponed game is played on its new date, it can get a new pick at the new date's prices.
- A pick still pending when its game finishes is settled as a paper bet at its locked book and price. The closing price is the same book's last snapshot taken after the pick and before puck drop. If that book has no such snapshot, CLV compares the median no-vig probability of every book's last quote in the same window with the no-vig fair probability the pick was issued at, so neither side of the comparison carries the book's margin. If no snapshot qualifies, CLV is left blank, not set to zero.
- **The void rule.** A pick is settled `VOID` instead, the way a book would void the bet, when its game is postponed or cancelled, or when the game finally starts more than 3 hours earlier or later than the start time it had when the pick was written (a postponed game played on its new date). Each pick stores that start time (`scheduled_start`), so a pick made a day ahead with `betting.recommend --date` is not voided when its game starts on time. A pick written before that column existed has no stored start, and falls back to the old rule: void when the game starts more than 36 hours after the pick was priced. A void has no profit or loss and no CLV. It isn't counted as a bet, a stake, a win or a loss in the bankroll ledger or the CLV report, and it releases its stake from the day's budget.

The edge and staking rules live in [betting/engine.py](betting/engine.py) and are unit-tested, and the live job and the backtest both call them. The daily cap is applied in `betting/recommend.py` and `betting/backtest.py`, and settlement is in `betting/settle.py`. The bet checker measures against the offered price with the margin left in, so it reads lower than the daily job for the same bet.

The code departs from PROJECT_CONTEXT §7 in five places, and the code is what runs. It removes the margin by proportional rescaling, not the power method. It accepts quotes up to 18 hours old, not 5 minutes. It has no limit of 3 correlated bets per game. The totals and props thresholds are unused, because neither market is bet. And when a pick's own book has no closing quote, CLV compares two no-vig probabilities instead of §7's two implied ones, because the consensus close has no single price with a margin in it.

## What this deliberately does not do

- **No real money yet.** Real stakes wait for 500+ paper bets with average CLV above 1 percentage point and a significance test on the claimed edge. A full backtest season produced 363 bets at a 2.5-point threshold and only 46 in the 6–9 point band, so at a higher threshold that gate is several seasons away.
- **No automatic bet placement.** Recommendations are for a person to act on. The dashboard is read-only; there is no approve, skip, or "I placed this" step yet.
- **No totals betting** until the totals model passes its gate.
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
python -m ingestion.espn_odds               # free historical reference lines; resumable
python pipeline.py features                 # every season
python -m models.lgbm                       # walk-forward evaluation; registers the moneyline model
python -m models.totals                     # registers the totals model
python pipeline.py status
```

- **The venue seed** adds arena coordinates and time zones, which the travel features need. It applies `db/seed_venues.sql` through Python, read as UTF-8, so it is one command on every system and "Montréal" arrives intact. It is safe to run before the backfill and safe to re-run. `python pipeline.py setup` already runs it when no team has coordinates yet, so on a new database this line only confirms it.
- **Run `python -m models.lgbm` before the first recommendation job.** The job retrains its model every run but can only save predictions once `lgbm_market` has a registry row. Without it, the job scores the slate, fails when it saves, logs a non-fatal error, and writes nothing. `python -m models.totals` is needed only for the totals predictions; without it, moneyline picks are still written.
- **ESPN no longer serves 2024-25 lines.** The original setup filled that season from a Kaggle mirror by hand, and there is no loader for it in the repo. See [docs/historical_odds.md](docs/historical_odds.md).

### Every day

On the live Mac these runs are automatic (see [Scheduling](#scheduling)). By hand, on a new machine or to catch up:

```bash
python pipeline.py daily                    # morning: picks are issued at this snapshot's prices
python pipeline.py odds                     # optional, midday: confirmed starters and alerts
python pipeline.py close --due              # every 15 minutes: a close only when a game is about to start
python pipeline.py close                    # by hand: a closing snapshot right now (2 credits)
streamlit run dashboard/app.py              # the control room
```

### Before each season

The season rolls over on July 1 without any change to the code.

1. Run `python pipeline.py setup`. Check that it prints the new season, your time zone, and today's date, and that the Odds API key is OK. If `NHL_SEASON` is still set in `.env` from a playoff run that went past July 1, remove it.
2. Build the new season's features now with `python pipeline.py features --season 20262027`, or let the next `daily` run do it.
3. Check that the scheduled jobs are loaded: `daily`, `close`, and `odds` if you use it (see [Scheduling](#scheduling)).
4. Decide `EDGE_MIN_ML` for the season (see Status).
5. Before using the promo calculator, re-check the Kalshi and Polymarket fees hard-coded in `betting/promo.py`.

### Upgrading an existing database

`db/schema.sql` runs only when the Docker volume is first created, so an older database can be missing up to five columns: `raw.games.start_time_utc` (puck drop), `raw.games.schedule_state` (postponed, suspended, or cancelled), `betting.recommendations.priced_at` (when a pick's price was captured), `betting.recommendations.scheduled_start` (the game's start time when the pick was written), and, on an older database where settlement has never run, `betting.placed_bets.is_paper`. Nothing needs doing by hand. The next pipeline command adds the missing columns, and so do the dashboard and the module commands that use them. No data is moved. `python -m config.migrate` does the same on demand.

Games already stored have no start time or schedule state until their schedule is loaded again. The next `daily` run fills the window from 3 days back to 7 days ahead. To fill a whole season at once, run `python -m ingestion.nhl_api season 20262027` (the current season). Older seasons can stay blank. An odds snapshot matches a game with no start time by its Eastern date instead, and settlement logs a warning that the game's closing window has no puck-drop limit.

Picks written before the upgrade have no `priced_at`, so their closing window has no lower limit, and the snapshot they were priced from can still grade them at zero CLV. They have no `scheduled_start` either, so the void rule checks them with the old 36-hour limit (see [Betting rules](#betting-rules)).

On a machine that already runs the pipeline, also reinstall with `python -m pip install -e ".[ml,dashboard,dev]"` to pick up the new dependencies (`tzdata`, `matplotlib`, `nhl-api-py` 3.2.0 or later, and LightGBM held below 5), install the `close` job that runs `close --due` every 15 minutes in place of any fixed-time close (see [Scheduling](#scheduling)), and set `LOCAL_TIMEZONE`, plus `BETTABLE_BOOKS` if wanted, in `.env`.

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

`betting.bankroll_log` holds one row per settled day and is rebuilt from `placed_bets` on every settle run. `models.predictions` holds every scored game, including the totals distributions. Arbitrage and middle alerts appear only as warning lines in the run log, plus an optional macOS notification. They aren't stored or shown on the dashboard.

## Commands

**Pipeline** (`python pipeline.py <command>`):

| Command | What it does |
|---|---|
| `setup` | Checks the database connection (adding any missing columns), `nhlpy`, and the Odds API key, applies the venue seed when no team has coordinates yet, and prints the season, the time zone, and today's local date |
| `status` | Row counts for the main tables, including upcoming games (any not yet final), and games per season |
| `backfill` | Loads every season from `BACKFILL_FIRST_SEASON` through the current one from the NHL API and MoneyPuck |
| `features [--season YYYYYYYY]` | Builds the feature store for one season or all |
| `daily` | The full daily chain described above |
| `odds` | Full odds snapshot (6 credits), starters, recommendations for games without a pick, and alerts |
| `close [--due]` | Moneyline-only odds snapshot (2 credits) for the closing price. No picks, no alerts. With `--due`, only when a game starts within 16 minutes and no moneyline snapshot is under 16 minutes old; otherwise it logs why and exits at no cost |
| `recommend` | Scores today's slate into `betting.recommendations` |
| `starters` | Starting goalies from Daily Faceoff |
| `settle` | Settles paper bets and rebuilds the bankroll and CLV ledger |

**Module tools** (`python -m <module>`):

| Module | Use |
|---|---|
| `betting.checker --leg "MTL@BUF ml away -125" --leg "SJS@WSH total over 6.5 -110"` | Is this bet or parlay +EV? Uses stored model probabilities; add `--date`, `--price` for a boosted parlay, `--bankroll` |
| `betting.recommend --date 2026-01-15 --simulate --dry-run` | Replays a past slate as if it were upcoming, writing nothing. Also takes `--bankroll` and `--edge-min` |
| `betting.settle --report` | CLV and ROI report, with a count of settled bets that have a CLV |
| `betting.backtest` | Payout backtest on DraftKings-era prices. Uses the 2.5 threshold in `betting/engine.py`. Redraws `models/artifacts/lgbm_calibration.png` |
| `betting.alerts` | Arbitrage and middle scan over the freshest quotes; results go to the log |
| `betting.promo free_bet --amount 100 --bonus-odds 400 --hedge-venue kalshi --contract-price 0.80` | Hedge stakes for a sportsbook promo, per bettor. `--contract-price` is the price of the contract that pays if the bonus leg loses. Details: [docs/promo_hedging_calculator.md](docs/promo_hedging_calculator.md) |
| `models.baseline`, `models.lgbm`, `models.totals` | Walk-forward evaluation and registry entry for each model. `models.baseline` and `models.lgbm` redraw their calibration plots in `models/artifacts/` (`baseline_calibration.png`, `lgbm_calibration.png`, both tracked by git); `models.totals` draws none |
| `ingestion.nhl_api teams\|daily\|season <YYYYYYYY>\|backfill-all` | NHL API loads, piece by piece. `season` also fills in start times for that whole season |
| `ingestion.moneypuck <start_year> [--download]` | One season of MoneyPuck shots, such as `2025` for 2025-26. `--download` fetches a fresh copy; without it the CSV must already be in `DATA_DIR` |
| `ingestion.espn_odds [season]` | Historical reference lines; all seasons when no season is given |
| `ingestion.dailyfaceoff [--date YYYY-MM-DD]` | Starters for one date |
| `ingestion.odds_api [--markets h2h,spreads,totals]` | One odds snapshot: 6 credits for the default markets, 2 for `h2h`. Skipped, at no cost, when no game starts in the next 24 hours |
| `config.migrate [--seed-venues]` | Adds any columns a database made from an older `db/schema.sql` is missing. Pipeline commands do this on their own. `--seed-venues` also applies `db/seed_venues.sql` |

`--help` prints the options and runs nothing for `betting.checker`, `betting.recommend`, `betting.settle`, `betting.alerts`, `betting.promo`, `ingestion.odds_api`, `ingestion.espn_odds`, `ingestion.dailyfaceoff`, and `config.migrate`, so it never spends credits or writes to the database. `models.baseline`, `models.lgbm`, `models.totals`, and `betting.backtest` take no options and start their full run whatever you pass them.

## Configuration

Settings come from `.env`. `.env.example` sets the first four rows and lists the next four commented out. The last eight are optional and not in `.env.example`.

| Variable | Default | Purpose |
|---|---|---|
| `POSTGRES_HOST`, `_PORT`, `_DB`, `_USER`, `_PASSWORD` | `localhost`, `5432`, `nhl_betting`, `nhl`, `nhl_dev_2026` | Database connection. Only the password also reaches the Docker container; `docker-compose.yml` fixes the database name, user, and port, so change those in both files |
| `ODDS_API_KEY` | none | Live odds snapshots. Each machine has its own `.env`, so each can use its own key and its own monthly credits |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `DATA_DIR` | `<repo>/data`; `.env.example` sets `./data` | Cache for MoneyPuck CSV downloads. A relative path is relative to the repo, whatever folder a command starts in |
| `LOCAL_TIMEZONE` | this machine's time zone | Your time zone as an IANA name, such as `America/Chicago`. It decides "today": the slate date, the schedule refresh window, and the season. The dashboard shows start times in it. Set it wherever the clock runs on UTC, such as a container or a cloud server. An unknown name, or a region folder such as `America` on its own, logs a warning and falls back to the machine's zone |
| `BETTABLE_BOOKS` | unset: every book | Comma-separated Odds API bookmaker keys you can bet at, such as `draftkings,fanduel`; case doesn't matter. Only these books can supply a pick's price, while the fair probability still uses every book. Each recommendation run logs which books it used |
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

A full snapshot (`daily`, `odds`, or `python -m ingestion.odds_api`) requests three markets (moneyline, puck line, totals) across two regions, and The Odds API bills markets times regions, so it costs 6 credits. A `close` snapshot requests the moneyline only and costs 2.

A request is free only when the API lists no NHL events at all, which in practice means the off-season. In season it also lists the coming days' games, so a request on a day without games still costs full price. So every snapshot command first checks `raw.games`: when no game starts in the next 24 hours, it logs why and makes no request. That keeps off-days free as long as the schedule is current, which the daily run takes care of.

Each machine reads its own `.env`, so a Mac and a PC can each use their own key, each with its own 500 free credits a month. Machines that share one key share its 500 credits.

| Plan | Credits a month, per key | What that allows in season |
|---|---|---|
| Free | 500 | The daily run plus `close --due` at its defaults: at most about 456 in any month of the 2026-27 schedule. The midday `odds` run (about 186 more in a full month) doesn't fit |
| 20K ($30 a month) | 20,000 | The whole [snapshot schedule](#snapshot-schedule), plus an `odds` run every 15 minutes through each game day |

After each request the log shows the credits remaining, the credits used, and what that call cost. The log never shows the API key. Adding a third region, such as `us_ex` for Kalshi, would raise a full snapshot to 9 credits and a close to 3.

## Snapshot schedule

A pick is locked at the price of the snapshot it was issued from, so grading it needs a later snapshot taken before its puck drop. The daily and midday times are defaults for a machine on Central time. The close job has no times: it reads each game's start time from the database.

| Run | When | Credits | What it is for |
|---|---|---|---|
| `daily` | Morning, 9:00 | 6, or 0 when no game starts in the next 24 hours | Refreshes everything and issues the day's picks at this snapshot's prices |
| `odds` (optional) | Midday, 13:00, once most starting goalies are confirmed | 6, or 0 the same way | Confirmed starters, picks for games that still have none, and arbitrage and middle alerts. It never changes a pick already issued |
| `close --due` | Every 15 minutes | 2 when due, otherwise 0 | Moneyline only. It snapshots when a game starts within 16 minutes and no moneyline snapshot is under 16 minutes old, so every start time, afternoon games included, gets one close, in the last run before its puck drop |

Why 16 and 16: any 16 minutes hold one of the 15-minute runs, so every start time has a run in its last 16 minutes, and the next run, 15 minutes later, finds a snapshot less than 16 minutes old and skips. The extra minute absorbs a run that starts a little late. When two start times are less than 16 minutes apart (22 times in 2026-27), the later game can share the earlier one's close, taken up to about 30 minutes before its puck drop.

What the close job costs depends on how many different start times each slate has. Replaying it against the 2026-27 schedule (1,344 regular-season games over 185 game days, 4.3 different start times a game day on average), with a run every 15 minutes, for each of the 15 minutes the cycle could start on:

| Setting | Closes a game day | Daily run plus closes, per calendar month (UTC) | Close before puck drop, median | Games with a close in their last 16 minutes |
|---|---|---|---|---|
| Default: `CLOSE_LEAD_MINUTES=16`, `CLOSE_MIN_GAP_MINUTES=16` | 4.2 (about 8 credits) | Oct 444, Nov 392, Dec 408, Jan 444, Feb 336, Mar 454 on average; never more than 456 | 9 minutes | 98.7%; every game has one in its last 30 |
| The old default, `CLOSE_LEAD_MINUTES=40`, `CLOSE_MIN_GAP_MINUTES=25` | 6.4 (about 13 credits) | Oct 571, Nov 507, Dec 537, Jan 587, Feb 439, Mar 583 on average; up to 658 | 8 minutes | 66%; every game has one in its last 30 |

The daily run is 6 credits a game day, about 186 in a month with a game every day; the rest is the close job. At the defaults the two stay under the free plan's 500 in every month, whatever minute the cycle starts on. Adding the midday `odds` run goes over, so leave it out on the free plan, and watch the credits-remaining figure in the log. When the credits run out, every snapshot fails for the rest of the month, including the morning one the picks come from. The old fixed-time closes (17:30 and 20:30) used about 120 credits a month but missed afternoon games.

- Each game's close is the last snapshot before its own puck drop. A close that runs after a game has started skips that game.
- `python pipeline.py close` without `--due` takes a snapshot right away, for 2 credits.
- A game with no snapshot between its pick and its puck drop settles with a blank CLV. A Mac that sleeps through the evening takes no closes, because launchd skips interval runs while the Mac is asleep.

## Scheduling

The live copy runs on macOS under launchd. Don't add cron entries as well, because the crontab is intentionally empty.

Templates for three agents are in [ops/launchd/](ops/launchd/), with install steps in its README: `com.nhlbetting.daily`, `com.nhlbetting.close` (`close --due` every 15 minutes), and the optional `com.nhlbetting.odds`. Each job logs to the repo's `logs/` folder, which git ignores. The live Mac already has `com.nhlbetting.daily` and `com.nhlbetting.odds` in `~/Library/LaunchAgents`, with their own times and log paths. Nothing replaces those files automatically: add the `close` agent by hand (unloading any earlier fixed-time close first), and compare the other two with the templates before replacing them.

- **Check the agents** with `launchctl list | grep com.nhlbetting`. The middle column is the last exit code.
- **Docker Desktop must be running**, or every run fails at the database check.
- **A Mac that was asleep** at the daily or midday time runs that job once when it wakes. The run waits up to 3 minutes for the network, and a late close skips the games already under way.
- **Pause them for the off-season** with `launchctl unload ~/Library/LaunchAgents/com.nhlbetting.odds.plist`, and the same for `.daily` and `.close`.

**On Windows**, [ops/windows/register-tasks.ps1](ops/windows/) registers the same three jobs in Task Scheduler, the scheduler built into Windows: `daily` at 9:00, `odds` at 13:00 with `-IncludeOdds`, and `close --due` every 15 minutes. Each starts in the repo folder, appends to `logs\<task>.log`, and runs hidden through `conhost.exe --headless`, so no console window pops up (Windows 10 21H2 or later, or Windows 11; `-VisibleConsole` for older Windows). It takes `-RepoPath`, `-PythonPath`, and `-Unregister`; its README has the details.

## Tests

```bash
pytest
```

**A plain `pytest` never touches a database.** `tests/conftest.py` runs before any test imports the settings. Unless you opt in, it points the database settings at a closed port, so every database test skips, whatever `.env` names. A line at the top of the output says which way it went, and why. Opt in only for a disposable database, one of two ways:

- **Its name ends in `_test`**, set as `POSTGRES_DB` in the environment or in `.env`.
- **`NHL_ALLOW_DB_TESTS=1` together with `POSTGRES_HOST`, `POSTGRES_PORT` and `POSTGRES_DB`**, all three set in the environment for that run, pointing at the copy. The flag alone does nothing, and so do overrides that name the same database as `.env` (`localhost` and `127.0.0.1` count as the same server): the tests still skip, and the line at the top says why. So the flag on its own can never send the suite to the database `.env` names.

The suite has 328 tests. Without a database, 259 pass, 60 skip, and 9 fail, in about a minute. The 9 failures are the `TestLegMath` and `TestParlayMath` tests in `tests/test_checker.py` (see [Known issues](#known-issues)). Ten files need no database at all: `test_betting_engine`, `test_promo`, `test_setup`, `test_feature_utils`, `test_espn_odds`, `test_odds_api`, `test_moneypuck`, `test_pipeline`, `test_migrate`, and `test_db_guard`; most other files have pure tests too.

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

With a database, all 328 pass against a copy of the live data, in about a minute. What the database tests write, all in the database they run against:

- **`test_settle`** adds synthetic picks (books `testbook`, `otherbook`, `voidbook`) and odds snapshots on 2020-21 games, then runs settlement, which settles or voids every due pick in that database, not only its own. It rebuilds `betting.bankroll_log` afterwards. It temporarily changes a few 2020-21 games' start times, and one game's schedule state, game state and score (to a postponed game that hasn't been played), and puts them back.
- **`test_recommend`** issues and deletes picks for 2026-01-15 games, deleting only the ones it created. The end-to-end test overwrites the moneyline and totals predictions of the 2026-01-15 games and deletes only the rows it added. One test temporarily marks a 2026-01-15 game as upcoming and postponed, then restores it.
- **Feature tests** (`test_build_all`, `test_build_vectors`, `test_team_features`, `test_goalie_features`, `test_schedule_features`, `test_elo`) rebuild `features.*` tables: 2020-21 and 2021-22 rolling features and matchups, every game vector, and every Elo rating.
- **`test_baseline`** re-registers `baseline_logreg` in `models.model_registry`.
- **`test_nhl_api`**, **`test_checker`**, and **`test_dailyfaceoff`** insert synthetic rows and delete them: games with ids 9999020001 to 9999020004 (the last three in January 2031), predictions, and a 2026-01-15 starting goalie (replacing, then deleting, any real row for that team and date).

`test_baseline` and `test_lgbm` also redraw `models/artifacts/baseline_calibration.png` and `lgbm_calibration.png`, which git tracks: restore them with `git checkout -- models/artifacts/` unless the model changed. Don't run the database tests while a scheduled job is writing to the same database.

## Repository layout

```
config/settings.py     Settings from .env, database engine, local date and season
config/migrate.py      Adds the columns an older database is missing; applies the venue seed
db/                    schema.sql (applied by Docker on first start), seed_venues.sql
ingestion/             nhl_api, moneypuck, odds_api, espn_odds, dailyfaceoff
features/              team, goalie, schedule, and Elo builders; build_all orchestrates
models/                baseline, lgbm (moneyline), totals; artifacts/ holds calibration plots
betting/               engine, recommend, settle, backtest, checker, alerts, promo
dashboard/app.py       Streamlit control room: Today, Model, Backtest, Bankroll tabs
ops/launchd/           launchd job templates for the Mac: daily, odds, close
ops/windows/           Task Scheduler registration script for Windows: the same three jobs
pipeline.py            Master command-line entry point
tests/                 Test suite; conftest.py keeps it off the live database
PROJECT_CONTEXT.md     Locked decisions, status, and lessons learned
docs/                  Phase results, odds audit, execution research, design documents
docs/PHASE2_PLAN.md    The Phase 2 build plan, kept for reference
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
