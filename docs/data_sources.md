# Data sources: what we use, what it costs, and what is still on the table

> Checked 2026-10-04 with live requests to every free endpoint named here, the installed `nhlpy` 3.3.0, the
> live database on the Windows PC (read-only queries), and the providers' own docs. Nothing in this survey
> spent an Odds API credit. Where a claim rests on one sample, it says so.

This page lists every place the system gets data from. For each one it says:

- what the source gives us
- what it costs
- which code loads it, and which database tables it fills
- what we use it for
- its pros and cons
- how far back it goes
- what it offers that we **don't** use yet, and what that would likely be worth

Terms used throughout (each is explained once, here):

- **Moneyline** → a bet on who wins the game, overtime and shootout included.
- **Puck line** → hockey's handicap bet: the favourite must win by 2+ goals (−1.5), the underdog can lose by 1 (+1.5).
- **Total / over-under (O/U)** → a bet on whether the two teams' combined goals go over or under a number such as 6.5.
- **Player prop** → a bet on one player's own numbers, e.g. "over 2.5 shots on goal".
- **Price / odds** → what a bet pays. American odds: −120 means risk 120 to win 100; +110 means risk 100 to win 110.
- **Opening line** → the first price a sportsbook posts. **Closing line** → the last price before the game starts.
- **CLV (closing-line value)** → whether the price we took was better than the closing price. Consistently beating the close is the best early sign that a betting model is really good.
- **Vig (margin)** → the sportsbook's built-in fee. It is why both sides of a bet add up to more than 100%.
- **No-vig price** → the sportsbook's price with the fee taken out: the market's honest guess at the probability.
- **xG (expected goals)** → how likely a shot was to go in, judged from where and how it was taken. A team's xG adds these up, so it measures chance quality, not luck.
- **API** → a web address that a program (not a person) asks for data and gets it back in a structured format.
- **Point-in-time** → what was known *before* the game. A feature that uses anything learned later is **leakage**, and it makes a model look better on paper than it can ever be in real betting.
- **Credit** → The Odds API's billing unit. The PC's key has 20,000 a month.

---

## 1. Summary table

| # | Source | What it gives us | Cost | Code → tables | Used for | History |
|---|---|---|---|---|---|---|
| 1 | **NHL web API** (`api-web.nhle.com`, through the `nhlpy` package) | Schedule, results, boxscores (every skater and goalie line), team game stats | Free, no key | `ingestion/nhl_api.py` → `raw.games`, `raw.teams`, `raw.players`, `raw.skater_games`, `raw.goalie_games`, `raw.team_games` | The backbone: every game, every result, who played, who started in goal | Decades; we load 2020-21 onward (9,289 games) |
| 2 | **NHL stats REST** (`api.nhle.com/stats/rest`) | Per-game power-play and penalty-kill ice time, power-play goals and assists, faceoffs | Free, no key | `ingestion/nhl_stats.py` → fills columns in `raw.skater_games` | Power-play features for the win/loss and totals models; props inputs | Decades (per game) |
| 3 | **MoneyPuck** | Every shot with its location, type and MoneyPuck's xG | Free for non-commercial use; credit MoneyPuck | `ingestion/moneypuck.py` → `raw.shots` (683,721 shots) | Team chance-quality features (xG for and against) and goalie quality (goals saved above expected) | 2007-08 onward |
| 4 | **The Odds API, live** | Prices from 10 named books, including Kalshi, Polymarket and Pinnacle: moneyline, puck line, totals; player props per game | Paid key, 20K credits/month (~$30). Full snapshot 3 credits, close 1, props 1 per game per market | `ingestion/odds_api.py` → `raw.odds_snapshots`; `ingestion/props_odds.py` → `raw.prop_snapshots` | The prices picks are made at, line shopping (finding the best price across books), and the close for CLV | Only what we save ourselves |
| 5 | **The Odds API, historical** | Snapshots of every book's prices, every 5 minutes (10 before Sept 2022) | Paid plans only: 10 credits per market per region per snapshot; props 10 per market per game | Being added on the `data/odds-history` branch (not on `main` yet) | Real two-way prices and closes for past seasons, so backtests can use prices we could have bet | Game markets from 2020-06-06; props from 2023-05-03 |
| 6 | **ESPN summary (pickcenter)** | One book's opening and closing moneyline, puck line, total, and over/under prices | Free, no key | `ingestion/espn_odds.py` → `raw.historical_odds` | The market's price as a model input (the win/loss model starts from it), and the history for over/under backtests | 2020-21 → 2023-24 (Unibet), 2025-26 (DraftKings); **no 2024-25** |
| 7 | **ESPN core (props)** | Past player-prop prices (DraftKings, ESPN BET) | Free, no key | `ingestion/espn_props.py` → `raw.prop_odds_hist`, `raw.prop_odds_fetches` | The props model's market check (did it beat the prop prices?) | Scattered dates of 2025-26 and the 2026 playoffs (59,044 rows, 486 games) |
| 8 | **ESPN injuries** | Today's injury list: status, injury type, expected return date, news text | Free, no key | `ingestion/espn_injuries.py` → `raw.injuries` | Collected daily to build our own history. **Not yet read by any model or by the pick job** | None from ESPN: it is today-only, so our history starts the day we start saving |
| 9 | **Daily Faceoff, starting goalies** | Each day's projected and confirmed starting goalies | Free, scraped (the site blocks non-browser clients) | `ingestion/dailyfaceoff.py` → `raw.starting_goalies` | The pick job uses the confirmed starter instead of guessing (guessing was right 40% of the time on a test slate; with Daily Faceoff, 80%) | Today's page is reliable; old dated pages are **not** (see §9) |
| 10 | **Daily Faceoff, line combinations** (being added) | Each team's current forward lines, defence pairs, power-play units 1 and 2, penalty-kill units, injured reserve, goalies, with a last-updated time and source link | Free, scraped | Being added by another track | Who plays on the first power-play unit (big for points and shots props), late lineup changes | Current state only: must be saved daily to build history |
| 11 | **NHL free odds feed** (`partner-game` + `schedule`) | DraftKings (US) and FanDuel Canada prices for the next game day; moneylines from up to 7 partner books | Free, no key | `ingestion/nhl_odds.py` → `raw.nhl_feed_snapshots` (kept apart on purpose) | Comparison only, to learn whether it could supply free closing prices | None: games lose their odds once played |
| 12 | **Kaggle copy of ESPN's lines** (`jonathanncoletti/nhl-historical-game-data`) | 2024-25 favourite's moneyline, puck line and total | Free, loaded by hand; **no loader in the repo** | → `raw.historical_odds` with `provider = 'espn-kaggle-onesided'` (on the Mac's database; **missing on the PC's**) | Fills the 2024-25 hole in the market input | 2024-25 only |
| 13 | **NHL shift charts** (`api.nhle.com/stats/rest/en/shiftcharts`) | Every shift of every player in every game: who was on the ice together | Free, no key | `ingestion/nhl_shifts.py` → `raw.shifts`, fetch log `raw.shift_fetches` | Real lines and power-play units for every past game (no model reads them yet) | 2020-21 onward loaded (see 2.13) |
| 14 | **NHL right-rail `gameInfo`** (`api-web.nhle.com/v1/gamecenter/{id}/right-rail`) | Scratched players, referees and linesmen, head coaches | Free, no key | `ingestion/nhl_game_info.py` → `raw.game_scratches`, `raw.game_officials`, `raw.game_info` (also the fetch log) | Who was missing from each lineup; who officiated (no model reads them yet) | 2020-21 onward loaded (see 2.13) |
| 15 | **Kalshi public API** (`api.elections.kalshi.com/trade-api/v2`) | Every NHL game-winner market: settlement, hourly candles, 1-minute candles for the 3 hours before puck drop, with bid and ask | Free, no key for market data | `ingestion/kalshi.py` → `raw.kalshi_markets`, `raw.kalshi_candles` | Prices and closes at a venue a Texas bettor can use, for backtests and CLV | From the 2025 playoffs (2025-04-19) onward |

**Status on the PC's database, morning of 2026-10-04** (other tracks may be filling these today):
`raw.odds_snapshots`, `raw.prop_snapshots`, `raw.nhl_feed_snapshots`, `raw.injuries` and `raw.starting_goalies` are empty (no live jobs have run on this machine yet). Power-play ice time is 0 on all 285,941 skater rows (the `nhl_stats` backfill has not run here). `raw.historical_odds` has 5,990 rows and **no 2024-25 rows** (the hand-loaded Kaggle fill never reached this machine). `raw.shifts` and `raw.rosters` exist but nothing in the repo writes them. **Update 2026-10-05:** `raw.shifts` is now filled, and scratches, officials and Kalshi prices are loaded too (2.13).

---

## 2. Source by source

### 2.1 NHL web API (`api-web.nhle.com`)

- **What:** the league's own data service, the one NHL.com runs on. We reach it through the `nhlpy` package (installs as `nhl-api-py`, imports as `nhlpy`).
- **Cost:** free, no key. It is undocumented, so the league can change it without notice (it did in 2023, when the old `statsapi.web.nhl.com` died).
- **Code → tables:** `ingestion/nhl_api.py` → `raw.games`, `raw.teams`, `raw.players`, `raw.skater_games`, `raw.goalie_games`, and `raw.team_games` (from the game's `right-rail` page: hits, blocks, faceoffs, power-play chances and more).
- **Used for:** everything. Results label every model. Boxscores tell us who played, ice time, shots, and which goalie started (the `is_starter` flag).
- **Pros:** official, complete, fast, free, consistent IDs across every table.
- **Cons:** undocumented. It also has no injuries, and its game page lists goalies by games played, not by who starts tonight.
- **History:** many decades. We hold 2020-21 to now.

**Available but unused:**

| What | Where | Likely benefit |
|---|---|---|
| **Scratches** (players on the roster who did not dress) and **referees and linesmen** | The same `right-rail` page we already download for team stats (`gameInfo.awayTeam.scratches`, `gameInfo.referees`). Checked on games from 2020-21 and 2022-23 | **High for lineup modelling.** With the boxscore (who dressed), this rebuilds "who was missing" for every past game. That is a free stand-in for the historical injury lists that don't exist (see the survey). Referees: small. Some crews call more penalties, which nudges power plays, totals and power-play props. **Now loaded (2.13)** |
| **Shift charts** (every shift by every player) | `api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId=…` (705 shifts for one 2022-23 game) | **High.** It shows which players actually skated together (real lines and power-play units) for every past game. That gives line-level and player-level team strength, and the history needed to judge Daily Faceoff's projected lines. **Now loaded into `raw.shifts` (2.13)** |
| **Play-by-play** | `gamecenter/{id}/play-by-play` | Medium. Penalties, faceoffs and hits with times, plus goalie pulls. We get shots from MoneyPuck already; our own copy would remove that dependency |
| **NHL EDGE** (puck and player tracking: skating speed, shot speed, zone time, shot location, a goalie's last-10-game save percentage) | 23 `nhlpy` methods under `client.edge` (e.g. `team_zone_time_details`, `goalie_save_percentage_detail`) | **Low to medium, with a leakage trap.** EDGE returns **season totals to date**, not per game. A season-total pull for a past season includes games after the one being predicted, so it is leakage. Only daily snapshots saved from now on are safe. History starts 2021-22. Zone time (how long a team keeps the puck in the attacking zone) might add a little to team strength; most of it is already captured by xG |
| **Schedule odds** | `/v1/schedule/{date}`; read by `nhl_odds.py` (source 11) | See 2.10 |

### 2.2 NHL stats REST (`api.nhle.com/stats/rest`)

- **What:** the NHL.com statistics tables, filterable by game, season, team and player.
- **Cost:** free, no key, undocumented. `limit=-1` returns all rows; an explicit limit above 100 is silently cut to 100.
- **Code → tables:** `ingestion/nhl_stats.py` uses 3 skater reports (`timeonice`, `powerplay`, `faceoffwins`) to fill `pp_toi_seconds`, `sh_toi_seconds`, `pp_goals`, `pp_assists`, `fow`, `fol` in `raw.skater_games`. The boxscore load always left those at 0.
- **Used for:** power-play and penalty-kill strength (the features `pp_xgf_per60`, `pk_xga_per60`: chances created per 60 minutes of power play and allowed per 60 minutes of penalty kill). Until they are filled, those features are a meaningless constant. Props use power-play time too.
- **Pros:** one request returns a month of every player's games. It has many splits the boxscore lacks.
- **Cons:** undocumented, and the paging quirk above.
- **History:** decades, per game.

**Available but unused.** The live config (`/stats/rest/en/config`) lists 18 skater, 8 goalie and 24 team reports, almost all of them per game:

| Report | What it adds | Likely benefit |
|---|---|---|
| `team/powerplaytime`, `team/penalties`, `skater/penalties` | Power-play chances per game; penalties drawn and taken per 60 | **Medium** for totals and power-play props. How often a team draws and takes penalties decides how much power-play time there will be, and that is where scoring and shot volume jump |
| `team/goalsforbystrengthgoaliepull`, `team/goalsagainstbystrengthgoaliepull` | Goals scored at 6-on-5 and 5-on-6, i.e. with the goalie pulled (taken off for an extra attacker late in a game) | **Medium for the puck line and totals.** Empty-net goals decide many −1.5 puck-line bets and late overs. A team that pulls its goalie early, or gives up empty-netters, shifts both |
| `goalie/startedVsRelieved`, `goalie/savesByStrength`, `goalie/advanced` | Starts vs relief appearances, saves split by even strength and power play, quality starts | Medium for the goalie model and saves props. Even-strength save percentage is a steadier skill measure than overall save percentage |
| `skater/realtime` | Hits, blocks, giveaways, takeaways, shot attempts (including missed and blocked) per game | Medium for **blocked-shots and shot-attempt props**; small for game models |
| `skater/percentages`, `skater/puckPossessions`, `team/percentages` | Shot-attempt share at 5-on-5, split by score state (ahead, tied, behind), zone starts | Low to medium. Score-state splits help adjust for "score effects" (teams that are behind shoot more) |
| `skater/bios`, `goalie/daysrest` | Handedness, age, rest days between goalie starts | Low. Rest is already computed in `features/schedule_features.py` |

### 2.3 MoneyPuck

- **What:** a public hockey-analytics site that publishes every shot since 2007-08 with its own xG model (124 columns per shot).
- **Cost:** free for non-commercial use, and MoneyPuck must be credited. Scraping anything not on the data page needs their approval. The download host `peter-tanner.com` is MoneyPuck's official link, not a mirror.
- **Code → tables:** `ingestion/moneypuck.py` → `raw.shots`.
- **Used for:** xG for and against (chance quality) and goalie goals saved above expected (GSAx → goals a goalie stopped beyond what an average goalie would have, given the shots he faced). These feed `features/team_features.py` and `features/goalie_features.py`. MoneyPuck is **not** a betting model; it is an input.
- **Pros:** the best free shot-level data; updated nightly in season.
- **Cons:** one person's project (if it stops, our xG stops). Its xG model is theirs, so we can't change it. The non-commercial terms matter if this ever becomes a business.
- **History:** 2007-08 onward. We load 2020-21 onward.

**Available but unused** (all on [moneypuck.com/data.htm](https://moneypuck.com/data.htm)):

| File | What it adds | Likely benefit |
|---|---|---|
| **Game-by-game lines/pairs** (`seasonPlayersSummary/lines/{year}.zip`) | Every forward line's and defence pair's ice time and xG, per game | **Medium to high.** Line-level strength for lineup-aware models; checks Daily Faceoff's projected lines against what was played |
| **Game-by-game skaters and goalies** (`seasonPlayersSummary/{skaters,goalies}/{year}.zip`) | Per player per game: xG, shot attempts, ice time by situation, goalie xG against | **Medium.** A ready-made per-player history for player strength and props, without rebuilding it from shots |
| **All teams, game by game** (`careers/gameByGame/all_teams.csv`) | Team xG per game split by situation, already summed | Low; a cross-check of our own sums |
| Historical shots before 2020-21 | 13 more seasons | Low for game models (old seasons played differently) but **useful if we train our own xG model** (see the survey) |

### 2.4 The Odds API, live (`the-odds-api.com`)

- **What:** a paid service that gathers prices from dozens of sportsbooks and exchanges (venues where bettors trade with each other instead of against a bookmaker).
- **Cost:** credits. Cost = markets × regions (a region is a group of books billed together), and any 10 named books bill as one region. Our default list: kalshi, polymarket, pinnacle, draftkings, fanduel, betmgm, betrivers, espnbet, hardrockbet, novig. A full snapshot (3 markets) is 3 credits, a moneyline-only close is 1, and props are 1 credit per game per market per snapshot. The event list is free, and so are empty responses.
- **Code → tables:** `ingestion/odds_api.py` → `raw.odds_snapshots`; `ingestion/props_odds.py` → `raw.prop_snapshots`.
- **Used for:** the fair price (the median no-vig price across books), the best bettable price, the close for CLV, arbitrage and middle alerts, and live prop lines.
- **Pros:** many books in one call, including Kalshi and Polymarket (the two venues legal for a Texas bettor) and Pinnacle (the sharpest book: its prices are the hardest to beat, so they are the best yardstick). Prices are about 60 seconds fresh before games. Storing the data and training on it is allowed; reselling it is not.
- **Cons:** costs credits. Exchange prices are shown without trading fees (Kalshi's fee is up to about 1.75 cents a contract). Pinnacle's prices come from its public website "which may incur a delay".
- **History:** only what we save. That is why the live jobs must run every day.

**Available but unused:**

| What | Likely benefit |
|---|---|
| More snapshots per day (an hourly "line watch" in the 3 hours before each game) | **High.** The paid key can afford it (about 1 credit per snapshot for moneylines). It lets us learn when to bet: how prices move after goalie news, and whether to bet in the morning or wait |
| `alternate_totals`, `team_totals`, period markets (`h2h_p1`, `totals_p1`), `h2h_3_way` | Medium. More markets where our goal-distribution model could be priced. Period and team-total markets get less attention from the books |
| More prop markets (`player_points`, `player_goals`, `player_assists`, `player_total_saves`, `player_blocked_shots`, `player_power_play_points`) | **Medium to high.** Props are the least sharply priced markets (books limit bet sizes on them for that reason), and saves props follow directly from the goalie model |

### 2.5 The Odds API, historical

- **What:** past snapshots of the same feed: every 10 minutes from 2020-06-06, every 5 minutes from September 2022; props and other extra markets from 2023-05-03.
- **Cost:** paid plans only. 10 credits per market per region for each snapshot time; one call returns every NHL game listed at that moment. Props cost 10 credits per market per region **per game**.
- **Rough budget**, from the live schedule (about 850 distinct start times and about 225 game days per season). This assumes the 10-book rule also applies to history, which is not verified:
  - one closing snapshot per start time, moneyline only: about 8,500 credits a season
  - all 3 markets: about 25,500
  - adding one morning snapshot a day (3 markets): about 6,750 more
- **Code → tables:** being built on the `data/odds-history` branch; not on `main`.
- **Used for (once loaded):** the first **real two-way prices for 2020-21 to 2024-25** from many books. Those are prices we could actually have bet, which the ESPN Unibet lines are not (§2.6). That allows honest payout backtests, CLV backtests and opening-vs-closing studies across 5 seasons instead of 1. Pinnacle's historical close is the sharpest yardstick we can get.
- **Pros:** the only multi-book, timestamped source of past prices.
- **Cons:** expensive in bulk. Every pull must be costed first and never repeated (the CLAUDE.md credit rules).

### 2.6 ESPN summary (`site.api.espn.com/.../summary?event=`), the pickcenter block

- **What:** ESPN's game page data. Its `pickcenter` block has one book's opening and closing lines.
- **Cost:** free, no key.
- **Code → tables:** `ingestion/espn_odds.py` → `raw.historical_odds` (close and open moneylines, puck line, total, over/under prices).
- **Used for:** the market's view of each past game. The win/loss model (`models/lgbm.py`) **starts from the market's no-vig probability and learns a correction** (it is "boosted from the market"). That is why it beats a model that ignores the market. The DraftKings rows also give the 2025-26 over/under price history for totals backtests.
- **Pros:** free and covers almost every game.
- **Cons:**
  - The book changes by era. 2020-21 to 2023-24 is **Unibet**, quoting **three-way** lines (home win, away win or draw after 60 minutes). Those can be normalised into a probability, but they are not prices for the two-way bets we place, so they can't be used to simulate payouts.
  - Some Unibet prices were captured **during** the game.
  - ESPN serves **no lines for 2024-25** (or October to November 2025).
  - One book only, and no timestamps.
- **History:** 2020-21 → 2023-24 and 2025-26.

**Available but unused in the summary:** the game's referees (`gameInfo.officials`; also in the NHL `right-rail` page) and attendance. Low value. **A trap:** the summary also has an `injuries` block, but on a past game it shows **today's** injuries, not the injuries on game day. Checked 2026-10-04 on a 2026-04-06 game: every entry was dated October 2026. Using it for history would be leakage.

### 2.7 ESPN core API (`sports.core.api.espn.com`), prop prices

- **What:** ESPN's raw data service. Its `propBets` list gives past prop prices.
- **Cost:** free, no key.
- **Code → tables:** `ingestion/espn_props.py` → `raw.prop_odds_hist` (59,044 rows, 486 games, all 2025-26).
- **Used for:** `models/props_market_check.py`, the test of whether the shots-on-goal model beats the prop market. It did not.
- **Pros:** the only free prop-price history.
- **Cons:** patchy:
  - ESPN BET from October to November 2025, but its last update came after puck drop, so only its opening prices are usable
  - DraftKings on a few dates only, all-or-nothing per date
  - DraftKings entries carry no over/under label
- **History:** 2025-26 only. **Unused:** the `odds/{provider}/history/0/movement` endpoint exists but returned nothing for a 2025-26 game, and ESPN's `predictor` and `probabilities` endpoints say "not supported" for the NHL.

### 2.8 ESPN injuries (`site.api.espn.com/.../nhl/injuries`)

- **What:** the league-wide injury list: status (Day-To-Day, Out, Injured Reserve, Suspension), injury type, expected return date, and short and long news text.
- **Cost:** free, no key.
- **Code → tables:** `ingestion/espn_injuries.py` → `raw.injuries`, one snapshot per day, names matched to NHL player IDs.
- **Used for:** **nothing yet.** No feature, model or pick-job code reads `raw.injuries` (checked by search on 2026-10-04). Its job today is to build a history that ESPN does not keep.
- **Pros:** the only free structured injury list. The NHL API has none.
- **Cons:** today-only, so every missed day is lost for good. ESPN's player IDs are not NHL IDs, so names have to be matched. Checked 2026-10-04: per-season history at `seasons/{y}/athletes/{id}/injuries` holds only the current season's open entries; past seasons return nothing, and the team-season injury lists return 404.

### 2.9 Daily Faceoff, starting goalies (`dailyfaceoff.com/starting-goalies/{date}`)

- **What:** a hockey news site's daily list of projected and confirmed starting goalies, with the source of each confirmation (usually a beat reporter or team).
- **Cost:** free. The site blocks plain programs (Cloudflare), so the loader identifies itself as a Chrome browser. That is a grey area, and it could break without notice.
- **Code → tables:** `ingestion/dailyfaceoff.py` → `raw.starting_goalies`; read by `betting/recommend.py`.
- **Used for:** the starting goalie in each live pick. A "Confirmed" starter clears the "we guessed the goalie" flag.
- **Pros:** the fastest free source for confirmed starters.
- **Cons:** scraped, so it is fragile. Matching its full names to the NHL's abbreviated ones needs care.
- **History:** dated pages still load, but **their content is not reliable for the past.** Checked 2026-10-04: for 2025-01-15 the page listed 2 games (the schedule had more), and the EDM @ MIN game showed Calvin Pickard (an Edmonton goalie) as Minnesota's starter and Devon Levi (Buffalo) as Edmonton's. For 2022-01-15 it listed Brandon Bussi for Carolina; he was not a Carolina player then. Our own daily snapshots are the only trustworthy starter history.

### 2.10 Daily Faceoff, line combinations (`dailyfaceoff.com/teams/{team}/line-combinations`), being added

- **What, from one page fetched 2026-10-04:** each team's 4 forward lines, 3 defence pairs, goalies, **power-play unit 1 and 2**, penalty-kill units and injured reserve. Each player has an injury status and a "game-time decision" flag, and the page has a last-updated time and the source link (e.g. a reporter's post).
- **Cost:** free, scraped (same caveats as 2.9). 32 pages a run.
- **Used for (planned):** power-play unit 1 membership is the single biggest driver of a skater's points and shots, after his own talent. The book's opening prop prices are often set before lines are confirmed. Also used for late lineup changes on the game models.
- **History:** none; current state only. Save it daily, and use **NHL shift charts** (2.1) to see what was actually played in the past.

### 2.11 NHL free odds feed (`api-web.nhle.com/v1/partner-game/{US,CA}/now` and `/v1/schedule/{date}`)

- **What:** the NHL's own betting-partner prices: DraftKings in the US, FanDuel's Canadian book in Canada, and moneylines from up to 7 partner books on the schedule.
- **Cost:** free, no key.
- **Code → tables:** `ingestion/nhl_odds.py` → `raw.nhl_feed_snapshots`, deliberately kept apart from `raw.odds_snapshots` so an unproven feed can't change picks.
- **Used for:** comparison only (`pipeline.py compare-feeds`). If it proves fresh, it could provide **free** closing prices by polling every few minutes before games.
- **Pros:** free, and it can be polled often.
- **Cons:**
  - only the next game day
  - no history
  - its "last updated" stamp is not a price time
  - Veikkaus prices looked swapped, so they are excluded
- **History:** none.

### 2.12 Kaggle copy of ESPN's lines (`jonathanncoletti/nhl-historical-game-data`)

- **What:** a dataset whose author saved ESPN's lines at the time, including 2024-25, which ESPN later removed.
- **Cost:** free; loaded once by hand on the Mac. **There is no loader in the repo**, and the PC's database does not have these rows.
- **Used for:** the 2024-25 market input for the win/loss model.
- **Cons:** only the **favourite's** moneyline (the other side is empty). 41 games are missing. It can't be used for payouts. The Odds API history (2.5) replaces it with real two-way prices from many books.

### 2.13 Free loaders: shift charts, scratches and officials, Kalshi (built 2026-10-04)

Three loaders, no key, no credits. Each keeps a fetch log, so a stopped run picks up where it left off, and each is polite: one request every 0.34 s at most to the NHL (about 3 a second) and 0.2 s to Kalshi, with retries on timeouts, HTTP 429 (→ "too many requests") and server errors. `python pipeline.py free-data` runs all three for new games, and `daily` runs it last, after the picks. Each has `--report` (coverage by season; fetches nothing). Their acceptance checks were written into each module's docstring before the backfill ran; the results are below.

**Rate limits seen (2026-10-05).** The NHL endpoints let about 35 quick requests through and then answer 429 with `Retry-After: 60` (sometimes `0`). The shared client (`ingestion/polite.py`) now widens its gap by half on every 429 and narrows it again after 200 answered requests, so each loader settled at about one request every 0.8-1.2 s, about 1.2 games a second. The rest of the backfill (about 6,700 games for each NHL loader) took about 85 minutes, both loaders running side by side. Kalshi answered an occasional 429 at 5 a second and recovered after a 2-second pause.

#### Shift charts → `raw.shifts`, `raw.shift_fetches`

- **What:** every shift (→ one stretch a player spends on the ice) with player, team, period, start and end time and duration, and the NHL's row id. Goal markers in the same response are only counted.
- **When it is known (point-in-time):** only after the game. A feature may use the shifts of earlier games only.
- **Coverage after the backfill (2026-10-05):**

| Season | Finished games | Fetched, status ok | Of those, from the HTML reports | Shifts | Player-games who played but have no shift | Median gap, summed shifts vs box-score ice time |
|---|---|---|---|---|---|---|
| 2020-21 | 952 | 952 | 0 | 737,845 | 0 of 36,276 | 0 s |
| 2021-22 | 1,401 | 1,401 | 0 | 1,083,846 | 0 of 53,426 | 0 s |
| 2022-23 | 1,400 | 1,400 | 0 | 1,069,433 | 0 of 53,359 | 0 s |
| 2023-24 | 1,400 | 1,400 | 0 | 1,062,112 | 0 of 53,368 | 0 s |
| 2024-25 | 1,398 | 1,398 | 57 | 1,062,042 | 0 of 53,273 | 0 s |
| 2025-26 | 1,394 | 1,394 | 0 | 1,052,966 | 1 of 53,124 | 0 s |
| 2026-27 (to 2026-10-05) | 39 | 39 | 0 | 29,936 | 0 of 1,483 | 0 s |
| **Total** | **7,984** | **7,984** | **57** | **6,098,180** | **1** | |

"Finished games" are regular-season and playoff games marked final in `raw.games` (`game_state` FINAL or OFF, as everywhere else in the repo; preseason is not loaded). This table is the backfill as first loaded; the repair below changes it.

- **Checks:** S1 at least 99% of each season's games end "ok": **pass** (100% in every season, after the fallback below). S2 players who played but have no shift at most 0.5%: **pass** (1 of 304,309; a skater credited with 1 second of ice time in NSH-COL on 2025-12-09). S3 median gap between summed shift durations and box-score ice time at most 5 seconds: **pass** (0 seconds in every season).
- **Odd:** the shift-chart API returns `{"data": [], "total": 0}` for 57 games at the end of the 2024-25 regular season (2025-04-08 to 2025-04-15; checked again by hand on 2024021235). The NHL's HTML time-on-ice reports for those games are complete, so the loader now falls back to them (three requests a game: the two reports and the boxscore for sweater numbers). On a test game the HTML shifts summed to every skater's box-score ice time exactly. Those 57 games are marked `source = 'html'` in `raw.shift_fetches` and their rows have no `nhl_shift_id`. Without the fallback 2024-25 would have been at 95.9% and failed S1.
- **Bad rows in the source (found 2026-10-06, after the table above):** S3 tests the median gap, and the median could not see a bad tail. The shift API sometimes sends the same shift twice (same player, period and start time under two NHL row ids) and, for two games, rows of other teams, all looking normal. The backfill stored them as sent:
  - 19,203 repeated rows in 1,537 games;
  - 1,371 rows of other teams in 2 games. In 2021020513 (NYI-WSH) the STL and MIN rows were a third copy of the game's own shifts under the wrong team codes. In 2025020565 (NJD-BUF) the VGK and SJS rows were game 2024020565's shifts: the same game number a season earlier;
  - 923 games with at least one skater more than 60 seconds from his box-score ice time, and 2.1% of skater-games more than 5 seconds off. The worst was 4,485 seconds of shifts against 1,495 seconds of ice time.
- **The fix:** the loader now drops rows of any team not in the game and keeps one row per player, period and start time (the lowest NHL row id). Then a per-game QA check (→ quality check) compares each skater's summed shifts with his box-score ice time. A full game where any skater is more than 60 seconds off, or that has no box score yet, is stored as **`suspect`**, not `ok`, and tries the HTML reports, which replace the API's shifts only when they pass the same check. **A feature should read `ok` games only.** `--recheck` cleans and re-checks the rows already stored without fetching anything; `--game ID` fetches one game again.
- **The repair (2026-10-07):** rehearsed on a full copy of the live tables; the numbers below are from that copy. The live tables are backed up (`data/backups/shifts_before_repair_20261007.dump`) but **not repaired yet**: running the same two commands on the live database does it (about 2 minutes, then about 2 minutes of requests; `--retry-empty` also loads any game finished since).

| Step | Rows deleted | ok | suspect | Shifts after |
|---|---|---|---|---|
| Before | | 7,984 | 0 | 6,098,180 |
| `--recheck` | 18,499 repeated, 1,371 other teams' | 7,908 | 76 | 6,078,310 |
| `--retry-empty` (the 76 suspect games fetched again) | | 7,981 | 3 | 6,078,021 |

  Of the 76, 72 passed from the HTML reports and 1 from the API. The 3 left (2020020124, 2020020252, 2021020326) fail from both sources. The two cross-game games keep 704 and 746 shifts of their own teams and pass the check; fetching them again gives the same rows. After the repair: no repeated or wrong-team rows, no skater-game in an `ok` game more than 60 seconds off, 889 of 287,225 (0.31%) more than 5 seconds off; S1 still passes (lowest: 2020-21, 950 of 952), S2 0 of 304,309, S3 median 0 seconds. `--report` now prints the tail too: skater-games more than 5 and 60 seconds off, repeated rows and wrong-team rows, per season.

#### Scratches and officials → `raw.game_scratches`, `raw.game_officials`, `raw.game_info`

- **What:** per game, the scratched players (→ on the roster but did not dress; injured and left-out players are not told apart), the two referees and two linesmen, and both head coaches.
- **When it is known:** scratches at the warm-up, about 30-60 minutes before puck drop (fine for the close, not for a morning pick); referees on the morning of the game.
- **Coverage after the backfill (2026-10-05):**

| Season | Finished games | Fetched, status ok | Scratched players | Officials (referees + linesmen) | Games without exactly 2 referees |
|---|---|---|---|---|---|
| 2020-21 | 952 | 952 | 7,932 | 3,808 | 0 |
| 2021-22 | 1,401 | 1,401 | 9,634 | 5,597 | 2 (1 referee) |
| 2022-23 | 1,400 | 1,400 | 9,107 | 5,601 | 2 (one with 1, one with 3) |
| 2023-24 | 1,400 | 1,400 | 9,211 | 5,604 | 3 (3 referees) |
| 2024-25 | 1,398 | 1,398 | 8,917 | 5,597 | 5 (3 referees) |
| 2025-26 | 1,394 | 1,394 | 9,435 | 5,574 | 1 (1 referee) |
| 2026-27 (to 2026-10-05) | 39 | 39 | 205 | 156 | 0 |
| **Total** | **7,984** | **7,984** | **54,441** | **31,937** | **13** |

About 6.5 to 7 scratches per game (both teams together); 2020-21 runs higher (8.3), probably because teams carried taxi squads that season (→ extra players kept with the team during COVID; a guess, not checked). Across the six seasons there are 55 different referees and 58 different linesmen.

- **Checks:** G1 at least 99% of each season's games "ok": **pass** (100%). G2 at least 98% of "ok" games list exactly 2 referees: **pass** (13 of 7,984 do not, 0.16%; probably a referee hurt and replaced, or one who could not work the game: not checked). G3 no scratched player appears in that game's box score: **pass** (0).

#### Kalshi → `raw.kalshi_markets`, `raw.kalshi_candles`

- **What:** every market in Kalshi's `KXNHLGAME` series (→ "Will this team win?", pays $1, so the price reads as a probability; overtime and shootout count, like a moneyline), two per game, with settlement; hourly candles (→ open, high, low and close of the bid, ask and traded price over the hour) over each market's life; and 1-minute candles for the 3 hours before puck drop. `ingestion.kalshi.closing_lines()` returns the last 1-minute candle that ends at or before `start_time_utc`: the pre-game close. Prices are before Kalshi's taker fee (0.07 × p × (1 − p) per contract).
- **Never use a candle that ends after `start_time_utc` as a pre-game price:** Kalshi trades during the game.
- **Coverage after the backfill (2026-10-05):**

| Season | Events (games) | Markets | Settled | Hourly candles | 1-minute candles | Games with a two-sided close |
|---|---|---|---|---|---|---|
| 2024-25 (playoffs only) | 83 | 166 | 166 | 6,786 | 13,233 | 83 |
| 2025-26 | 1,392 | 2,784 | 2,784 | 195,133 | 447,653 | 1,392 |
| 2026-27 (to 2026-10-05) | 89 | 178 | 86 | 11,037 | 12,924 | 43 (the rest not played yet) |
| Not matched to a game | 122 | 244 | 244 | 16,267 | 0 | — |

- **Checks:** K1 every event has exactly two markets: **pass** (0 exceptions). K2 at least 95% of 2025-26 and 2026-27 games played have an event: **pass** (2025-26: 1,392 of 1,394, 99.9%; 2026-27: 39 of 39). K3 the settled result agrees with `raw.games`: **pass** (3,028 of 3,028 markets). K4 at least 90% of matched settled games have a two-sided close (bid above 0 and ask below 1 on both sides): **pass** (100% in each season).
- **Close against DraftKings:** on the 1,012 games of 2025-26 with a DraftKings closing moneyline in `raw.historical_odds`, Kalshi's close (mid-price, both sides scaled to sum to 1) and DraftKings' no-vig close differ by 0.55 points of probability at the median, 1.6 points at the 95th percentile, and never by more than 4.8 points; the mean difference is −0.04 points. So the close is a pre-game price, not an in-play one. The median bid-ask spread at the close is 1 cent, and buying both sides at the ask costs about 1-2 cents over $1 (the exchange's equivalent of the vig) before the fee.
- **Odd:** two 2025-26 playoff games are missing from Kalshi's game series (2026-05-03 MTL at TBL and 2026-05-18 MTL at BUF, both game 7s). The 122 unmatched events are preseason games (September and early October, not in `raw.games`) and LA at CBJ on 2026-01-26, which has no game on that date in `raw.games` (Kalshi's rule: a game not started within 48 hours settles both sides at a "fair price"); Kalshi settled it (0.48 / 0.52, result `scalar`). A market fetched before its game is in `raw.games` gets hourly candles only (`candles_problem` says so); it is fetched again once its game is matched, so its 1-minute close is never missing (none were, when checked). Kalshi's history moved to a `/historical/` path for markets settled before its cutoff (2026-08-05 when checked); `--backfill` lists both.

---

## 3. Free sources we don't use yet, ranked by likely benefit

| Rank | Source | What it is | Why it would help |
|---|---|---|---|
| 1 | **Kalshi public market data** (`api.elections.kalshi.com/trade-api/v2`, no key for market data) | Kalshi is one of the two venues legal for a Texas bettor. Its API gives every NHL game market's prices, and hourly candles (a candle → the open, high, low and close price over one period) with bid and ask (the best price buyers offer and the best price sellers accept) under `/historical/markets/{ticker}/candlesticks`, plus every trade | **High.** Checked 2026-10-04: 1,631 settled NHL game events from 2025-04-19 to 2026-10-03, so the **whole 2025-26 season at the actual venue's own prices**, for free. That gives a backtest at prices a Texas bettor could really have taken, a free Kalshi closing price for CLV, and how Kalshi prices compare with the books. Kalshi also lists NHL spread, period-total and player markets (series `KXNHLSPREAD`, `KXNHLGOAL`, `KXNHLSAVES`, `KXNHLPTS` and more). Old trades moved to a `/historical/` path (cutoff 2026-08-05). **Game-winner markets now loaded (2.13)**; the other series are not |
| 2 | **NHL shift charts + `right-rail` scratches** | Who skated with whom in every past game; who was scratched | **High.** Rebuilds historical lineups and lines for free. This is the base for a lineup-aware model, and it stands in for the missing injury history. **Now loaded (2.13)** |
| 3 | **NHL stats REST: goalie-pull, penalties, power-play-time reports** | §2.2 table | Medium. Puck line and totals (empty-net goals, power-play volume); goalie model |
| 4 | **MoneyPuck game-by-game lines and players** | §2.3 table | Medium. Per-line and per-player strength without rebuilding from shots |
| 5 | **Polymarket public prices** (`gamma-api.polymarket.com`; price history through its order-book API) | The other venue legal in Texas | Medium. The same idea as Kalshi. Checked only that NHL markets are listed (2026-10-04); the price-history depth is not checked |
| 6 | **Sportsbook Reviews Online archive** ([link](https://www.sportsbookreviewsonline.com/scoresoddsarchives/nhl/nhloddsarchives.htm)) | Excel files of opening and closing moneyline, puck line and total for every game, 2007-08 to 2022-23. No longer updated | Low to medium now that the Odds API history exists. A free second opinion on 2020-21 to 2022-23 closes and long history for open-vs-close studies. The book behind the numbers is not named |
| 7 | **NHL EDGE tracking** | §2.1 table | Low; needs daily snapshots to be safe from leakage |
| 8 | **Referees** (NHL `right-rail`, ESPN) | Who officiates | Low. A small penalty-rate effect on totals and power-play props. **Now loaded (2.13)** |

## 4. Checked and rejected

- **DraftKings' own site:** blocked (HTTP 403 from this PC, 2026-09-28). Its terms ban automated access, and it adds nothing the Odds API lacks.
- **ESPN's per-game injuries block for past games:** shows today's list, not game day's (leakage), §2.6.
- **Daily Faceoff's dated archive pages for past seasons:** wrong goalies and missing games, §2.9.
- **ESPN predictor, win probabilities and line movement:** not supported or empty for the NHL.
- **Hockey-Scraper:** listed in `PROJECT_CONTEXT.md` but not used by any code. The NHL shift-chart and play-by-play endpoints cover the same ground, and they are current.

## 5. Why not one source for everything?

No single free source has all of it.

- **The NHL** has results, players and shifts, but no injuries and no odds history.
- **MoneyPuck** has the best shot data, but no prices or injuries.
- **ESPN** has injuries and one book's past lines, but no history of injuries, and it lost 2024-25.
- **The Odds API** has many books' prices, history included, but it costs credits and has no game stats.
- **Daily Faceoff** has confirmed goalies and lines, but no reliable past.

So each source does the job it is best at. We save our own copy of anything that is today-only (injuries, starters, lines, live prices), because once a day is missed it is gone. The paid Odds API key now covers the one hole that mattered most: real past prices from many books.

## 6. Sources

Live checks by this survey, 2026-10-04: `api.nhle.com/stats/rest/en/config`, `api-web.nhle.com/v1/gamecenter/{id}/right-rail` and `shiftcharts`, the ESPN site and core endpoints named above, Daily Faceoff team and dated starter pages, the Kalshi trade API, Polymarket's gamma API, `nhlpy` 3.3.0's `client.edge` methods, and the PC's database.

- MoneyPuck data page and terms: <https://moneypuck.com/data.htm> (read 2026-10-04)
- The Odds API historical data: <https://the-odds-api.com/historical-odds-data/> (read 2026-10-04): 2020-06-06 featured markets, 5-minute snapshots from September 2022, props from 2023-05-03, 10 credits per region per market, paid plans only
- Sportsbook Reviews Online NHL archive: <https://www.sportsbookreviewsonline.com/scoresoddsarchives/nhl/nhloddsarchives.htm> (2007-08 to 2022-23; "will not be updated"; read 2026-10-04)
- Earlier repo research: `docs/research/2026-09-28-data-feeds.md`, `docs/historical_odds.md`
