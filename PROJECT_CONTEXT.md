# PROJECT_CONTEXT.md

> **Purpose:** This file is the single source of truth for the NHL Sports Betting Predictive System. It captures locked architectural decisions, the technology stack, current phase status, and hard-won learnings so that any developer — or any AI assistant (e.g., Claude Code) — can pick up the project with full context. Keep this file updated as decisions change.

**Last updated:** October 1, 2026
**Project owner:** the owner (personal details live in the git-ignored `private/` folder)
**Assistant role convention:** Chief Data Scientist / Chief Software AI Developer

---

## 1. Project Mission

Build a professional-grade, implementation-focused NHL sports betting system that identifies positive expected-value (+EV) wagers across **all major markets**: moneyline, puck line, totals (over/under), and goalie/player props.

**Design targets (locked):**
- **Markets:** Moneyline + puck line + totals + goalie/player props
- **Workflow:** Semi-automated (system surfaces picks → user approves → system logs results), with an explicit path to full automation (API-based bet placement)
- **Infrastructure:** Local-first (PostgreSQL + Python on a single machine), designed for eventual cloud migration (AWS Lambda + RDS/DynamoDB)
- **Primary KPI:** Closing Line Value (CLV), not short-term P&L

---

## 2. Locked Architectural Decisions

These decisions are settled. Revisit only with explicit justification.

1. **Own the core, borrow the periphery.** Our prediction models, feature store, and betting engine are custom-built. The only runtime data dependency is `nhl-api-py`. All reviewed modeling repos are *methodology references*, not imported libraries.
2. **Probability-first, not prediction-first.** Every model emits calibrated probabilities. All downstream decisions consume probabilities.
3. **Separation of concerns.** Five independent layers: Data → Features → Models → Strategy → Interface. Any model can be swapped without touching the betting engine.
4. **Time-series discipline everywhere.** No random splits. Walk-forward validation with purge/embargo gaps. Point-in-time-correct features. No lookahead leakage.
5. **Local-first, cloud-ready.** Everything containerizable. Schema, config, and pipeline designed for migration.
6. **Measure edge before risking capital.** Paper-trade 500+ bets. CLV is the north-star metric. Statistical significance testing on every claimed edge.

---

## 3. Technology Stack

| Component | Local (Phase 1) | Cloud (Future) |
|-----------|-----------------|----------------|
| Language | Python 3.11+ | Python 3.11+ (Lambda/ECS) |
| Database | PostgreSQL 16 (Docker) | AWS RDS PostgreSQL / DynamoDB |
| ML | LightGBM, XGBoost, scikit-learn, Optuna | Same (SageMaker optional) |
| Data ingestion | `nhl-api-py` + requests | Same (Lambda-triggered) |
| Odds feed | The Odds API (REST) | Same |
| Scheduler | cron / APScheduler | EventBridge / Step Functions |
| Dashboard | Streamlit | Streamlit Cloud / Django |
| Containerization | Docker Compose | ECS Fargate / Lambda |
| Model registry | MLflow (local) or file-based | MLflow on EC2 / SageMaker |
| Versioning | Git + DVC | Same + S3 DVC backend |

### CRITICAL: Package import name
`nhl-api-py` (the pip package) imports as **`nhlpy`**, NOT `nhl_api_py`.
```python
from nhlpy import NHLClient   # correct
```
This bit us once during Phase 1 smoke tests. Do not "fix" it back.

---

## 4. Data Sources

| Source | Purpose | Access | Notes |
|--------|---------|--------|-------|
| NHL API (api-web.nhle.com) | Schedule, results, boxscores, rosters, EDGE stats | Free via `nhlpy` | Undocumented; can change without notice. New API since 2023 (old statsapi.web.nhl.com is dead) |
| MoneyPuck | Shot-level data w/ pre-computed xG (2007–present) | Free CSV | Gold standard. 1.84M+ shots. Updated nightly in-season |
| The Odds API | Live odds from 10 named books (`ODDS_BOOKMAKERS`: kalshi, polymarket, pinnacle, draftkings, fanduel, betmgm, betrivers, espnbet, hardrockbet, novig); player props per game | Free tier 500 credits/mo per key (full snapshot 3, close 1 since 2026-09-29; was 6 and 2 with us,us2); historical odds are paid plans only | Uses full team names — see mapping in `ingestion/odds_api.py` |
| NHL odds feed (api-web.nhle.com partner-game US/CA + schedule) | DraftKings (US) and FanDuel Canada ml/pl/total/3-way; schedule moneylines from partner books | Free, no key (`ingestion/nhl_odds.py`) | Next game date only, no history; lastUpdatedUTC is not a price timestamp; stored in raw.nhl_feed_snapshots for comparison only |
| NHL stats REST (api.nhle.com/stats/rest) | Per-game PP/PK TOI, PP goals/assists, faceoffs for every skater | Free (`ingestion/nhl_stats.py`) | Fills raw.skater_games columns the boxscore load always left at 0 |
| ESPN (site summary, injuries, sports.core) | Historical reference lines with open/close and O/U and puck-line prices; daily injury list; 2025-26 player-prop prices | Free, no key (`ingestion/espn_odds.py`, `espn_injuries.py`, `espn_props.py`) | Book varies by era; DraftKings 2025-26 is the backtestable one |
| Hockey-Scraper | Historical PBP + shifts (2007–2023) | pip (`hockey_scraper`) | Historical backfill ONLY. Not for live/current use |
| Daily Faceoff / LeftWingLock | Confirmed starting goalies, lineups | Web scrape / manual | Needed for goalie-conditioned predictions. Injuries now come from ESPN |

---

## 5. Repository Research Findings (Track A — COMPLETE)

We evaluated 25 GitHub repos, kept 11, and produced full T2 analyses with RQI scores (7-dimension weighted index: Reproducibility 20%, Maintenance 15%, Data Freshness 15%, Model Characteristics 15%, Feature Engineering 15%, Data Quality 10%, Operational Readiness 10%).

| Rank | Repo | RQI | Role in our system |
|------|------|-----|--------------------|
| 1 | coreyjs/nhl-api-py | 4.03 | **Direct dependency** — primary data layer |
| 2 | HarryShomer/Hockey-Scraper | 3.40 | Historical PBP backfill only |
| 3 | evjrob/bayes-bet | 3.23 | Architecture reference (AWS/Django); Bayesian team-strength priors |
| 4 | Zmalski/NHL-API-Reference | 3.23 | API endpoint documentation |
| 5 | TonyAllenPrice/nhldata | 2.58 | Optional MoneyPuck convenience wrapper |
| 6 | JNoel71/NHL-xG-Model | 2.50 | Methodology ref — LightGBM xG, dual venue adjustment |
| 7 | JNoel71/NHL-Game-Prediction | 2.43 | Feature blueprint — 600+ feature taxonomy |
| 8 | gschwaeb/NHL_Game_Prediction | 2.28 | Betting layer ref — value-bet logic, honest backtest |
| 9 | andrewderango/NHL-Projections-2023 | 2.08 | Methodology ref — stacked ensemble, Monte Carlo |
| 10 | saiemgilani/Goalie_Model_NHL | 1.53 | Methodology ref — Buhlmann credibility, PMF output |
| 11 | miltonleung/Bookie | 1.35 | Educational only |

**Key takeaways:**
- Data-infrastructure repos score high; modeling repos score low (frozen, built on deprecated API).
- No single repo is a production system. We build custom, harvesting methodology.
- Goaltending modeling is the single biggest feature gap across all repos — saiemgilani fills the methodology void (Buhlmann shrinkage + goals-allowed PMF).
- **License risk:** `nhl-api-py`, Hockey-Scraper, and both JNoel71 repos are GPL-3.0 (copyleft). Two repos have no license. Legal review before any redistribution; clean-room reimplementation for methodology ports.

---

## 6. System Architecture (5 Layers)

```
Layer 1: DATA        → ingestion/ → raw.* tables in PostgreSQL
Layer 2: FEATURES    → features/  → features.* (point-in-time vectors)
Layer 3: MODELS      → models/    → models.* (calibrated probabilities)
Layer 4: STRATEGY    → betting/   → betting.* (recommendations, stakes, CLV)
Layer 5: INTERFACE   → dashboard/ → Streamlit (daily slate, bankroll, CLV report)
```

### Model stack (layered, each feeds the next)
- **Layer A — xG:** LightGBM per-shot goal probability. Blueprint: JNoel71/xG-Model. ~28 shot features, dual venue adjustment (Krzywicki + Schuckers-Curro). Target AUC > 0.77.
- **Layer B — Goalie quality:** XGBoost save-prob + Monte Carlo PMF + Buhlmann shrinkage. Blueprint: saiemgilani. Outputs shrunk quality rating + goals-allowed PMF.
- **Layer C — Game outcome:** LightGBM home-win probability + isotonic calibration. Blueprints: JNoel71/Game-Prediction (features), gschwaeb (betting), evjrob (Bayesian priors). Target log loss < 0.675, ECE < 0.02.
- **Layer D — Totals:** Convolve home/away goal PMFs → total-goals distribution → price any O/U line.
- **Layer E — Props:** Goalie saves/GAA/shutout from PMF. Skater props via Poisson regression (later phase).

### Database schema (4 namespaces, 28 tables)
- `raw.*` — games, teams, players, rosters, shots, team_games, skater_games, goalie_games, odds_snapshots, historical_odds, starting_goalies, shifts; since 2026-09-29 also nhl_feed_snapshots, injuries, prop_odds_hist, prop_odds_fetches, prop_snapshots, and since 2026-10-04 odds_history and odds_history_fetches (created on old databases by `config/migrate.ensure_schema`)
- `features.*` — team_rolling, goalie_rolling, matchup, game_vector
- `models.*` — model_registry, predictions
- `betting.*` — recommendations, placed_bets, bankroll_log

Full column definitions in `db/schema.sql`.

---

## 7. Betting Engine Rules (locked defaults)

- **Edge thresholds:** ML ≥ 2.5%, totals ≥ 3.0%, props ≥ 4.0%
- **Staking:** Quarter-Kelly (f = 0.25). `stake = 0.25 × kelly × bankroll`
- **Exposure caps:** Max 2% bankroll per bet; max 10% per day; max 3 correlated bets per game and max 4% of bankroll staked on one game, every market counted (enforced since 2026-09-29: `MAX_BETS_PER_GAME`, `MAX_GAME_STAKE_PCT` in `betting/engine.py`, env-overridable)
- **Line shopping:** Best price across all books; exclude stale lines (>5 min old)
- **CLV:** `clv = implied_prob(closing) − implied_prob(placed)`. Target avg CLV > 1.0% over 500+ bets.
- **No-vig conversion:** Power method

---

## 8. Implementation Roadmap & Status

| Phase | Scope | Status |
|-------|-------|--------|
| **Track A (Docs)** | File 1 (T2 analyses), File 2 (Catalog), File 3 (Design Proposal) | ✅ COMPLETE |
| **Phase 1** | Data foundation: schema, 3 ingestion modules, master pipeline, smoke tests | ✅ COMPLETE (6 seasons backfilled: 7,945 games, 683k shots) |
| **Phase 2** | Feature store + baseline logistic regression model | ✅ COMPLETE — **gate passed: walk-forward log loss 0.6829 < 0.69** (see `docs/phase2_results.md`) |
| **Phase 3 (modeling half)** | Historical odds (99.5% coverage, free — `docs/historical_odds.md`) + market feature + LightGBM boosted from the market + temperature calibration | ✅ COMPLETE — **log loss 0.6607, ECE 0.0146** (`lgbm_market v2`, see `docs/phase3_results.md`) |
| **Phase 3 (betting half)** | Edge engine + quarter-Kelly staking + payout backtest + Streamlit dashboard | ✅ COMPLETE — backtest says raise edge threshold to ~5-6% (validate via paper trading); PMF totals + daily recommendation job remain | 
| **Phase 3 (remaining)** | Daily recommendation job, PMF totals model, bet checker + parlay evaluator, arb/middle alerts | ✅ COMPLETE (2026-07-17) — recs job simulated + verified vs stored vectors; **totals gate FAILED honestly** (predictions-only, betting off; see `models/totals.py` STATUS); strategy-layer isotonic deferred until pooled live predictions exist |
| **Phase 4** | Live-season ops: **paper trading first** (validate the 5-6% edge threshold), cloud migration, Daily Faceoff confirmed starters, player props | 🔶 IN PROGRESS — paper settlement + CLV ledger (`betting/settle.py`) and Daily Faceoff starters (`ingestion/dailyfaceoff.py`, starter accuracy 40%→80% on test slate) DONE 2026-07-17; 2026-09-29: named-bookmaker odds (Kalshi/Polymarket in, half the credits), free NHL odds feed + `compare-feeds`, ESPN open/O-U prices, ESPN injuries, PP stats, props data (ESPN history + live Odds API on the props machine), per-game caps, totals v2 (still fails its gate); 2026-10-04: Odds API historical backfill started (2024-25 closes complete, 2024-25 mornings 149/223; `raw.odds_history`); remaining: cloud migration, props model, rec digest, historical closes for 2023-24 and 2022-23 |
| **Phase 5** | Multi-sport expansion (NBA/CBB first), correlated same-game parlays via a joint model | ⬜ Pending |

### Phase 1 deliverables (done)
- `docker-compose.yml`, `db/schema.sql` (4 schemas, 15 tables)
- `config/settings.py` (DB connection, config)
- `ingestion/nhl_api.py`, `ingestion/moneypuck.py`, `ingestion/odds_api.py`
- `pipeline.py` (setup / status / backfill / daily)
- `tests/test_setup.py` (5 passing)

### Phase 2 deliverables (done — see `docs/phase2_results.md`)
1. ✅ `features/team_features.py` — 14 rolling stats × 5 windows, point-in-time correct (shift-then-roll)
2. ✅ `features/goalie_features.py` — Buhlmann shrinkage, k = 66 estimated empirically (ANOVA over per-start SV%)
3. ✅ `features/schedule_features.py` + `db/seed_venues.sql` — rest/b2b/travel/DST-aware tz shift
4. ✅ `features/elo.py` — K=20, home ice 50 pts, pre-game values, 1/3 season regression, ARI→UTA continuity
5. ✅ `features/build_vectors.py` — 107-feature home−away vectors, fully finite (market feature deferred: no odds snapshots yet)
6. ✅ `features/build_all.py` + `pipeline.py features [--season]` — orchestration, wired into daily chain
7. ✅ `models/baseline.py` — walk-forward CV (expanding seasons, 7-day purge), **pooled OOF log loss 0.6829 — GATE PASSED**; calibration plot + `models.model_registry` entry. 83 tests green.

---

## 9. Key Learnings (running log)

- **Import name gotcha:** `nhl-api-py` → `from nhlpy import NHLClient`. (Cost us a smoke-test failure in Phase 1.)
- **NHL API migration:** Old `statsapi.web.nhl.com` was deprecated in 2023. Anything built on it is broken. Current API is `api-web.nhle.com`.
- **Accuracy ceiling:** ~62% is the public-model ceiling for NHL game prediction. JNoel71 hit 63.2% with 600+ features. Do not chase accuracy past this — chase *calibration* and *CLV*.
- **Realistic ROI:** 2–4% for top public models. Anything claiming 10%+ is overfit or mismeasured.
- **Calibration > accuracy** for betting. Most reviewed repos have zero calibration diagnostics. This is our edge.
- **Fractional Kelly** (quarter) is the consensus for NHL's high variance. Full Kelly will blow up the bankroll in normal drawdowns.
- **Goalie data is the gap.** No game-prediction repo conditions on confirmed starter. We must integrate Daily Faceoff scraping.
- **Every kept repo has bus factor 1.** Pin and fork critical dependencies.
- **MoneyPuck team codes:** four franchises use dotted codes (`L.A/N.J/S.J/T.B`) that don't match NHL API abbrevs (`LAK/NJD/SJS/TBL`). Normalized in the loader. Symptom when broken: league-wide xGF% averaged .543 instead of .500.
- **For/Against shares need symmetric masking.** Rolling sums skip NULLs per column, so a game missing one side's data biases every share metric. If either side is missing, drop both.
- **Goalie SV% is mostly noise:** empirical Buhlmann k ≈ 66 starts to reach Z = 0.5. Shrunk ratings (never NULL) are what go in the vectors, not raw rolling SV%.
- **psycopg2 can't adapt `np.float64`** — cast numpy scalars to Python `float` before executemany, or the array literal parses as a schema reference.
- **Elo extremes are real, not bugs:** 2023-24 Sharks bottom ~1274, 2022-23 Bruins peak ~1720. Sanity bounds: [1250, 1750].
- **Historical odds are free:** ESPN's public summary API (pickcenter) serves both MLs for most seasons; Kaggle mirror `jonathanncoletti/nhl-historical-game-data` fills 2024-25 (which ESPN deleted; favorite side only, ML value is a junk constant −105). Unibet-era rows are 3-WAY regulation lines (implied sums ≈ 0.83) — fine normalized as a feature, never for payout simulation. See `docs/historical_odds.md`.
- **The market beats us on average (LL 0.6529 vs our 0.6607)** — expected. Boost FROM it (`init_score = logit(market)`), don't feed it as a feature (trees can't fully recover it); route no-line games to a market-blind fallback trained/calibrated on its own regime only.
- **Isotonic overfits small calibration tails** (~150-1,000 games; +0.02 LL measured). Temperature scaling per fold; isotonic only on pooled predictions (1000s).
- **Small model-market disagreements are noise:** backtest flat ROI by claimed edge: 2.5-4% → −16.8%, 4-6% → −1.6%, 6-9% → +26.6% (n=46). Raise ML edge threshold to ~5-6%, but VALIDATE via 2026-27 paper trading — do not lock a threshold tuned on the season it was measured on.
- **Ops:** scheduling is launchd (`~/Library/LaunchAgents/com.nhlbetting.{daily,odds}.plist`, plus `com.nhlbetting.close` from 2026-09-28), NOT cron — crontab is empty, don't re-add. Snapshot schedule (2026-09-28, Central): morning `pipeline.py daily` (full 6-credit snapshot; picks are issued and frozen at its prices), optional midday `pipeline.py odds` (6 credits: confirmed starters, picks for games still without one, alerts), and the `close` job running `pipeline.py close --due` every 15 minutes (StartInterval 900): a moneyline-only 2-credit snapshot only when a game starts within 16 min and no ml snapshot is under 16 min old (CLOSE_LEAD_MINUTES / CLOSE_MIN_GAP_MINUTES, defaults 16/16 since the third review pass; malformed values log an error and fall back), i.e. one close per start time in the last cycle before puck drop, replacing the fixed ~17:30/~20:30 closes that missed afternoon games. Every snapshot command skips the request when no game starts in the next 24h (the API bills in season even on off-days). Credits, replayed on the 2026-27 schedule for every cycle start minute (UTC calendar months): daily + close --due at 16/16 ≈ 336–454/month, never above 456 (median close 9 min before puck drop; 98.7% of games have a close in their last 16 min, all in their last 30); the old 40/25 default needed 507–587 on average in Oct–Jan and Mar, up to 658, and running out kills the 9:00 pick snapshot too; the midday odds run (~186/month) does not fit the free plan. Windows tasks run hidden via `conhost.exe --headless` (Win10 21H2+/Win11; `-VisibleConsole` fallback), and their LastTaskResult reads 0 even on failure, so check the logs. Each machine has its own `.env`, so the Mac and the PC can each use their own key (500 credits each). Plist templates live in `ops/launchd/`, Windows Task Scheduler registration in `ops/windows/register-tasks.ps1`; the live Mac's existing daily/odds plists are not replaced automatically, and the close job must be installed by hand. `gh` CLI is installed and authenticated. Odds API key is valid in `.env` (gitignored).
- **Ops credits halved (2026-09-29):** with `ODDS_BOOKMAKERS` (10 named books = 1 region) every credit figure in the bullet above halves: full snapshot 3, close 1, daily + close --due ≤ ~228/month at 16/16 (Oct 222 … Mar 227 on average), and the midday odds run (~93/month) now fits the free plan.
- **Diff features destroy totals information.** The shared game_vector stores home−away differentials — right for win probability, wrong for totals (high-vs-high and low-vs-low matchups both diff to ~0). Totals need per-side LEVEL features (attack rows: off_*/def_*/goalie_*).
- **The scoring environment owns totals; public features add ~nothing.** Walk-forward: attack-row Poisson NLL 2.1867 vs trailing-environment baseline 2.1815; over/under log loss at the DK line 0.7053 vs 0.693 naive. GATE FAILED and recorded as such — totals betting stays OFF until live O/U prices + boost-from-market-total (2026-27) get it through. Season-level scoring shifts are the dominant error and are not knowable ex ante from public stats. (Corrected 2026-09-29: confirmed starters will NOT get it through — the historical test already used the actual starters via `is_starter`.)
- **The calibration tail is the playoffs.** Any time-ordered tail of a training window that ends a season is playoff-heavy — a different scoring regime. Mean-scale corrections fit there swing ±0.4 goals/game; measured net-harmful even playoff-filtered (helps long-train folds, wrecks short ones).
- **Slate vectors == historical vectors, provably.** The daily job builds pre-game vectors by appending stats-less rows to the shift-then-roll builders; verified equal to the stored historical vectors within DB NUMERIC rounding (tests/test_recommend.py). Starter projection (most starts in last 10 team games) hit only ~40% on the test slate — Daily Faceoff integration (Phase 4) is the fix.
- **Checker edges are conservative by design:** measured vs the offered vig-inclusive price (one side of one book visible), vs the daily job's consensus no-vig. The same bet can read +1.7% in the checker and +4.0% in the job.
- **raw.players stores ABBREVIATED names** ('J. Swayman'), but Daily Faceoff publishes full names ('Jeremy Swayman') — exact matching resolves 0/19; first-initial + surname keys (team-disambiguated) resolve 20/20. Watch for the same trap anywhere player names join external sources.
- **Utah renamed again**: 'Utah Hockey Club' → 'Utah Mammoth' (2025-26). Both map to UTA in `ingestion/odds_api._TEAM_NAME_TO_ABBREV` (shared by the odds feed and the Daily Faceoff scraper).
- **Confirmed starters are worth exactly what we thought**: Daily Faceoff integration took slate starter accuracy from 40% (most-starts heuristic) to 80% on the 2026-01-15 test slate; `starter_fallback=0` only on DF-'Confirmed' rows.
- **Texas execution (researched 2026-07-17, `docs/texas_execution_options.md`):** no legal TX sportsbook before 2028+. Live execution = the TWO-BETTOR structure: bettor A (a legal online-sportsbook state) with full sportsbook + promo access **in their own name, their own funds, their own bets**, plus bettor B on Kalshi/Polymarket from TX. The load-bearing legal line: no proxy placement, no cross-funding — shared analytics, never a shared wallet. Promo-hedging calculator (`betting/promo.py`, design in `docs/promo_hedging_calculator.md`) reports per-bettor P&L only. Re-verify fee schedules + OK ballot measure before Oct 2026.
- **Match odds to games on start instants, not dates (2026-09-28).** `raw.games.date` is the league's Eastern schedule date, but the Odds API's `commence_time` is UTC, so a 7pm ET January game lands on the NEXT UTC day. Matching on the UTC date silently dropped most evening games (no pick, no alert, no CLV; debug-level log only; credits still spent). Now: home team + `raw.games.start_time_utc` (NHL `startTimeUTC`) within 6h, nearest first; Eastern-date fallback only for rows with no start time; unmatched events log a WARNING with a count.
- **The /odds endpoint returns in-play games (2026-09-28).** Games stay listed after puck drop, with live prices. Skip any event whose `commence_time <= now`, or an in-play price becomes a pick's price or its "close". The live slate also drops LIVE/CRIT games and any game past its `start_time_utc`.
- **Freeze picks at issue and bound the close by puck drop, or CLV is zero by construction (2026-09-28).** Every run used to delete pending picks and re-price them at the newest snapshot, and settlement took the same book's latest snapshot as the close — usually the very snapshot the pick came from, so same-book CLV read 0. Now a pick is written once per game (`betting.recommendations.priced_at` = its snapshot's `captured_at`) and never re-priced or deleted; the close is the same book's last snapshot with `priced_at < captured_at < start_time_utc`, consensus fallback in the same window, else CLV NULL (never 0). A 2-credit moneyline-only `pipeline.py close` run supplies that snapshot.
- **Void a pick on a moved start, not on how early it was priced (2026-09-28).** Voiding every pick priced more than 36h before puck drop also voided picks made a day ahead (`betting.recommend --date <tomorrow>`). Each pick now stores the game's start when written (`betting.recommendations.scheduled_start`); settlement voids when the actual start is more than 3h away from it, or the game is PPD/CNCL, and keeps the 36h rule only for picks with no stored start. A voided pick no longer counts as the game's frozen pick, so a postponed game gets a new pick on its new date.
- **The DB-test opt-in must name the copy, not just flip a flag (2026-09-28).** `NHL_ALLOW_DB_TESTS=1` alone ran the whole DB suite against whatever `.env` named, i.e. the live database. `tests/conftest.py` now enables DB tests only for a `*_test` database or for the flag plus POSTGRES_HOST/PORT/DB set in the environment and naming a different database than `.env`; otherwise it forces 127.0.0.1:1 and says why.
- **SQLAlchemy 2.1 changed the default PostgreSQL driver (2026-09-28).** A bare `postgresql://` URL now means psycopg (v3); only psycopg2 is installed, so every command failed with `No module named 'psycopg'`. Name the driver: `postgresql+psycopg2://` (`config/settings.py`).
- **ESPN's pickcenter carried opening and over/under prices all along (2026-09-29).** The summary block `ingestion/espn_odds.py` always downloaded holds opening moneylines and total, closing and opening O/U prices and puck-line prices; the old parser kept only the closing moneyline, spread and total line, so `models/totals.py` believed "no O/U prices exist historically". Now stored in 14 new `raw.historical_odds` columns; `--refresh` fills old rows (never overwrites a stored close, never mixes books). 2025-26 DraftKings: ~95% of games have open and close O/U prices (71/74 sample), the opening total sits on a different line from the close in about a quarter of games (so open prices pair with `total_open`). Unibet eras: closing O/U only (2020-23), everything in 2023-24, but 3-way moneylines and some in-play quotes — backtests filter `provider = 'DraftKings'`. Median DK O/U overround ≈ 4.5%.
- **`bookmakers=` halves the Odds API bill (2026-09-29).** Docs: cost = markets × regions, "every group of 10 bookmakers is the equivalent of 1 region", "bookmakers takes priority" and can mix regions. 10 named books (kalshi, polymarket first — the owner's legal Texas venues, which us,us2 never returned) cost 3/1 instead of 6/2. The median fair price now spans those 10 (offshore/sweepstakes books gone). Exchange quotes are treated as fee-free (docs silent): set `BETTABLE_BOOKS` and remember the Kalshi fee 0.07·p·(1−p). Not yet exercised live: check `x-requests-last` = 3 on the first run.
- **The NHL's own odds feed is free but limited (2026-09-29).** partner-game/US = DraftKings, /CA = FanDuel Canada (not the Odds API's US FanDuel), schedule = moneylines from partner books; date lookups 404. Only the next game date is priced; played games lose their odds (no history); `lastUpdatedUTC` stayed a month old while DK moved in 10/20 series, so it is not a price time; at 01:09 Central on 2026-10-01 the feed still listed the previous night's finished games and that day's 8 games were unpriced, so its daily rollover time is unknown. Veikkaus contradicted every other book on the favourite in 2/3 games → excluded by default. Stored in `raw.nhl_feed_snapshots` (never `raw.odds_snapshots`), one free snapshot after every Odds API snapshot; `pipeline.py compare-feeds` decides whether it can add free closes. Reference only until a season of pairs says it is fresh.
- **Odds API historical odds (2026-10-04).** `/v4/historical/sports/icehockey_nhl/odds?date=` returns the snapshot at or before `date` (5-minute snapshots since Sept 2022, 10-minute before; NHL from 2020-06-29), costs 10 × markets × regions (confirmed: `x-requests-last` 10 for h2h, 20 for h2h+totals with 10 named books), and lists in-play games with live prices, so loads drop any event whose API `commence_time` or NHL start has passed. The API's `commence_time` ran a few minutes after the NHL schedule. Old seasons had no `start_time_utc` (filled free from the NHL schedule by `odds_history starts`). Bought: 2024-25 closing snapshots per 75-minute start cluster (549 calls, all 1,398 games, h2h+totals at 10 books, median 14 min before puck drop) and 149 of 223 10:00 CT morning snapshots, 14,000 credits with probes. Closing no-vig log loss 0.657–0.658 at every book; Pinnacle margin 2.6%, US books 4.0–4.8%. lowvig and betonlineag matched 99.2% of the time (swap one out for later seasons). The fetch log (`raw.odds_history_fetches`) stops a snapshot ever being bought twice, whatever the book list.
- **Props data sources (2026-09-29).** Live lines: only The Odds API's per-event endpoint (1 credit/game/market; empty responses free); the PC's key collects shots on goal at a morning + pre-game snapshot, 308–464 credits/month. History: ESPN sports.core propBets (free) — DraftKings on scattered 2025-26 dates and the playoffs (pre-game, 2–5 h before puck drop), ESPN BET Oct–Nov 2025 (last updates hours after puck drop: opening prices only); DK pairs are unlabelled with the over first (Brier check: stored 0.236 vs swapped 0.296). Odds API historical props need a paid plan (back to 2023-05). raw.skater_games PP/PK TOI, PP goals/assists and faceoffs were ALWAYS 0 from the boxscore load — now filled from api.nhle.com/stats/rest (`ingestion/nhl_stats.py`, a month per request). DK props overround ≈ 6.2% vs 4.3% for DK moneylines, so the 4% props threshold is probably too low. League shots per 60 fell 6.39 → 5.61 (2021-22 → 2025-26): props models must correct for drift.
- **Filling PP stats changes features: backfill every season, then rebuild and retrain (2026-09-29).** With PP TOI = 0, `pp_xgf_per60`/`pk_xga_per60` were a constant 99 (clipped), so the models never used them. Filling only some games makes them real there and 99 elsewhere — a feature that means different things in different seasons, and stored vectors that stop matching a live rebuild (seen on the clone: `TestSlateVectors` failed until 2025-26 features were rebuilt). Measured on the clone with a quarter of 2025-26 filled: walk-forward scores for 2025-26 unchanged to 4 decimals (training folds were all-constant, so the models ignore it), ~0.001 on the first 8 games of 2026-27 (training now includes filled rows). Upgrade order: `nhl_stats --season` for every season → `pipeline.py features` → `models.lgbm` / `models.totals`.
- **`test_baseline`'s DB test breaks at each season rollover (2026-10-01).** It requires folds[1:-1] to beat naive; once the new season has finished games it becomes the last fold and 2025-26 (baseline 0.6949 vs naive 0.6932) a middle one. Test fragility, not a model change.
- **Totals v2: more accurate, still no edge (2026-09-29).** Margin reweighting (ties 1.15×, one-goal 0.51×, two 0.72×, three 1.34× the independent joint; fitted per fold) plus a point-in-time within-season drift correction: NLL 2.1867 → 2.1801, DK-line O/U log loss 0.7051 → 0.6958 (n=1,011). The hardened gate gives the baseline the same margin fix: baseline 2.1787 still wins (model ahead in 2/5 seasons). GATE_PASSED stays False; `log_totals_gate()` logs it each run. `total_pmf` applies the weights by default, so the checker and alerts changed too. Next route: boost from the market total once 2026-27 O/U snapshots exist, judged by `market_check` (≥200 games, 95% confidence). `poisson_totals` is now v2: run `python -m models.totals` once after upgrading or totals predictions fail (non-fatally) for lack of a registry row.
- **Machine roles (2026-10-01, replacing the 2026-09-29 split).** The goal is that both machines run every job, each on its own key and its own DB (every job needs a paid key: picks + one props market is about 540–790 credits/month; on a free 500-credit key a machine runs `-Role picks` only — the Windows PC is on a free key as of 2026-10-02): the Mac installs all five `ops/launchd/` templates (daily, optional odds, close --due, props, props --due); the Windows PC runs `register-tasks.ps1 -Role all` (the same five; no `refresh`, which `daily` covers). `-Role picks` / `-Role props` remain for a single-job machine. Each machine issues and grades its own picks; a pick is graded only against closes on the machine that made it.
- **Never log a request URL that carries a key (2026-09-28).** `requests` puts the full URL, `?apiKey=` included, into HTTPError/ConnectionError messages, and urllib3 logs every request URL at DEBUG. Log a summary (status, reason, the API's own message), pass anything logged through a redactor, and filter urllib3's DEBUG lines. Logs written before this date may hold the key.

---

## 10. How to Resume Work

```bash
# Extract / clone, then:
python -m venv .venv && source .venv/bin/activate
pip install -e ".[ml,dashboard,dev]"
cp .env.example .env          # fill in ODDS_API_KEY if available

docker compose up -d          # start PostgreSQL (schema auto-applies)
python pipeline.py setup      # verify prerequisites
python pipeline.py backfill   # pull 5 seasons (~30-60 min)
python pipeline.py status     # confirm data populated

pytest tests/ -v              # 5 smoke tests should pass
```

**When switching to Claude Code:** point it at this file first. It contains every locked decision and the full Phase 2 plan. The three design docs (`File1`, `File2`, `File3`) should live in `/docs` for deep reference.

---

## 11. Reference Documents

- `File1_*` — Repository T2 template analyses (11 repos, full RQI scoring)
- `File2_Repository_Catalog_and_Cross_Reference_Analysis.docx` — Consolidated catalog, cross-reference matrices, gap analysis, integration map, risk register
- `File3_System_Design_Proposal.docx` — Full system design: architecture, schema, model specs, betting engine, roadmap, success criteria
- `README.md` — Quickstart and project structure
