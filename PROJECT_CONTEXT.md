# PROJECT_CONTEXT.md

> **Purpose:** This file is the single source of truth for the NHL Sports Betting Predictive System. It captures locked architectural decisions, the technology stack, current phase status, and hard-won learnings so that any developer — or any AI assistant (e.g., Claude Code) — can pick up the project with full context. Keep this file updated as decisions change.

**Last updated:** July 9, 2026
**Project owner:** Gavin
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
| The Odds API | Live odds from 15+ books | Free tier 500 req/mo | Uses full team names — see mapping in `ingestion/odds_api.py` |
| Hockey-Scraper | Historical PBP + shifts (2007–2023) | pip (`hockey_scraper`) | Historical backfill ONLY. Not for live/current use |
| Daily Faceoff / LeftWingLock | Confirmed starting goalies, lineups, injuries | Web scrape / manual | Needed for goalie-conditioned predictions |

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

### Database schema (4 namespaces, 15 tables)
- `raw.*` — games, teams, players, rosters, shots, skater_games, goalie_games, odds_snapshots, shifts
- `features.*` — team_rolling, goalie_rolling, matchup, game_vector
- `models.*` — model_registry, predictions
- `betting.*` — recommendations, placed_bets, bankroll_log

Full column definitions in `db/schema.sql`.

---

## 7. Betting Engine Rules (locked defaults)

- **Edge thresholds:** ML ≥ 2.5%, totals ≥ 3.0%, props ≥ 4.0%
- **Staking:** Quarter-Kelly (f = 0.25). `stake = 0.25 × kelly × bankroll`
- **Exposure caps:** Max 2% bankroll per bet; max 10% per day; max 3 correlated bets per game
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
| **Phase 3** | Production LightGBM + PMF totals + betting engine + Streamlit dashboard | ⬜ NEXT |
| **Phase 4** | Live betting, iteration, player props, live/in-game model, cloud migration | ⬜ Pending |

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
