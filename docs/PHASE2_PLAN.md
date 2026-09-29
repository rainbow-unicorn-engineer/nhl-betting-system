# PHASE2_PLAN.md — Feature Store + Baseline Model

> **For:** Claude Code (or any developer) picking up Phase 2.
> **Prerequisite:** Phase 1 verified end-to-end (`python pipeline.py status` shows populated `raw.*` tables with ≥5 seasons of games + boxscores).
> **Read first:** `PROJECT_CONTEXT.md` (locked decisions), `docs/File3_System_Design_Proposal.docx` §4–5 (feature store + model specs).
> **Success gate for Phase 2:** baseline logistic regression achieves walk-forward CV log loss **< 0.69** (must beat the naive ~0.693 coin-flip baseline) with a calibration plot produced.

Each task below is sized to be one commit. Work sequentially — later tasks depend on earlier ones. Write a test with each task and commit only when it passes.

---

## Task 0 — Phase 2 branch + module skeleton
- [ ] Create branch `phase-2-features`
- [ ] Create `features/` module files: `team_features.py`, `goalie_features.py`, `schedule_features.py`, `elo.py`, `build_vectors.py`
- [ ] Add a `features/README.md` describing the module layout
- [ ] Commit: `chore: phase 2 feature module skeleton`

---

## Task 1 — Feature engineering helpers + config
- [ ] Create `features/util.py` with:
  - `per60(stat, toi_seconds)` helper
  - `safe_div(num, den, default=None)` helper
  - Rolling-window constants: `WINDOWS = [5, 10, 20, 40, 82]`
  - Season-stage classifier: `EARLY (games 1-20)`, `MID (21-62)`, `LATE (63-82)`
- [ ] Unit test each helper (edge cases: zero TOI, zero denominator, game 1 of season)
- [ ] Commit: `feat: feature engineering helpers`

---

## Task 2 — Team rolling features
Populates `features.team_rolling` (one row per game_id × team × window).

- [ ] For each team, compute rolling stats over each window using **only games prior to the target game** (point-in-time correct — no leakage):
  - Scoring: `gf_per60`, `ga_per60`
  - Shot quality: `xgf_pct` (needs shots table), `cf_pct`, `ff_pct`
  - Efficiency: `sh_pct`, `sv_pct`, `pdo`
  - Special teams: `pp_pct`, `pk_pct`, `pp_xgf_per60`, `pk_xga_per60`
  - Other: `fow_pct`, `pim_per60`
  - `games_played` (actual number of prior games available in window — may be < window early in season)
- [ ] Handle early-season case: if fewer than window games exist, use what's available and record actual `games_played`
- [ ] Test: pick one team + one mid-season game, verify a hand-computed rolling GF/60 matches
- [ ] Test: verify NO row uses data from the target game or later (leakage check)
- [ ] Commit: `feat: team rolling features`

---

## Task 3 — Goalie rolling features + Buhlmann credibility
Populates `features.goalie_rolling`. This is the module that fills the biggest gap in the reviewed repos.

- [ ] Compute per-goalie rolling stats over last N starts: `sv_pct`, `gsax_per60`, `hd_sv_pct`, `fenwick_sv_pct`, `xga_per60`
- [ ] Implement Buhlmann credibility shrinkage:
  - `Z = n / (n + k)` where `n` = starts in window
  - Estimate `k` empirically from between-goalie variance vs within-goalie variance across a training season (document the estimate in code comments)
  - `shrunk_sv_pct = Z * goalie_sv_pct + (1 - Z) * league_avg_sv_pct`
  - Same shrinkage for `shrunk_gsax`
- [ ] This automatically handles rookies/backups/early-season (low n → shrinks toward league mean)
- [ ] Test: goalie with 1 start should shrink heavily toward league average; goalie with 40 starts should stay near their raw number
- [ ] Commit: `feat: goalie rolling features with Buhlmann shrinkage`

---

## Task 4 — Schedule + context features
Populates part of `features.matchup`.

- [ ] `rest_days` per team (days since that team's previous game; 0 = back-to-back)
- [ ] `b2b` boolean flag
- [ ] `travel_km`: great-circle distance from previous game venue to current venue (need team venue lat/long — add to `raw.teams` seed data)
- [ ] `tz_shift`: hours of timezone change from last game
- [ ] `home_game_num` / `away_game_num`: game number within season for each team
- [ ] `season_stage`: EARLY/MID/LATE
- [ ] Seed `raw.teams` with venue coordinates + timezones (32 teams — one-time data file `db/seed_venues.sql`)
- [ ] Test: a team playing consecutive calendar days has `b2b=True, rest_days=0`
- [ ] Commit: `feat: schedule and travel features`

---

## Task 5 — Elo rating system
Populates `home_elo` / `away_elo` in `features.matchup`.

- [ ] Implement standard Elo: `K=20`, home-ice advantage ≈ 50 points
- [ ] Process all games chronologically, updating ratings after each
- [ ] Store the pre-game Elo for each team (point-in-time correct)
- [ ] Carry ratings across seasons with regression to mean (e.g., 1/3 toward 1500 at season start)
- [ ] Test: a team that wins should gain rating; ratings should be pre-game not post-game
- [ ] Commit: `feat: Elo rating system`

---

## Task 6 — game_vector materialization
Populates `features.game_vector` — the final model-ready table.

- [ ] Join team_rolling (both teams, all windows) + goalie_rolling (both starters) + matchup features
- [ ] Compute **home − away differentials** for each feature (this is the modeling representation)
- [ ] Add market feature: opening-line no-vig implied probability (from `raw.odds_snapshots`, earliest snapshot per game)
- [ ] Store ordered `feature_names[]` and `feature_vector[]` (double precision arrays)
- [ ] Set label `home_win` = (home_score > away_score) for completed games; NULL for future games
- [ ] Handle missing starters gracefully (use team-level goalie average as fallback, flag it)
- [ ] Test: verify vector length matches feature_names length; verify no NULLs in feature_vector for completed games with known starters
- [ ] Test: leakage check — every feature must be computable strictly before puck drop
- [ ] Commit: `feat: game_vector materialization`

---

## Task 7 — Feature pipeline orchestration
- [ ] Add `features/build_all.py` that runs Tasks 2–6 in dependency order for a given season or date range
- [ ] Wire into `pipeline.py`: add command `python pipeline.py features [--season YYYYYYYY | --all]`
- [ ] Add `python pipeline.py features` to the daily refresh chain (after data ingestion)
- [ ] Test: run on one season end-to-end, confirm `features.game_vector` populated with expected row count
- [ ] Commit: `feat: feature build orchestration + pipeline integration`

---

## Task 8 — Baseline logistic regression model
Populates `models.model_registry` and validates the approach.

- [ ] Create `models/baseline.py`
- [ ] Load `features.game_vector` for completed games into a training matrix
- [ ] Implement **walk-forward CV** (this is mandatory — no random splits):
  - Expanding window: train on seasons up to N, validate on season N+1
  - Apply a 1-week purge gap between train and validation to prevent leakage
  - Roll forward through all available seasons
- [ ] Fit `sklearn.LogisticRegression` (start simple; standardize features first)
- [ ] Report per-fold and aggregate: **log loss**, accuracy, Brier score, AUC
- [ ] Produce a calibration/reliability plot (save to `models/artifacts/`)
- [ ] Register the model in `models.model_registry` with CV metrics
- [ ] **GATE:** aggregate walk-forward log loss must be < 0.69. If not, debug feature leakage or data quality before proceeding.
- [ ] Test: verify walk-forward never trains on data at or after validation period (the critical leakage test)
- [ ] Commit: `feat: baseline logistic regression with walk-forward CV`

---

## Task 9 — Phase 2 wrap-up
- [ ] Write `models/README.md` documenting the baseline results (log loss, accuracy, calibration)
- [ ] Update `PROJECT_CONTEXT.md` §8 status table: mark Phase 2 complete, note the baseline log loss achieved
- [ ] Add a `docs/phase2_results.md` with the calibration plot and per-season CV metrics
- [ ] Open a PR from `phase-2-features` → `main`, summarizing what was built and the gate result
- [ ] Commit + PR: `docs: phase 2 results and status update`

---

## Guardrails (apply to every task)

- **No lookahead leakage, ever.** Every feature must be computable strictly before puck drop. When in doubt, write the leakage test.
- **Point-in-time correctness.** Rolling features use only prior games. Elo and goalie ratings are pre-game values.
- **`nhl-api-py` imports as `nhlpy`.** Do not "fix" it.
- **Calibration matters more than accuracy.** A well-calibrated 58% model beats an overconfident 61% model for betting.
- **Commit at each working milestone.** Small, tested, reversible commits.
- **Target metrics (from File 3 §9):** baseline log loss < 0.69 now; production target < 0.675 in Phase 3.

---

## What Phase 2 explicitly does NOT include (deferred to Phase 3)

- The LightGBM production model (Phase 3)
- xG model training (Phase 3 — use MoneyPuck pre-computed xG for now)
- PMF totals pricing (Phase 3)
- The betting engine, Kelly staking, CLV tracking (Phase 3)
- Streamlit dashboard (Phase 3)

Stay scoped. Phase 2 is: **features + a calibrated baseline that beats naive.** That's the whole job.
