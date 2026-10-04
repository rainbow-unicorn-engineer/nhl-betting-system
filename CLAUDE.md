# CLAUDE.md: read me first

## The mission

**Build the most optimal NHL betting engine possible: one that beats the sportsbooks' prices and proves it.**

This is not a demo or a hobby script. The goal is a system that consistently gets better prices than the closing market, sizes bets well, and grows a bankroll. Use whatever works:
- advanced statistics and forecasting
- machine learning, including pretrained models from Hugging Face where they help
- Monte Carlo simulation
- large backtests

This machine has the hardware for it: an Intel Core Ultra 9 285K (24 cores), 64 GB of RAM and an RTX 5080 (16 GB). Don't shy away from heavy computation.

**Act, don't just advise.** When something would make the engine better (more data, a better model, a missing feature), build it, test it honestly, and explain why at the end. Don't hand the owner a list of things that "could" be done. Decide, do, report.

**Winning is proven, not claimed.** A model that looks good on its own history but can't beat the market loses money. So every model claim is earned by these rules:

1. **Pre-register the test.** Write the pass bar before running anything, and report every variant tried, failures included.
2. **Walk forward.** Train only on the past and test on seasons the model has never seen (`models/baseline.py` `walk_forward_folds`).
3. **No leakage.** Nothing from the game itself or later may feed a prediction. Prove it with point-in-time tests that rewrite or delete later rows.
4. **Beat the market, not just a baseline.** The real bar is the sportsbooks' no-vig price and closing line value (CLV). Beating our own baselines is a step, not the finish.
5. **Get independent review.** A second reviewer reproduces the numbers and hunts for leakage before anything is merged.

## Who reads the output

The owner is new to sports betting. In every chat reply, doc, docstring and dashboard label, explain betting and statistics jargon in plain English the first time it appears, marked with →. Examples: vig → the sportsbook's built-in fee; CLV → whether we got a better price than the final pre-game price; log loss → a score for how wrong probabilities were, lower is better.

## How the system runs

- **The record machine is the Windows PC.** Picks to bet come from the PC's dashboard. The Mac runs the same jobs as a backup and its own paper ledger, but it is not where bets are taken from. Its own Odds API key keeps it off the PC's credits.
- **Odds API:** the PC's key is on the paid 20K plan (20,000 credits a month). Spend credits deliberately:
  - log what every bulk pull will cost before running it
  - keep a reserve for the live jobs
  - never re-pull what is already stored
- **Scheduling:** `ops/windows/register-tasks.ps1 -Role all -IncludeOdds` on the PC; the `ops/launchd/` templates on the Mac.
- **Database:** Postgres in Docker (`docker compose up -d`). It is never committed. Tests stay off the live database unless pointed at a disposable copy (`tests/conftest.py`).

## Repo conventions

- **Commits:** the author identity to use is in `CLAUDE.local.md`, which is git-ignored.
  - add NO co-author or Claude/AI lines
  - make one accurate commit per concern, with prefixes `feat:` `fix:` `docs:` `test:` `chore:`
- **Never commit:** `.env`, databases, `logs/`, `data/`, or anything in `private/`.
- **`private/` and `CLAUDE.local.md` are for anything personal:** the bettors, their platforms and balances, answer documents, notes. It is git-ignored. Repo docs refer to "bettor 1 / bettor 2", never real names.
- **Experiments:** dated, pre-registered, with results kept even when they fail. They are described in `experiments/README.md`; their code lives in the normal modules behind opt-in flags.
- **Agent/LLM features** (`docs/design/agents-governance-eval-rag.md`): don't merge until the design and implementation are finished and the owner signs off.
- **Freshness:** `README.md` and `PROJECT_CONTEXT.md` describe the live system. Update them in the same change as the code.

## Where things are

| Path | What it is |
|---|---|
| `PROJECT_CONTEXT.md` | Architecture, locked rules, phase status, learnings |
| `README.md` | How to run everything; every command and setting |
| `docs/data_sources.md` | Every data source, what it is used for, and what is still unused |
| `experiments/README.md` | Every model experiment and its result |
| `ingestion/` | One module per data source |
| `features/` | Point-in-time feature builders |
| `models/` | `lgbm` (win/loss), `totals` (over/under), `props_sog` (player shots), `baseline` (the original benchmark) |
| `betting/` | Engine (edge, Kelly, caps), recommend, settle, backtest, checker, promo, alerts |
| `dashboard/app.py` | The Streamlit dashboard. Open it with `ops/windows/open-dashboard.bat` |
