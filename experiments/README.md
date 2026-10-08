# Experiments

Each experiment wrote down its pass bar before running, reported every variant it tried, and was checked by an independent reviewer for leakage (→ accidentally using information from the game itself or later) and for reproducibility.

The code stays in the normal modules, so it keeps working as they change. A model that passed became the default. A model that didn't sits behind an opt-in flag, so the result can be re-run. This page is the index. Each module's `STATUS` docstring has the full numbers.

Terms used below:
- **Log loss / NLL** → a score for how surprised a model's probabilities were by what actually happened. Lower is better.
- **Paired SE** → the size of the random wobble ("noise") in a difference between two models, measured on the same games. A gap of 2 SE or more is unlikely to be luck.
- **Calibration / ECE** → whether "40%" really happens about 40% of the time. ECE is the average miss in percentage points.
- **No-vig market price** → the sportsbook's probability with its built-in fee taken out.

## Index

| Date | Experiment | Code | Result |
|---|---|---|---|
| 2026-07 | Totals model v1 (Poisson goals per team) | `models/totals.py` | Failed its gate: lost to "recent league scoring rate" (NLL 2.1867 vs 2.1815). No totals bets. |
| 2026-10-01 | Totals model v2 (tie-margin reweighting + in-season drift correction) | `models/totals.py` | Still fails, narrowly: 2.1801 vs 2.1787. No totals bets. |
| 2026-10-01 | Goalie starter-role inputs for totals (variants A-D) | `features/goalie_role.py`, `models/totals.py` (`variant=`, opt-in) | No variant passes. Best was C at +0.0003 ± 0.0010 vs baseline. Backups really do allow ~0.12 more goals, but it doesn't improve predictions. |
| 2026-10-02 | Shots-on-goal props model v1 | `models/props_sog.py`, `features/player_shots.py` | Passed as a forecaster: beat both baselines in 5/5 seasons, about 19 SE. |
| 2026-10-03 | Props model v2: in-season drift correction | `models/props_sog.py` (`DRIFT_CORRECT`) | Adopted by its rule, but marginally: −1.8 SE vs v1, with worse pooled calibration. The drift's real cause is in the baseline. |
| 2026-10-03 | Props market check vs ESPN BET and DraftKings | `models/props_market_check.py` | Does not beat the market: the book's no-vig price had lower log loss, about 3 SE pooled. Flat-bet ROI intervals all include 0. |
| 2026-10-04 | Moneyline v3: real 2024-25 market input (V1), + power-play form (V2), + goalie roles (V3); priced backtests; morning-vs-close timing | `models/moneyline_v3.py`, `features/market_prices.py`, `features/power_play.py` | No variant passes; V0 stays. The experiment's V0 scored 0.6616 on all 6,993 games, flattered by 106 2023-24 market prices taken during the game (a leak found by the review); leaving those games out of the scoring gives 0.66484, but the model was still trained on them. The production model, re-trained with them masked, scores 0.66479 on all games (0.66520 on the clean ones) and is 0.0047 worse than the no-vig close (SE 0.0025); its 2025-26 DraftKings backtest is 487 bets, -3.5%. Best is V3 at +0.0004 ± 0.0012 vs V0, and V1-V3 are each worse than the no-vig close at 95%. Backtests: every V0 ROI interval includes 0, closing EV negative. Timing: neither morning nor close is better. See `docs/backtest_results.md`. |

## Re-running

```powershell
.venv\Scripts\python -m models.totals                 # totals walk-forward and gate (registers the production model)
.venv\Scripts\python -m models.props_sog --evaluate    # props walk-forward and gate (read-only)
.venv\Scripts\python -m models.props_market_check     # props vs the market (read-only)
.venv\Scripts\python -m models.moneyline_v3           # moneyline v3, priced backtests, timing (read-only)
```

The goalie variants run from Python: `models.totals.run_totals(register=False, variant="C", with_roles=True)`.
