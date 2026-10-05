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
| 2026-10-04 | Our own expected-goals (xG) model, Layer A (variants X1 core, X2 + prior attempt) | `features/xg.py`, `models/xg.py` (opt-in) | Fails its gate vs MoneyPuck's xG on 605k held-out shots: AUC 0.761 vs 0.787, log loss +0.0086 (SE 0.0002); better calibrated (ECE 0.005 vs 0.011). Downstream: moneyline log loss +0.0018 (SE 0.0008) worse with our xG; props unchanged, and player xG features don't help props at all. Not adopted; MoneyPuck's xG stays. |

## Re-running

```powershell
.venv\Scripts\python -m models.totals                 # totals walk-forward and gate (registers the production model)
.venv\Scripts\python -m models.props_sog --evaluate    # props walk-forward and gate (read-only)
.venv\Scripts\python -m models.props_market_check     # props vs the market (read-only)
.venv\Scripts\python -m models.xg --downstream          # our xG vs MoneyPuck, then the moneyline and props test (read-only)
```

The goalie variants run from Python: `models.totals.run_totals(register=False, variant="C", with_roles=True)`.
