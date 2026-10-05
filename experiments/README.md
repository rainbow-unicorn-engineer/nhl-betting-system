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
| 2026-10-04 | Same-game parlay joint pricer: win x over/under (variants A-F) | `betting/sgp.py` | Fails: no variant beats multiplying the two chances (A +0.00005 ± 0.00046 log loss over 2,408 games; best z +0.10). Underpowered by nature: an exactly right model would need ~30,000 games to pass. The checker keeps withholding same-game verdicts. |

## Re-running

```powershell
.venv\Scripts\python -m models.totals                 # totals walk-forward and gate (registers the production model)
.venv\Scripts\python -m models.props_sog --evaluate    # props walk-forward and gate (read-only)
.venv\Scripts\python -m models.props_market_check     # props vs the market (read-only)
.venv\Scripts\python -m betting.sgp --evaluate          # same-game parlay pricer vs independence (read-only)
```

The goalie variants run from Python: `models.totals.run_totals(register=False, variant="C", with_roles=True)`.
