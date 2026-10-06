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
| 2026-10-04 | Totals v3: start from the market's over/under price (T1), plus a Dixon-Coles low-score term (T2; → one number that makes 0-0, 1-0, 0-1 and 1-1 scores more or less likely than independent team scores would), plus goalie-role inputs (T3) | `models/totals.py` (`--v3`, opt-in) | No variant passes. Best was T3 at −0.0023 ± 0.0016 vs baseline (3/5 seasons), and every variant's over/under log loss is worse than the no-vig market (T3 +0.0024 ± 0.0008, 6,143 games). The market's price alone beats the baseline in 5/5 seasons; the model's corrections on top only make it worse. The low-score term came out at about 0. v2 stays the default; no totals bets. |
| 2026-10-05 | Props model v3: power-play (PP → his team has an extra skater after an opponent's penalty) features (P1), usage proxies (P2), a fixed exposure baseline B3 (P3) | `models/props_sog.py` (`VARIANTS`, `--v3`), `features/player_shots.py`, `models/props_market_check.py` | P3 adopted by its rule: beats v2 by 0.0018 log loss a player-game (~13 SE, 5/5 seasons). P1 helped narrowly (−2.4 SE); P2 added nothing. B3 is a better baseline but did not remove the +0.05 shots-a-game over-prediction. Market check re-run on every shots price row (incl. late playoffs): still does NOT beat the market (pooled +0.0021 log loss, SE 0.0012; v2 was +0.0053). The gap is in per-player means, not level or spread. Registration stays off. |

## Re-running

```powershell
.venv\Scripts\python -m models.totals                 # totals walk-forward and gate (registers the production model)
.venv\Scripts\python -m models.props_sog --evaluate    # props walk-forward and gate (read-only)
.venv\Scripts\python -m models.props_sog --v3          # every v3 variant and the adoption rule (read-only)
.venv\Scripts\python -m models.props_market_check     # props vs the market (read-only)
.venv\Scripts\python -m models.totals --v3           # totals v3 experiment vs the market (read-only, ~20 s)
```

The goalie variants run from Python: `models.totals.run_totals(register=False, variant="C", with_roles=True)`.
