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
| 2026-10-04 | Same-game parlay joint pricer: win x over/under (variants A-F) | `betting/sgp.py` | Fails: no variant beats multiplying the two chances (A +0.00005 ± 0.00046 log loss over 2,408 games; best z +0.10). Underpowered by nature: an exactly right model would need ~30,000 games to pass. The checker keeps withholding same-game verdicts. |
| 2026-10-04 | TabPFN (pretrained table model from Hugging Face) for the moneyline: A feature, B market residual, C average with lgbm v2 | `experiments/tabpfn/` (separate `tabpfn` dependency group) | No variant passes: 0/5 seasons beat lgbm v2 by 2 SE. Best was C, a tie (+0.0001 ± 0.0005). All lose to the no-vig market. lgbm v2 stays. |
| 2026-10-04 | Our own expected-goals (xG) model, Layer A (variants X1 core, X2 + prior attempt) | `features/xg.py`, `models/xg.py` (opt-in) | Fails its gate vs MoneyPuck's xG on 605k held-out shots: AUC 0.761 vs 0.787, log loss +0.0086 (SE 0.0002); better calibrated (ECE 0.005 vs 0.011). Downstream: moneyline log loss +0.0018 (SE 0.0008) worse with our xG; props unchanged, and player xG features don't help props at all. Not adopted; MoneyPuck's xG stays. |
| 2026-10-05 | Free data loaders: shift charts, scratches and officials, Kalshi prices (acceptance checks S1-S3, G1-G3, K1-K4) | `ingestion/nhl_shifts.py`, `ingestion/nhl_game_info.py`, `ingestion/kalshi.py` | Data, not a model. Every check passes on the full backfill (7,984 games; 1,686 Kalshi events). A later check found repeated and wrong-game shift rows that the median in S3 could not see; the shift loader now cleans them and marks games that fail a per-game ice-time check 'suspect' (repair rehearsed on a copy: 7,981 ok, 3 suspect; the live run is pending). Kalshi's pre-game close sits within 0.55 points of DraftKings' no-vig close at the median. Details in `docs/data_sources.md` 2.13. |
| 2026-10-04 | Moneyline v3: real 2024-25 market input (V1), + power-play form (V2), + goalie roles (V3); priced backtests; morning-vs-close timing | `models/moneyline_v3.py`, `features/market_prices.py`, `features/power_play.py` | No variant passes; V0 stays. The experiment's V0 scored 0.6616 on all 6,993 games, flattered by 106 2023-24 market prices taken during the game (a leak found by the review); leaving those games out of the scoring gives 0.66484, but the model was still trained on them. The production model, re-trained with them masked, scores 0.66479 on all games (0.66520 on the clean ones) and is 0.0047 worse than the no-vig close (SE 0.0025); its 2025-26 DraftKings backtest is 487 bets, -3.5%. Best is V3 at +0.0004 ± 0.0012 vs V0, and V1-V3 are each worse than the no-vig close at 95%. Backtests: every V0 ROI interval includes 0, closing EV negative. Timing: neither morning nor close is better. See `docs/backtest_results.md`. |

## Re-running

```powershell
.venv\Scripts\python -m models.totals                 # totals walk-forward and gate (registers the production model)
.venv\Scripts\python -m models.props_sog --evaluate    # props walk-forward and gate (read-only)
.venv\Scripts\python -m models.props_sog --v3          # every v3 variant and the adoption rule (read-only)
.venv\Scripts\python -m models.props_market_check     # props vs the market (read-only)
.venv\Scripts\python -m models.totals --v3           # totals v3 experiment vs the market (read-only, ~20 s)
.venv\Scripts\python -m betting.sgp --evaluate          # same-game parlay pricer vs independence (read-only)
.venv\Scripts\python -m experiments.tabpfn.run         # TabPFN trial (read-only; needs the tabpfn group and its weights, see experiments/tabpfn/README.md)
.venv\Scripts\python -m models.xg --downstream          # our xG vs MoneyPuck, then the moneyline and props test (read-only)
.venv\Scripts\python -m models.moneyline_v3           # moneyline v3, priced backtests, timing (read-only)
```

The goalie variants run from Python: `models.totals.run_totals(register=False, variant="C", with_roles=True)`.
