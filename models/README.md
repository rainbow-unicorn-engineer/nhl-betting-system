# models/ — Model Layer (Layer 3)

Consumes `features.game_vector`, emits calibrated probabilities into
`models.*`. Every model here must respect walk-forward discipline: no random
splits, scaling/fitting on training folds only, validation strictly in the
future of training.

## Modules

| Module | What it does |
|---|---|
| `baseline.py` | Phase 2 baseline: logistic regression over the 107-feature game vectors, expanding-window walk-forward CV with a 7-day purge gap. Registers itself in `models.model_registry` and writes a reliability plot to `artifacts/`. |

## Baseline results (Phase 2 gate: log loss < 0.69)

**PASSED — pooled out-of-fold log loss 0.6829** over 6,993 validation games
(5 season folds, 2021-22 → 2025-26). Accuracy 0.582, Brier 0.2433,
AUC 0.611, ECE 0.051. Per-fold table and analysis: `docs/phase2_results.md`.

Run it:

```bash
python -m models.baseline        # trains, evaluates, plots, registers
```

Registry entry: `baseline_logreg v1` (`is_active = FALSE` — it is a
validation baseline, not a production model). The feature-set hash pins the
exact 107-feature ordering the metrics were computed on.

## Phase 3 (next)

- LightGBM game-outcome model + isotonic calibration (target LL < 0.675, ECE < 0.02)
- Goals-allowed PMFs → totals pricing
- Market features (opening no-vig implied probability) once odds snapshots accumulate
