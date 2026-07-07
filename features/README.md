# features/ — Feature Store (Layer 2)

Computes **point-in-time correct** features from `raw.*` tables and materializes
them into the `features.*` schema. Every value in every row must be computable
strictly **before puck drop** of the game it describes — no lookahead leakage,
ever. When in doubt, write the leakage test.

## Module layout

| Module | Populates | What it computes |
|--------|-----------|------------------|
| `util.py` | — | Shared helpers: `per60`, `safe_div`, `WINDOWS`, `season_stage` |
| `team_features.py` | `features.team_rolling` | Rolling team stats (GF/60, xGF%, PDO, special teams, …) over 5/10/20/40/82-game windows, prior games only |
| `goalie_features.py` | `features.goalie_rolling` | Rolling goalie stats (SV%, GSAx/60, …) with Buhlmann credibility shrinkage toward league mean |
| `schedule_features.py` | `features.matchup` (partial) | Rest days, back-to-backs, travel km, timezone shift, game number, season stage |
| `elo.py` | `features.matchup` (partial) | Pre-game Elo ratings (K=20, home-ice ≈ 50 pts, cross-season regression to mean) |
| `build_vectors.py` | `features.game_vector` | Joins everything, computes home−away differentials, adds market feature, stores the model-ready vector + label |
| `build_all.py` | all of the above | Orchestrates the builders in dependency order |

## Dependency order

```
team_features ─┐
goalie_features ─┼─> build_vectors -> features.game_vector
schedule_features ─┤
elo ─┘
```

## Invariants (enforced by tests)

1. **No lookahead:** rolling windows use only games with an earlier date
   (and earlier game_id on same-date ties). Elo/goalie ratings are pre-game.
2. **Early-season honesty:** if fewer than `window` prior games exist, use what
   exists and record the actual `games_played` — never pad or peek.
3. **Label discipline:** `home_win` is set only for completed games; NULL otherwise.
