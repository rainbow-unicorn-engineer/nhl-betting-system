# Historical odds: sources, coverage, and quality audit

`raw.historical_odds` holds one reference line per game for the full
backfill window. Built 2026-07-10 at zero cost. 7,904 of 7,945 completed
games covered (99.5%).

## Sources

1. **ESPN public summary API** (`pickcenter` block) — 6,094 games.
   Both moneylines + puck line + total. Ingester: `ingestion/espn_odds.py`
   (idempotent/resumable; re-run to top up new games).
2. **Kaggle mirror of the same ESPN data**
   (`jonathanncoletti/nhl-historical-game-data`) — 1,810 games, used to fill
   2024-25, which ESPN no longer serves (their Unibet→DraftKings provider
   transition year). Scraped contemporaneously by the dataset author, so it
   preserves what ESPN dropped. **Favorite's moneyline only** (`away_ml` or
   `home_ml` is NULL; side inferred from spread sign);
   `provider = 'espn-kaggle-onesided'`.

## Coverage by season

| Season   | Coverage | Provider |
|----------|---------:|----------|
| 2020-21  | 100%     | Unibet |
| 2021-22  | 100%     | Unibet |
| 2022-23  | 100%     | Unibet |
| 2023-24  | 100%     | Unibet |
| 2024-25  | 97.1%    | espn-kaggle-onesided (41 games unavailable anywhere free) |
| 2025-26  | 100%     | DraftKings + kaggle fill |

## Era caveat: Unibet lines are 3-way

Verified by implied-probability sums: Unibet rows (2020-21 → 2023-24)
average **0.829** (± .028) — these are 60-minute three-way lines (the
missing ~0.17 is the regulation-draw outcome). DraftKings rows average
**1.043** — true two-way moneylines with normal vig.

Implications:
- **Market feature**: normalize home/(home+away) implied probability —
  valid in both eras (audit below confirms).
- **Payout backtests**: only the DraftKings era (2025-26) plus our own
  `raw.odds_snapshots` going forward carry true bettable two-way prices.
  Do not simulate moneyline payouts against Unibet-era rows.

## Predictive quality audit

No-vig home implied probability vs actual outcomes (log loss; lower is
better; our Phase 2 model OOF = 0.6829, naive = 0.693):

| Season   | Market LL | Market acc |
|----------|----------:|-----------:|
| 2020-21  | 0.6548 | .620 |
| 2021-22  | 0.6409 | .644 |
| 2022-23  | 0.6567 | .605 |
| 2023-24  | 0.6399 | .625 |
| 2025-26  | 0.6795 | .560 |
| **Pooled** | **0.6529** | **.613** |

The market beats our current model by ~0.03 log loss everywhere —
including fold 5 (2025-26), where our model regressed to near-naive but
the market held 0.6795. This is the strongest single feature available
and the Phase 3 priority.
