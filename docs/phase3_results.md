# Phase 3 modeling results — market feature + LightGBM + calibration

**Run date:** 2026-07-10 · **Model:** `lgbm_market v2` · **Data:** 7,945
games, 2020-21 → 2025-26, 109 features (107 Phase 2 + 2 market)

## Headline

| Metric (pooled OOF, walk-forward) | Phase 2 baseline | Phase 3 | Target |
|---|---:|---:|---:|
| Log loss | 0.6829 | **0.6607** | beat 0.6829 ✅ |
| Accuracy | 0.582 | **0.601** | — |
| Brier    | 0.2433 | **0.2345** | — |
| AUC      | 0.611 | **0.636** | — |
| ECE      | 0.0506 | **0.0146** | < 0.02 ✅ |

## Per fold

| Val season | LL | naive | market | acc | AUC | ECE |
|---|---:|---:|---:|---:|---:|---:|
| 2021-22 | 0.6497 | 0.6899 | 0.6409 | .633 | .679 | .050 |
| 2022-23 | 0.6573 | 0.6930 | 0.6567 | .609 | .654 | .028 |
| 2023-24 | 0.6442 | 0.6904 | 0.6399 | .617 | .666 | .038 |
| 2024-25 | 0.6637 | 0.6867 | (no line) | .600 | .623 | .037 |
| 2025-26 | 0.6886 | 0.6932 | 0.6795 | .543 | .570 | .050 |

Every fold beats naive. Market-covered folds land within ~0.01 of the
market itself; 2024-25 (market-blind fallback) beats the Phase 2 baseline's
own fold there (0.6683).

## Architecture (what the walk-forward evidence forced)

1. **Boost from the market, don't feed it as a feature.** As feature #108,
   LightGBM couldn't fully recover the market's signal (fold-1 raw 0.679 vs
   market 0.641). With `init_score = logit(market prob)` the booster learns
   residual corrections and market-level performance becomes the floor.
2. **Two models routed by `market_available`.** A market-offset model
   cannot predict from scratch, and 2024-25 has no free line anywhere
   (`docs/historical_odds.md`). M scores lined games; a market-blind
   fallback F scores the rest. Each is trained/early-stopped/calibrated
   only on its own regime's rows (regime-mixed calibration tails cost
   +0.017 on fold 5 before this).
3. **Temperature scaling, not per-fold isotonic.** Isotonic on ~150-1,000
   game calibration tails overfit and *worsened* pooled OOF by +0.02
   (0.674 raw → 0.695). One-parameter temperature scaling gets ECE to
   0.0146. A final isotonic pass, if ever needed, belongs at the strategy
   layer, fit on pooled OOF predictions.

## Honest reading

- The model ≈ market + small corrections on lined games. That is the
  expected ceiling for public features; the betting edge must come from
  calibration + selective disagreement + line shopping, not from
  out-predicting the closing line wholesale.
- 2025-26 remains the weakest fold (AUC .570) — flagged in Phase 2, still
  true with market data. Watch whether 2026-27 continues the pattern.
- Unibet-era market values are 3-way regulation lines; they normalize into
  valid probabilities (audited: LL 0.6529 pooled) but payout backtests must
  use the DraftKings era + live snapshots only.

## Betting half: engine + payout backtest

`betting/engine.py` implements the locked §7 rules as pure, hand-verified
functions (no-vig fair probs, full/quarter Kelly, 2%-per-bet + 10%-per-day
caps, edge threshold); `betting/backtest.py` runs walk-forward OOF
probabilities through that exact engine against true DraftKings prices
(the only bettable-price era — 1,018 games, 2025-26), compounding a
bankroll chronologically. `dashboard/app.py` (Streamlit) shows
recommendations, model registry, backtest, and bankroll.

Backtest at the locked 2.5% threshold: 363 bets, hit 51.0%, quarter-Kelly
ROI +1.4%, flat-stake ROI −6.0%, max drawdown 17.9%. Kelly-positive with
flat-negative on 363 bets = no demonstrated edge at that threshold (both
within ±5% noise). The informative result is the edge-bucket breakdown:

| Claimed edge | n | hit | flat ROI |
|---|---:|---:|---:|
| 2.5–4% | 198 | .460 | −16.8% |
| 4–6%   | 118 | .534 | −1.6% |
| 6–9%   | 46  | .652 | **+26.6%** |

Strictly monotonic: small model-market disagreements are model noise;
large ones carry real signal. Read against the fold analysis, that's
coherent — the model ≈ market + corrections, so only *confident*
corrections are bets. **Implication: the 2.5% moneyline threshold is too
permissive for this model; ~5-6% looks right.** That number must NOT be
locked from this sample (choosing a threshold on the season being scored
is curve-fitting): validate via 2026-27 paper trading before real stakes,
and note the test season was also the model's weakest fold, betting into
near-closing prices with no line shopping.

## Remaining Phase 3 scope

- PMF totals model (over/under markets) — not started
- Daily recommendation job writing betting.recommendations in-season
- Isotonic pass at the strategy layer once pooled live predictions exist

## Reproduce

```
python -m models.lgbm            # walk-forward + registry upsert
python -m betting.backtest       # strategy simulation (true-price era)
streamlit run dashboard/app.py   # control room
python -m pytest tests/
```
