# Promo-hedging calculator — design

> Basis: the two-bettor legal structure in `docs/texas_execution_options.md`
> (§ Two-bettor execution structure). Bettor A (partner, Louisiana) has
> full sportsbook + promo access in his own name with his own funds;
> Bettor B (Gavin, Texas) has Kalshi/Polymarket. Every leg is placed by
> its own bettor with his own bankroll; the calculator never pools stakes
> — it reports per-bettor P&L per outcome, plus an informational
> household net.

## What it answers

Given a promo (free bet, risk-free/first-bet-insurance, deposit match
with playthrough, profit boost) on some market at some odds, and one or
more hedge venues with their cost models:

1. the hedge stake that equalizes profit across outcomes,
2. the guaranteed conversion (dollars extracted per promo dollar),
3. which available market/odds maximizes conversion,
4. per-bettor cash flow in each outcome (who is up, who is down, when).

## Core math (implemented in `betting/promo.py`, hand-verified tests)

All in decimal odds. `d_b` = bonus-leg odds, `d_h` = hedge-side
*effective* decimal after venue costs.

- **Free bet (stake not returned)** of size `F`:
  `h = F(d_b − 1)/d_h`; guaranteed `= F(d_b − 1)(1 − 1/d_h)`;
  conversion rate `= guaranteed / F`. Conversion rises with longshot
  bonus odds — the classic ~70-80% at d_b ≥ 4 with a cheap hedge.
- **Risk-free bet** `B` with a refund-if-lose free bet valued at `c`
  per dollar (c ≈ the free-bet conversion above, ~0.7):
  `h = B(d_b − c)/d_h`; guaranteed `= B(d_b − 1) − h`.
- **Profit boost** β on a cash stake `B`: boosted `d' = 1 + (d_b−1)(1+β)`,
  `h = B·d'/d_h`, guaranteed `= B(d' − 1) − B·d'/d_h`; positive iff the
  boost outruns the round-trip vig.
- **Deposit match / site credit with rollover** `R×`: modeled as value
  `≈ match × (1 − hold_per_leg)^R_legs` grinding near-even legs; v2
  refinement, reported as an estimate, not a guarantee.

## Venue cost models (`effective_decimal`)

| Venue | Effective decimal for the hedge side |
|---|---|
| Sportsbook | `decimal(american)` as offered |
| Kalshi taker | `1 / (q + 0.07·q(1−q))`, q = contract price |
| Kalshi maker | `1 / (q + maker_fee)` (fee ≈ 0, queue risk instead) |
| Polymarket taker | `1 / (q + 0.05·q(1−q))` |
| Polymarket maker | `1 / q` (zero fee + rebate; queue risk) |

Fee coefficients are parameters, not constants — Polymarket already
moved 0.03 → 0.05 in July 2026; re-pull each season.

## Constraints the calculator models

- **Per-leg bettor + venue tags**: a hedge plan is a list of legs, each
  `(bettor, venue, market, price, stake)`. Legality is structural: legs
  tagged bettor-A must be LA-book legs; bettor-B legs must be
  TX-prediction-market legs. Single-bettor plans (A hedges A on another
  LA book) are the default and cleanest.
- **Slippage between legs**: prices move between placements; a slippage
  parameter widens the guaranteed band. (With A at home in LA, both
  legs can be placed near-simultaneously — this shrinks vs the old
  trip-based design but stays an input.)
- **Binary-contract mismatch**: Kalshi settles 2-way incl. OT; NHL
  3-way or puck-line promos need synthetic hedges (ML + total legs) —
  the calculator flags residual exposure rather than pretending
  a perfect hedge exists.
- **Account longevity (commercial, not legal)**: books limit promo
  hedgers; an expected-account-lifetime input discounts long promo
  sequences (e.g. rollover grinds) vs instant conversions.
- **Withdrawal float**: per-venue lockup days on each bettor's funds.

## What it deliberately does not do

- No pooled bankroll, no "transfer" leg between bettors, no
  stake-netting across A and B — per-bettor P&L only.
- No modeling of proxy placement in any form.

## Roadmap

- v1 (implemented): free bet, risk-free, profit boost + venue cost
  models + best-odds conversion search + per-bettor outcome table. CLI.
- v2: rollover/playthrough sequencing, model-aware skewed hedges (leave
  +EV exposure unhedged when `models.predictions` disagrees with the
  market), correlation-aware synthetic hedges priced from the totals
  PMFs, promo-inventory tracker per book.
