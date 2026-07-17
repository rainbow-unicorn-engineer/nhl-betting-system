# Legal sports-betting execution options for a Texas resident (as of July 2026)

> Research deliverable for the execution layer + promo-hedging calculator.
> Texas has no legal online sportsbook and none before 2028 at the earliest;
> the system's live execution therefore runs on CFTC-regulated prediction
> markets (legal in TX) plus trip-based sportsbook accounts in drivable
> states. Re-verify the items in the last section before October 2026.

## 1. Prediction markets legal in Texas

### Kalshi — the anchor option
- **TX legality:** Legal and available. Texas has sent no cease-and-desist
  and filed no lawsuit (unlike ~15 other states); the AG declined to join
  multi-state briefs against CFTC authority. Watch item: Lt. Gov. Patrick
  directed a Senate committee (March 2026) to study closing the
  "prediction market loophole" for the **2027 session** — no hearings
  scheduled yet. Age 18+.
  ([Texas Tribune 2026-05-01](https://www.texastribune.org/2026/05/01/texas-prediction-market-regulations-kalshi-gambling-sports-betting/),
  [Covers](https://www.covers.com/betting/prediction-sites/usa/texas-kalshi),
  [SI](https://www.si.com/prediction-markets/reviews/kalshi-texas))
- **NHL markets:** Multiyear **official NHL partnership since Oct 2025**.
  Single-game moneylines, puck lines, totals, player props (goals, assists,
  points, SOG, hits, blocks, TOI, faceoffs), goal-scorer markets, futures,
  live trading. Moneylines are liquid; **props/totals thinner than NBA/NFL**
  — treat the bid-ask spread as a real cost.
  ([Kalshi NHL](https://kalshi.com/category/sports/hockey/nhl),
  [RotoWire](https://www.rotowire.com/prediction-markets/nhl))
- **Fees:** taker ≈ `0.07 × p × (1−p)` per contract (max ~1.75¢ at 50¢ ≈
  3.5% of stake at even odds; ~0.7% at 90¢). **Maker fees far lower**
  (~0.08–2%), no settlement fee. vs sportsbook −110/−110 ≈ 4.5% hold:
  maker fills + line shopping can beat sportsbook pricing.
  ([fee schedule](https://kalshi.com/fee-schedule))
- **Rails:** free ACH both directions; ~2% debit card fee; USD, no crypto.

### Polymarket US — cheaper taker, second book
- Legal in TX: bought QCX ($112M, Jul 2025), CFTC amended designation
  2025-11-25, launched **Polymarket US 2025-12-03** as a regulated DCM.
  Challenged in ~9 states, **not Texas**. Full KYC, USD via FCM.
- Also an official NHL partner (late 2025). Game markets + futures;
  props thinner than Kalshi.
- **Fees:** sports taker `0.05 × p × (1−p)` (max ~2.5% of stake at 50¢ —
  cheaper than Kalshi taker). Rate **rose 0.03 → 0.05 in July 2026** —
  fee drift is real; re-pull each season. **Makers pay zero** + 15% rebate
  of taker fees. ([docs.polymarket.us/fees](https://docs.polymarket.us/fees))

### Robinhood event contracts
- Kalshi under the hood (KalshiEX LLC is the DCM; may migrate to LedgerX
  in 2026). Available in TX. **$0.01 Robinhood + $0.01 Kalshi per
  contract per side ≈ 4% at 50¢ — strictly worse than direct Kalshi.**
  No analytical reason to use it.

### Crypto.com ("OG", also powers DraftKings Predict / Fanatics)
- TX-available, CFTC-regulated; withdrew from 9 hostile states, not TX.
  NHL coverage thinner than Kalshi; viable line-shopping venue only.

### ForecastEx (Interactive Brokers)
- **Exited sports entirely (March 2026); never listed NHL.** Irrelevant.

### Novig & ProphetX — near-zero-vig P2P exchanges
- Both available in TX. **Novig got CFTC approval 2026-06-16** (sweepstakes
  → regulated prediction market); ProphetX runs as a sports-only prediction
  market nationwide. Near-zero commission; **NHL liquidity unproven —
  verify on October slates.**

## 2. Drivable states with legal online sportsbooks

Geolocation restricts only bet *placement*: register, deposit, withdraw,
and manage accounts from Texas; be physically in-state only at bet time.
All sportsbook states are 21+.

| State | Status (Jul 2026) | Books | Drive | Notes |
|---|---|---|---|---|
| **Louisiana** | Legal, 55/64 parishes | FD, DK, BetMGM, Caesars, bet365, Fanatics, theScore | Houston→Lake Charles ~2h; Dallas→Shreveport ~3h | Primary trip state. Sabine Parish (Toledo Bend) opted out — cross on I-10/I-20 |
| **Arkansas** | Legal; **DK+FD launched 2026-03-20** | DK, FD, BetSaracen | Dallas→Texarkana ~2.75h | Freshest market = fresh new-user promos |
| **Kansas** | Legal, 6 apps | DK, FD, BetMGM, Caesars, Fanatics, ESPN BET | Dallas→border ~5h | Licenses run to 2027-08-31; market may restructure |
| **Colorado** | Legal, deepest menu | 9+ apps | Amarillo→border ~3.5h | Biggest promo stack (~$5,600 aggregate); Panhandle/ski trips only |
| **Arizona** | Legal | all majors | El Paso→border ~4.5h | El Paso only |
| **Oklahoma** | **Illegal.** HB 1047 failed Senate 21–27 (Apr 2026); **ballot measure possible 2026-11-03** | — | Dallas→border ~1.25h | Re-check after Nov 2026 election |
| **New Mexico** | Retail tribal only, no mobile | 5 casinos | nearest book ~2h from El Paso | Retail-only kills hedging workflow |

## 3. DFS / pick'em (PrizePicks, Underdog)

Operating openly in TX for a decade despite the unrescinded 2016 Paxton AG
opinion; zero enforcement. Pick'em is the closest thing to TX props action
but throttles winners and can't isolate single-game sides — poor hedge
legs. PrizePicks "Predict" (Kalshi-powered) runs in TX — the two worlds
are converging.

## 4. Sweepstakes books (Fliff et al.)

Legal-in-TX today, no ban pending, but the most fragile category (bans in
NY/NJ/CT/MT/CA during 2025-26; ~12 states now prohibit). Redemption
friction, poor pricing. **Promo-harvesting only, never core execution.**

## 5. Texas legislative outlook

- 2025 session passed nothing; adjourned sine die 2025-06-02.
- Next window: **2027 session** (needs 2/3 both chambers + Nov 2027
  referendum) → realistic sportsbook launch **2028+**. Plan through the
  2027-28 season with no TX sportsbook.
- Same 2027 session may try to *restrict* prediction markets; CFTC
  preemption litigation (heading toward SCOTUS) is the federal shield.

## 6. Implications for the promo-hedging calculator

Accessible promo inventory:
1. **Trip-based sportsbook new-user promos** (LA primary, AR freshest):
   one trip can open 5–8 books; promo EV dwarfs vig.
2. **Prediction-market signup promos**: small (Kalshi ~$20, Polymarket ~$50)
   but zero travel.
3. **Sweepstakes signup bonuses**: couch-harvestable, capped value.

Constraints to model:
- **Cross-platform hedging is the norm**: bonus leg placed in-state, hedge
  leg on Kalshi/Polymarket from home. Per-leg pricing: sportsbook American
  odds + free-bet SNR conversion vs contract price + taker fee
  `f = k·p·(1−p)` with k = 0.07 Kalshi taker / 0.05 Polymarket taker /
  ~0 maker & Novig/ProphetX — plus bid-ask spread as an explicit input.
- **Geofencing timing**: all same-book legs must be queued for the trip
  window ("batch mode"); after driving home the TX-legal platform is the
  only live rebalance venue.
- **Slippage between legs** (line moves between Lake Charles and Kalshi).
- **Withdrawal friction / bankroll lockup** per platform (sportsbook ACH
  days; Fliff slow with minimums; Kalshi/Polymarket fast + free).
- **Mismatched settlement**: binary contracts may need ML+total synthetics
  to hedge a puck-line free bet; support partial hedges + residual exposure.
- **Age/KYC**: prediction markets 18+, sportsbooks 21+, SSN KYC everywhere.

## Two-bettor execution structure (Louisiana partner)

Gavin's partner is resident in Louisiana through ~mid-2028, which upgrades
the execution layer from "trip-based promo harvesting" to a standing
two-bettor structure. This is the documented plan, and its legality rests
on the two actors staying independent:

**Bettor A (partner, Louisiana):** full online sportsbook access (FanDuel,
DraftKings, BetMGM, Caesars, bet365, Fanatics, theScore — 55 of 64
parishes). Accounts **in his name, funded by him, bets placed by him while
physically in Louisiana, winnings his** (and his taxable income — LA
withholds on gambling winnings; he files as the bettor). He runs the
system's recommendations like any user of betting software — sharing
software, picks, and strategy is legal.

**Bettor B (Gavin, Texas):** Kalshi + Polymarket US from home, exactly as
ranked below. His own accounts, his own funds.

**The line that keeps this legal (do not cross it):** no proxy placement.
If A's accounts are funded by B, or A places bets at B's direction with
B's economics, that is messenger betting — illegal in Louisiana as in
other legal states, a universal sportsbook T&C violation, and the
specific pattern KYC/source-of-funds reviews are built to catch
(withdrawal freezes + confiscation, before any legal question). Each
bettor's bets are his own decisions with his own bankroll; the system is
shared analytics, not a shared wallet.

Practical notes for this structure:
- **Promo inventory**: A can open 7-8 LA books from home, no travel —
  the new-user promo stack that previously required Houston/Dallas trips
  is now fully and continuously accessible, plus ongoing reload/boost
  offers that trip-based access always missed.
- **Hedging modes** (calculator supports both):
  1. *Single-bettor hedge (cleanest)*: A hedges his own promo leg on
     another LA book — one person, one bankroll, all in-state.
  2. *Coordinated cross-venue*: A takes a promo leg in LA; B
     independently takes the offsetting side on Kalshi/Polymarket with
     his own funds. Each bet is legal where placed; the household-level
     netting is their personal finance. The calculator prices each leg
     against its own venue's cost model and reports per-bettor P&L,
     never a pooled stake.
- **Book risk (not legal risk)**: consistent promo-hedged play gets
  accounts limited or promo-restricted ("bonus abuse" T&Cs). That is a
  commercial risk input for the calculator (expected account lifetime),
  not a legality issue.
- **Geofence timing**: A must be in-state at placement (he lives there —
  non-issue); B's hedge venue has no geofence. The old "batch mode for
  trip windows" constraint drops out of the design.

## Bottom line (ranked for an NHL bettor in Texas)

1. **Kalshi direct** — always-on TX-legal execution + hedge venue (maker
   orders where possible).
2. **Polymarket US** — second book: line shopping + cheaper taker fills.
3. **Trip sportsbooks (LA, AR)** — where the promo-hedging calculator
   earns its keep.
4. **Novig / ProphetX** — near-zero-vig supplement, pending NHL liquidity
   check.
5. Crypto.com/DK Predict → Robinhood → pick'em → sweepstakes: niche or
   avoid.

## Re-verify before October 2026

1. Oklahoma ballot measure result (2026-11-03).
2. Kalshi/state litigation in three federal appeals courts → SCOTUS;
   TX interim committee posture.
3. Kalshi + Polymarket fee schedules (Polymarket already hiked July 2026).
4. Kalshi NHL props/totals spreads + depth on real October slates.
5. Novig regulated-exchange rollout: fees + NHL from opening night?
6. Robinhood LedgerX migration.
7. Kansas restructuring (licenses to 2027-08-31).
8. TX 2027-session pre-filed bills (appear Nov 2026) targeting prediction
   markets / DFS / sweepstakes.
9. Current LA/AR promo offers immediately before any trip.
