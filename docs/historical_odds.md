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

## The Odds API historical endpoint (bought 2026-10-04)

The free sources above give one line per game, and for 2024-25 only the
favourite's price. Two-way prices (→ a bet with two outcomes, overtime
and shootout included, which is what the models price) from several books
(→ sportsbooks), with the time each price was taken, come from The Odds
API's paid historical endpoint (→ the part of the API that serves past
prices). `ingestion/odds_history.py` buys them into `raw.odds_history` and
logs every purchase in `raw.odds_history_fetches`. The markets bought are
the moneyline (`h2h` in the API → who wins) and the total (→ over or under
the combined goals line).

**Endpoint facts, confirmed against the v4 docs and live calls:**

- `GET /v4/historical/sports/icehockey_nhl/odds?date=<ISO time>` returns
  the snapshot (→ every book's prices for every listed game at one
  moment) taken at or just before `date`, wrapped as `timestamp`,
  `previous_timestamp`, `next_timestamp` and `data` (the events). A
  snapshot holds every game listed at that moment: later games that day,
  the next days, and games already in play (→ already started, priced
  live; those are dropped on load).
- Snapshots every 10 minutes from June 2020, every 5 minutes from
  September 2022 (seen: 23:40:38, 23:45:38, 23:50:38 on 2024-12-10).
  `icehockey_nhl` history starts 2020-06-29. Paid plans only.
- Cost: 10 credits (→ The Odds API's billing unit) x markets x regions
  (→ the API's groups of books, such as `us`), where up to 10 named
  `bookmakers` bill as one region. The first call (h2h, `regions=us`)
  read `x-requests-last: 10`; every h2h + totals call with 10 named books
  read 20. An empty response costs nothing. Credits are reported in the
  `x-requests-last`, `x-requests-used` and `x-requests-remaining` headers.
- The API's `commence_time` was often a few minutes after the NHL's
  scheduled start (17:10 vs 17:00 for the 2024 Prague opener), so a price
  is treated as in play once either time has passed.

**Books (10 = one region):** pinnacle (the sharpest reference price),
draftkings, fanduel, betmgm, betrivers and espnbet (also in the live
default list, so history and live compare book for book), williamhill_us
(Caesars), and the offshore books (→ sportsbooks licensed outside the
US) lowvig, betonlineag and bovada, priced in every season. Four probe
calls (→ one-off test calls, 40 credits) on 2022-12-13 and 2024-12-10
showed which books each era had: 2022-23 also had pointsbetus,
barstool, unibet_us, twinspires, wynnbet, superbook, foxbet and
sugarhouse (since closed or renamed); espnbet starts in November 2023.
In the 2024-25 data lowvig and betonlineag quoted the same price 99.2% of
the time, so for 2023-24 and 2022-23 swap betonlineag for another book
(for example `mybookieag`, or `pointsbetus` in 2022-23) with
`--bookmakers`, and name the seasons with `--steps` so the run buys only
those:

```bash
BOOKS=pinnacle,draftkings,fanduel,betmgm,williamhill_us,betrivers,espnbet,lowvig,mybookieag,bovada
python -m ingestion.odds_history plan  --steps close:20232024,close:20222023 --bookmakers $BOOKS
python -m ingestion.odds_history fetch --steps close:20232024,close:20222023 --bookmakers $BOOKS --max-credits 2000
```

By default a snapshot already bought with other books counts as bought,
so the 2024-25 snapshots are skipped whatever `--bookmakers` says. **Do not
add `--same-books-only` to this command:** that flag counts a snapshot as
bought only when it was bought with exactly the same books, so every
snapshot bought with the old list would be bought again (all of 2024-25
is 13,960 credits). For that reason `--same-books-only` refuses to run
without explicit `--steps`.

**The plan, and what was bought on 2026-10-04 (14,000 credits including
the probes; 6,000 left on the account for the live jobs):**

| Purpose | Season | Snapshots | Credits | Coverage |
|---|---|---:|---:|---|
| close | 2024-25 | 549 of 549 | 10,980 | all 1,398 games (1,312 regular season + 86 playoff), moneyline and total, at Pinnacle, DraftKings and FanDuel; at least 1,392 games at every one of the 10 books |
| morning | 2024-25 | 149 of 223 | 2,980 | 889 games on 149 dates, spread evenly over the season |
| close | 2023-24 | 0 of 548 | (10,960 needed) | not bought |
| close | 2022-23 | 0 of 540 | (10,800 needed) | not bought |

- **close** → for each game date the start times are grouped into
  clusters (a cluster takes every start within 75 minutes of its first
  one) and one snapshot is bought at the cluster's first puck drop (→ the
  game's start) minus 10 minutes. The closing price (→ the last price
  before the game starts, the market's sharpest opinion) is what closing
  line value is measured against. The last price stored before a game is a median 14 minutes
  before its scheduled start, at most 89.
- **morning** → one snapshot per game date at 10:00 Central, for the
  bet-timing study (does the price move between the morning and the
  close?). A plan the budget cuts short is bought in a spread order
  (bit-reversed → a fixed order that takes the first date, then the
  middle one, then the quarter points, and so on), so the 149 dates cover
  the whole season, not its first five months.
- Sanity check: the closing no-vig home probability (→ the book's chance
  with its margin removed) scores a log loss (→ how far probabilities
  are from what happened; lower is better) of 0.657 to 0.658 against
  2024-25 results at every book; the average margin (→ the book's
  built-in fee) is 2.6% at Pinnacle and 4.0 to 4.8% at the US books.
  Pinnacle's closing total was 5.5 or 6.0 in 1,044 of 1,398 games.

**Commands** (`plan` costs nothing; `fetch` needs `--max-credits`):

```bash
python -m ingestion.odds_history starts 20232024 20222023    # free: fill old seasons' start times from the NHL schedule
python -m ingestion.odds_history plan                        # what is left to buy, and its cost
python -m ingestion.odds_history fetch --max-credits 2000 --reserve 6000 --max-minutes 8
python -m ingestion.odds_history rematch                     # match stored rows whose game was missing
python -m ingestion.odds_history reparse                     # load paid_unparsed calls from their raw copies (no API call)
```

`fetch` stops before a call that would pass `--max-credits` (this run),
`--cap-total` (all logged purchases together) or leave fewer than
`--reserve` credits on the account (default 6,000). It reads the account's
remaining credits from the free `/sports` endpoint first and from every
response after; when that first read fails it does not start (pass
`--allow-unknown-remaining` to rely on `--max-credits` alone). A call that
comes back without credit headers (a timeout or a dropped connection) is
counted at its full expected cost, since the API may have billed it, and
a run stops after 5 failed calls in a row. Every paid response is also
kept, gzipped, in `data/odds_history/` (git-ignored), so the data
survives a database loss.
A paid call that fails after payment (its response could not be parsed or
stored) is still logged, as status `paid_unparsed` with the credits it
cost, and counts as bought, so it is never paid for twice; `reparse`
loads it later from the raw copy without another call.
A copy is named `<purpose>_<requested time>_<hash>.json.gz`, where the
hash is a short fingerprint of the markets and the book list, so the same
time bought with other books gets its own file; an existing file is never
overwritten (a repeat gets `_2`, `_3`, ...). The 698 copies from the
2024-25 purchase predate the hash and are named
`<purpose>_<requested time>.json.gz`.

**The fetch log** (`raw.odds_history_fetches`, one line per call):

| status | Meaning | Counts as bought? |
|---|---|---|
| `ok` | a snapshot with games, rows stored | yes |
| `empty` | a snapshot with no games listed (costs nothing) | yes |
| `paid_unparsed` | paid for, but parsing or storing failed; raw copy kept for `reparse` | yes |
| `error` | the call failed, or the response had no snapshot `timestamp`; its credits are logged | no: the next run retries it |
| `probe` | the four test calls of 2026-10-04, logged by hand (purpose `probe` too), not by `fetch` | no |

Every line's credits, the probes' included, count toward `--cap-total`.
