# Data feeds: what the Odds API key buys and what is free

> Research snapshot, 2026-09-28. Produced by read-only investigation (live requests to free endpoints, the
> official docs, and a disposable copy of the database). File and line references point at the code as it was
> that day, before later fixes; see README.md and PROJECT_CONTEXT.md for the current state.

Checks the NHL API odds feed, the NHL stats API, DraftKings direct, ESPN (injuries, open/close odds), The Odds API (regions, exchanges, props, history, credits), MoneyPuck and Daily Faceoff against live requests and the docs, and maps each to the repo.

## Summary
I checked all five claims against live requests, the docs and the repo code. Everything was read-only: no repo edits, no Odds API key used, and the only database I touched was the throwaway clone on port 55432.

- **(a) NHL feed: mostly true.** `partner-game/US/now` is free JSON with DraftKings moneyline, puck line and over/under. The Canada version gives FanDuel. The NHL schedule endpoint goes further and carries moneylines from up to 7 partner books. It has three limits. It covers only the next day with games. It keeps no history, because finished games lose their odds. And its "lastUpdatedUTC" stamp was a month old, so how fresh the prices are is unproven.
- **(b) DraftKings direct: true.** It returned 403 "Access Denied" even from this home PC, not just from cloud servers. One of the owner's cited sources documents DraftKings' fantasy-contest API, not sportsbook odds.
- **(c) ESPN: true, plus something useful.** The injuries endpoint works, and the NHL API has nothing like it. The sports.core odds resource has open, close and current prices, for DraftKings only. The summary block the repo already downloads has the same open/close data and the over/under prices. `ingestion/espn_odds.py` throws both away.
- **(d) The Odds API key buys:** prices from about 20 US books, plus Kalshi and Polymarket in the `us_ex` region (a region is a group of bookmakers the API bills together). The repo requests only `us,us2`, so it never sees the two platforms the owner can legally use in Texas. Player props work through the per-game endpoint. Historical odds are paid plans only.
- **(e) MoneyPuck and Daily Faceoff are both in use.** MoneyPuck's peter-tanner.com host is its official download link, not a mirror. Daily Faceoff blocks non-browser clients, so the repo presents itself as a Chrome browser.
## Details
Terms used below:
- **Moneyline:** a bet on who wins.
- **Puck line:** a bet with a 1.5-goal handicap.
- **Over/under (totals):** a bet on total goals.
- **Closing line:** the last price before puck drop.
- **CLV (closing-line value):** how much better your price was than the closing line. The repo treats it as its main quality score (betting/settle.py:15-27).
- **Credit:** The Odds API's billing unit. The free plan gives 500 a month.

VERIFIED means I saw it in a response or in code. INFERRED means it is my judgement.

=== (a) Official NHL API (api-web.nhle.com) ===

VERIFIED: GET /v1/partner-game/US/now returned 200 with no key.
- Top-level fields: currentOddsDate="2026-09-29", lastUpdatedUTC="2026-08-28T18:00:38Z", bettingPartner={name:"DraftKings", partnerId:9}, and games[].
- Each game has gameId (the same ID the repo uses), gameType, startTimeUTC, and homeTeam/awayTeam each with an odds[] list of {description, value, qualifier}.
- Markets present: MONEY_LINE_2_WAY, PUCK_LINE (qualifier ±1.5), OVER_UNDER (qualifier O6.5/U6.5), MONEY_LINE_3_WAY (includes a "Draw" price), and on one game MONEY_LINE_2_WAY_TNB (tie-no-bet).
- One book per country:
  - /partner-game/CA/now returned FanDuel (partnerId 7).
  - A date in the path (/partner-game/US/2026-09-29) returned 404, so there is no date lookup.

VERIFIED, and better than the claim: /v1/schedule/{date} carries moneylines from several books at once.
- Its "oddsPartners" list names Unibet, Tipsport, Veikkaus, FanDuel, Sportradar, DraftKings and Doxxbet.
- Each team carries odds [{providerId, value}]. FLA@CAR had 5 books quoting, e.g. DraftKings +105/-125 and FanDuel +104/-125.

VERIFIED coverage limits:
- Only the next game date has odds. All 5 games on 2026-09-29 were priced; 0 of the 3 on 2026-09-30 and 0 of the later ones.
- No history: completed games from 2026-04-06 show odds=None in /v1/schedule.
- The repo's ingestion/nhl_api.py never reads these odds (no "odds" in the file).
- The NHL schedule says the regular season starts 2026-09-29, so the first real slate is tomorrow.

Freshness:
- VERIFIED: Cache-Control max-age=12, so the CDN caches for 12 seconds.
- VERIFIED: the lastUpdatedUTC field is a month old for the US feed and 2026-09-19 for Canada.
- INFERRED: either that field doesn't track price changes, or the opening-night prices really haven't moved. One fetch can't tell which.
- INFERRED data quality: Veikkaus's decimal prices favored the away team in all 3 sampled games while every other book favored home. It looks swapped, so don't use it unchecked.

Could it replace The Odds API?

(1) Morning pick price. It can supplement, not replace.
- It gives DraftKings and FanDuel lines, plus a few more moneylines, for free.
- It cannot give line shopping (picking the best price across about 20 books, recommend.py:25-27).
- It cannot give Kalshi or Polymarket.
- Its freshness is unproven.

(2) Closing lines. Only if polled live before puck drop, since it keeps no history.
- Because it is free, it could be polled every 10-15 minutes before each game at zero credits.
- That would be denser than the repo's two 2-credit close runs (README.md:312-313).
- The catch: DraftKings and FanDuel only, and freshness must be proven first.

Stats REST API (api.nhle.com/stats/rest/en/...). VERIFIED: /config lists these reports:
- Skaters: realtime (hits, blocks, giveaways), timeonice, powerplay, faceoffs, shottype, scoringRates.
- Goalies: daysrest, startedVsRelieved, savesByStrength, advanced.
- Teams: daysbetweengames, goalsbyperiod, leadingtrailing, goalsforbystrengthgoaliepull, and more.

VERIFIED: goalie/summary?isGame=true returned per-game rows, 2,768 for 2025-26 regular season (e.g. Comrie, 2026-04-16, 27 saves on 33 shots).

What it adds over api-web:
- League-wide filterable tables in one call instead of one boxscore per game.
- Power-play time on ice and similar splits that are useful for props.
- It has no odds.
- The repo does not use it (no "stats/rest" anywhere).

=== (b) DraftKings direct ===

VERIFIED: GET sportsbook-nash.draftkings.com/sites/US-SB/api/v5/eventgroups/42133?format=json with a browser user-agent returned HTTP 403 "Access Denied" from Server: AkamaiGHost.
- This came from the owner's Windows PC, not a cloud server.
- The response doesn't say whether it was geo-blocking or bot-blocking.

VERIFIED: the repo docs say Texas has no legal online sportsbook (docs/texas_execution_options.md:3-6).

Terms of service:
- I could not read DraftKings' terms page (it also returned 403).
- INFERRED: sportsbook terms routinely ban automated access. Getting around Akamai with fake headers or proxies (the Apify-style scrapers) means dodging an access control, and could put any DraftKings account at risk.
- INFERRED: DK-direct adds nothing the NHL feed and The Odds API don't already give.

VERIFIED: the cited SeanDrum/Draft-Kings-API-Documentation repo covers DraftKings fantasy-contest endpoints (api.draftkings.com/contests, draftgroups, lineups). It says nothing about eventgroups or sportsbook odds.

The repo does not use DraftKings directly.

=== (c) ESPN ===

Injuries: VERIFIED that site.api.espn.com/apis/site/v2/sports/hockey/nhl/injuries returned 200.
- Contents: 31 teams, 102 entries, stamped 2026-09-28T17:47:55Z, season 2026-27.
- Each entry: status (Day-To-Day 38, Out 33, Injured Reserve 29, Suspension 2), date, details.type (e.g. "Hip"), details.returnDate, shortComment/longComment (news text), and athlete (ESPN id, name, position, team).
- 12 goalies are listed, e.g. Demko on IR (hip) and Hellebuyck suspended until 2026-10-17.
- It is current-state only. Players carry ESPN IDs, so matching them to NHL IDs means name matching, like dailyfaceoff.py:110-120.
- The repo has no injury ingestion (no "injur" in any .py file).
- VERIFIED: the NHL API has no injury endpoint and its game page doesn't know about injuries. /v1/gamecenter/2026020004/landing lists F. Andersen as EDM's top goalie, while ESPN has him on IR and Daily Faceoff has Jarry "Confirmed".

Open/close odds: VERIFIED on ESPN event 401803580 (TB@BUF, 2026-04-06, game 2025021229).
- sports.core.api.espn.com/v2/.../events/401803580/competitions/401803580/odds returned count=1: one provider, DraftKings (provider id 100).
- That one item has open, close and current prices for moneyline, puck line and total:

| Market | Open | Close |
|---|---|---|
| BUF (home) moneyline | -105 | +102 |
| TB (away) moneyline | -115 | -122 |
| Over 6.5 | +110 | -115 |
| Under 6.5 | -130 | -105 |

- It has no timestamps, and "current" equals "close".

VERIFIED against the repo:
- The site summary pickcenter block the repo already downloads (espn_odds.py:28, 48-53) holds the same open/close data in moneyline, pointSpread and total sub-blocks, plus closing overOdds -115 / underOdds -105.
- parse_pickcenter (espn_odds.py:56-67) keeps only the closing moneylines, the spread, the total line and the display text. It drops every opening price and both over/under prices.
- The clone shows the stored row for game 2025021229 is (DraftKings, home_ml 102, away_ml -122, 1.5, 6.5), which is the close.
- The table has no columns for open or over/under prices (db/schema.sql:199-208).
- This matters because models/totals.py:70-76 says totals can't be backtested because "No O/U prices exist historically".
- VERIFIED for one game, INFERRED for the rest: the 2025-26 DraftKings era (1,014 rows in the clone) likely has free open and close over/under prices in ESPN.

=== (d) The Odds API: what the key buys (docs only; no call made) ===

VERIFIED from the v4 guide, the markets page, the bookmakers page, the historical page and update-intervals:

Credit costs:
- /odds costs markets × regions. The repo's full snapshot is 3 × 2 = 6 credits and a close is 2 (odds_api.py:6-8, 111).
- "Every group of 10 bookmakers is the equivalent of 1 region", and "bookmakers takes priority" over regions and "can be from any region". So a hand-picked list of up to 10 books costs half of us,us2.
- /events and /sports are free. "Responses with empty data do not count."
- Response headers x-requests-remaining, x-requests-used and x-requests-last report credit use. The repo already logs them (odds_api.py:135-139).
- Featured-market update intervals: about 60 seconds before games, 40 seconds in-play.

Bookmakers by region:
- us: DraftKings, FanDuel, BetMGM, BetRivers, Caesars, Fanatics, Bovada, and others.
- us2: theScore Bet (espnbet), Hard Rock, and others.
- us_ex (exchanges): Kalshi, Polymarket, Novig, ProphetX, BetOpenly.
- Pinnacle is in eu.
- Paid plans only: Caesars, Fanatics, Courtside, ReBet.
- The repo never requests us_ex or Kalshi/Polymarket, even though betting/promo.py:8 names them as the owner's venues.

NHL player props:
- Markets: player_points, player_assists, player_goals, player_shots_on_goal, player_blocked_shots, player_power_play_points, player_total_saves, anytime/first/last goal scorer, and alternate versions.
- They are available "one event at a time using the /events/{eventId}/odds endpoint".
- Cost is unique markets returned × regions, for US books only.
- Period markets, team totals and alternate lines exist too.

Historical odds:
- "only available on paid usage plans" (both doc pages).
- Costs 10 credits per region per market; props cost 10 per region per market per event.
- Snapshots are every 5 minutes since Sept 2022.
- Game markets go back to 2020-06-06; props back to 2023-05-03.

Pricing and terms:
- From a summary of a JavaScript-rendered page: Starter is free at 500 credits; 20K is $30/month; 100K is $59/month.
- That summary also said Starter includes "Historical Odds", which contradicts the two doc pages. Trust the docs.
- Terms: storing data indefinitely and using it to train models is permitted. Resale as a raw feed is not.
- The terms say nothing about one person holding several free accounts.

=== (e) MoneyPuck and Daily Faceoff ===

MoneyPuck. VERIFIED:
- moneypuck.com/data.htm links its shot files at peter-tanner.com/moneypuck/downloads/shots_YYYY.zip, exactly what moneypuck.py:36 downloads.
- It says 2026-27 data is "updated nightly". shots_2026.zip isn't linked yet, which is expected before opening night; refresh_season skips until a game is final (moneypuck.py:169-176).
- So README.md:61's worry that the file comes from an unchecked "mirror" is resolved: it is the official download link.
- Terms: "free to use for non-commercial purposes… Please clearly credit MoneyPuck.com". Scraping beyond the listed files needs approval.
- The repo uses it for shot-level expected goals (xG, the chance a shot becomes a goal) and shot features.

Daily Faceoff. VERIFIED:
- The repo uses it for confirmed starting goalies (dailyfaceoff.py:40, 63-67), called from pipeline.py:118-124 in the daily and odds runs.
- A plain curl got Cloudflare "Sorry, you have been blocked", even for robots.txt.
- With the repo's Chrome-on-Mac user-agent (dailyfaceoff.py:41-42), /starting-goalies/2026-09-29 returned 200 with __NEXT_DATA__: 5 games, EDM "Tristan Jarry Confirmed", the rest not yet confirmed.
- The docstring says "identified UA" (line 14), but the code presents itself as a Chrome browser.
- The site is © The Nation Network Ltd. The page links only a privacy policy; I found no public terms page.
- INFERRED: this is low-volume scraping past bot protection. It's a grey area and could break without notice (README.md:84 already says so).

=== What the repo uses today ===

| Feed | In use? | Free or keyed | What for |
|---|---|---|---|
| NHL api-web (via the nhlpy package) | Yes: standings, schedule, boxscores, plus right-rail directly (nhl_api.py:22, 36, 86, 184, 350) | Free | Games and stats; odds fields ignored |
| NHL stats REST | No | Free | — |
| ESPN site summary pickcenter | Yes (espn_odds.py) | Free | Closing reference line; open and over/under prices discarded |
| ESPN injuries | No | Free | — |
| ESPN sports.core | No | Free | — |
| The Odds API | Yes (odds_api.py) | Keyed | us,us2; moneyline, puck line, totals; 6 or 2 credits per run; no us_ex, props or historical |
| MoneyPuck | Yes | Free, non-commercial, credit required | Shot data and xG |
| Daily Faceoff | Yes | Free, scraped | Starting goalies |
| DraftKings direct | No | — | — |

=== Recommended feed plan ===

(1) Opening/morning lines:
- Keep The Odds API as the main source.
- Replace regions=us,us2 with bookmakers= listing at most 10 books. Put kalshi and polymarket first, since they are the owner's legal venues, then pinnacle, draftkings, fanduel, betmgm, betrivers, espnbet, hardrockbet, novig.
- VERIFIED rule, INFERRED fit: that halves the cost (full snapshot 3 credits, close 1).
- Add a zero-credit NHL snapshot (the schedule endpoint's moneylines plus partner-game US/CA for puck line and totals) as extra books.

(2) Closing lines:
- The Odds API close runs for the books you actually bet.
- Add free NHL-feed polling in the last hour before each puck drop, once its freshness is proven.
- Keep ESPN's DraftKings close (and open) as a free after-the-fact backup. It can fill the blank CLVs on afternoon games (README.md:318).

(3) Injuries: ESPN's injuries endpoint.
- Save a snapshot every day, because it keeps no history.
- Match names the way Daily Faceoff names are matched.

(4) Starting goalies:
- Daily Faceoff stays the only free source of confirmed starters. The NHL game page lists goalies by games played, not starters.
- Use ESPN injuries to rule out injured goalies.

(5) Player props lines:
- The Odds API's per-game endpoint is the only source among these.
- Start saving lines now. Historical props need a paid plan.
- INFERRED: Kalshi also lists NHL props (texas doc:24-26), but I didn't check its API.

Per-machine keys:
- VERIFIED: credits are counted per key, and the terms don't forbid a second free account.
- INFERRED: each machine has its own database, so a pick can only be graded against close snapshots taken on the same machine.
- Best split: the Mac's key runs the moneyline schedule. The PC's key collects props:
  - At 1 credit per game per snapshot, about 7 games a day × 30 × 2 snapshots comes to about 420 a month, for one prop market such as shots on goal.
  - Two or more prop markets would need the $30 plan.
## Recommendations
1. **Cheapest win:** change `ingestion/espn_odds.py` `parse_pickcenter` and add columns to `raw.historical_odds` to keep ESPN's opening prices and over/under prices. Then re-run the free 2025-26 backfill, about one request per game. That gives the totals model a DraftKings over/under price history for one season. `models/totals.py:70-76` says that history doesn't exist today. Spot-check 20 or so games first, because I verified only one.
2. **Stop paying for two regions.** Replace `regions: "us,us2"` (odds_api.py:111) with `bookmakers=` listing at most 10 books. Include `kalshi` and `polymarket`, the owner's legal Texas venues, which the repo never collects today. Also include `pinnacle` and the main US books. Full snapshots drop from 6 to 3 credits and closes from 2 to 1, so the four-run month falls from about 480 credits to about 240. Set `BETTABLE_BOOKS=kalshi,polymarket`. Before trusting exchange prices, check whether they include Kalshi's and Polymarket's trading fees; `betting/promo.py` already models those fees.
3. **Prove the free NHL feed on opening night (2026-09-29).** Take the NHL `partner-game/US/now` and `schedule` odds and an Odds API DraftKings moneyline at the same minute, a few times during the day. If they match and move together, add a zero-credit NHL snapshot source writing to `raw.odds_snapshots` under distinct book names. Poll it in the last hour before each puck drop for denser closing lines. If it lags, use it only as a reference. Either way, don't use the Veikkaus prices unchecked.
4. **Add an ESPN injuries job:** one call a day, stored with the date, with names matched to NHL player IDs. Use it to rule out injured goalies in the starter logic and as a future feature.
5. **Give each machine its own Odds API key, with separate roles.** The Mac's key covers the moneyline picks and closes. The PC's key starts collecting NHL shots-on-goal props at a morning and a pre-game snapshot, about 420 credits a month. A picks machine needs its own closes, because the grading uses the same database.
6. **Props backtest later:** one month of the $30 plan (20K credits) roughly covers historical shots-on-goal props for 2025-26. That is about 1,312 games × 10 credits at one snapshot per game, from the doc prices; confirm plan details in the account dashboard.
7. **Leave DraftKings direct alone.** It is blocked (403 from this PC), its terms are unreadable, and it offers nothing extra.
8. **Doc fixes:**
   - README.md:61 should say peter-tanner.com is MoneyPuck's official download host, updated nightly.
   - The `dailyfaceoff.py` docstring should say it sends a browser user agent, not an "identified UA".
   - `ingestion/espn_odds.py:7-10` should note that pickcenter also holds opening lines.
