"""
ingestion/nhl_odds.py
The NHL's own free odds feed, stored beside The Odds API so the two can be
compared before anything relies on the free one (first test: opening
night, 2026-09-29).
Populates: raw.nhl_feed_snapshots, a table of its own. It never writes
raw.odds_snapshots: the recommendation job takes its fair price (the
median no-vig price across books) and its best price (line shopping) from
that table, so an unproven feed there would change the picks.

Terms:
  moneyline (ml)   a bet on who wins, overtime and shootout included
  puck line (pl)   the ±1.5-goal handicap; `line` is the home side's (-1.5)
  total            over/under on combined goals; `line` is the total (6.5)
  3-way (ml3)      regulation (60 minutes) only: home, away, or a draw
  American odds    -125 → stake 125 to win 100; +105 → stake 100 to win 105
  decimal odds     2.12 → total return per 1 staked (the same as +112)
  implied prob.    the win chance a price stands for, bookmaker margin included

Sources (api-web.nhle.com: free, no key, undocumented, can change):
  partner-US  /v1/partner-game/US/now — the NHL's US betting partner,
              DraftKings: MONEY_LINE_2_WAY (ml), PUCK_LINE (pl),
              OVER_UNDER (total), MONEY_LINE_3_WAY (ml3). Tie-no-bet lines
              (MONEY_LINE_2_WAY_TNB → stake refunded on a regulation tie)
              are ignored.
  partner-CA  /v1/partner-game/CA/now — the Canadian partner, FanDuel, same
              markets. It is FanDuel's Canadian book, while The Odds API's
              fanduel is the US one, so small gaps between them are normal.
  schedule    /v1/schedule/<today> — one moneyline per partner book, keyed
              by providerId and named from the response's oddsPartners list
              (Unibet, Tipsport, Veikkaus, FanDuel, Sportradar, DraftKings,
              Doxxbet). Some books quote decimal odds; every price is stored
              as American.

A book is named by the partner's name, lower-cased without spaces
("DraftKings" → draftkings), which matches The Odds API's keys for the two
books both sources carry: draftkings and fanduel.

What research found (one fetch, 2026-09-28): only the next game date
carries odds; finished games lose theirs, so there is no history and the
feed must be polled; partner-game has no date lookup (404); the feed's
lastUpdatedUTC was weeks old although the CDN caches for only 12 seconds,
so how fresh the prices are is unproven. compare_feeds() measures it.

Veikkaus: in research its decimal prices favoured the away team in games
every other book had the home team favoured, so they look side-swapped.
It is left out by default: NHL_FEED_EXCLUDE_BOOKS, comma-separated book
names, default "veikkaus"; set it empty to keep every book. Each snapshot
still checks every book, excluded ones too, and logs any whose favourite
disagrees with the other books' in most games (swapped_books()), so a
fixed feed shows up in the log.

In-play: a game whose start time has passed, or whose schedule state is
LIVE, CRIT, FINAL or OFF, is skipped, as in ingestion/odds_api.py.

compare_feeds(date) pairs each DraftKings and FanDuel price from this feed
with the nearest raw.odds_snapshots price from the same book, game and
market taken within 10 minutes either way, and reports the gap in American
odds (cents) and in implied probability. It also counts how often each
price changed across snapshots in both sources (freshness): a feed that
never moves while The Odds API does is stale.

CLI: python -m ingestion.nhl_odds snapshot | compare [--date YYYY-MM-DD]
"""
import argparse
import logging
import math
import os
import re
import statistics
import time
from datetime import date as date_cls, datetime, timedelta, timezone
from typing import Optional

import requests
from sqlalchemy import text

from config.migrate import ensure_schema
from config.settings import engine, local_today
from features.util import american_implied_prob
from ingestion.odds_api import SNAPSHOT_HORIZON, parse_commence, upcoming_start_times

logger = logging.getLogger("nhl.ingestion.nhl_odds")

API_WEB = "https://api-web.nhle.com/v1"
PARTNER_URL = API_WEB + "/partner-game/{country}/now"
SCHEDULE_URL = API_WEB + "/schedule/{date}"
PARTNER_SOURCES = (("partner-US", "US"), ("partner-CA", "CA"))
SCHEDULE_SOURCE = "schedule"
TIMEOUT_S = 30
REQUEST_PAUSE_S = 1.0      # between the three requests: polite to a free API

PARTNER_MARKETS = {"MONEY_LINE_2_WAY": "ml", "PUCK_LINE": "pl",
                   "OVER_UNDER": "total", "MONEY_LINE_3_WAY": "ml3"}
DEFAULT_EXCLUDED_BOOKS = "veikkaus"
IN_PLAY_STATES = frozenset({"LIVE", "CRIT", "FINAL", "OFF"})

# A two-sided price's implied probabilities add up to a little over 1 (the
# margin). Well under 1 is a regulation 3-way price passed off as a 2-way
# one (the ESPN history's Unibet era summed to about 0.83); well over is a
# broken quote. Either way the row is dropped, with a warning.
IMPLIED_SUM_OK = {"ml": (0.97, 1.25), "pl": (0.97, 1.25),
                  "total": (0.97, 1.25), "ml3": (0.97, 1.40)}

# swapped_books(): a game counts when the other books' median no-vig home
# probability is at least this far from 50%, i.e. they agree who is favoured
CLEAR_FAVOURITE = 0.02

PAIR_WINDOW = timedelta(minutes=10)
COMPARE_BOOKS = ("draftkings", "fanduel")     # the books both sources carry
COMPARE_MARKETS = ("ml", "pl", "total")       # The Odds API side has no ml3
SIDES = {"ml": ("home", "away"), "pl": ("home", "away"),
         "total": ("over", "under"), "ml3": ("home", "away", "draw")}
PRICE_FIELDS = ("home_price", "away_price", "over_price", "under_price",
                "draw_price", "line")
ODDS_API_SOURCE = "odds-api"                  # how compare labels raw.odds_snapshots

# The table: config/migrate.py (TABLES) and db/schema.sql carry the same
# DDL (tests/test_migrate.py checks); ensure_table() applies it on first use.
DDL = [
    """
    CREATE TABLE IF NOT EXISTS raw.nhl_feed_snapshots (
        id                BIGSERIAL PRIMARY KEY,
        captured_at       TIMESTAMP NOT NULL,       -- naive UTC, like raw.odds_snapshots
        game_id           BIGINT NOT NULL REFERENCES raw.games(game_id),
        source            VARCHAR(12) NOT NULL,     -- partner-US, partner-CA, schedule
        book              VARCHAR(40) NOT NULL,     -- draftkings, fanduel, tipsport, ...
        market            VARCHAR(6) NOT NULL,      -- ml, pl, total, ml3
        home_price        INTEGER,                  -- American odds, e.g. -125, +105
        away_price        INTEGER,
        over_price        INTEGER,
        under_price       INTEGER,
        draw_price        INTEGER,                  -- ml3 only: a regulation tie
        line              NUMERIC(4,1),             -- pl: home handicap (-1.5); total: 6.5
        feed_updated_utc  TIMESTAMP                 -- the feed's lastUpdatedUTC (naive UTC); NULL for schedule
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_nhl_feed_game ON raw.nhl_feed_snapshots(game_id)",
    "CREATE INDEX IF NOT EXISTS idx_nhl_feed_time ON raw.nhl_feed_snapshots(captured_at)",
]

_INSERT = text("""
    INSERT INTO raw.nhl_feed_snapshots
        (captured_at, game_id, source, book, market, home_price, away_price,
         over_price, under_price, draw_price, line, feed_updated_utc)
    VALUES (:captured_at, :game_id, :source, :book, :market, :home_price,
            :away_price, :over_price, :under_price, :draw_price, :line,
            :feed_updated_utc)
""")

_table_ready = False


def ensure_table() -> None:
    """Create raw.nhl_feed_snapshots and its indexes if missing. Once per
    process; safe to repeat."""
    global _table_ready
    if _table_ready:
        return
    with engine.begin() as conn:
        for stmt in DDL:
            conn.execute(text(stmt))
    _table_ready = True


# ── Prices ─────────────────────────────────────────────────────────

def book_key(name) -> str:
    """'DraftKings' -> 'draftkings': lower case, letters and digits only."""
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


def decimal_to_american(d) -> Optional[int]:
    """Decimal odds -> American, rounded half away from zero: 2.12 -> +112,
    1.72 -> -139, 2.0 -> +100. None for anything that is not above 1."""
    try:
        d = float(d)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(d) or d <= 1.0:
        return None
    if d >= 2.0:
        return math.floor((d - 1.0) * 100.0 + 0.5)
    return -math.floor(100.0 / (d - 1.0) + 0.5)


def to_american(value) -> Optional[int]:
    """One feed price as American odds. Signed or three-digit values are
    American already ('-125', '+104', -125.0, 105.0); a bare number above
    1 and under 100 is decimal ('2.12' -> +112). None when missing or not
    a price (American odds are never strictly between -100 and +100)."""
    if value is None or isinstance(value, bool):
        return None
    s = str(value).strip()
    try:
        x = float(s)
    except ValueError:
        return None
    if not math.isfinite(x):
        return None
    if abs(x) >= 100:
        return math.floor(abs(x) + 0.5) * (1 if x > 0 else -1)
    if s[:1] in ("+", "-"):
        return None
    return decimal_to_american(x)


def _line(qualifier) -> Optional[float]:
    try:
        v = float(str(qualifier).strip())
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _naive_utc(value) -> Optional[datetime]:
    t = parse_commence(value)
    return None if t is None else t.replace(tzinfo=None)


def _implied_sum(row: dict) -> Optional[float]:
    probs = [american_implied_prob(row.get(f"{side}_price"))
             for side in SIDES[row["market"]]]
    return None if any(p is None for p in probs) else sum(probs)


def _row(base: dict, market: str, **prices) -> dict:
    return {**base, "market": market, **{f: prices.get(f) for f in PRICE_FIELDS}}


def _keep_plausible(rows: list, label: str) -> list:
    """Rows whose implied probabilities add up like a real price
    (IMPLIED_SUM_OK); the rest are dropped with one warning."""
    kept, odd = [], []
    for r in rows:
        total = _implied_sum(r)
        lo, hi = IMPLIED_SUM_OK[r["market"]]
        if total is not None and lo <= total <= hi:
            kept.append(r)
        else:
            odd.append(f"{r['book']} {r['market']} game {r['game_id']} "
                       f"(implied probabilities add up to "
                       f"{'?' if total is None else f'{total:.2f}'})")
    if odd:
        logger.warning(f"NHL feed {label}: dropped {len(odd)} price(s) that don't "
                       f"add up like a real price: " + "; ".join(odd[:5])
                       + (" ..." if len(odd) > 5 else ""))
    return kept


# ── Parsing (pure) ─────────────────────────────────────────────────

def _partner_entries(team) -> list:
    """[(description, American price or None, qualifier)] for one team."""
    out = []
    for o in ((team or {}).get("odds") or []):
        if isinstance(o, dict):
            out.append((str(o.get("description") or ""), to_american(o.get("value")),
                        str(o.get("qualifier") or "").strip()))
    return out


def _first(entries: list, description: str, draw: bool = False) -> tuple:
    for d, price, q in entries:
        if d == description and (q.lower() == "draw") == draw:
            return price, q
    return None, None


def _partner_markets(home: list, away: list) -> dict:
    """{market: price fields} for one partner-game game; a market is left
    out unless every side has a price."""
    out = {}
    h, _ = _first(home, "MONEY_LINE_2_WAY")
    a, _ = _first(away, "MONEY_LINE_2_WAY")
    if h is not None and a is not None:
        out["ml"] = {"home_price": h, "away_price": a}

    h, hq = _first(home, "PUCK_LINE")
    a, aq = _first(away, "PUCK_LINE")
    line, away_line = _line(hq), _line(aq)
    if (h is not None and a is not None and line is not None
            and (away_line is None or away_line == -line)):
        out["pl"] = {"home_price": h, "away_price": a, "line": line}

    over = under = None
    for d, price, q in home + away:
        if d == "OVER_UNDER" and q[:1].upper() in ("O", "U"):
            side = (price, _line(q[1:]))
            if q[:1].upper() == "O":
                over = over or side
            else:
                under = under or side
    if (over and under and None not in (over[0], under[0], over[1])
            and over[1] == under[1]):
        out["total"] = {"over_price": over[0], "under_price": under[0],
                        "line": over[1]}

    h, _ = _first(home, "MONEY_LINE_3_WAY")
    a, _ = _first(away, "MONEY_LINE_3_WAY")
    draw, _ = _first(home, "MONEY_LINE_3_WAY", draw=True)
    if draw is None:
        draw, _ = _first(away, "MONEY_LINE_3_WAY", draw=True)
    if None not in (h, a, draw):
        out["ml3"] = {"home_price": h, "away_price": a, "draw_price": draw}
    return out


def parse_partner(payload, source: str) -> list:
    """Rows from one /partner-game/<country>/now response. Pure.

    One row per game and market (ml, pl, total, ml3) with every side
    priced. Each row also carries `start` (the game's startTimeUTC, aware)
    and `state` (None: the partner feed has no game state), which decide
    the in-play skip but are not stored."""
    if not isinstance(payload, dict):
        return []
    partner = payload.get("bettingPartner") or {}
    book = book_key(partner.get("name")) or f"partner{partner.get('partnerId', '')}"
    updated = _naive_utc(payload.get("lastUpdatedUTC"))
    rows = []
    for g in payload.get("games") or []:
        if not isinstance(g, dict) or not g.get("gameId"):
            continue
        base = {"game_id": int(g["gameId"]), "source": source, "book": book,
                "feed_updated_utc": updated,
                "start": parse_commence(g.get("startTimeUTC")), "state": None}
        markets = _partner_markets(_partner_entries(g.get("homeTeam")),
                                   _partner_entries(g.get("awayTeam")))
        rows += [_row(base, m, **prices) for m, prices in markets.items()]
    return _keep_plausible(rows, source)


def parse_schedule(payload) -> list:
    """Moneyline rows from one /schedule/<date> response. Pure.

    One row per game and partner book that priced both teams. providerId
    is named from the response's oddsPartners list (an unlisted id becomes
    'provider<id>'); decimal prices become American. Rows carry `start`
    and `state` (gameState) for the in-play skip."""
    if not isinstance(payload, dict):
        return []
    names = {p.get("partnerId"): book_key(p.get("name"))
             for p in payload.get("oddsPartners") or [] if isinstance(p, dict)}
    rows = []
    for day in payload.get("gameWeek") or []:
        for g in (day or {}).get("games") or []:
            if not isinstance(g, dict) or not g.get("id"):
                continue
            prices = {}
            for side in ("home", "away"):
                for o in ((g.get(f"{side}Team") or {}).get("odds") or []):
                    if isinstance(o, dict) and o.get("providerId") is not None:
                        prices.setdefault(o["providerId"], {})[side] = to_american(o.get("value"))
            base = {"game_id": int(g["id"]), "source": SCHEDULE_SOURCE,
                    "feed_updated_utc": None,
                    "start": parse_commence(g.get("startTimeUTC")),
                    "state": g.get("gameState")}
            for pid, p in prices.items():
                if p.get("home") is None or p.get("away") is None:
                    continue
                book = names.get(pid) or f"provider{pid}"
                rows.append(_row({**base, "book": book}, "ml",
                                 home_price=p["home"], away_price=p["away"]))
    return _keep_plausible(rows, SCHEDULE_SOURCE)


# ── Checks before storing (pure) ───────────────────────────────────

def excluded_books(environ=None) -> frozenset:
    """NHL_FEED_EXCLUDE_BOOKS as book keys. Unset = {'veikkaus'}; set
    empty = nothing excluded."""
    env = os.environ if environ is None else environ
    raw = env.get("NHL_FEED_EXCLUDE_BOOKS")
    if raw is None:
        raw = DEFAULT_EXCLUDED_BOOKS
    return frozenset(k for k in (book_key(b) for b in raw.split(",")) if k)


def swapped_books(rows: list, min_games: int = 2) -> dict:
    """{book: (games it disagreed on, games judged)} for each book whose
    moneyline favourite is the other team from the rest of the books' in
    at least min_games games and at least half of those judged. Pure.

    A game is judged for a book when at least two other books price it and
    their median no-vig home probability is CLEAR_FAVOURITE or more from
    50%. The sign of a swap: prices quoted for the wrong teams."""
    probs = {}
    for r in rows:
        if r["market"] != "ml":
            continue
        ph = american_implied_prob(r["home_price"])
        pa = american_implied_prob(r["away_price"])
        if ph is None or pa is None:
            continue
        probs.setdefault(r["game_id"], {})[r["book"]] = ph / (ph + pa)
    tally = {}
    for books in probs.values():
        for book, p in books.items():
            others = [q for b, q in books.items() if b != book]
            if len(others) < 2:
                continue
            consensus = statistics.median(others)
            if abs(consensus - 0.5) < CLEAR_FAVOURITE:
                continue
            judged, disagreed = tally.get(book, (0, 0))
            tally[book] = (judged + 1, disagreed + ((p - 0.5) * (consensus - 0.5) < 0))
    return {b: (d, j) for b, (j, d) in sorted(tally.items())
            if d >= min_games and 2 * d >= j}


def select_rows(rows: list, starts: dict, now: datetime, excluded) -> tuple:
    """(rows to store, counts of the rest). Pure.

    starts: game_id -> start_time_utc (aware, or None) for the games
    raw.games has. Dropped: excluded books; games raw.games doesn't know
    (the table's game_id references it); games under way or over (start
    passed or an in-play state); games with no start time anywhere, since
    they can't be shown to be pre-game."""
    kept = []
    skipped = {"excluded": 0, "unknown_game": 0, "in_play": 0, "no_start": 0}
    for r in rows:
        if r["book"] in excluded:
            skipped["excluded"] += 1
            continue
        if r["game_id"] not in starts:
            skipped["unknown_game"] += 1
            continue
        start = r.get("start") or starts[r["game_id"]]
        if r.get("state") in IN_PLAY_STATES or (start is not None and start <= now):
            skipped["in_play"] += 1
            continue
        if start is None:
            skipped["no_start"] += 1
            continue
        kept.append(r)
    return kept, skipped


# ── Fetching and storing ───────────────────────────────────────────

def _fetch_json(url: str, label: str):
    """GET one feed URL: the parsed JSON, or None after any failure
    (logged, never raised, so one broken source can't stop the others)."""
    try:
        resp = requests.get(url, timeout=TIMEOUT_S,
                            headers={"Accept": "application/json"})
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as e:
        r = e.response
        logger.error(f"NHL odds feed {label} failed: HTTP "
                     f"{getattr(r, 'status_code', '?')} {getattr(r, 'reason', '')}")
    except requests.Timeout:
        logger.error(f"NHL odds feed {label} failed: timed out after {TIMEOUT_S}s")
    except requests.ConnectionError as e:
        logger.error(f"NHL odds feed {label} failed: could not reach "
                     f"api-web.nhle.com ({type(e).__name__})")
    except ValueError:
        logger.error(f"NHL odds feed {label} failed: the response is not JSON")
    except Exception as e:
        logger.error(f"NHL odds feed {label} failed: {type(e).__name__}: {e}")
    return None


def fetch_feeds(schedule_date: date_cls) -> list:
    """Parsed rows from partner-US, partner-CA and the schedule for
    schedule_date, pausing REQUEST_PAUSE_S between requests. A source that
    fails or changes layout is logged and skipped."""
    rows = []
    requests_made = [(source, PARTNER_URL.format(country=country), parse_partner)
                     for source, country in PARTNER_SOURCES]
    requests_made.append((SCHEDULE_SOURCE,
                          SCHEDULE_URL.format(date=schedule_date.isoformat()),
                          lambda payload, _source: parse_schedule(payload)))
    for i, (source, url, parse) in enumerate(requests_made):
        if i:
            time.sleep(REQUEST_PAUSE_S)
        payload = _fetch_json(url, source)
        if payload is None:
            continue
        try:
            got = parse(payload, source)
        except Exception as e:
            logger.error(f"NHL odds feed {source}: could not read the response "
                         f"({type(e).__name__}: {e}); the layout may have changed")
            continue
        logger.debug(f"NHL odds feed {source}: {len(got)} price rows")
        rows += got
    return rows


def store(rows: list, now: datetime) -> int:
    """Write the rows select_rows() keeps into raw.nhl_feed_snapshots, all
    stamped captured_at = now (naive UTC). Logs any side-swapped book and
    what was skipped. Returns the number of rows written."""
    excluded = excluded_books()
    for book, (disagreed, judged) in swapped_books(rows).items():
        say = logger.info if book in excluded else logger.warning
        say(f"NHL feed: {book}{' (excluded)' if book in excluded else ''} looks "
            f"side-swapped: its moneyline favourite is the other team from the "
            f"other books' in {disagreed} of {judged} games")

    ids = sorted({r["game_id"] for r in rows})
    captured_at = now.astimezone(timezone.utc).replace(tzinfo=None)
    with engine.begin() as conn:
        starts = dict(conn.execute(text("""
            SELECT game_id, start_time_utc FROM raw.games WHERE game_id = ANY(:ids)
        """), {"ids": ids}).fetchall()) if ids else {}
        kept, skipped = select_rows(rows, starts, now, excluded)
        if kept:
            conn.execute(_INSERT, [
                {"captured_at": captured_at, "game_id": r["game_id"],
                 "source": r["source"], "book": r["book"], "market": r["market"],
                 **{f: r[f] for f in PRICE_FIELDS},
                 "feed_updated_utc": r["feed_updated_utc"]} for r in kept])

    by_source = {}
    for r in kept:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    extra = ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in skipped.items() if v)
    if skipped["unknown_game"]:
        logger.warning(f"NHL feed: {skipped['unknown_game']} price row(s) for games "
                       f"raw.games doesn't have (refresh the schedule: python "
                       f"pipeline.py daily)")
    logger.info(f"NHL feed: stored {len(kept)} price rows for "
                f"{len({r['game_id'] for r in kept})} games at {captured_at:%Y-%m-%d %H:%M:%S} UTC ("
                + (", ".join(f"{k} {v}" for k, v in sorted(by_source.items())) or "none")
                + ")" + (f"; skipped {extra}" if extra else ""))
    return len(kept)


def snapshot(skip_when_idle: bool = False) -> int:
    """One snapshot of the three free sources into raw.nhl_feed_snapshots.
    Costs nothing. skip_when_idle=True (for scheduled runs) skips the
    requests when raw.games has no game in the next 24 hours, so an
    off-season scheduler doesn't poll the NHL for nothing. Returns the
    number of rows written."""
    ensure_schema()
    ensure_table()
    if skip_when_idle and not upcoming_start_times(datetime.now(timezone.utc),
                                                   SNAPSHOT_HORIZON):
        logger.info("NHL feed: no game in raw.games starts in the next 24 hours; "
                    "no snapshot")
        return 0
    rows = fetch_feeds(local_today())
    if not rows:
        logger.warning("NHL feed: no prices came back (none posted yet, or every "
                       "request failed)")
    return store(rows, datetime.now(timezone.utc))


# ── Comparing with The Odds API (pure core) ────────────────────────

def _num(x) -> Optional[float]:
    return None if x is None else float(x)


def _scale(american) -> float:
    """American odds on a continuous scale: -105 -> -5, +105 -> +5, so
    prices either side of even money are as far apart as they look."""
    a = float(american)
    return a - 100.0 if a >= 100 else a + 100.0


def cents_diff(feed, odds) -> Optional[float]:
    """feed minus odds in 'cents' on that scale: -120 vs -125 is +5,
    +105 vs -105 is +10. Positive = the feed pays more."""
    if feed is None or odds is None:
        return None
    return _scale(feed) - _scale(odds)


def prob_diff(feed, odds) -> Optional[float]:
    """feed minus odds implied probability, in percentage points. Positive
    = the feed gives that side a higher chance (it pays less)."""
    pf, po = american_implied_prob(feed), american_implied_prob(odds)
    if pf is None or po is None:
        return None
    return (pf - po) * 100.0


def pair_prices(feed_rows: list, odds_rows: list, window: timedelta = PAIR_WINDOW) -> list:
    """One dict per feed price side, paired with the nearest odds row of
    the same game, book and market captured within `window` either way.

    Only COMPARE_BOOKS and COMPARE_MARKETS. status: 'ok'; 'line_differs'
    (a puck line or total at a different number, so the prices aren't
    comparable and no gap is given); 'unpaired' (no odds row in the
    window). Rows are dicts with game_id, captured_at (naive UTC), book,
    market and the PRICE_FIELDS."""
    by_key = {}
    for o in odds_rows:
        by_key.setdefault((o["game_id"], o["book"], o["market"]), []).append(o)
    out = []
    for f in feed_rows:
        if f["book"] not in COMPARE_BOOKS or f["market"] not in COMPARE_MARKETS:
            continue
        near = min(by_key.get((f["game_id"], f["book"], f["market"]), []),
                   key=lambda o: abs(o["captured_at"] - f["captured_at"]), default=None)
        if near is not None and abs(near["captured_at"] - f["captured_at"]) > window:
            near = None
        if near is None:
            status = "unpaired"
        elif f["market"] != "ml" and _num(f.get("line")) != _num(near.get("line")):
            status = "line_differs"
        else:
            status = "ok"
        for side in SIDES[f["market"]]:
            fp = f.get(f"{side}_price")
            op = near.get(f"{side}_price") if near else None
            out.append({
                "game_id": f["game_id"], "source": f["source"], "book": f["book"],
                "market": f["market"], "side": side,
                "feed_at": f["captured_at"],
                "odds_at": near["captured_at"] if near else None,
                "minutes_apart": (round((near["captured_at"] - f["captured_at"])
                                        .total_seconds() / 60.0, 1) if near else None),
                "feed_line": _num(f.get("line")),
                "odds_line": _num(near.get("line")) if near else None,
                "feed_price": fp, "odds_price": op,
                "cents": cents_diff(fp, op) if status == "ok" else None,
                "prob_pp": prob_diff(fp, op) if status == "ok" else None,
                "status": status,
            })
    return out


def summarize_pairs(pairs: list) -> list:
    """Per (source, book, market): sides paired, how many identical, mean
    and max absolute probability gap (points), mean absolute cents, and
    the sides left unpaired or at a different line."""
    groups = {}
    for p in pairs:
        groups.setdefault((p["source"], p["book"], p["market"]), []).append(p)
    out = []
    for (source, book, market), ps in sorted(groups.items()):
        ok = [p for p in ps if p["status"] == "ok" and p["prob_pp"] is not None]
        out.append({
            "source": source, "book": book, "market": market,
            "paired": len(ok),
            "identical": sum(p["cents"] == 0 for p in ok),
            "mean_abs_pp": (statistics.fmean(abs(p["prob_pp"]) for p in ok) if ok else None),
            "max_abs_pp": max((abs(p["prob_pp"]) for p in ok), default=None),
            "mean_abs_cents": (statistics.fmean(abs(p["cents"]) for p in ok) if ok else None),
            "unpaired": sum(p["status"] == "unpaired" for p in ps),
            "line_differs": sum(p["status"] == "line_differs" for p in ps),
        })
    return out


def freshness(rows: list) -> list:
    """Per (game, source, book, market) price series: snapshots, how many
    times the prices (or line) changed from one snapshot to the next, when
    they last changed, and how many different feed timestamps
    (lastUpdatedUTC) came with them. Pure; rows need captured_at."""
    groups = {}
    for r in sorted(rows, key=lambda r: r["captured_at"]):
        groups.setdefault((r["game_id"], r["source"], r["book"], r["market"]), []).append(r)
    out = []
    for (game_id, source, book, market), rs in sorted(groups.items()):
        prices = [tuple(_num(r.get(f)) for f in PRICE_FIELDS) for r in rs]
        changed = [rs[i]["captured_at"] for i in range(1, len(rs))
                   if prices[i] != prices[i - 1]]
        out.append({
            "game_id": game_id, "source": source, "book": book, "market": market,
            "snapshots": len(rs), "changes": len(changed),
            "first_at": rs[0]["captured_at"], "last_at": rs[-1]["captured_at"],
            "last_change_at": changed[-1] if changed else None,
            "feed_stamps": len({r.get("feed_updated_utc") for r in rs
                                if r.get("feed_updated_utc") is not None}),
        })
    return out


def summarize_freshness(series: list) -> list:
    """Per (source, book, market): price series, snapshots, changes, and
    how many series moved at least once."""
    groups = {}
    for s in series:
        groups.setdefault((s["source"], s["book"], s["market"]), []).append(s)
    return [{"source": source, "book": book, "market": market,
             "series": len(ss), "snapshots": sum(s["snapshots"] for s in ss),
             "changes": sum(s["changes"] for s in ss),
             "moved": sum(s["changes"] > 0 for s in ss),
             "feed_stamps": max(s["feed_stamps"] for s in ss)}
            for (source, book, market), ss in sorted(groups.items())]


# ── Comparing with The Odds API (database) ─────────────────────────

def compare_feeds(on_date: Optional[date_cls] = None) -> dict:
    """NHL feed against The Odds API for the games on one schedule
    (Eastern) date, default today's local date. Reads only.

    Returns {"date", "games", "feed_rows", "odds_rows", "pairs",
    "pair_summary", "freshness", "freshness_summary"}: see pair_prices,
    freshness and their summaries. Freshness covers every book in the
    feed, and The Odds API's draftkings and fanduel (source 'odds-api')."""
    ensure_schema()
    ensure_table()
    on_date = on_date or local_today()
    with engine.connect() as conn:
        games = [dict(r) for r in conn.execute(text("""
            SELECT game_id, away_team, home_team, start_time_utc FROM raw.games
            WHERE date = :d ORDER BY start_time_utc, game_id
        """), {"d": on_date}).mappings().all()]
        ids = [g["game_id"] for g in games]
        feed = [dict(r) for r in conn.execute(text("""
            SELECT game_id, captured_at, source, book, market, home_price,
                   away_price, over_price, under_price, draw_price, line,
                   feed_updated_utc
            FROM raw.nhl_feed_snapshots WHERE game_id = ANY(:ids)
            ORDER BY captured_at, id
        """), {"ids": ids}).mappings().all()] if ids else []
        odds = [dict(r) for r in conn.execute(text("""
            SELECT game_id, captured_at, book_name AS book, market_type AS market,
                   home_price, away_price, over_price, under_price, line
            FROM raw.odds_snapshots
            WHERE game_id = ANY(:ids) AND book_name = ANY(:books)
            ORDER BY captured_at, snapshot_id
        """), {"ids": ids, "books": list(COMPARE_BOOKS)}).mappings().all()] if ids else []
    for o in odds:
        o["source"] = ODDS_API_SOURCE
    pairs = pair_prices(feed, odds)
    series = freshness(feed + odds)
    return {"date": on_date, "games": games, "feed_rows": len(feed),
            "odds_rows": len(odds), "pairs": pairs,
            "pair_summary": summarize_pairs(pairs), "freshness": series,
            "freshness_summary": summarize_freshness(series)}


def _fmt(x, spec: str = ".2f") -> str:
    return "-" if x is None else format(x, spec)


def format_report(result: dict, detail: bool = False) -> str:
    """compare_feeds() as text for the command line."""
    lines = [f"NHL feed vs The Odds API, games of {result['date']}: "
             f"{len(result['games'])} games, {result['feed_rows']} NHL feed rows, "
             f"{result['odds_rows']} Odds API rows ({', '.join(COMPARE_BOOKS)})", ""]
    if not result["games"]:
        lines.append("No games on this date in raw.games.")
        return "\n".join(lines)

    lines += ["Price gaps: NHL feed minus The Odds API, same book, game and market, "
              f"snapshots at most {int(PAIR_WINDOW.total_seconds() // 60)} minutes apart.",
              "  prob = implied probability in points (+ = the feed gives that side "
              "a higher chance, so it pays less)",
              "  cents = American odds on a scale where -105 and +105 are 10 apart "
              "(+ = the feed pays more)"]
    head = (f"  {'source':<11}{'book':<12}{'market':<7}{'paired':>7}{'same':>6}"
            f"{'mean|prob|':>11}{'max|prob|':>10}{'mean|cents|':>12}"
            f"{'unpaired':>9}{'line!=':>8}")
    if result["pair_summary"]:
        lines.append(head)
        for s in result["pair_summary"]:
            lines.append(f"  {s['source']:<11}{s['book']:<12}{s['market']:<7}"
                         f"{s['paired']:>7}{s['identical']:>6}"
                         f"{_fmt(s['mean_abs_pp']):>11}{_fmt(s['max_abs_pp']):>10}"
                         f"{_fmt(s['mean_abs_cents'], '.1f'):>12}"
                         f"{s['unpaired']:>9}{s['line_differs']:>8}")
    else:
        lines.append("  No DraftKings or FanDuel prices from the NHL feed for these games.")
    if not result["odds_rows"]:
        lines.append("  No Odds API snapshots of draftkings or fanduel for these games: "
                     "take one (python -m ingestion.odds_api) within 10 minutes of an "
                     "NHL feed snapshot to get pairs.")

    lines += ["", "Freshness: how often each price series changed between snapshots "
              "(a series = one game, source, book and market).",
              f"  {'source':<11}{'book':<12}{'market':<7}{'series':>7}{'snapshots':>10}"
              f"{'changes':>8}{'moved':>6}{'feed stamps':>12}"]
    for s in result["freshness_summary"]:
        lines.append(f"  {s['source']:<11}{s['book']:<12}{s['market']:<7}{s['series']:>7}"
                     f"{s['snapshots']:>10}{s['changes']:>8}{s['moved']:>6}"
                     f"{s['feed_stamps']:>12}")
    if not result["freshness_summary"]:
        lines.append("  No snapshots yet.")

    if detail and result["pairs"]:
        lines += ["", "Every paired price:",
                  f"  {'game':<11}{'source':<11}{'book':<12}{'market':<7}{'side':<6}"
                  f"{'feed at':<20}{'feed':>6}{'odds':>6}{'min':>6}{'cents':>7}"
                  f"{'prob':>7}  status"]
        for p in result["pairs"]:
            lines.append(f"  {p['game_id']:<11}{p['source']:<11}{p['book']:<12}"
                         f"{p['market']:<7}{p['side']:<6}"
                         f"{p['feed_at']:%Y-%m-%d %H:%M:%S} "
                         f"{_fmt(p['feed_price'], 'd'):>6}{_fmt(p['odds_price'], 'd'):>6}"
                         f"{_fmt(p['minutes_apart'], '.1f'):>6}{_fmt(p['cents'], '.0f'):>7}"
                         f"{_fmt(p['prob_pp']):>7}  {p['status']}")
    return "\n".join(lines)


# ── Command line ───────────────────────────────────────────────────

def _date_arg(value: str) -> date_cls:
    try:
        return date_cls.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a date like 2026-09-29")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ingestion.nhl_odds",
        description="The NHL's free odds feed (DraftKings, FanDuel and the "
                    "schedule's partner books), stored in raw.nhl_feed_snapshots "
                    "beside The Odds API for comparison. Free: no key, no credits.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("snapshot", help="store one snapshot of the three NHL feed "
                                    "sources (three requests, a second apart)")
    cmp = sub.add_parser("compare", help="compare stored NHL feed prices with "
                                         "raw.odds_snapshots for one date")
    cmp.add_argument("--date", type=_date_arg, default=None,
                     help="schedule date YYYY-MM-DD (default: today's local date)")
    cmp.add_argument("--detail", action="store_true",
                     help="also list every paired price")
    args = parser.parse_args(argv)

    if args.command == "snapshot":
        n = snapshot()
        print(f"Stored {n} NHL feed price rows in raw.nhl_feed_snapshots.")
    else:
        print(format_report(compare_feeds(args.date), detail=args.detail))
    return 0


if __name__ == "__main__":
    main()
