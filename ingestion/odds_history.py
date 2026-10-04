"""
ingestion/odds_history.py
Past two-way prices from The Odds API's paid historical endpoint, stored
in raw.odds_history, with every purchase logged in raw.odds_history_fetches.

Terms:
- Two-way price → a bet with only two outcomes (home or away including
  overtime and shootout; over or under the total). ESPN's older lines are
  three-way (a regulation draw is a third outcome), which is not the bet
  the models price.
- Snapshot → every book's prices for every listed game at one moment.
- Closing price → the last price before puck drop; the market's final and
  sharpest opinion, and the yardstick for closing line value (CLV → whether
  a bet got a better price than the close).
- Moneyline (h2h) → who wins; totals → over/under the combined goals line.
- American odds → -120 means risk 120 to win 100; +110 means risk 100 to
  win 110.
- Credits → The Odds API's billing unit.

The endpoint (the v4 docs, confirmed 2026-10-04):
  GET /v4/historical/sports/icehockey_nhl/odds?date=<ISO time>
  returns the snapshot taken at or just BEFORE `date` ("the closest
  snapshot equal to or earlier than the provided date"), wrapped as
  {timestamp, previous_timestamp, next_timestamp, data: [events]}.
  Snapshots every 10 minutes from June 2020, every 5 minutes from
  September 2022; icehockey_nhl history starts 2020-06-29. Paid plans only.
  Cost: 10 x markets x regions per call, where up to 10 named bookmakers
  bill as one region. A test call on 2024-12-10 (h2h, regions=us) read
  x-requests-last = 10, so h2h + totals with 10 named books costs 20.
  An empty response costs nothing.

What one snapshot holds: every game the books list at that moment,
including later games that day and the next days. In-play games are
listed too, with live prices: those are dropped (commence_time at or
before the snapshot), so nothing stored here was priced after puck drop.

The purchase plan (`plan` prints it with its cost; `fetch` buys it in
order, stopping at the credit cap):
  close   → per game date, the start times are grouped into clusters (a
            cluster starts at the earliest unclaimed puck drop and takes
            every start within 75 minutes of it); one snapshot per cluster
            at its first puck drop minus 10 minutes. Every game in the
            cluster gets a price at most 10 + 75 minutes before its start,
            usually 10 to 40.
  morning → one snapshot per game date at 10:00 America/Chicago, for the
            bet-timing study (does the price move between the morning and
            the close?).
The plan needs raw.games.start_time_utc; seasons loaded before that column
existed have none, so `starts` fills it from the free NHL schedule API.

Never bought twice: before a call, a logged fetch (holding every requested
market, whatever its books; --same-books-only to require the same books,
which re-buys snapshots bought with other books and so needs explicit
--steps)
whose snapshot covers the requested time (snapshot_ts <= t <
next_timestamp, or the same requested time) means the API would return the
same snapshot, so the call is skipped. Re-runs resume where the last one
stopped. Stored rows are unique per (snapshot, event, book, market, side).

Budget: `fetch` stops before a call that would take this run's spending
past --max-credits, the fetch log's all-time total past --cap-total, or
the account's x-requests-remaining below --reserve (default 6,000, kept
for the live jobs). It reads the remaining credits from the free /sports
endpoint before the first call (and will not start when it cannot, unless
--allow-unknown-remaining) and from every response's headers; a response
without those headers is counted at its full expected cost. It also stops
after 5 failed calls in a row.

A paid call that cannot be parsed or stored is still logged, as
'paid_unparsed' with its credits, and counts as bought; its raw copy is
kept, and `reparse` loads it later without another call.

Every paid response is also saved, gzipped, under --raw-dir (default
data/odds_history/, git-ignored), so paid data survives a database loss.
A copy is named <purpose>_<requested time>_<hash>.json.gz, where the hash
is a short fingerprint of the markets and book list, and never overwrites
an existing file (a second copy of the same request gets _2, _3, ...).

The API key never reaches the logs (odds_api._redact on everything
logged; exceptions from requests, whose messages carry the URL, are never
logged themselves).
"""
import argparse
import gzip
import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import requests
from sqlalchemy import text

from config.settings import PROJECT_ROOT, engine
from ingestion.odds_api import (BASE_URL, SPORT, _TEAM_NAME_TO_ABBREV, _api_key,
                                _http_error_summary, _key_list, _redact,
                                match_event, parse_commence, region_units)

logger = logging.getLogger("nhl.ingestion.odds_history")

TIMEOUT_S = 60
PAUSE_S = 0.2
HIST_PATH = f"/historical/sports/{SPORT}/odds"
CREDITS_PER_MARKET_REGION = 10          # historical featured markets (the v4 docs)
DEFAULT_MARKETS = "h2h,totals"
# Chosen 2026-10-04 from probe calls on 2022-12-13 and 2024-12-10 (see
# docs/historical_odds.md): pinnacle is the sharp reference; draftkings,
# fanduel, betmgm, betrivers and espnbet are also in the live default list,
# so history and live snapshots compare book for book; williamhill_us
# (Caesars) is the other large US book; lowvig, betonlineag and bovada are
# offshore market books priced in every season since 2020 (lowvig is
# BetOnline at a reduced margin, close to a no-vig price). Lesson from the
# 2024-25 pull: lowvig and betonlineag quoted the same price 99.2% of the
# time, so betonlineag adds almost nothing; for seasons not yet bought,
# pass --bookmakers with it swapped for another book (see
# docs/historical_odds.md). Kept as the default so 2024-25 stays one list.
DEFAULT_BOOKMAKERS = ("pinnacle", "draftkings", "fanduel", "betmgm", "williamhill_us",
                      "betrivers", "espnbet", "lowvig", "betonlineag", "bovada")
CLUSTER_SPAN = timedelta(minutes=75)
CLOSE_LEAD = timedelta(minutes=10)
MORNING_TZ = ZoneInfo("America/Chicago")
MORNING_TIME = (10, 0)
DEFAULT_RESERVE = 6000
STOP_STATUSES = (401, 403, 429)          # bad key, plan/quota, rate limit after a retry
MAX_CONSECUTIVE_ERRORS = 5               # a run stops after this many failed calls in a row
_SIDES_H2H = ("home", "away")
_MARKET_KEYS = {"h2h": "h2h", "totals": "totals"}
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "odds_history"

# The same DDL is in config/migrate.py (TABLES) and db/schema.sql.
DDL = [
    """
    CREATE TABLE IF NOT EXISTS raw.odds_history (
        id              BIGSERIAL PRIMARY KEY,
        snapshot_ts     TIMESTAMP NOT NULL,            -- the API's snapshot time, naive UTC
        requested_ts    TIMESTAMP NOT NULL,            -- the date= asked for, naive UTC
        event_id        VARCHAR(64) NOT NULL,          -- The Odds API event id
        game_id         BIGINT REFERENCES raw.games(game_id),  -- NULL when unmatched
        commence_time   TIMESTAMP,                     -- the API's puck drop, naive UTC
        home_name       VARCHAR(40),                   -- as the API names the teams
        away_name       VARCHAR(40),
        book            VARCHAR(40) NOT NULL,          -- Odds API bookmaker key
        market          VARCHAR(10) NOT NULL,          -- h2h or totals
        side            VARCHAR(5) NOT NULL,           -- home, away, over, under
        price           INTEGER NOT NULL,              -- American odds
        point           NUMERIC(4,1),                  -- the total line; NULL for h2h
        book_updated_at TIMESTAMP,                     -- the market's last_update, naive UTC
        UNIQUE (snapshot_ts, event_id, book, market, side)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_odds_history_game ON raw.odds_history(game_id, market)",
    "CREATE INDEX IF NOT EXISTS idx_odds_history_snapshot ON raw.odds_history(snapshot_ts)",
    """
    CREATE TABLE IF NOT EXISTS raw.odds_history_fetches (
        id              BIGSERIAL PRIMARY KEY,
        requested_ts    TIMESTAMP NOT NULL,            -- the date= asked for, naive UTC
        purpose         VARCHAR(12) NOT NULL,          -- close, morning; probe = hand-logged test call
        season          INTEGER,
        markets         VARCHAR(60) NOT NULL,
        bookmakers      VARCHAR(200) NOT NULL,
        snapshot_ts     TIMESTAMP,                     -- what the API returned
        next_ts         TIMESTAMP,                     -- the API's next_timestamp
        credits         INTEGER NOT NULL DEFAULT 0,    -- x-requests-last
        n_events        INTEGER NOT NULL DEFAULT 0,
        n_rows          INTEGER NOT NULL DEFAULT 0,
        status          VARCHAR(16) NOT NULL,          -- ok, empty, error, paid_unparsed; probe = hand-logged
        fetched_at      TIMESTAMP NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_odds_history_fetches_ts "
    "ON raw.odds_history_fetches(requested_ts)",
]


def ensure_tables(db=None) -> None:
    """Create raw.odds_history and raw.odds_history_fetches when missing,
    and widen the fetch log's status from its first VARCHAR(10) to
    VARCHAR(16) ('paid_unparsed' is 13 characters). Widening a VARCHAR is a
    catalog change in PostgreSQL, with no table rewrite, and runs once."""
    with (db or engine).begin() as conn:
        for stmt in DDL:
            conn.execute(text(stmt))
        width = conn.execute(text("""
            SELECT character_maximum_length FROM information_schema.columns
            WHERE table_schema = 'raw' AND table_name = 'odds_history_fetches'
              AND column_name = 'status'
        """)).scalar()
        if width is not None and width < 16:
            conn.execute(text("ALTER TABLE raw.odds_history_fetches "
                              "ALTER COLUMN status TYPE VARCHAR(16)"))


# ── Pure helpers ──────────────────────────────────────────────────

def _naive(t: Optional[datetime]) -> Optional[datetime]:
    """Aware -> naive UTC (the storage convention); naive passes through."""
    if t is None:
        return None
    if t.tzinfo is not None:
        t = t.astimezone(timezone.utc).replace(tzinfo=None)
    return t


def _aware(t: Optional[datetime]) -> Optional[datetime]:
    """Naive UTC or aware -> aware UTC."""
    if t is None:
        return None
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)


def iso_z(t: datetime) -> str:
    """The date= form the API takes: 2024-12-10T23:50:00Z."""
    return _aware(t).strftime("%Y-%m-%dT%H:%M:%SZ")


def _price(value) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return int(round(v)) if math.isfinite(v) else None


def _point(value) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def call_cost(markets: str, bookmakers: str) -> int:
    """Credits one historical call should cost: 10 x markets x regions,
    up to 10 named books billing as one region."""
    return (CREDITS_PER_MARKET_REGION * len(_key_list(markets))
            * region_units({"bookmakers": bookmakers}))


def cluster_starts(starts: Iterable[datetime], span: timedelta = CLUSTER_SPAN) -> List[List[datetime]]:
    """Group puck drops: each cluster opens at the earliest start not yet
    taken and holds every start within `span` of that first one."""
    out: List[List[datetime]] = []
    for s in sorted(set(_aware(s) for s in starts if s is not None)):
        if out and s - out[-1][0] <= span:
            out[-1].append(s)
        else:
            out.append([s])
    return out


@dataclass(frozen=True)
class PlannedFetch:
    requested_ts: datetime       # aware UTC
    purpose: str                 # close | morning
    season: int
    game_date: date
    n_games: int


def morning_time(d: date) -> datetime:
    """10:00 America/Chicago on schedule date d, as aware UTC."""
    local = datetime(d.year, d.month, d.day, *MORNING_TIME, tzinfo=MORNING_TZ)
    return local.astimezone(timezone.utc)


def plan_requests(games: Sequence[dict], purpose: str) -> List[PlannedFetch]:
    """The snapshots one purpose needs. games: dicts with season, date
    (schedule date) and start_time_utc (aware). Games without a start time
    are left out (fill them first with `starts`)."""
    by_date: Dict[Tuple[int, date], List[datetime]] = {}
    for g in games:
        if g.get("start_time_utc") is None:
            continue
        by_date.setdefault((g["season"], g["date"]), []).append(_aware(g["start_time_utc"]))
    out: List[PlannedFetch] = []
    for (season, d), starts in sorted(by_date.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if purpose == "close":
            for cluster in cluster_starts(starts):
                out.append(PlannedFetch(cluster[0] - CLOSE_LEAD, "close", season, d,
                                        sum(1 for s in starts if s in cluster)))
        elif purpose == "morning":
            t = morning_time(d)
            n = sum(1 for s in starts if s > t)
            if n:
                out.append(PlannedFetch(t, "morning", season, d, n))
        else:
            raise ValueError(f"unknown purpose {purpose!r} (close or morning)")
    return out


def is_covered(requested: datetime, done: Iterable[dict]) -> bool:
    """True when a logged fetch already holds the snapshot a request for
    `requested` would return: the same requested time, or a snapshot with
    snapshot_ts <= requested < next_ts (the API returns the latest snapshot
    at or before the requested time). done: rows with requested_ts,
    snapshot_ts, next_ts (naive UTC or aware)."""
    r = _aware(requested)
    for f in done:
        if _aware(f.get("requested_ts")) == r:
            return True
        snap, nxt = _aware(f.get("snapshot_ts")), _aware(f.get("next_ts"))
        if snap is not None and nxt is not None and snap <= r < nxt:
            return True
    return False


def parse_snapshot(body: dict, requested: datetime, candidates: List[dict]) -> Tuple[List[dict], dict]:
    """Rows for raw.odds_history from one historical response, and counts.

    Events that started at or before the snapshot are dropped (in-play
    prices): by the API's commence_time, or by the matched game's NHL
    start time when that is earlier. Each pre-game event is matched to raw.games with
    odds_api.match_event (home team, start within 6 hours); an unmatched
    event is still stored, with game_id NULL and the team names, so a
    later schedule fix can match it. h2h outcomes become home/away by team
    name; totals outcomes over/under with the line in point."""
    snap = parse_commence(body.get("timestamp"))
    stats = {"events": 0, "in_play": 0, "matched": 0, "unmatched": 0, "rows": 0}
    if snap is None:
        return [], stats
    rows: List[dict] = []
    starts = {g["game_id"]: _aware(g["start_time_utc"]) for g in candidates
              if g.get("start_time_utc") is not None}
    for ev in body.get("data") or []:
        stats["events"] += 1
        commence = parse_commence(ev.get("commence_time"))
        if commence is not None and commence <= snap:
            stats["in_play"] += 1
            continue
        game_id, status = match_event(ev, candidates, snap)
        start = starts.get(game_id)
        if start is not None and start <= snap:
            # the NHL's scheduled puck drop has passed even if the API's
            # commence_time (often a few minutes later) has not
            stats["in_play"] += 1
            continue
        stats["matched" if game_id else "unmatched"] += 1
        home, away = ev.get("home_team", ""), ev.get("away_team", "")
        for bm in ev.get("bookmakers") or []:
            book = bm.get("key")
            if not book:
                continue
            for mk in bm.get("markets") or []:
                market = _MARKET_KEYS.get(mk.get("key", ""))
                if market is None:
                    continue
                updated = _naive(parse_commence(mk.get("last_update") or bm.get("last_update")))
                for oc in mk.get("outcomes") or []:
                    name = oc.get("name", "")
                    if market == "h2h":
                        side = "home" if name == home else "away" if name == away else None
                    else:
                        side = {"Over": "over", "Under": "under"}.get(name)
                    price = _price(oc.get("price"))
                    if side is None or price is None:
                        continue
                    rows.append({
                        "snapshot_ts": _naive(snap), "requested_ts": _naive(requested),
                        "event_id": str(ev.get("id", "")), "game_id": game_id,
                        "commence_time": _naive(commence),
                        "home_name": home[:40] or None, "away_name": away[:40] or None,
                        "book": book[:40], "market": market, "side": side, "price": price,
                        "point": _point(oc.get("point")) if market == "totals" else None,
                        "book_updated_at": updated,
                    })
    stats["rows"] = len(rows)
    return rows, stats


# ── HTTP (never logs the key) ─────────────────────────────────────

def http_get(path: str, params: dict) -> Tuple[Optional[object], dict, Optional[int]]:
    """GET BASE_URL + path with the key added: (body, headers, status).
    body None after any failure; status None when no response came back.
    A 429 is retried once after a pause."""
    key = _api_key()
    if not key:
        logger.error("ODDS_API_KEY is not set (missing, or still the .env.example "
                     "placeholder): no historical request made")
        return None, {}, None
    url = f"{BASE_URL}/{path.lstrip('/')}"
    for attempt in (1, 2):
        try:
            resp = requests.get(url, params={"apiKey": key, **params}, timeout=TIMEOUT_S)
            if resp.status_code == 429 and attempt == 1:
                logger.warning("Odds API: too many requests (HTTP 429); retrying once")
                time.sleep(3)
                continue
            if resp.status_code >= 400:
                logger.error(f"Odds API historical request failed: {_http_error_summary(resp)}")
                return None, dict(resp.headers), resp.status_code
            return resp.json(), dict(resp.headers), resp.status_code
        except requests.Timeout:
            logger.error(f"Odds API historical request failed: timed out after {TIMEOUT_S}s")
            return None, {}, None
        except requests.ConnectionError as e:
            logger.error(f"Odds API historical request failed: could not reach "
                         f"api.the-odds-api.com ({type(e).__name__})")
            return None, {}, None
        except Exception as e:
            logger.error(f"Odds API historical request failed: {type(e).__name__}: {_redact(e)}")
            return None, {}, None
    return None, {}, 429


def _int_header(headers: dict, name: str) -> Optional[int]:
    for k, v in (headers or {}).items():
        if k.lower() == name:
            try:
                return int(float(v))
            except (TypeError, ValueError):
                return None
    return None


# ── Database ──────────────────────────────────────────────────────

def load_games(seasons: Sequence[int], db=None) -> List[dict]:
    """Regular-season and playoff games of these seasons (not postponed,
    suspended or cancelled), with their start times."""
    with (db or engine).connect() as conn:
        rows = conn.execute(text("""
            SELECT game_id, season, date, start_time_utc, home_team, away_team
            FROM raw.games
            WHERE season = ANY(:seasons) AND game_type IN (2, 3)
              AND COALESCE(schedule_state, 'OK') NOT IN ('PPD', 'SUSP', 'CNCL')
        """), {"seasons": list(seasons)}).mappings().all()
    return [dict(r) for r in rows]


BOUGHT_STATUSES = ("ok", "empty", "paid_unparsed")


def covering_fetches(rows: Iterable[dict], markets: str, bookmakers: str,
                     same_books: bool = False) -> List[dict]:
    """The logged fetches (ok, empty, or paid_unparsed: paid for, its raw
    copy kept, waiting for `reparse`) that count as already bought for
    a request of these markets: every requested market was in the bought
    call. By default the book list is ignored, so changing the books never
    re-buys a stored snapshot; same_books=True counts only identical lists
    (for deliberately buying other books at the same times)."""
    want, books = set(_key_list(markets)), _key_list(bookmakers)
    out = []
    for r in rows:
        if r.get("status") not in BOUGHT_STATUSES:
            continue
        if not want <= set(_key_list(r.get("markets", ""))):
            continue
        if same_books and _key_list(r.get("bookmakers", "")) != books:
            continue
        out.append(r)
    return out


def load_done(markets: str, bookmakers: str, same_books: bool = False, db=None) -> List[dict]:
    """covering_fetches() over the fetch log."""
    with (db or engine).connect() as conn:
        rows = conn.execute(text("""
            SELECT requested_ts, snapshot_ts, next_ts, markets, bookmakers, status
            FROM raw.odds_history_fetches
        """)).mappings().all()
    return covering_fetches([dict(r) for r in rows], markets, bookmakers, same_books)


def credits_logged(db=None) -> int:
    with (db or engine).connect() as conn:
        return int(conn.execute(text(
            "SELECT COALESCE(SUM(credits), 0) FROM raw.odds_history_fetches")).scalar())


def _candidates(games: List[dict], snap: datetime) -> List[dict]:
    """Games that could match a snapshot's pre-game events: starting from
    the snapshot to 10 days after it (before a season opens the API lists
    its first week)."""
    lo, hi = snap - timedelta(hours=6), snap + timedelta(days=10)
    return [g for g in games if g.get("start_time_utc") is not None
            and lo <= _aware(g["start_time_utc"]) <= hi]


INSERT_ROW = text("""
    INSERT INTO raw.odds_history
        (snapshot_ts, requested_ts, event_id, game_id, commence_time, home_name,
         away_name, book, market, side, price, point, book_updated_at)
    VALUES (:snapshot_ts, :requested_ts, :event_id, :game_id, :commence_time,
            :home_name, :away_name, :book, :market, :side, :price, :point,
            :book_updated_at)
    ON CONFLICT (snapshot_ts, event_id, book, market, side) DO NOTHING
""")

INSERT_FETCH = text("""
    INSERT INTO raw.odds_history_fetches
        (requested_ts, purpose, season, markets, bookmakers, snapshot_ts, next_ts,
         credits, n_events, n_rows, status)
    VALUES (:requested_ts, :purpose, :season, :markets, :bookmakers, :snapshot_ts,
            :next_ts, :credits, :n_events, :n_rows, :status)
""")


def store(rows: List[dict], fetch: dict, db=None) -> None:
    """Rows and their fetch-log line in one transaction."""
    with (db or engine).begin() as conn:
        if rows:
            conn.execute(INSERT_ROW, rows)
        conn.execute(INSERT_FETCH, fetch)


def season_of(t: datetime) -> int:
    """The NHL season a moment falls in: August onwards starts a season
    (2024-10-04 -> 20242025, 2025-03-01 -> 20242025)."""
    y = t.year if t.month >= 8 else t.year - 1
    return y * 10000 + y + 1


def rematch_pairs(events: Iterable[dict], games: List[dict]) -> Dict[str, int]:
    """{event_id: game_id} for stored events that had no game, by the same
    rule as a fetch (home team, start within 6 hours). events: dicts with
    event_id, home_name, away_name, commence_time (naive UTC or aware);
    games: as load_games returns. Pure. Events with no commence_time, an
    unknown team or no game in reach are left out."""
    out: Dict[str, int] = {}
    for e in events:
        commence = _aware(e.get("commence_time"))
        if commence is None:
            continue
        ev = {"home_team": e.get("home_name") or "", "away_team": e.get("away_name") or "",
              "commence_time": iso_z(commence)}
        game_id, _ = match_event(ev, _candidates(games, commence),
                                 commence - timedelta(seconds=1))
        if game_id:
            out[e["event_id"]] = game_id
    return out


def rematch_unmatched(db=None) -> int:
    """Give stored rows with no game_id one (rematch_pairs): for events
    bought before their game was in raw.games or had a start time.
    Returns the rows updated."""
    db = db or engine
    with db.connect() as conn:
        events = [dict(r) for r in conn.execute(text("""
            SELECT DISTINCT event_id, home_name, away_name, commence_time
            FROM raw.odds_history WHERE game_id IS NULL
        """)).mappings()]
    if not events:
        return 0
    seasons = {season_of(e["commence_time"]) for e in events if e["commence_time"] is not None}
    pairs = rematch_pairs(events, load_games(sorted(seasons), db=db))
    n = 0
    with db.begin() as conn:
        for event_id, game_id in pairs.items():
            n += conn.execute(text("""
                UPDATE raw.odds_history SET game_id = :g
                WHERE event_id = :e AND game_id IS NULL
            """), {"g": game_id, "e": event_id}).rowcount or 0
    logger.info(f"Matched {n} stored row(s) that had no game")
    return n


UPDATE_FETCH = text("""
    UPDATE raw.odds_history_fetches
    SET snapshot_ts = :snapshot_ts, next_ts = :next_ts, n_events = :n_events,
        n_rows = :n_rows, status = :status
    WHERE id = :id AND status = 'paid_unparsed'
""")


def reparse_file(path: Path, requested: datetime, games: List[dict]) -> Tuple[List[dict], dict]:
    """(rows, fetch-log fields) from a raw copy, as a fetch would have
    stored them."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        body = json.load(f)
    if not isinstance(body, dict):
        raise ValueError(f"{Path(path).name} does not hold a historical response")
    rows, update, _stats = parse_body(body, requested, games)
    return rows, update


def reparse_unparsed(raw_dir: Path = DEFAULT_RAW_DIR, db=None) -> Dict[str, int]:
    """Load every paid_unparsed fetch from its raw copy: store its rows and
    set its log line to what the fetch would have written (ok or empty).
    A fetch whose copy is missing or still fails stays paid_unparsed."""
    db = db or engine
    with db.connect() as conn:
        fetches = [dict(r) for r in conn.execute(text("""
            SELECT id, requested_ts, purpose, season, markets, bookmakers
            FROM raw.odds_history_fetches WHERE status = 'paid_unparsed' ORDER BY id
        """)).mappings()]
    out = {"unparsed": len(fetches), "loaded": 0, "missing": 0, "failed": 0, "rows": 0}
    if not fetches:
        return out
    games = load_games(sorted({f["season"] for f in fetches if f["season"]}), db=db)
    for f in fetches:
        stem = raw_stem(f["purpose"], f["requested_ts"], f["markets"], f["bookmakers"])
        path = find_raw(raw_dir, stem)
        if path is None:
            out["missing"] += 1
            logger.warning(f"No raw copy {stem}*.json.gz in {raw_dir} for fetch {f['id']}")
            continue
        try:
            rows, update = reparse_file(path, f["requested_ts"], games)
            with db.begin() as conn:
                if rows:
                    conn.execute(INSERT_ROW, rows)
                conn.execute(UPDATE_FETCH, {**update, "id": f["id"]})
        except Exception as e:
            out["failed"] += 1
            logger.error(f"Fetch {f['id']} ({path.name}) still fails: "
                         f"{type(e).__name__}: {_redact(e)[:300]}")
            continue
        out["loaded"] += 1
        out["rows"] += len(rows)
    logger.info(f"Reparse: {out}")
    return out


# ── Start times for older seasons (free NHL schedule API) ────────

def collect_start_times(lo: date, hi: date, fetch_week: Callable,
                        pause: float = 0.3) -> Dict[int, datetime]:
    """{game_id: puck drop (aware UTC)} from the NHL weekly schedule, one
    fetch_week(YYYY-MM-DD) call per 7 days from lo while <= hi. A week that
    fails is skipped with a warning."""
    from ingestion.nhl_api import _parse_start_time
    found: Dict[int, datetime] = {}
    d = lo
    while d <= hi:
        try:
            week = fetch_week(d.strftime("%Y-%m-%d")) or {}
        except Exception as e:
            logger.warning(f"NHL schedule fetch failed for {d}: {type(e).__name__}")
            week = {}
        for day in week.get("gameWeek", []):
            for g in day.get("games", []):
                t = _parse_start_time(g.get("startTimeUTC"))
                if g.get("id") and t is not None:
                    found[int(g["id"])] = t
        d += timedelta(days=7)
        if pause:
            time.sleep(pause)
    return found


def fill_start_times(season: int, db=None, fetch_week=None, pause: float = 0.3) -> int:
    """Fill raw.games.start_time_utc where it is NULL for one season, from
    the NHL weekly schedule (free). Only that column is written, and only
    where empty and in that season. Returns the number of games filled."""
    if fetch_week is None:
        from ingestion.nhl_api import client
        fetch_week = lambda d: client.schedule.weekly_schedule(date=d)   # noqa: E731
    db = db or engine
    with db.connect() as conn:
        lo, hi = conn.execute(text("""
            SELECT MIN(date), MAX(date) FROM raw.games
            WHERE season = :s AND start_time_utc IS NULL
        """), {"s": season}).one()
    if lo is None:
        logger.info(f"Season {season}: every game already has a start time")
        return 0
    found = collect_start_times(lo, hi, fetch_week, pause)
    with db.begin() as conn:
        n = 0
        for gid, t in found.items():
            n += conn.execute(text("""
                UPDATE raw.games SET start_time_utc = :t
                WHERE game_id = :g AND season = :s AND start_time_utc IS NULL
            """), {"t": t, "g": gid, "s": season}).rowcount or 0
    logger.info(f"Season {season}: filled {n} start times from the NHL schedule")
    return n


# ── The run ───────────────────────────────────────────────────────

@dataclass
class Budget:
    max_credits: Optional[int] = None    # this run
    cap_total: Optional[int] = None      # all logged fetches, all runs
    reserve: int = DEFAULT_RESERVE       # x-requests-remaining floor
    spent: int = 0
    logged_before: int = 0
    remaining: Optional[int] = None

    def blocks(self, cost: int) -> Optional[str]:
        """Why a call costing `cost` may not be made, or None."""
        if self.max_credits is not None and self.spent + cost > self.max_credits:
            return f"this run's cap of {self.max_credits} credits ({self.spent} spent)"
        if self.cap_total is not None and self.logged_before + self.spent + cost > self.cap_total:
            return (f"the all-time cap of {self.cap_total} credits "
                    f"({self.logged_before + self.spent} logged)")
        if self.remaining is not None and self.remaining - cost < self.reserve:
            return (f"the reserve: {self.remaining} credits left on the account, "
                    f"{self.reserve} kept for the live jobs")
        return None


@dataclass
class RunReport:
    calls: int = 0
    skipped: int = 0
    credits: int = 0
    rows: int = 0
    errors: int = 0
    stopped: Optional[str] = None
    by_purpose: Dict[str, Dict[str, int]] = field(default_factory=dict)


def describe_plan(plan: List[PlannedFetch], done: List[dict], cost: int) -> List[str]:
    """One line per (purpose, season): calls, games, credits, done."""
    groups: Dict[Tuple[str, int], List[PlannedFetch]] = {}
    for p in plan:
        groups.setdefault((p.purpose, p.season), []).append(p)
    lines, total = [], 0
    for (purpose, season), ps in groups.items():
        todo = [p for p in ps if not is_covered(p.requested_ts, done)]
        total += len(todo) * cost
        lines.append(f"  {purpose:8s} {season}: {len(ps)} snapshots over "
                     f"{len({p.game_date for p in ps})} game dates "
                     f"({sum(p.n_games for p in ps)} games); {len(ps) - len(todo)} already "
                     f"bought; to buy {len(todo)} x {cost} = {len(todo) * cost} credits")
    lines.append(f"  total still to buy: {total} credits")
    return lines


def spread_order(n: int) -> List[int]:
    """0..n-1 reordered so that any prefix is spread evenly over the range
    (bit-reversed order: 0, n/2, n/4, 3n/4, ...). A step the budget cuts
    short then covers the whole season thinly instead of its first months."""
    if n <= 0:
        return []
    bits = max(1, (n - 1).bit_length())

    def rev(i: int) -> int:
        return int(format(i, f"0{bits}b")[::-1], 2)

    return sorted(range(n), key=rev)


def build_plan(steps: Sequence[Tuple[str, int]], games: List[dict]) -> List[PlannedFetch]:
    """The plan in priority order: steps are (purpose, season) pairs, each
    step's snapshots in spread_order (deterministic, so re-runs resume)."""
    plan: List[PlannedFetch] = []
    for purpose, season in steps:
        step = plan_requests([g for g in games if g["season"] == season], purpose)
        plan += [step[i] for i in spread_order(len(step))]
    return plan


def raw_stem(purpose: str, requested: datetime, markets: str, bookmakers: str) -> str:
    """The raw copy's file name without its extension:
    close_2024-10-04T165000Z_1a2b3c4d, the last part a short hash of the
    markets and the book list (order-blind), so copies of the same time
    bought with different books never share a name."""
    key = (",".join(sorted(_key_list(markets))) + "|"
           + ",".join(sorted(_key_list(bookmakers))))
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    return f"{purpose}_{iso_z(requested).replace(':', '')}_{digest}"


def save_raw(raw_dir: Path, stem: str, body: dict) -> Path:
    """Write body gzipped as <stem>.json.gz, or <stem>_2.json.gz, _3, ...
    when that name is taken: an existing copy is never overwritten (the
    file is opened in exclusive-create mode)."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        path = raw_dir / (f"{stem}.json.gz" if n == 1 else f"{stem}_{n}.json.gz")
        try:
            with gzip.open(path, "xt", encoding="utf-8") as f:
                json.dump(body, f)
            return path
        except FileExistsError:
            n += 1


def find_raw(raw_dir: Path, stem: str) -> Optional[Path]:
    """The newest raw copy saved under this stem (the highest _N), or None."""
    best, best_n = None, 0
    for path in Path(raw_dir).glob(f"{stem}*.json.gz"):
        rest = path.name[len(stem):-len(".json.gz")]
        if rest == "":
            n = 1
        elif rest.startswith("_") and rest[1:].isdigit():
            n = int(rest[1:])
        else:
            continue
        if n > best_n:
            best, best_n = path, n
    return best


def parse_body(body: dict, requested: datetime, games: List[dict]) -> Tuple[List[dict], dict, dict]:
    """(rows, fetch-log fields, stats) for one paid response: the rows for
    raw.odds_history and the snapshot_ts, next_ts, n_events, n_rows and
    status to log. Used by a fetch and by `reparse` on a raw copy.

    Status: ok (events listed), empty (a snapshot with no events), or error
    when the response has no snapshot `timestamp`: that is not a snapshot
    at all, so it is not counted as bought and a later run retries it."""
    snap = parse_commence(body.get("timestamp"))
    rows, stats = parse_snapshot(body, requested, _candidates(games, snap) if snap else [])
    status = "error" if snap is None else "ok" if body.get("data") else "empty"
    update = {"snapshot_ts": _naive(snap),
              "next_ts": _naive(parse_commence(body.get("next_timestamp"))),
              "n_events": stats["events"], "n_rows": len(rows), "status": status}
    return rows, update, stats


def run_fetch(plan: List[PlannedFetch], markets: str, bookmakers: str, budget: Budget,
              games: List[dict], done: List[dict], getter: Callable = http_get,
              saver: Callable = store, raw_dir: Optional[Path] = None,
              max_minutes: Optional[float] = None, limit: Optional[int] = None,
              pause: float = PAUSE_S) -> RunReport:
    """Buy the plan's snapshots in order until done or a cap is reached.

    Stops early after MAX_CONSECUTIVE_ERRORS failed calls in a row. A
    response without credit headers (a timeout, a dropped connection) is
    counted as having cost the full expected price, against this run's cap
    and the account's remaining credits, since the API may have billed it."""
    cost = call_cost(markets, bookmakers)
    report = RunReport()
    started = time.monotonic()
    streak = 0                           # failed calls in a row
    for p in plan:
        if is_covered(p.requested_ts, done):
            report.skipped += 1
            continue
        if streak >= MAX_CONSECUTIVE_ERRORS:
            report.stopped = (f"{streak} failed calls in a row: stopping before "
                              f"{iso_z(p.requested_ts)} (see the log)")
            break
        if limit is not None and report.calls >= limit:
            report.stopped = f"--limit {limit} calls reached"
            break
        if max_minutes is not None and time.monotonic() - started > max_minutes * 60:
            report.stopped = f"--max-minutes {max_minutes:g} reached"
            break
        why = budget.blocks(cost)
        if why:
            report.stopped = f"stopped before {iso_z(p.requested_ts)}: {why}"
            break
        body, headers, status = getter(HIST_PATH, {
            "date": iso_z(p.requested_ts), "markets": markets,
            "bookmakers": bookmakers, "oddsFormat": "american"})
        report.calls += 1
        charged = _int_header(headers, "x-requests-last")
        remaining = _int_header(headers, "x-requests-remaining")
        if charged is None:
            charged = cost
            logger.warning(f"No x-requests-last header for {iso_z(p.requested_ts)}: "
                           f"counting the expected {cost} credits as spent")
        if remaining is not None:
            budget.remaining = remaining
        elif budget.remaining is not None:
            budget.remaining -= charged
        budget.spent += charged
        report.credits += charged
        fetch = {"requested_ts": _naive(p.requested_ts), "purpose": p.purpose,
                 "season": p.season, "markets": markets, "bookmakers": bookmakers,
                 "snapshot_ts": None, "next_ts": None, "credits": charged,
                 "n_events": 0, "n_rows": 0, "status": "error"}
        if body is None or not isinstance(body, dict):
            report.errors += 1
            streak += 1
            saver([], fetch)
            if status in STOP_STATUSES:
                report.stopped = f"HTTP {status}: every further call would fail"
                break
            continue
        # The credits are spent: keep the raw copy first, then parse and
        # store. Whatever fails after this point, the call is logged.
        if raw_dir is not None:
            try:
                save_raw(raw_dir, raw_stem(p.purpose, p.requested_ts, markets, bookmakers), body)
            except Exception as e:
                logger.error(f"Raw copy of {iso_z(p.requested_ts)} not saved: "
                             f"{type(e).__name__}: {_redact(e)[:300]}")
        try:
            rows, update, stats = parse_body(body, p.requested_ts, games)
            fetch.update(update)
            saver(rows, fetch)
        except Exception as e:
            report.errors += 1
            streak += 1
            fetch.update(n_rows=0, status="paid_unparsed")
            logger.error(f"{p.purpose} {p.season} {iso_z(p.requested_ts)}: the paid response "
                         f"could not be parsed or stored ({type(e).__name__}: "
                         f"{_redact(e)[:300]}); logged as paid_unparsed with {charged} "
                         f"credits, and `reparse` loads it from the raw copy")
            try:
                saver([], fetch)
            except Exception:
                logger.error("The fetch log could not be written either: stopping, so no "
                             "further call is bought without being logged")
                raise
            done.append(fetch)
            continue
        if fetch["status"] == "error":
            report.errors += 1
            streak += 1
            logger.error(f"{p.purpose} {p.season} {iso_z(p.requested_ts)}: the response has "
                         f"no snapshot timestamp; logged as error ({charged} credits), "
                         f"to be retried")
            continue
        done.append(fetch)
        streak = 0
        report.rows += len(rows)
        agg = report.by_purpose.setdefault(f"{p.purpose} {p.season}",
                                           {"calls": 0, "credits": 0, "rows": 0})
        agg["calls"] += 1
        agg["credits"] += charged
        agg["rows"] += len(rows)
        logger.info(f"{p.purpose} {p.season} {p.game_date} asked {iso_z(p.requested_ts)} "
                    f"got {body.get('timestamp')}: {stats['events']} events "
                    f"({stats['in_play']} in play dropped, {stats['unmatched']} unmatched), "
                    f"{len(rows)} rows, {charged} credits, "
                    f"{budget.remaining if budget.remaining is not None else '?'} left")
        if pause:
            time.sleep(pause)
    return report


def remaining_credits() -> Optional[int]:
    """x-requests-remaining from the free /sports endpoint."""
    _body, headers, _status = http_get("/sports", {})
    return _int_header(headers, "x-requests-remaining")


def parse_steps(raw: str) -> List[Tuple[str, int]]:
    """'close:20242025,morning:20242025' -> [('close', 20242025), ...]."""
    steps = []
    for part in _key_list(raw):
        purpose, _, season = part.partition(":")
        if purpose not in ("close", "morning") or not season.isdigit() or len(season) != 8:
            raise argparse.ArgumentTypeError(
                f"{part!r}: expected close:<season> or morning:<season>, e.g. close:20242025")
        steps.append((purpose, int(season)))
    return steps


DEFAULT_STEPS = "close:20242025,morning:20242025,close:20232024,close:20222023"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ingestion.odds_history",
        description="Past h2h and totals prices from The Odds API's historical "
                    "endpoint (paid key; 10 credits x markets per call with up to "
                    "10 named books). `plan` prints what a fetch would buy and "
                    "costs nothing; `fetch` buys it, resumably, and never buys a "
                    "snapshot twice; `starts` fills old seasons' puck-drop times "
                    "from the free NHL schedule.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "fetch"):
        p = sub.add_parser(name)
        p.add_argument("--steps", default=None, type=parse_steps,
                       help=f"purpose:season pairs in priority order (default {DEFAULT_STEPS})")
        p.add_argument("--markets", default=DEFAULT_MARKETS)
        p.add_argument("--bookmakers", default=",".join(DEFAULT_BOOKMAKERS))
        p.add_argument("--same-books-only", action="store_true",
                       help="count a snapshot as bought only if it was bought with "
                            "exactly these books (default: any books). Re-buys every "
                            "snapshot bought with other books, so it needs explicit --steps")
    f = sub.choices["fetch"]
    f.add_argument("--max-credits", type=int, required=True,
                   help="most credits this run may spend")
    f.add_argument("--cap-total", type=int, default=None,
                   help="most credits all logged fetches together may reach")
    f.add_argument("--reserve", type=int, default=DEFAULT_RESERVE,
                   help=f"stop when the account would drop below this many credits "
                        f"(default {DEFAULT_RESERVE})")
    f.add_argument("--max-minutes", type=float, default=8.0)
    f.add_argument("--allow-unknown-remaining", action="store_true",
                   help="fetch even when the account's remaining credits cannot be "
                        "read first (the --reserve floor is then unchecked until a "
                        "response reports it)")
    f.add_argument("--limit", type=int, default=None, help="most calls this run")
    f.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR),
                   help="where to keep a gzipped copy of every paid response")
    s = sub.add_parser("starts")
    s.add_argument("seasons", nargs="+", type=int)
    sub.add_parser("rematch", help="match stored rows that have no game yet")
    r = sub.add_parser("reparse", help="load paid_unparsed fetches from their raw copies "
                                       "(no API call)")
    r.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    args = parser.parse_args(argv)

    if args.cmd == "starts":             # writes raw.games only
        for season in args.seasons:
            fill_start_times(season)
        return 0
    if args.cmd in ("plan", "fetch"):
        if args.same_books_only and args.steps is None:
            parser.error("--same-books-only re-buys every snapshot already bought with "
                         "other books, so it needs explicit --steps naming only the "
                         "seasons to buy (e.g. --steps close:20232024)")
        if args.steps is None:
            args.steps = parse_steps(DEFAULT_STEPS)
    ensure_tables()
    if args.cmd == "rematch":
        print(f"Matched {rematch_unmatched()} row(s)")
        return 0
    if args.cmd == "reparse":
        out = reparse_unparsed(Path(args.raw_dir))
        print(f"paid_unparsed fetches {out['unparsed']}: loaded {out['loaded']} "
              f"({out['rows']} rows), raw copy missing {out['missing']}, "
              f"still failing {out['failed']}")
        if out["rows"]:
            rematch_unmatched()
        return 0

    bookmakers = ",".join(_key_list(args.bookmakers))
    markets = ",".join(_key_list(args.markets))
    seasons = sorted({s for _, s in args.steps})
    games = load_games(seasons)
    missing = [g for g in games if g["start_time_utc"] is None]
    if missing:
        logger.warning(f"{len(missing)} game(s) have no start time and are left out of "
                       f"the plan: run `python -m ingestion.odds_history starts "
                       f"{' '.join(map(str, seasons))}` first")
    plan = build_plan(args.steps, games)
    done = load_done(markets, bookmakers, same_books=args.same_books_only)
    cost = call_cost(markets, bookmakers)
    print(f"Plan ({markets}; {len(_key_list(bookmakers))} books: {bookmakers}; "
          f"{cost} credits a call):")
    for line in describe_plan(plan, done, cost):
        print(line)
    if args.cmd == "plan":
        return 0

    budget = Budget(max_credits=args.max_credits, cap_total=args.cap_total,
                    reserve=args.reserve, logged_before=credits_logged(),
                    remaining=remaining_credits())
    print(f"Account credits remaining before the run: "
          f"{budget.remaining if budget.remaining is not None else 'unknown'}")
    if budget.remaining is None and not args.allow_unknown_remaining:
        print("Not fetching: the account's remaining credits could not be read (no key, "
              "no connection, or no x-requests-remaining header), so the --reserve "
              "floor cannot be checked. Fix that, or pass --allow-unknown-remaining "
              "to rely on --max-credits alone.")
        return 2
    report = run_fetch(plan, markets, bookmakers, budget, games, done,
                       raw_dir=Path(args.raw_dir), max_minutes=args.max_minutes,
                       limit=args.limit)
    print(f"Calls {report.calls}, credits {report.credits}, rows {report.rows}, "
          f"errors {report.errors}, already bought {report.skipped}; account "
          f"remaining {budget.remaining}")
    for k, v in report.by_purpose.items():
        print(f"  {k}: {v}")
    if report.stopped:
        print(f"Stopped: {report.stopped}")
    if report.rows:
        rematch_unmatched()
    return 0


if __name__ == "__main__":
    main()
