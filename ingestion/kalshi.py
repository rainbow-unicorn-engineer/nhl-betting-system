"""
ingestion/kalshi.py
Kalshi's NHL game-winner markets and their price history, from Kalshi's
free public API (no key), into raw.kalshi_markets and raw.kalshi_candles.

Terms:
- Kalshi → a US exchange regulated by the CFTC, where people trade with
  each other instead of against a bookmaker. One of the two venues a Texas
  bettor can legally use.
- Contract / market → "Will <team> win?". It pays $1 if yes and $0 if no,
  so its price in dollars (0.01 to 0.99) reads directly as the market's
  probability. Each game (an "event") has two markets, one per team.
  They behave like a moneyline (→ a bet on who wins, overtime and shootout
  included).
- Bid / ask → the best price a buyer is offering and the best price a seller
  will accept right now. We buy YES at the ask. Mid → halfway between.
  An empty order book shows as bid 0.00 and ask 1.00: no price, not a price.
- Candle (candlestick) → the open, high, low and close of a price over one
  period (here 60 minutes or 1 minute), for the bid, the ask and the traded
  price, plus the volume traded in that period.
- Settlement → the final payout: 1.00 to the winner's YES, 0.00 to the
  loser's. A game that is cancelled, or not started within 48 hours of its
  scheduled time, settles both sides at a "fair price" (result 'scalar';
  seen once, LA at CBJ on 2026-01-26, settled 0.48 / 0.52).
- Closing price / CLV → the last price before puck drop; closing-line value
  is whether a bet got a better price than that. Kalshi trades DURING the
  game too, so any candle that ends after start_time_utc is in-play and
  must never be used as a pre-game price.
- Fees → Kalshi charges a taker fee of 0.07 x p x (1 - p) per $1 contract
  (p = price). Prices stored here are before fees.

The API (https://api.elections.kalshi.com/trade-api/v2, checked 2026-10-04):
  GET /markets?series_ticker=KXNHLGAME            markets settled after the
                                                  historical cutoff, and open ones
  GET /historical/markets?series_ticker=KXNHLGAME markets settled before the
                                                  cutoff (GET /historical/cutoff;
                                                  2026-08-05 when checked)
  GET /series/KXNHLGAME/markets/{ticker}/candlesticks   (live markets)
  GET /historical/markets/{ticker}/candlesticks         (historical markets)
      ?start_ts=&end_ts=&period_interval=60|1  (unix seconds)
Historical candles name their fields close/volume/open_interest; live ones
close_dollars/volume_fp/open_interest_fp. Both are read.
On 2026-10-04 the series held 1,682 events (3,364 markets), 2025-04-19 to
2026-10-11: the 2024-25 playoffs, all of 2025-26 including preseason, and
2026-27 so far. Every event has exactly two markets.

Tickers: event KXNHLGAME-25DEC10DETCGY = the game of 2025-12-10 (the US
schedule date), DET at CGY (away team first); market ...-DET backs Detroit.
Kalshi abbreviates four teams differently from the NHL (LA, TB, NJ, SJ for
LAK, TBL, NJD, SJS; a few early markets use the NHL codes). map_games()
matches each event to raw.games by the two teams and the date (exact date
first, then one day either side); preseason games are not in raw.games and
stay unmatched (game_id NULL).

What is stored:
- raw.kalshi_markets: one row per market (both listings), refreshed on
  every run, with the full API object in `raw`.
- raw.kalshi_candles: for each settled market, hourly candles over its
  whole life (period_minutes 60) and, for markets matched to a game,
  1-minute candles over the CLOSE_WINDOW_MIN (180) minutes up to
  start_time_utc (period_minutes 1). closing_lines() reads the last
  1-minute candle at or before puck drop: Kalshi's free closing price.
Candles are fetched once per market, after it settles; the market row's
candles_status logs it (ok, empty, error; error is retried every run).

Pace: at most one request every MIN_INTERVAL_S (0.2 s, 5 a second; Kalshi's
basic tier allows 20 reads a second), retried on 429 and 5xx. A full
backfill is about 4 listing requests plus 2 candle requests per market.

Acceptance checks (written 2026-10-04, before the backfill ran; each miss
is reported):
  K1  every event has exactly two markets;
  K2  at least 95% of 2025-26 and 2026-27 regular-season and playoff games
      already played have a matched Kalshi event (preseason events are
      expected to stay unmatched);
  K3  for every matched market settled 'yes' or 'no', the result agrees
      with raw.games (the YES team won) — 100%, any disagreement listed;
  K4  at least 90% of matched settled games have a usable two-sided
      pre-game close (bid > 0 and ask < 1 on both markets).

`python -m ingestion.kalshi --help`.
"""
import argparse
import json
import logging
import re
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from sqlalchemy import text

from config.settings import engine
from ingestion.polite import PoliteClient

logger = logging.getLogger("nhl.ingestion.kalshi")

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXNHLGAME"
MIN_INTERVAL_S = 0.2
PAGE_LIMIT = 1000
CLOSE_WINDOW_MIN = 180
HOURLY, MINUTE = 60, 1
FINAL_RESULTS = ("yes", "no", "scalar")

# Kalshi code → NHL code, where they differ
KALSHI_TO_NHL = {"LA": "LAK", "TB": "TBL", "NJ": "NJD", "SJ": "SJS"}

_MONTHS = {m: i for i, m in enumerate(
    ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"), 1)}
_EVENT_RE = re.compile(r"^(?P<series>[A-Z0-9]+)-(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<dd>\d{2})"
                       r"(?P<teams>[A-Z]+)$")

# The same DDL is in config/migrate.py (TABLES) and db/schema.sql.
DDL = [
    """
        CREATE TABLE IF NOT EXISTS raw.kalshi_markets (
            ticker              VARCHAR(64) PRIMARY KEY,
            event_ticker        VARCHAR(64) NOT NULL,
            series_ticker       VARCHAR(32) NOT NULL,
            game_id             BIGINT REFERENCES raw.games(game_id),
            event_date          DATE,
            kalshi_team         VARCHAR(8),
            team                VARCHAR(3),
            is_home             BOOLEAN,
            title               TEXT,
            yes_sub_title       VARCHAR(80),
            status              VARCHAR(16),
            result              VARCHAR(10),
            settlement_value    NUMERIC(6,4),
            settlement_ts       TIMESTAMPTZ,
            open_time           TIMESTAMPTZ,
            close_time          TIMESTAMPTZ,
            expected_expiration TIMESTAMPTZ,
            last_price          NUMERIC(6,4),
            volume              NUMERIC(18,2),
            open_interest       NUMERIC(18,2),
            source              VARCHAR(10) NOT NULL,          -- live, historical
            raw                 JSONB NOT NULL,
            listed_at           TIMESTAMP NOT NULL,
            candles_status      VARCHAR(10),                   -- NULL = not fetched; ok, empty, error
            candles_attempts    SMALLINT NOT NULL DEFAULT 0,
            candles_problem     TEXT,
            candles_fetched_at  TIMESTAMP
        )""",
    "CREATE INDEX IF NOT EXISTS idx_kalshi_markets_game ON raw.kalshi_markets(game_id)",
    "CREATE INDEX IF NOT EXISTS idx_kalshi_markets_event ON raw.kalshi_markets(event_ticker)",
    """
        CREATE TABLE IF NOT EXISTS raw.kalshi_candles (
            ticker          VARCHAR(64) NOT NULL REFERENCES raw.kalshi_markets(ticker),
            period_minutes  SMALLINT NOT NULL,
            end_period_ts   TIMESTAMPTZ NOT NULL,
            yes_bid_open    NUMERIC(6,4),
            yes_bid_high    NUMERIC(6,4),
            yes_bid_low     NUMERIC(6,4),
            yes_bid_close   NUMERIC(6,4),
            yes_ask_open    NUMERIC(6,4),
            yes_ask_high    NUMERIC(6,4),
            yes_ask_low     NUMERIC(6,4),
            yes_ask_close   NUMERIC(6,4),
            price_open      NUMERIC(6,4),
            price_high      NUMERIC(6,4),
            price_low       NUMERIC(6,4),
            price_close     NUMERIC(6,4),
            price_mean      NUMERIC(6,4),
            volume          NUMERIC(18,2),
            open_interest   NUMERIC(18,2),
            PRIMARY KEY (ticker, period_minutes, end_period_ts)
        )""",
]

_ensured = False


def ensure_tables(db=None) -> None:
    """Apply DDL when a table is missing (information_schema first). Once
    per process."""
    global _ensured
    if _ensured and db is None:
        return
    with (db or engine).begin() as conn:
        n = conn.execute(text("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'raw' AND table_name IN ('kalshi_markets', 'kalshi_candles')
        """)).scalar()
        if n != 2:
            logger.info("Schema upgrade: raw.kalshi_markets / raw.kalshi_candles")
            for stmt in DDL:
                conn.execute(text(stmt))
    if db is None:
        _ensured = True


# ── Pure parsing ──────────────────────────────────────────────────

def parse_event_ticker(event_ticker: str) -> Optional[Tuple[date, str]]:
    """'KXNHLGAME-25DEC10DETCGY' → (date(2025, 12, 10), 'DETCGY'); None if
    the ticker doesn't follow that pattern."""
    m = _EVENT_RE.match(event_ticker or "")
    if not m or m["mon"] not in _MONTHS:
        return None
    try:
        d = date(2000 + int(m["yy"]), _MONTHS[m["mon"]], int(m["dd"]))
    except ValueError:
        return None
    return d, m["teams"]


def market_code(ticker: str) -> Optional[str]:
    """'KXNHLGAME-25DEC10DETCGY-DET' → 'DET'."""
    parts = (ticker or "").rsplit("-", 2)
    return parts[-1] if len(parts) == 3 and parts[-1] else None


def nhl_code(kalshi_code: Optional[str]) -> Optional[str]:
    if not kalshi_code:
        return None
    return KALSHI_TO_NHL.get(kalshi_code, kalshi_code)


def _dollars(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ts(value) -> Optional[datetime]:
    """ISO time → aware UTC; None for missing or Go's zero time."""
    if not value or str(value).startswith("0001-"):
        return None
    try:
        t = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return (t if t.tzinfo else t.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def market_row(m: dict, source: str) -> dict:
    """A raw.kalshi_markets row (without game_id/team/is_home) from one API
    market object."""
    parsed = parse_event_ticker(m.get("event_ticker", ""))
    return {
        "ticker": m["ticker"],
        "event_ticker": m.get("event_ticker"),
        "series_ticker": (m.get("event_ticker") or "").split("-")[0] or SERIES,
        "event_date": parsed[0] if parsed else None,
        "kalshi_team": market_code(m["ticker"]),
        "title": m.get("title"),
        "yes_sub_title": (m.get("yes_sub_title") or None),
        "status": m.get("status"),
        "result": (m.get("result") or None),
        "settlement_value": _dollars(m.get("settlement_value_dollars")),
        "settlement_ts": _ts(m.get("settlement_ts")),
        "open_time": _ts(m.get("open_time")),
        "close_time": _ts(m.get("close_time")),
        "expected_expiration": _ts(m.get("expected_expiration_time")),
        "last_price": _dollars(m.get("last_price_dollars")),
        "volume": _dollars(m.get("volume_fp", m.get("volume"))),
        "open_interest": _dollars(m.get("open_interest_fp", m.get("open_interest"))),
        "source": source,
        "raw": json.dumps(m, sort_keys=True),
    }


def _ohlc(block: Optional[dict], key: str) -> Optional[float]:
    if not isinstance(block, dict):
        return None
    return _dollars(block.get(f"{key}_dollars", block.get(key)))


def parse_candles(body: dict, ticker: str, period_minutes: int) -> List[dict]:
    """raw.kalshi_candles rows from one candlesticks response (either the
    live or the historical field names)."""
    out: Dict[int, dict] = {}
    for c in (body or {}).get("candlesticks") or []:
        end = c.get("end_period_ts")
        try:
            end = int(end)
        except (TypeError, ValueError):
            continue
        bid, ask, price = c.get("yes_bid"), c.get("yes_ask"), c.get("price")
        out[end] = {
            "ticker": ticker, "period_minutes": int(period_minutes),
            "end_period_ts": datetime.fromtimestamp(end, timezone.utc),
            "yes_bid_open": _ohlc(bid, "open"), "yes_bid_high": _ohlc(bid, "high"),
            "yes_bid_low": _ohlc(bid, "low"), "yes_bid_close": _ohlc(bid, "close"),
            "yes_ask_open": _ohlc(ask, "open"), "yes_ask_high": _ohlc(ask, "high"),
            "yes_ask_low": _ohlc(ask, "low"), "yes_ask_close": _ohlc(ask, "close"),
            "price_open": _ohlc(price, "open"), "price_high": _ohlc(price, "high"),
            "price_low": _ohlc(price, "low"), "price_close": _ohlc(price, "close"),
            "price_mean": _ohlc(price, "mean"),
            "volume": _dollars(c.get("volume_fp", c.get("volume"))),
            "open_interest": _dollars(c.get("open_interest_fp", c.get("open_interest"))),
        }
    return [out[k] for k in sorted(out)]


def map_games(rows: Sequence[dict], games: Sequence[dict]) -> Dict[str, dict]:
    """{event_ticker: {'game_id', 'home_team', 'away_team', 'start_time_utc'}}
    for each event whose two markets' teams and date match one game in
    `games` (dicts with game_id, date, home_team, away_team,
    start_time_utc). Exact date first, then one day either side; an event
    with two candidate games on the same offset stays unmatched."""
    by_event: Dict[str, dict] = {}
    for r in rows:
        ev = by_event.setdefault(r["event_ticker"], {"date": r.get("event_date"), "teams": set()})
        code = nhl_code(r.get("kalshi_team"))
        if code:
            ev["teams"].add(code)
    by_teams: Dict[frozenset, List[dict]] = {}
    for g in games:
        by_teams.setdefault(frozenset((g["home_team"], g["away_team"])), []).append(g)
    out: Dict[str, dict] = {}
    for event, ev in by_event.items():
        if ev["date"] is None or len(ev["teams"]) != 2:
            continue
        candidates = by_teams.get(frozenset(ev["teams"]), [])
        for offset in (0, 1):
            hits = [g for g in candidates if abs((g["date"] - ev["date"]).days) == offset]
            if len(hits) == 1:
                out[event] = hits[0]
                break
            if len(hits) > 1:
                break
    return out


def candle_windows(row: dict, start_time_utc: Optional[datetime]) -> List[Tuple[int, int, int]]:
    """[(period_minutes, start_ts, end_ts)] to request for one settled
    market: hourly over its life; 1-minute for the CLOSE_WINDOW_MIN minutes
    up to puck drop when the game's start time is known."""
    out = []
    lo = row.get("open_time")
    hi = row.get("settlement_ts") or row.get("close_time")
    if lo and hi and hi > lo:
        out.append((HOURLY, int(lo.timestamp()), int(hi.timestamp())))
    if start_time_utc is not None:
        end = int(start_time_utc.timestamp())
        out.append((MINUTE, end - CLOSE_WINDOW_MIN * 60, end))
    return out


# ── Network ───────────────────────────────────────────────────────

def _client() -> PoliteClient:
    return PoliteClient("Kalshi", min_interval_s=MIN_INTERVAL_S, timeout_s=60)


def list_markets(client: PoliteClient, source: str) -> Optional[List[dict]]:
    """Every KXNHLGAME market under one listing ('live' or 'historical');
    None when a page failed (nothing is then stored from this listing)."""
    path = "/markets" if source == "live" else "/historical/markets"
    out, cursor = [], None
    while True:
        params = {"series_ticker": SERIES, "limit": PAGE_LIMIT}
        if cursor:
            params["cursor"] = cursor
        reply = client.get_json(BASE_URL + path, params)
        if reply.status != "ok" or not isinstance(reply.body, dict):
            logger.error(f"Kalshi {source} listing failed: {reply.problem}")
            return None
        page = reply.body.get("markets") or []
        out.extend(page)
        cursor = reply.body.get("cursor")
        if not cursor or not page:
            return out


def candle_url(ticker: str, source: str) -> str:
    if source == "historical":
        return f"{BASE_URL}/historical/markets/{ticker}/candlesticks"
    return f"{BASE_URL}/series/{SERIES}/markets/{ticker}/candlesticks"


def fetch_candles(client: PoliteClient, row: dict,
                  start_time_utc: Optional[datetime]) -> Tuple[str, List[dict], Optional[str]]:
    """(status, candle rows, problem) for one market: 'ok' when every window
    answered (rows may come from either the live or the historical path;
    a 404 on one is retried on the other), 'empty' when all answered with
    no candles, 'error' otherwise."""
    rows: List[dict] = []
    windows = candle_windows(row, start_time_utc)
    if not windows:
        return "empty", [], "no open/close time"
    for period, lo, hi in windows:
        params = {"start_ts": lo, "end_ts": hi, "period_interval": period}
        reply = client.get_json(candle_url(row["ticker"], row["source"]), params)
        if reply.status == "not_found":
            other = "live" if row["source"] == "historical" else "historical"
            reply = client.get_json(candle_url(row["ticker"], other), params)
        if reply.status != "ok" or not isinstance(reply.body, dict):
            return "error", [], f"{period}-minute candles: {reply.problem}"
        rows.extend(parse_candles(reply.body, row["ticker"], period))
    return ("ok" if rows else "empty"), rows, None


# ── Database ──────────────────────────────────────────────────────

_MARKET_COLS = ["ticker", "event_ticker", "series_ticker", "game_id", "event_date",
                "kalshi_team", "team", "is_home", "title", "yes_sub_title", "status",
                "result", "settlement_value", "settlement_ts", "open_time", "close_time",
                "expected_expiration", "last_price", "volume", "open_interest", "source",
                "raw"]

UPSERT_MARKET = text(f"""
    INSERT INTO raw.kalshi_markets ({", ".join(_MARKET_COLS)}, listed_at)
    VALUES ({", ".join(":" + c if c != "raw" else "CAST(:raw AS jsonb)" for c in _MARKET_COLS)},
            (now() AT TIME ZONE 'UTC'))
    ON CONFLICT (ticker) DO UPDATE SET
        {", ".join(f"{c} = EXCLUDED.{c}" for c in _MARKET_COLS
                   if c not in ("ticker", "game_id", "team", "is_home"))},
        game_id = COALESCE(EXCLUDED.game_id, raw.kalshi_markets.game_id),
        team = COALESCE(EXCLUDED.team, raw.kalshi_markets.team),
        is_home = COALESCE(EXCLUDED.is_home, raw.kalshi_markets.is_home),
        listed_at = EXCLUDED.listed_at
""")

_CANDLE_COLS = ["ticker", "period_minutes", "end_period_ts", "yes_bid_open", "yes_bid_high",
                "yes_bid_low", "yes_bid_close", "yes_ask_open", "yes_ask_high", "yes_ask_low",
                "yes_ask_close", "price_open", "price_high", "price_low", "price_close",
                "price_mean", "volume", "open_interest"]

INSERT_CANDLES = text(f"""
    INSERT INTO raw.kalshi_candles ({", ".join(_CANDLE_COLS)})
    VALUES ({", ".join(":" + c for c in _CANDLE_COLS)})
""")

UPDATE_CANDLE_STATUS = text("""
    UPDATE raw.kalshi_markets SET
        candles_status = CASE WHEN :status = 'error' AND candles_status IN ('ok', 'empty')
                              THEN candles_status ELSE :status END,
        candles_attempts = LEAST(candles_attempts + 1, 32000),
        candles_problem = :problem,
        candles_fetched_at = (now() AT TIME ZONE 'UTC')
    WHERE ticker = :ticker
""")


def load_games(lo: date, hi: date, db=None) -> List[dict]:
    with (db or engine).connect() as conn:
        return [dict(r) for r in conn.execute(text("""
            SELECT game_id, date, home_team, away_team, start_time_utc FROM raw.games
            WHERE game_type IN (2, 3) AND date BETWEEN :lo AND :hi
        """), {"lo": lo - timedelta(days=2), "hi": hi + timedelta(days=2)}).mappings()]


def attach_games(rows: List[dict], games: Sequence[dict]) -> int:
    """Set game_id, team and is_home on each row in place; returns the
    number of matched events."""
    matched = map_games(rows, games)
    for r in rows:
        g = matched.get(r["event_ticker"])
        code = nhl_code(r.get("kalshi_team"))
        r["team"] = code if code and len(code) <= 3 else None
        if g:
            r["game_id"] = g["game_id"]
            r["is_home"] = (code == g["home_team"])
        else:
            r["game_id"], r["is_home"] = None, None
    return len(matched)


def store_markets(rows: List[dict], db=None) -> None:
    if not rows:
        return
    with (db or engine).begin() as conn:
        conn.execute(UPSERT_MARKET, rows)


def store_candles(ticker: str, status: str, candles: List[dict],
                  problem: Optional[str] = None, db=None) -> None:
    """Replace a market's candles (only when some came back) and log the
    fetch, in one transaction."""
    with (db or engine).begin() as conn:
        if candles:
            conn.execute(text("DELETE FROM raw.kalshi_candles WHERE ticker = :t"),
                         {"t": ticker})
            conn.execute(INSERT_CANDLES, candles)
        conn.execute(UPDATE_CANDLE_STATUS, {"ticker": ticker, "status": status,
                                            "problem": problem})


def markets_needing_candles(retry_empty: bool = False, limit: Optional[int] = None,
                            db=None) -> List[dict]:
    sql = """
        SELECT m.ticker, m.source, m.open_time, m.close_time, m.settlement_ts,
               g.start_time_utc
        FROM raw.kalshi_markets m
        LEFT JOIN raw.games g ON g.game_id = m.game_id
        WHERE m.result IN ('yes', 'no', 'scalar')
          AND (m.candles_status IS NULL OR m.candles_status = 'error'
               OR (CAST(:retry_empty AS boolean) AND m.candles_status = 'empty'))
        ORDER BY m.event_date, m.ticker
    """
    params = {"retry_empty": bool(retry_empty)}
    if limit:
        sql += " LIMIT :limit"
        params["limit"] = int(limit)
    with (db or engine).connect() as conn:
        return [dict(r) for r in conn.execute(text(sql), params).mappings()]


# ── Runs ──────────────────────────────────────────────────────────

def refresh_markets(include_historical: bool, client: Optional[PoliteClient] = None,
                    db=None) -> Dict[str, int]:
    """List markets (the live listing; the historical one too when asked),
    match them to games and upsert them. A market in both listings keeps
    the historical copy (it has settled for good)."""
    ensure_tables(db)
    client = client or _client()
    by_ticker: Dict[str, dict] = {}
    counts = {"listed_live": 0, "listed_historical": 0, "markets": 0,
              "events": 0, "matched_events": 0, "listing_failed": 0}
    for source in (("live", "historical") if include_historical else ("live",)):
        markets = list_markets(client, source)
        if markets is None:
            counts["listing_failed"] += 1
            continue
        counts[f"listed_{source}"] = len(markets)
        for m in markets:
            if m.get("ticker") and m.get("event_ticker"):
                by_ticker[m["ticker"]] = market_row(m, source)
    rows = list(by_ticker.values())
    dates = [r["event_date"] for r in rows if r["event_date"]]
    if rows and dates:
        counts["matched_events"] = attach_games(rows, load_games(min(dates), max(dates), db))
    else:
        for r in rows:
            r.update(game_id=None, team=nhl_code(r.get("kalshi_team")), is_home=None)
    store_markets(rows, db)
    counts["markets"] = len(rows)
    counts["events"] = len({r["event_ticker"] for r in rows})
    logger.info(f"Kalshi: {counts['markets']} markets in {counts['events']} events "
                f"stored, {counts['matched_events']} events matched to raw.games")
    return counts


def fetch_all_candles(retry_empty: bool = False, limit: Optional[int] = None,
                      client: Optional[PoliteClient] = None, db=None,
                      max_consecutive_errors: int = 25) -> Dict[str, int]:
    ensure_tables(db)
    client = client or _client()
    todo = markets_needing_candles(retry_empty, limit, db)
    counts = {"markets": len(todo), "ok": 0, "empty": 0, "error": 0, "candles": 0,
              "stopped_early": 0}
    streak = 0
    for i, row in enumerate(todo, 1):
        status, candles, problem = fetch_candles(client, row, row.get("start_time_utc"))
        store_candles(row["ticker"], status, candles, problem, db)
        counts[status] += 1
        counts["candles"] += len(candles)
        streak = streak + 1 if status == "error" else 0
        if i % 200 == 0 or i == len(todo):
            logger.info(f"Kalshi candles {i}/{len(todo)}: {counts['ok']} ok, "
                        f"{counts['empty']} empty, {counts['error']} error, "
                        f"{counts['candles']:,} candles")
        if streak >= max_consecutive_errors:
            logger.error(f"Kalshi candles: {streak} failures in a row; stopping "
                         f"(a re-run resumes from here)")
            counts["stopped_early"] = 1
            break
    return counts


def run(include_historical: bool = False, retry_empty: bool = False,
        limit: Optional[int] = None) -> Dict[str, int]:
    """Refresh the market list, then fetch candles for every settled market
    without them. The daily form lists only live markets (a few requests);
    --backfill adds the historical listing."""
    client = _client()
    counts = refresh_markets(include_historical, client)
    counts.update({f"candles_{k}": v for k, v in
                   fetch_all_candles(retry_empty, limit, client).items()})
    return counts


# ── Reading ───────────────────────────────────────────────────────

CLOSING_SQL = """
    WITH last_pre AS (
        SELECT DISTINCT ON (c.ticker) c.ticker, c.end_period_ts,
               c.yes_bid_close, c.yes_ask_close, c.price_close
        FROM raw.kalshi_candles c
        JOIN raw.kalshi_markets m ON m.ticker = c.ticker
        JOIN raw.games g ON g.game_id = m.game_id
        WHERE c.period_minutes = 1 AND c.end_period_ts <= g.start_time_utc
        ORDER BY c.ticker, c.end_period_ts DESC
    )
    SELECT m.game_id, g.season, g.date, g.start_time_utc, m.ticker, m.team, m.is_home,
           m.result, m.settlement_value, l.end_period_ts AS close_ts,
           l.yes_bid_close AS bid, l.yes_ask_close AS ask, l.price_close AS last_trade
    FROM raw.kalshi_markets m
    JOIN raw.games g ON g.game_id = m.game_id
    LEFT JOIN last_pre l ON l.ticker = m.ticker
    WHERE (:season IS NULL OR g.season = CAST(:season AS integer))
    ORDER BY g.date, m.game_id, m.is_home DESC
"""


def closing_lines(season: Optional[int] = None, db=None) -> List[dict]:
    """Per matched market: the last 1-minute candle ending at or before
    puck drop (bid, ask, last trade). Never an in-play price."""
    with (db or engine).connect() as conn:
        return [dict(r) for r in conn.execute(text(CLOSING_SQL), {"season": season}).mappings()]


def summarize_closes(lines: Iterable[dict]) -> Dict[int, dict]:
    """Per season: games with a usable two-sided close (bid > 0, ask < 1 on
    both markets), the median bid-ask spread, the median overround (→ the
    two asks' sum minus 1: what buying both sides would cost over $1, the
    exchange's equivalent of the vig), and the median minutes between the
    close candle and puck drop."""
    import statistics as st
    games: Dict[int, List[dict]] = {}
    for r in lines:
        games.setdefault(r["game_id"], []).append(r)
    out: Dict[int, dict] = {}
    for gid, rs in games.items():
        season = rs[0]["season"]
        s = out.setdefault(season, {"games": 0, "with_close": 0, "spreads": [],
                                    "overrounds": [], "lead_min": []})
        s["games"] += 1
        ok = [r for r in rs if r["bid"] is not None and r["ask"] is not None
              and float(r["bid"]) > 0 and float(r["ask"]) < 1]
        if len(rs) == 2 and len(ok) == 2:
            s["with_close"] += 1
            s["spreads"] += [float(r["ask"]) - float(r["bid"]) for r in ok]
            s["overrounds"].append(sum(float(r["ask"]) for r in ok) - 1)
            s["lead_min"] += [(r["start_time_utc"] - r["close_ts"]).total_seconds() / 60
                              for r in ok]
    for season, s in out.items():
        for key in ("spreads", "overrounds", "lead_min"):
            vals = s.pop(key)
            s[f"median_{key[:-1] if key != 'lead_min' else key}"] = (
                round(st.median(vals), 4) if vals else None)
    return out


def coverage(db=None) -> List[dict]:
    ensure_tables(db)
    with (db or engine).connect() as conn:
        rows = conn.execute(text("""
            SELECT COALESCE(g.season::text, 'unmatched') AS season,
                   COUNT(DISTINCT m.event_ticker) AS events,
                   COUNT(*) AS markets,
                   COUNT(*) FILTER (WHERE m.result IN ('yes', 'no')) AS settled,
                   COUNT(*) FILTER (WHERE m.result = 'scalar') AS scalar,
                   COUNT(*) FILTER (WHERE m.candles_status = 'ok') AS candles_ok,
                   COUNT(*) FILTER (WHERE m.candles_status = 'empty') AS candles_empty,
                   COUNT(*) FILTER (WHERE m.candles_status = 'error') AS candles_error,
                   (SELECT COUNT(*) FROM raw.kalshi_candles c
                    JOIN raw.kalshi_markets m2 ON m2.ticker = c.ticker
                    LEFT JOIN raw.games g2 ON g2.game_id = m2.game_id
                    WHERE COALESCE(g2.season::text, 'unmatched')
                          = COALESCE(g.season::text, 'unmatched')) AS candles
            FROM raw.kalshi_markets m LEFT JOIN raw.games g ON g.game_id = m.game_id
            GROUP BY 1 ORDER BY 1
        """)).mappings().all()
    return [dict(r) for r in rows]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ingestion.kalshi",
        description="Load Kalshi's NHL game-winner markets (free public API, no "
                    "key) into raw.kalshi_markets, and hourly plus pre-game "
                    "1-minute price candles into raw.kalshi_candles. Safe to re-run.")
    parser.add_argument("--backfill", action="store_true",
                        help="also list markets settled before Kalshi's historical "
                             "cutoff (the full history); default: live listing only")
    parser.add_argument("--limit", type=int, default=None,
                        help="fetch candles for at most this many markets this run")
    parser.add_argument("--retry-empty", action="store_true",
                        help="also re-fetch markets whose candles came back empty")
    parser.add_argument("--report", action="store_true",
                        help="print coverage and closing-price summary; fetch nothing")
    args = parser.parse_args(argv)
    if args.report:
        for r in coverage():
            print(r)
        for season, s in sorted(summarize_closes(closing_lines()).items()):
            print(season, s)
        return 0
    counts = run(args.backfill, args.retry_empty, args.limit)
    print(counts)
    return 1 if counts.get("candles_stopped_early") or counts.get("listing_failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
